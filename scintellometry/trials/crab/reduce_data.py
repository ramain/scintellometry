""" work in progress: need to do lofar-style waterfall and foldspec """
from __future__ import division, print_function

import numpy as np
import astropy.units as u

from scintellometry.meta.reduction import reduce, CL_parser

MAX_RMS = 2.
_fref = 610. * u.MHz  # ref. freq. for dispersion measure
PULSE_WIDTH = 6e-5 #Pulse width in seconds at 610MHz
TIME_RES = 4e-6


#def rfi_filter_raw(power):
#    nchan = power.shape[1]
#    if power.shape[-1] == 4:
#        chans = power.sum(0)[:, (0,3)].sum(-1)
#    else:
#        chans = power.sum(0)
#    nzap = int(nchan*0.3)
#    clean = power
#    for i in range(nzap):
#        zap = chans.argmax()
#        chans = np.delete(chans, zap)
#        clean = np.delete(clean, zap, 1)
#    return clean

def rfi_filter_raw(power):
    # rfi filter to remove RFI contaminated frequency channels
    # NOTE: Does not remove impulsive RFI
    # Input array in units [time, frequency, polarization]
    if power.shape[-1] == 4:
        chans = power[:, :, (0,3)].sum(-1, keepdims=True)
    else:
        chans = power
    std = chans.std(0, keepdims=True)
    ok = std < MAX_RMS*np.median(std)
    power *= ok
    return power, ok


def rfi_filter_power(power, tsr, phase, *args, **kwargs):
    # Detect and store giant pulses in each block through simple S/N test
    buff = 50 # Time range in bins to store of each giant pulse
    bscrunch = int(PULSE_WIDTH/TIME_RES) #binning factor
    if power.shape[-1] == 4:
        freq_av = power.sum(1)[:, (0,3)].sum(-1)
    else:
        freq_av = power.sum(1)
    
    #bin by factor, ~one bin per giant pulse
    #check size of new binning
    size = int(len(freq_av)/bscrunch)*bscrunch
    profile = freq_av[:size]
    profile = profile.reshape(int(len(freq_av)/bscrunch),bscrunch).sum(-1)
    
    #calculate signal to noise
    sn = (profile - profile.mean()) / profile.std()
    peaks = np.argwhere(sn > 6) #peak bins in binned profile
    peaks = peaks*bscrunch #uncrunch peaks
    #take only peaks corresponding to main/inter pulse - phase currently hardcoded
    peaks = peaks[((phase[peaks]>0.06)&(phase[peaks]<0.08))|((phase[peaks]>0.46)&(phase[peaks]<0.50))] 
    #write information
    with open('giant_pulses.txt', 'a') as f:
        f.writelines(['{0} {1} {2} {3}\n'.format(tsr[peak], sn[int(peak/bscrunch)], peak, phase[peak]) for peak in peaks])
    for peak in peaks:
        np.save('GP%s' % (tsr[peak]), power[max(peak-buff,0):min(peak+buff,power.shape[0]),:,:] )
        #np.save('pulse%s' % (tsr[peak]), sn[max(int(peak/bscrunch)-buff,0):min(int(peak/bscrunch)+buff,sn.shape[0])])
    return power


if __name__ == '__main__':
    args = CL_parser()
    args.verbose = 0 if args.verbose is None else sum(args.verbose)
    if args.fref is None:
        args.fref = _fref

    if args.rfi_filter_raw:
        args.rfi_filter_raw = rfi_filter_raw
    else:
        args.rfi_filter_raw = None

    if args.rfi_filter_power:
        args.rfi_filter_power = rfi_filter_power
    else:
        args.rfi_filter_power = None

    if args.reduction_defaults == 'gmrt':
        args.telescope = 'gmrt'
        args.nchan = 512
        args.ngate = 512
        args.ntbin = 5
        # 170 /(100.*u.MHz/6.) * 512 = 0.0052224 s = 256 bins/pulse
        args.ntw_min = 170
        args.rfi_filter_raw = None
        args.verbose += 1
    reduce(
        args.telescope, args.date, tstart=args.tstart, tend=args.tend,
        nchan=args.nchan, ngate=args.ngate, ntbin=args.ntbin,
        ntw_min=args.ntw_min, rfi_filter_raw=args.rfi_filter_raw,
        rfi_filter_power=args.rfi_filter_power,
        do_waterfall=args.waterfall, do_foldspec=args.foldspec,
        dedisperse=args.dedisperse, fref=args.fref, verbose=args.verbose)
