#!/usr/bin/env python

""" Emily's script for analysing polarization of the Crab pulsar """

import sys
import numpy as np
import matplotlib.pylab as plt

# used to center pulse profile
# just to make things look better
# use for param with dimensions (time, freq, phase)
def rollToMid(param):
    paramtemp = param.sum(0)
    parammax = np.argmax(paramtemp.sum(0))
    phasebin = paramtemp.shape[1]
    midpt = phasebin/2.
    rollfact = int(abs(parammax-midpt))
    return np.roll(param, rollfact, axis=-1)

# used to find the location of the precursor pulse
# use for param with dimensions (time, freq, phase)
def findPrecursor(param):
    paramprof = param.sum(0).sum(0)
    parammax = np.argmax(paramprof)
    phasebin = param.shape[2]
    start = int(parammax - 0.0859375*phasebin)
    end = int(parammax - 0.0234375*phasebin)
    output = range(start, end)
    return output

# function to give Stokes parameter based on waterfall file
# parameter should be input as a string e.g. 'I'
# this is used for GIANT PULSES
def stokeswf(wf, param):
    # Stokes Parameter
    xx = wf[...,0]
    Rexy = wf[...,1]
    Imxy = wf[...,2]
    yy = wf[...,3]
    I = (xx+yy)/2.
    if param == "I":
        n = (xx+yy)/2.
    if param == "Q":
        n = (xx-yy)/2.
    if param == "U":
        n = Rexy
    if param == "V":
        n = Imxy
    # Pulse Finding
    tbins = n.shape[0] # number of time bins
    ncut = I[1:(tbins-2),...] # cut to find max while avoiding edge
    pulse = np.argmax(ncut.sum(-1)) # location of giant pulse
    tstart = pulse - (0.003*tbins) # cutting around giant pulse
    tend = pulse + (0.004*tbins) # cutting around giant pulse
    n = n[tstart:tend,...] # cutting around giant pulse
    # Cleaning
    nmax = np.argmax(I.sum(-1)) # finding location of giant pulse
    n_back = n[nmax-(0.002*tbins):nmax-(0.001*tbins),...].mean(axis=0)
    n_back = n_back[np.newaxis,...] # finding off-pulse to subtract
    n = n-n_back # subtracting off-pulse
    output = n.sum(0) # summing over time
    return output

# function to give Stokes parameter based on foldspec, icount file
# parameter should be input as a string e.g. 'I'
# this is used for the PRECURSOR PULSE
def stokes_liam(f, ic, param):
    # Stokes Parameter
    xx = f[...,0]
    Rexy = f[...,1]
    Imxy = f[...,2]
    yy = f[...,3]
    I = (xx + yy)/2./ic
    if param == "I":
        n = (xx+yy)/2./ic
    if param == "Q":
        n = (xx-yy)/2./ic
    if param == "U":
        n = Rexy/ic
    if param == "V":
        n = Imxy/ic
    # Roll to Mid
    if '2015-07' in sys.argv[1]:
        n = rollToMid(n)
    range_select = range(100,118)#findPrecursor(I)
    # Cleaning
    pprof_1D = n[:,:,:].sum(0).sum(0) # phase profile, used for subtracting
    pmax = np.argmax(I.sum(0).sum(0))        # max point in phase profile
    n_back1 = n[:,:,pmax-50:pmax-30].mean(axis=-1) # background gate
    n_back = n_back1[...,np.newaxis]                # background gate
    n_clean = (n[:,:,:]-n_back) # subtracting the background 
    n_clean = n_clean[:,:,range_select].sum(-1).sum(0) # only using precursor
    return n_clean

# function to rebin data given data, shape to rebin to
# note that right now this is just for 1-D data
def rebin(a, shape):
    a = a[:,np.newaxis]
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    output = a.reshape(sh).mean(-1).mean(1)
    return output.sum(-1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s foldspec (icounts) (optional params)" % sys.argv[0]
        sys.exit(1)

  # wf = (time, frequency, polarization)
    if 'waterfall' in sys.argv[1]:
        wf = np.load(sys.argv[1])
        title = 'Giant Pulse (%s)' % sys.argv[1].split('_')[-1].split('+')[0] 
  # f = (time, frequency, phase, polarization)
    elif 'foldspec' in sys.argv[1]:
        f = np.load(sys.argv[1])
        ic = np.load(sys.argv[1].replace('foldspec', 'icount'))
        title = 'Precursor Pulse (%s)' %sys.argv[1].split('_')[-1].split('+')[0]

    # params from filename
    nfreq = float(sys.argv[1].split('_')[1].split('chan')[0])
    freq = np.linspace(400,800,nfreq)
    lamsq = (300./freq)**2.
    tobs = float(sys.argv[1].split('+')[-1].split('sec')[0])
    
    if 'waterfall' in sys.argv[1]:
        I = stokeswf(wf, "I")
        Q = stokeswf(wf, "Q")
        U = stokeswf(wf, "U")   
        V = stokeswf(wf, "V")
    elif 'foldspec' in sys.argv[1]:
        I = stokes_liam(f, ic, "I")
        Q = stokes_liam(f, ic, "Q")
        U = stokes_liam(f, ic, "U")   
        V = stokes_liam(f, ic, "V")
    
    Ucomp = U + 1j*V # complex U, will look at this below
    
    # plots of Stokes Parameters vs Frequency
    if '--SvF' in sys.argv:
        plt.suptitle('Stokes Parameters for B0531+21 %s' % title)
        plt.subplot(221)
        plt.plot(I, 'm')
        plt.title('I')
        plt.ylabel('Intensity')
        plt.xlim(0,1024)
        plt.xticks(np.linspace(0,1024,5),np.linspace(400,800,5))
        plt.subplot(222)
        plt.plot(Q, 'm')
        plt.title('Q')
        plt.xlim(0,1024)
        plt.xticks(np.linspace(0,1024,5),np.linspace(400,800,5))
        plt.subplot(223)
        plt.plot(U, 'm')
        plt.title('U')
        plt.ylabel('Intensity')
        plt.xlabel('Frequency (MHz)')
        plt.xlim(0,1024)
        plt.xticks(np.linspace(0,1024,5),np.linspace(400,800,5))
        plt.subplot(224)
        plt.plot(V, 'm')
        plt.title('V')
        plt.xlabel('Frequency (MHz)')
        plt.xlim(0,1024)
        plt.xticks(np.linspace(0,1024,5),np.linspace(400,800,5))
        plt.show()
        plt.clf()

    # plots of Stokes Parameters vs Wavelength Squared
    if '--SvW' in sys.argv:
        plt.suptitle('Stokes Parameters for B0531+21 %s' % title)
        plt.subplot(221)
        plt.plot(lamsq, I, 'm')
        plt.title('I')
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylabel('Intensity')
        plt.subplot(222)
        plt.plot(lamsq, Q, 'm')
        plt.title('Q')
        plt.xlim(max(lamsq), min(lamsq))
        plt.subplot(223)
        plt.plot(lamsq, U, 'm')
        plt.title('U')
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength Squared (m^2)')
        plt.subplot(224)
        plt.plot(lamsq, V, 'm')
        plt.title('V')
        plt.xlim(max(lamsq), min(lamsq))
        plt.xlabel('Wavelength Squared (m^2)')
        plt.show()
        plt.clf()
    
    
    RM_o = -42.3 # RM found online
                 # http://iopscience.iop.org/0004-637X/517/1/460/fulltext/
    P = Q + 1j*U # polarization vector


    if '2015-04' in sys.argv[1]:
            # parameters from FaradayTools for April 2015 data
            # params for de-rotation (Faraday Rotation)
        paramsFR = [np.array([  -0.9231338, -42.99465735,
                                     2.20884495,   0.08746663])]
            # params for undoing Cable Delay
            # comes from FaradayTools for April 2015 data
        paramsCD = [np.array([-0.03643979, -5.40430987,  
                                   1.50817018,  0.01116632])]

    elif '2015-07' in sys.argv[1]:
            # params from FaradayTools for July 2015 data
            # params for de-rotation (Faraday Rotation)
        paramsFR = [np.array([  0.80011356,  -4.38546764e+01,   
                                    1.25454162,  -1.73598027e-02])]

            # params for undoing Cable Delay
            # comes from FaradayTools for July 2015 data
        paramsCD = [np.array([ 0.05604661, -0.26770184,
                                   -0.04650346,  0.05152873])]
    else:
            # currently this isn't going to work on BGQ
            # except maybe if you import a better version of Python? 
                                  # (that works with scipy.optimize)
        import faraday_tools as ft
        paramsFR = ft.run_RM_fits(Q[..., np.newaxis], RM_o)
        paramsCD = ft.run_RM_fits2(Ucomp[..., np.newaxis], 0.0)


            
    # for undoing Cable Delay
    vecCD = paramsCD[0][0] * np.exp(-2j*np.pi*(np.arange(1024)/1024. * 
                                  paramsCD[0][1]+paramsCD[0][2]))+paramsCD[0][3]
    # for Faraday de-rotation
    vecFR = paramsFR[0][0] * np.exp(2j*(paramsFR[0][1]*lamsq + 
                                  paramsFR[0][2])) + paramsFR[0][3]
    
    # first undo Cable Delay
    Ucomp_nocd = Ucomp*vecCD
    U_nocd = Ucomp_nocd.real
    V_nocd = Ucomp_nocd.imag
    
    # plot of U and V, before and after undoing cable delay
    if '--UVCD' in sys.argv:
        plt.suptitle('U/V Before and After Cable Delay')
        plt.subplot(121)
        plt.title('Stokes U')
        before = plt.plot(lamsq, U, 'm')
        after = plt.plot(lamsq, U_nocd, 'k')
        plt.legend([before, after], ['Before', 'After'])
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength Squared')
        plt.subplot(122)
        plt.title('Stokes V')
        before = plt.plot(lamsq, V, 'm')
        after = plt.plot(lamsq, V_nocd, 'k')
        plt.legend([before, after], ['Before', 'After'])
        plt.xlim(max(lamsq), min(lamsq))
        plt.xlabel('Wavelength Squared')
        plt.show()
        plt.clf()   

    # rebinning, as per Ue-Li's suggestion
    if '--UVCDrebin' in sys.argv:
        plt.suptitle('U/V Before and After Cable Delay, Rebinned')
        plt.subplot(121)
        plt.title('Stokes U')
        before = plt.plot(rebin(lamsq, (64,1)), rebin(U, (64,1)), 'b')
        after = plt.plot(rebin(lamsq, (64,1)), rebin(U_nocd, (64,1)), 'r')
        plt.legend([before, after], ['Before', 'After'])
        plt.xlim(max(rebin(lamsq,(64,1))), min(rebin(lamsq, (64,1))))
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength Squared')
        plt.subplot(122)
        plt.title('Stokes V')
        before = plt.plot(rebin(lamsq, (64,1)), rebin(V, (64,1)), 'b')
        after = plt.plot(rebin(lamsq, (64,1)), rebin(V_nocd, (64,1)), 'r')
        plt.legend([before, after], ['Before', 'After'])
        plt.xlim(max(rebin(lamsq,(64,1))), min(rebin(lamsq, (64,1))))
        plt.xlabel('Wavelength Squared')
        plt.show()
        plt.clf()
    
    # undo Faraday rotation second
    P_nocd = Q+1j*U_nocd
    P_noFR_nocd = P*vecFR
    Q_noFR = P_noFR_nocd.real
    U_noFR_nocd = U_nocd*vecFR#P_noFR_nocd.imag
    
    # plot of Q and U, before and after FR, Cable Delay
    if '--QUFR' in sys.argv:
        plt.suptitle('Q/U Before and After De-Rotation')
        plt.subplot(121)
        plt.title('Stokes Q')
        before = plt.plot(lamsq, Q, 'm')
        after = plt.plot(lamsq, Q_noFR, 'k')
        plt.legend([before, after], ['Before', 'After'])
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength Squared')
        plt.subplot(122)
        plt.title('Stokes U')
        before = plt.plot(lamsq, U_nocd, 'm')
        after = plt.plot(lamsq, U_noFR_nocd, 'k')
        plt.legend([before, after], ['Before', 'After'])
        plt.xlim(max(lamsq), min(lamsq))
        plt.xlabel('Wavelength Squared')
        plt.show()
        plt.clf()
    
    # rebinned, as per Ue-Li's suggestion
    if '--QUFRrebin' in sys.argv:
        plt.suptitle('Q/U Before and After De-Rotation, Rebinned')
        plt.subplot(121)
        plt.title('Stokes Q')
        before = plt.plot(rebin(lamsq, (64,1)), rebin(Q, (64,1)), 'b')
        after = plt.plot(rebin(lamsq, (64,1)), rebin(Q_noFR, (64,1)), 'r')
        plt.legend([before, after], ['Before', 'After'])
        plt.xlim(max(rebin(lamsq,(64,1))), min(rebin(lamsq, (64,1))))
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength Squared')
        plt.subplot(122)
        plt.title('Stokes U')
        before = plt.plot(rebin(lamsq, (64,1)), rebin(U_nocd, (64,1)), 'b')
        after = plt.plot(rebin(lamsq, (64,1)), rebin(U_noFR_nocd, (64,1)), 'r')
        plt.legend([before, after], ['Before', 'After'])
        plt.xlim(max(rebin(lamsq,(64,1))), min(rebin(lamsq, (64,1))))
        plt.xlabel('Wavelength Squared')
        plt.show()
        plt.clf()
    
    # plot of new Stokes params vs freq
    if '--newstokes' in sys.argv:
        plt.suptitle('Stokes Parameters for B0531+21 %s' % title)
        plt.subplot(221)
        plt.plot(I, 'm')
        plt.title('I')
        plt.ylabel('Intensity')
        plt.xlim(0,1024)
        plt.xticks(np.linspace(0,1024,5),np.linspace(400,800,5))
        plt.subplot(222)
        plt.plot(Q_noFR, 'm')
        plt.title('Q')
        plt.xlim(0,1024)
        plt.xticks(np.linspace(0,1024,5),np.linspace(400,800,5))
        plt.subplot(223)
        plt.plot(U_noFR_nocd, 'm')
        plt.title('U')
        plt.ylabel('Intensity')
        plt.xlabel('Frequency (MHz)')
        plt.xlim(0,1024)
        plt.xticks(np.linspace(0,1024,5),np.linspace(400,800,5))
        plt.subplot(224)
        plt.plot(V_nocd, 'm')
        plt.title('V')
        plt.xlabel('Frequency (MHz)')
        plt.xlim(0,1024)
        plt.xticks(np.linspace(0,1024,5),np.linspace(400,800,5))
        plt.show()
        plt.clf()
    
    # plot of new Stokes params vs lambda squared
    if '--newstokes' in sys.argv:
        plt.suptitle('Stokes Parameters for B0531+21 %s' % title)
        plt.subplot(221)
        plt.plot(lamsq, I, 'm')
        plt.title('I')
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylabel('Intensity')
        plt.subplot(222)
        plt.plot(lamsq, Q_noFR, 'm')
        plt.title('Q')
        plt.xlim(max(lamsq), min(lamsq))
        plt.subplot(223)
        plt.plot(lamsq, U_noFR_nocd, 'm')
        plt.title('U')
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength Squared (m^2)')
        plt.subplot(224)
        plt.plot(lamsq, V_nocd, 'm')
        plt.title('V')
        plt.xlim(max(lamsq), min(lamsq))
        plt.xlabel('Wavelength Squared (m^2)')
        plt.show()
        plt.clf()
        
    if '--sameplot' in sys.argv:
        plt.suptitle('Stokes Parameters for B0531+21 %s' % title)
        I = plt.plot(lamsq, I)
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylabel('Intensity')
        Q = plt.plot(lamsq, Q_noFR)
        U = plt.plot(lamsq, U_noFR_nocd)
        plt.xlabel('Wavelength Squared (m^2)')
        V = plt.plot(lamsq, V_nocd)
        plt.legend([I, Q, U, V], ['I', 'Q', 'U', 'V'])
        plt.show()
        plt.clf()

    if '--paramsafter' in sys.argv:
        ymin = min(I)
        ymax = max(I)
        plt.suptitle('Stokes Parameters for B0531+21 %s' % title)
        plt.subplot(221)
        plt.plot(lamsq, I, 'c')
        plt.title('I')
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylim(ymin, ymax)
        plt.ylabel('Intensity')
        plt.subplot(222)
        plt.plot(lamsq, Q_noFR, 'c')
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylim(ymin, ymax)
        plt.title('Q')
        plt.subplot(223)
        plt.plot(lamsq, U_noFR_nocd, 'c')
        plt.title('U')
        plt.xlim(max(lamsq), min(lamsq))
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength Squared (m^2)')
        plt.ylim(ymin, ymax)
        plt.subplot(224)
        plt.plot(lamsq, V_nocd, 'c')
        plt.title('V')
        plt.xlim(max(lamsq), min(lamsq))
        plt.xlabel('Wavelength Squared (m^2)')
        plt.ylim(ymin, ymax)
        plt.show()
        plt.clf()
        

    if '--experimental' in sys.argv:
        Q = (wf[...,0]-wf[...,3])/2.
        I = (wf[...,0]+wf[...,3])/2.
        tbins = I.shape[0] # number of time bins
        ncut = I[1:(tbins-2),...] # cut to find max 
        pulse = np.argmax(ncut.sum(-1)) # location of giant pulse
        tstart = pulse - (0.003*tbins) # cutting around giant pulse
        tend = pulse + (0.004*tbins) # cutting around giant pulse
        Qcut = Q[tstart:tend,...] # cutting around giant pulse
        nmax = np.argmax(I.sum(-1)) # finding location of giant pulse
        n_back = Qcut[nmax-(0.002*tbins):nmax-(0.001*tbins),...].mean(axis=0)
        n_back = n_back[np.newaxis,...] # finding off-pulse to subtract
        Qcut = Qcut-n_back # subtracting off-pulse
        Qtest = Qcut*vecFR
        Qold = plt.plot(Qcut.sum(-1) - np.median(Qcut.sum(-1)))
        Q = plt.plot(Qtest.sum(-1) - np.median(Qtest.sum(-1)))
        Icut = I[tstart:tend,...]
        n_back2 = Icut[nmax-(0.002*tbins):nmax-(0.001*tbins),...].mean(axis=0)
        n_back2 = n_back2[np.newaxis,...]
        Icut = Icut - n_back
        I = plt.plot(Icut.sum(-1) - np.median(Icut.sum(-1)))
        plt.legend([I, Q, Qold], ['Stokes I', 'Stokes Q After', 
                                  'Stokes Q Before'])
        plt.ylabel('Intensity')
        plt.xlabel('Time (s)')
        plt.title('Stokes I and Q for B0531+21 %s' % title)
        plt.xticks(np.linspace(0, len(Icut.sum(-1)), 6), 
                   np.linspace((float(tstart/tbins)*2.), 
                               float((tend/tbins)*2.), 6))
        plt.show()
        plt.clf()
        
        plt.title('XX and YY for B0531+21 %s Before' % title)
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.xticks(np.linspace(0, len(Icut.sum(-1)), 6), 
                   np.linspace((float(tstart/tbins)*2.), 
                               float((tend/tbins)*2.), 6))
        X = plt.plot((Icut.sum(-1) + Qcut.sum(-1)) - 
                     np.median((Icut.sum(-1) + Qcut.sum(-1))))
        Y = plt.plot((Icut.sum(-1) - Qcut.sum(-1)) - 
                     np.median((Icut.sum(-1) - Qcut.sum(-1))))
        plt.legend([X, Y], ['XX', 'YY'])
        plt.show()
        plt.clf()
        
        plt.title('XX and YY for B0531+21 %s After' % title)
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.xticks(np.linspace(0, len(Icut.sum(-1)), 6), 
                   np.linspace((float(tstart/tbins)*2.), 
                               float((tend/tbins)*2.), 6))
        X = plt.plot((Icut.sum(-1) + Qtest.sum(-1)) - 
                     np.median((Icut.sum(-1) + Qtest.sum(-1))))
        Y = plt.plot((Icut.sum(-1) - Qtest.sum(-1)) - 
                     np.median((Icut.sum(-1) - Qtest.sum(-1))))
        plt.legend([X, Y], ['XX', 'YY'])
        plt.show()
        plt.clf()
  
