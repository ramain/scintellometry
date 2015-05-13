# ______           ______
# | ___ \    /\    | ___ \    /\
# | |  | |  /  \   | |  | |  /  \
# | |  | | / /\ \  | |  | | / /\ \
# | |__| |/ ____ \ | |__| |/ ____ \
# |_____//_/    \_\|_____//_/    \_\

from __future__ import division
import os

import numpy as np
from scipy.fftpack import fftfreq, fftshift
from astropy.io.fits import Header
from astropy.time import Time
import astropy.units as u

from . import SequentialFile, header_defaults


class DADAData(SequentialFile):

    telescope = 'dada'

    def __init__(self, raw_files, blocksize, comm=None):
        """Pulsar data stored in the DADA format"""

        header = read_header(raw_files[0])
        if header['NBIT'] != 8:
            raise ValueError("Can only deal with 8-bit dada data so far")
        self.data_is_complex = header['NDIM'] == 2
        if self.data_is_complex:
            dtype = 'ci1,ci1'
        else:
            raise ValueError("Can only deal with complex dada data so far")

        nchan = header['NCHAN']
        utc_start = header['UTC_START']
        # replace '-' between date and time with a 'T' and convert to Time
        self.time0 = Time(utc_start[:10]+'T'+utc_start[11:],
                          scale='utc', format='isot')
        self.npol = header['NPOL']
        self.samplerate = header['NDIM']/(header['TSAMP'] * u.microsecond)
        self.fedge = header['FREQ'] * u.MHz
        self.fedge_at_top = header['BW'] < 0.
        f = fftshift(fftfreq(nchan, (2./self.samplerate).to(u.s).value)) * u.Hz
        # the below just assigns fedge to self.frequencies for NCHAN=1;
        # for NCHAN>1, this has *not* been tested.
        if self.data_is_complex:
            if self.fedge_at_top:
                self.frequencies = self.fedge - f
            else:
                self.frequencies = self.fedge + f
        # Commented out real-data case, since *not* tested
        # else:
        #     self.fedge = (header['FREQ'] - 0.5 * header['BW']) * u.MHz
        #     if self.fedge_at_top:
        #         self.frequencies = self.fedge - (f-f[0])
        #     else:
        #         self.frequencies = self.fedge + (f-f[0])
        self.dtsample = (nchan * 2 / self.samplerate).to(u.s)
        if comm.rank == 0:
            print("In DADAData, calling super")
            print("Start time: ", self.time0.iso)
        super(DADAData, self).__init__(raw_files, blocksize, dtype, nchan,
                                       comm=comm)
        self.header_size = header['HDR_SIZE']
        if self.filesize != header['FILE_SIZE'] + self.header_size:
            raise ValueError("File size is not equal to file size given in "
                             "header")
        self['SUBINT'].header.update(header)

    def ntint(self, nchan):
        return (self.blocksize // self.recordsize // nchan //
                (1 if self.data_is_complex else 2))

    def __str__(self):
        return ('<DADAData nchan={0} dtype={1} blocksize={2}\n'
                'current_file_number={3}/{4} current_file={5}>'
                .format(self.nchan, self.dtype, self.blocksize,
                        self.current_file_number, len(self.files),
                        self.files[self.current_file_number]))

# DADA defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['dada'] = {
    'PRIMARY': {'TELESCOP':'DADA',
                'IBEAM':1, 'FD_POLN':'LIN',
                'OBS_MODE':'SEARCH',
                'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                'TRK_MODE':'TRACK',
                'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                'OBSNCHAN':0, 'CHAN_DM':0,
                'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                'SCANLEN':1, 'FA_REQ':0,
                'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0},
    'SUBINT': {'INT_TYPE': 'TIME',
               'SCALE': 'FluxDen',
               'POL_TYPE': 'AABB',
               'NPOL':1,
               'NBIN':1, 'NBIN_PRD':1,
               'PHS_OFFS':0,
               'NBITS':1,
               'ZERO_OFF':0, 'SIGNINT':0,
               'NSUBOFFS':0,
               'NCHAN':1,
               'CHAN_BW':1,
               'DM':0, 'RM':0, 'NCHNOFFS':0,
               'NSBLK':1}}


def read_header(filename):
    """
    Reads in dada header block and returns results as a FITS header

    Parameters
    ----------
    filename : str
        full path to .dada file

    Returns
    -------
    header : `~astropy.io.fits.Header`
        FITS header with appropriate keys
    """
    with open(filename) as f:
        hdr_size = 4096
        header = Header()
        while f.tell() < hdr_size:
            lin = f.readline()
            if lin == '\n':
                continue
            try:
                comment_start = lin.index('#')
            except ValueError:
                comment = None
            else:
                if comment_start == 0:
                    if "end of header" in lin:
                        break
                    else:
                        continue
                else:
                    comment = lin[comment_start+1:].strip()
                    lin = lin[:comment_start]

            key, value = lin.strip().split()
            if key in ('FILE_SIZE', 'FILE_NUMBER', 'HDR_SIZE',
                       'OBS_OFFSET', 'OBS_OVERLAP',
                       'NBIT', 'NDIM', 'NPOL', 'NCHAN', 'RESOLUTION', 'DSB'):
                value = int(value)
                if key == 'HDR_SIZE':
                    hdr_size = value
            elif key in ('FREQ', 'BW', 'TSAMP'):
                value = float(value)

            header[key] = (value, comment)

    return header
