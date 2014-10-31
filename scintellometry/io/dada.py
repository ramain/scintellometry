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
from mpi4py import MPI

from . import MultiFile, header_defaults


class DADAData(MultiFile):

    telescope = 'dada'

    def __init__(self, raw_files, blocksize, comm=None):
        """Pulsar data stored in the DADA format"""

        self.header = read_header(raw_files[0])
        if self.header['NBIT'] != 8:
            raise ValueError("Can only deal with 8-bit dada data so far")
        if self.header['NDIM'] != 2:
            raise ValueError("Can only deal with complex dada data so far")
        dtype = 'ci1,ci1'
        filesize = os.path.getsize(self.files[0])
        self.header_size = self.header['HEADER_SIZE']
        if filesize != self.header['FILESIZE'] + self.header_size:
            raise ValueError("File size is not equal to file size given in "
                             "header")
        self.filesize = self.header['FILESIZE']
        nchan = self.header['NCHAN']
        utc_start = self.header['UTC_START']
        # replace '-' between date and time with a 'T' and convert to Time
        self.time0 = Time(utc_start[:10]+'T'+utc_start[11:],
                          scale='utc', format='isot')
        self.npol = self.header['NPOL']
        self.samplerate = self.header['NDIM']/(self.header['TSAMP'] *
                                               u.microsec)
        self.fedge_at_top = self.header['BW'] < 0.
        self.fedge = (self.header['FREQ'] + 0.5 * self.header['BW']) * u.MHz
        f = fftshift(fftfreq(nchan, (2./self.samplerate).to(u.s).value)) * u.Hz
        if self.fedge_at_top:
            self.frequencies = self.fedge - (f-f[0])
        else:
            self.frequencies = self.fedge + (f-f[0])

        self.dtsample = (nchan * 2 / self.samplerate).to(u.s)
        if comm.rank == 0:
            print("In DADAData, calling super")
            print("Start time: ", self.time0.iso)

        self.files = raw_files
        self.current_file_number = None
        super(DADAData, self).__init__(raw_files, blocksize, dtype, nchan,
                                       comm=comm)

    def open(self, files, file_number=0):
        if file_number == self.current_file_number:
            return
        if self.current_file_number is not None:
            self.fh_raw.Close()
        self.fh_raw = MPI.File.Open(self.comm, files[file_number],
                                    amode=MPI.MODE_RDONLY)
        self.fh_raw.Seek(self.header_size)
        self.current_file_number = file_number

    def close(self):
        self.fh_raw.Close()

    def read(self, size):
        """Read size bytes, returning an ndarray with np.int8 dtype.

        Incorporate information from multiple underlying files if necessary.
        The current file pointer are assumed to be pointing at the right
        locations, i.e., just before the first bit of data that will be read.
        """

        if size % self.recordsize != 0:
            raise ValueError("Cannot read a non-integer number of records")

        # ensure we do not read beyond end
        size = min(size, len(self.files) * self.filesize - self.offset)
        if size <= 0:
            raise EOFError('At end of file in DADA.read')

        # allocate buffer for MPI read
        z = np.empty(size, dtype=np.int8)

        # read one or more pieces
        iz = 0
        while(iz < size):
            block, already_read = divmod(self.offset, self.filesize)
            fh_size = min(size - iz, self.filesize - already_read)
            self.fh_raw.Iread(z[iz:iz+fh_size])
            self._seek(self.offset + fh_size)
            iz += fh_size

        return z

    def _seek(self, offset):
        assert offset % self.recordsize == 0
        file_number = offset // self.filesize
        file_offset = offset % self.filesize
        self.open(self.files, file_number)
        self.fh_raw.Seek(file_offset + self.header_size)
        self.offset = offset

    def ntint(self, nchan):
        return self.setsize

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
