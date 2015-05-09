
# __      __  _____    _   _____
# \ \    / / | ___ \  | | |   __|
#  \ \  / /  | |  | | | | |  |_
#   \ \/ /   | |  | | | | |   _]
#    \  /    | |__| | | | |  |
#     \/     |_____/  |_| |__|

from __future__ import division, unicode_literals
import os
import warnings

import numpy as np
from astropy.utils.compat.odict import OrderedDict
from astropy.time import Time, TimeDelta
import astropy.units as u

from . import SequentialFile, header_defaults
from .vlbi_helpers import (get_frame_rate, make_parser, four_word_struct,
                           eight_word_struct)

# the high mag value for 2-bit reconstruction
OPTIMAL_2BIT_HIGH = 3.3359
FOUR_BIT_1_SIGMA = 2.95


class VDIFData(SequentialFile):

    telescope = 'vdif'

    def __init__(self, raw_files, channels, fedge, fedge_at_top,
                 blocksize=None, comm=None):
        """VDIF Data reader.

        Parameters
        ----------
        raw_files : list of string
            full file names of the Mark 4 data
        channels : list of int
            channel numbers to read; should be at the same frequency,
            i.e., 1 or 2 polarisations.
        fedge : Quantity
            Frequency at the edge of the requested VLBI channel
        fedge_at_top : bool
            Whether the frequency is at the top of the channel.
        blocksize : int or None
            Number of bytes typically read in one go
            (default: nthread*framesize, though for VDIF data this is rather
            low, so better to pass on a larger number).
        comm : MPI communicator
            For consistency with other readers.
        """
        if len(raw_files) > 1:
            raise ValueError("Can only handle single file for now.")
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        with open(raw_files[0], 'rb') as checkfile:
            header = VDIFFrameHeader.fromfile(checkfile)
            self.header0 = header
            self.thread_ids = get_thread_ids(checkfile, header.framesize)
            self.nthread = len(self.thread_ids)

        if header.nchan > 1:
            # This needs more thought, though a single thread with multiple
            # channels should be easy, as it is similar to other formats
            # (just need to calculate frequencies).  But multiple channels
            # over multiple threads may not be so easy.
            raise ValueError("Multi-channel vdif not yet supported.")

        self.channels = channels
        # For normal folding, channels should be given, but for other
        # reading, it may be useful to have all channels available.
        # Of course, for that case, the frequencies will be wrong.
        try:
            self.npol = len(channels)
        except TypeError:
            self.npol = header.nchan if channels is None else 1
        if not (1 <= self.npol <= 2):
            warnings.warn("Should use 1 or 2 channels for folding!")

        # Decoder for given bits per sample; see bottom of file.
        self._decode = DECODERS[header.bps, header['complex_data']]

        self.framesize = header.framesize
        self.payloadsize = header.payloadsize
        if blocksize is None:
            blocksize = header.payloadsize
        # Each "virtual record" is one sample for every thread.
        record_bps = header.bps * self.nthread
        if record_bps in (1, 2, 4):
            dtype = '{0}bit'.format(record_bps)
        elif record_bps % 8 == 0:
            dtype = '({0},)u1'.format(record_bps // 8)
        else:
            raise ValueError("VDIF with {0} bits per sample is not supported."
                             .format(header.bps))
        # SOMETHING LIKE THIS NEEDED FOR MULTIPLE FILES!
        # PROBABLY SHOULD MAKE SEQUENTIALFILE READER SEEK
        # OFFSET IN TOTAL BYTE SIZE, IGNORING HEADERS
        # self.totalfilesizes = np.array([os.path.getsize(fil)
        #                                 for fil in raw_files], dtype=np.int)
        # assert np.all(self.totalfilesizes // header.framesize == 0)
        # self.totalpayloads = (self.totalfilesizes // header.framesize *
        #                       header.payloadsize)
        # self.payloadranges = self.totalpayloads.cumsum()
        self.time0 = header.time()
        bandwidth = header.bandwidth
        if bandwidth:
            self.samplerate = bandwidth * (1 if header['complex_data']
                                           else 2)
        else:  # bandwidth not known (e.g., legacy header)
            frame_rate = get_frame_rate(checkfile, VDIFFrameHeader) * u.Hz
            self.samplerate = ((header.payloadsize // 4) * (8 // header.bps) *
                               frame_rate).to(u.MHz)
        self.dtsample = (header.nchan / self.samplerate).to(u.ns)
        if comm is None or comm.rank == 0:
            print("In VDIFData, calling super")
            print("Start time: ", self.time0.iso)
        super(VDIFData, self).__init__(raw_files, blocksize, dtype,
                                       header.nchan, comm=comm)

    def _seek(self, offset):
        assert offset % self.recordsize == 0
        # Seek in the raw file using framesize, i.e., including headers.
        self.fh_raw.seek(offset // self.payloadsize * self.framesize)
        self.offset = offset

    def record_read(self, count):
        """Read and decode count bytes.

        The range retrieved can span multiple frames and files.

        Parameters
        ----------
        count : int
            Number of bytes to read.

        Returns
        -------
        data : array of float
            Dimensions are [sample-time, vlbi-channel].
        """
        # for now only allow integer number of frames
        # assert count % (self.recordsize * self.nthread) == 0
        assert count % (self.payloadsize * self.nthread) == 0
        data = np.empty((count // self.recordsize, self.npol),
                        dtype=np.float32)
        sample = 0
        # With the payloadoffset applied, as we do, the invalid part from
        # VALIDEND to PAYLOADSIZE is also at the start.  Thus, the total size
        # at the start is this one plus the part before VALIDSTART.
        while count > 0:
            # Validate frame we're reading from.
            full_set, full_set_offset = divmod(
                self.offset, self.payloadsize * self.nthread)
            payload_offset = full_set_offset // self.nthread
            self.seek(self.payloadsize * self.nthread * full_set)
            frame_start = self.fh_raw.tell()
            to_read = min(count, self.payloadsize - payload_offset)
            for i in range(self.nthread):
                self.fh_raw.seek(frame_start + self.framesize * i)
                header = VDIFFrameHeader.fromfile(self.fh_raw,
                                                  self.header0.edv)
                # this leaves raw_file pointer at start of payload.
                try:
                    index = self.channels.index(header['thread_id'])
                except ValueError:
                    continue

                if payload_offset > 0:
                    self.fh_raw.seek(payload_offset, 1)

                raw = np.fromstring(self.fh_raw.read(to_read), np.uint8)
                nsample = len(raw) * self.nthread // self.recordsize
                data[sample:sample + nsample, index] = self._decode(raw)

            self.offset += to_read * self.nthread
            sample += nsample
            count -= to_read * self.nthread

        # ensure offset pointers from raw and virtual match again,
        # and are at the end of what has been read.
        if self.npol == 2:
            data = data.view('{0},{0}'.format(data.dtype.str))

        return data

    # def _seek(self, offset):
    #     assert offset % self.recordsize == 0
    #     # Find the correct file.
    #     file_number = np.searchsorted(self.payloadranges, offset)
    #     self.open(self.files, file_number)
    #     if file_number > 0:
    #         file_offset = offset
    #     else:
    #         file_offset = offset - self.payloadranges[file_number - 1]
    #     # Find the correct frame within the file.
    #     frame_nr, frame_offset = divmod(file_offset, self.payloadsize)
    #     self.fh_raw.seek(frame_nr * self.framesize + self.header_size)
    #     self.offset = offset

    def ntint(self, nchan):
        """Number of samples per block after channelizing."""
        return self.blocksize // self.recordsize // nchan // 2

    def __str__(self):
        return ('<VDIFData nthread={0} dtype={1} blocksize={2}\n'
                'current_file_number={3}/{4} current_file={5}>'
                .format(self.nthread, self.dtype, self.blocksize,
                        self.current_file_number, len(self.files),
                        self.files[self.current_file_number]))


# VDIF defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['vdif'] = {
    'PRIMARY': {'TELESCOP':'VDIF',
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


VDIF_header = {  # tuple has word-index, start-bit-index, bit-length
    'standard': {
        'invalid_data': (0, 31, 1),
        'legacy_mode': (0, 30, 1),
        'seconds': (0, 0, 29),
        'ref_epoch': (1, 24, 6),
        'frame_nr': (1, 0, 24),
        'vdif_version': (2, 29, 3),
        'lg2_nchan': (2, 24, 5),
        'frame_length': (2, 0, 24),
        'complex_data': (3, 31, 1),
        'bits_per_sample': (3, 26, 5),
        'thread_id': (3, 16, 10),
        'station_id': (3, 0, 16),
        'edv': (4, 24, 8)},
    1: {'sampling_unit': (4, 23, 1),
        'sample_rate': (4, 0, 23),
        'sync_pattern': (5, 0, 32),
        'das_id': (6, 0, 32),
        'ua': (7, 0, 32)},
    3: {'sampling_unit': (4, 23, 1),
        'sample_rate': (4, 0, 23),
        'sync_pattern': (5, 0, 32),
        'loif_tuning': (6, 0, 32),
        'dbe_unit': (7, 24, 4),
        'if_nr': (7, 20, 4),
        'subband': (7, 17, 3),
        'sideband': (7, 16, 1),
        'major_rev': (7, 12, 4),
        'minor_rev': (7, 8, 4),
        'personality': (7, 0, 8)},
    4: {'sampling_unit': (4, 23, 1),
        'sample_rate': (4, 0, 23),
        'sync_pattern': (5, 0, 32)}}


VDIF_header_parsers = {vk: {k: make_parser(*v) for k, v in vv.items()}
                       for vk, vv in VDIF_header.items()}

ref_max = int(2. * (Time.now().jyear - 2000.)) + 1
ref_epochs = Time(['{y:04d}-{m:02d}-01'.format(y=2000 + ref // 2,
                                               m=1 if ref % 2 == 0 else 7)
                   for ref in range(ref_max)], format='isot', scale='utc')


class VDIFFrameHeader(object):
    def __init__(self, data, edv=None, verify=True):
        """Interpret a tuple of words as a VDIF Frame Header."""
        self.data = data
        if edv is None:
            self.edv = False if self['legacy_mode'] else self['edv']
        else:
            self.edv = edv

        if verify:
            self.verify()

    def verify(self):
        """Verify header integrity."""
        if self.edv is False:
            assert self['legacy_mode']
            assert len(self.data) == 4
        else:
            assert not self['legacy_mode']
            assert self.edv == self['edv']
            assert len(self.data) == 8

    @classmethod
    def frombytes(cls, s, edv=None, verify=True):
        """Read VDIF Header from bytes."""
        try:
            return cls(eight_word_struct.unpack(s), edv, verify)
        except:
            return cls(four_word_struct.unpack(s), False, verify)

    @classmethod
    def fromfile(cls, fh, edv=None, verify=True):
        """Read VDIF Header from file."""
        # Assume non-legacy header to ensure those are done fastest.
        s = fh.read(32)
        if len(s) != 32:
            raise EOFError
        self = cls(eight_word_struct.unpack(s), edv, False)
        if not self.edv:
            # Legacy headers are 4 words, so rewind, and remove excess data.
            fh.seek(-16, 1)
            self.data = self.data[:4]
        if verify:
            self.verify()

        return self

    def __getitem__(self, item):
        try:
            return VDIF_header_parsers['standard'][item](self.data)
        except KeyError:
            if self.edv:
                try:
                    edv_parsers = VDIF_header_parsers[self.edv]
                except KeyError:
                    raise KeyError("VDIF Header of unsupported edv {0}"
                                   .format(self.edv))
                try:
                    return edv_parsers[item](self.data)
                except KeyError:
                    pass

        raise KeyError("VDIF Frame Header does not contain {0}".format(item))

    def keys(self):
        for key in VDIF_header_parsers['standard'].keys():
            yield key
        if self.edv:
            for key in VDIF_header_parsers[self.edv].keys():
                yield key

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        return ("<VDIFFrameHeader {0}>"
                .format(",\n                 ".join(
                    ["{0}: {1}".format(k, self[k]) for k in self.keys()])))

    @property
    def bps(self):
        bps = self['bits_per_sample'] + 1
        if self['complex_data']:
            bps *= 2
        return bps

    @property
    def size(self):
        return len(self.data) * 4

    @property
    def framesize(self):
        return self['frame_length'] * 8

    @property
    def payloadsize(self):
        return self.framesize - self.size

    @property
    def nchan(self):
        return 2**self['lg2_nchan']

    @property
    def station(self):
        msb = self['station_id'] >> 8
        if 48 <= msb < 128:
            return chr(msb) + chr(self['station_id'] & 0xff)
        else:
            return self['station_id']

    @property
    def bandwidth(self):
        if not self['legacy_mode'] and self.edv:
            return (self['sample_rate'] *
                    (u.MHz if self['sampling_unit'] else u.kHz))
        else:
            return None

    @property
    def seconds(self):
        return self['seconds']

    def time(self, samplerate=None):
        """
        Convert ref_epoch, seconds, and frame_nr to Time object.

        Uses 'ref_epoch', which stores the number of half-years from 2000,
        and 'seconds'.  For non-zero frame_nr, needs to have a
        samplerate.  By default, it will be attempted to take this from the
        header; it can be passed on if this is not available (e.g., for a
        legacy VDIF header)
        """
        frame_nr = self['frame_nr']
        if frame_nr == 0:
            offset = 0.
        else:
            if samplerate is None:
                samplerate = self.samplerate
            offset = (self.payloadsize // 4 * (32 // self.bps) /
                      self.bandwidth.to(u.Hz).value * 2) * frame_nr
        return (ref_epochs[self['ref_epoch']] +
                TimeDelta(self.seconds, offset, format='sec', scale='tai'))


def get_thread_ids(infile, framesize, searchsize=None):
    """
    Get the number of threads and their ID's in a vdif file.
    """
    # go to end of infile to get its size on disk
    if searchsize is None:
        searchsize = 1024 * framesize

    n_total = searchsize // framesize

    thread_ids = set()
    for n in range(n_total):
        infile.seek(n * framesize)
        try:
            thread_ids.add(VDIFFrameHeader.fromfile(infile)['thread_id'])
        except:
            break

    return thread_ids


def init_luts():
    """Set up the look-up tables for levels as a function of input byte."""
    lut2level = np.array([-1.0, 1.0], dtype=np.float32)
    lut4level = np.array([-OPTIMAL_2BIT_HIGH, -1.0, 1.0, OPTIMAL_2BIT_HIGH],
                         dtype=np.float32)
    lut16level = (np.arange(16) - 8.)/FOUR_BIT_1_SIGMA

    b = np.arange(256)[:, np.newaxis]
    # 1-bit mode
    i = np.arange(8)
    lut1bit = lut2level[(b >> i) & 1]
    # 2-bit mode
    i = np.arange(0, 8, 2)
    lut2bit = lut4level[(b >> i) & 3]
    # 4-bit mode
    i = np.arange(0, 8, 4)
    lut4bit = lut16level[(b >> i) & 0xf]
    return lut1bit, lut2bit, lut4bit

lut1bit, lut2bit, lut4bit = init_luts()


# Decoders keyed by bits_per_sample, complex_data:
DECODERS = {
    (2, False): lambda x: lut2bit[x].ravel(),
    (4, True): lambda x: lut2bit[x].reshape(-1, 2).view(np.complex64).squeeze()
}
