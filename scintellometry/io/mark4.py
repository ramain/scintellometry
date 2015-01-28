import sys
import os
import warnings
import numpy as np
from astropy.time import Time
import astropy.units as u

from . import SequentialFile, header_defaults

PAYLOADSIZE = 20000
VALIDSTART = 96
VALIDEND = 19936

# the high mag value for 2-bit reconstruction
OPTIMAL_2BIT_HIGH = 3.3359

# Check code on 2015-JAN-22.
# m5d /work/mhvk/scintillometry/gp052d_ar_no0021 MKIV1_4-512-8-2 1000
# data at 17, nonzero at line 657 -> item 640.
# Initially this seemed strange, since PAYLOADSIZE=20000 leads to 80000
# elements, so one would have expected VALIDSTART*4=96*4=384.
# But the mark5 code has PAYLOAD_OFFSET=(VALIDEND-20000)*f->ntrack/8 = 64*8
# Since each sample takes 2 bytes, one thus expects 384+64*8/2=640. OK.
# So, lines 639--641:
#  0  0  0  0  0  0  0  0
# -1  1  1 -3 -3 -3  1 -1
#  1  1 -3  1  1 -3 -1 -1
# Compare with my code:
# m4 = Mark4Data(['/work/mhvk/scintillometry/gp052d_ar_no0021'],
#                channels=None, fedge=0, fedge_at_top=True)
# data = m4.record_read(m4.framesize)
# data[639:642].astype(int)
# array([[ 0,  0,  0,  0,  0,  0,  0,  0],
#        [-1,  1,  1, -3, -3, -3,  1, -1],
#        [ 1,  1, -3,  1,  1, -3, -1, -1]])


class Mark4Data(SequentialFile):

    telescope = 'mark4'

    def __init__(self, raw_files, channels, fedge, fedge_at_top,
                 blocksize=None, Mbps=512, nvlbichan=8, nbit=2, fanout=4,
                 decimation=1, reftime=Time('J2010.', scale='utc'), comm=None):
        """Mark 4 Data reader.

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
            Number of bytes typically read in one go (default: framesize).
        Mbps : Quantity
            Total bit rate.  Only used to check consistency with the data.
        nvlbichan : int
            Number of VLBI channels encoded in Mark 4 data stream.
        nbit : int
            Number of bits per sample.
        fanout : int
            Number of samples per channel encoded in one record.
        decimation : int
            Number by which the samples should be decimated (default: 1, i.e.,
            no decimation).
        reftime : `~astropy.time.Time` instance
            Time close(ish) to the observation time, to resolve decade
            ambiguities in the times stored in the Mark 4 data frames.
        comm : MPI communicator
            For consistency with other readers; not used in this one.
        """
        assert nbit == 1 or nbit == 2
        assert fanout == 1 or fanout == 2 or fanout == 4
        assert decimation == 1 or decimation == 2 or decimation % 4 == 0
        ntrack = nvlbichan * nbit * fanout
        assert ntrack in (1, 2, 4, 8, 16, 32, 64)
        self.Mbps = u.Quantity(Mbps, u.Mbit / u.s)
        self.nvlbichan = nvlbichan
        self.nbit = nbit
        self.fanout = fanout
        self.decimation = decimation
        self.ntrack = ntrack
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        # assert 1 <= len(channels) <= 2
        self.channels = channels
        try:
            self.npol = len(channels)
        except TypeError:
            self.npol = self.nvlbichan if channels is None else 1
        if not (1 <= self.npol <= 2):
            warnings.warn("Should use 1 or 2 channels for folding!")
        # Comment from C code:
        # /* YES: the following is a negative number.  This is OK because the
        #    mark4 blanker will prevent access before element 0. */
        # The offset will also be relative to a positive frame position.
        self.framesize = PAYLOADSIZE * self.ntrack // 8
        self.payloadoffset = (VALIDEND - PAYLOADSIZE) * self.ntrack // 8
        self.invalid = ((VALIDSTART + (PAYLOADSIZE - VALIDEND)) *
                        self.ntrack // 8)
        self._decode = DECODERS[self.nbit, self.ntrack, self.fanout]
        # Initialize standard reader, setting self.files, self.blocksize,
        # dtype, nchan, itemsize, recordsize, setsize.
        # PAYLOADSIZE refers to the number of bits per frame per VLBI channel.
        if blocksize is None:
            blocksize = self.framesize
        dtype = '{0:d}u1'.format(self.ntrack // 8)
        self.filesize = os.path.getsize(raw_files[0])
        super(Mark4Data, self).__init__(raw_files, blocksize=blocksize,
                                        dtype=dtype, nchan=1, comm=comm)
        # Above also opened first file, so use it now to determine
        # the time of the first frame, which is also the start time.
        # But Mark IV cannot store full time, so first determine decade.
        self.decade = int((reftime.mjd - self.frame_time().mjd + 1826) /
                          3652.4) * 10
        # With this in place, reread and reinterpret the time.
        self.time0 = self.frame_time()
        # Find time difference between frames.
        self.seek(self.framesize)
        self.frame_duration = (round((self.frame_time() - self.time0)
                                     .to(u.ns).value) * u.ns).to(u.s)
        self.seek(0)
        # Calculate time associated with record, and the channel sample rate.
        # time corresponding to minimum useful record.
        self.dtsample = (self.frame_duration * self.recordsize /
                         self.framesize).to(u.ns)
        self.samplerate = (round((self.framesize // self.recordsize *
                                  self.fanout / self.frame_duration)
                                 .to(u.Hz).value) * u.Hz).to(u.MHz)
        # Check that the Mbps passed in is consistent with this file.
        mbps_est = self.samplerate * self.nvlbichan * self.nbit * u.bit
        if mbps_est != self.Mbps:
            warnings.warn("Warning: the data rate passed in ({0}) disagrees "
                          "with that calculated ({1})."
                          .format(self.Mbps, mbps_est))

    def open(self, number=0):
        """Open a raw file, and search for the start of the first frame."""
        if number == self.current_file_number:
            return self.fh_raw

        super(Mark4Data, self).open(number)
        frame = self.find_frame()
        if frame is None:
            raise IOError("Cannot find a frame start sequence.")
        if self.header_size and frame != self.header_size - self.payloadoffset:
            warnings.warn('File {0} has frame offset of {1}, which differs '
                          'from the old one of {2}.  Things may fail.'
                          .format(self.files[number], frame,
                                  self.header_size - self.payloadoffset))
        # Ensure reader is at the start of the frame.
        self.header_size = frame + self.payloadoffset
        self.seek(0)
        return self.fh_raw

    def record_read(self, count, blank=True):
        """Read and decode count bytes.

        The range retrieved can span multiple frames and files.

        Parameters
        ----------
        count : int
            Number of bytes to read.
        blank: bool
            If ``True`` (default), set invalid regions to 0.

        Returns
        -------
        data : array of float
            Dimensions are [sample-time, vlbi-channel].
        """
        assert count % self.recordsize == 0
        data = np.empty((count // self.recordsize * self.fanout, self.npol),
                        dtype=np.float32)
        sample = 0
        # With the payloadoffset applied, as we do, the invalid part from
        # VALIDEND to PAYLOADSIZE is also at the start.  Thus, the total size
        # at the start is this one plus the part before VALIDSTART.
        while count > 0:
            # Validate frame we're reading from.
            frame, frame_offset = divmod(self.offset, self.framesize)
            self.seek(frame * self.framesize)
            self.validate()
            if frame_offset > 0:
                self.seek(frame * self.framesize + frame_offset)
            to_read = min(count, self.framesize - frame_offset)
            raw = np.fromstring(self.read(to_read), np.uint8)
            nsample = len(raw) // self.recordsize * self.fanout
            data[sample:sample + nsample] = self._decode(raw, self.channels)
            # Blank invalid header samples.
            nblank = ((self.invalid - frame_offset) //
                      self.recordsize * self.fanout)
            if nblank > 0:
                data[sample:sample + nblank] = 0.
            sample += nsample
            count -= to_read

        if self.npol == 2:
            data = data.view('f4,f4')

        return data if self.decimation == 1 else data[::self.decimation]

    def ntint(self, nchan):
        """Number of samples per block after channelizing."""
        return self.blocksize // self.recordsize * self.fanout // nchan // 2

    @property
    def frame(self):
        """Return the file location of the frame marker.

        Assumes the current pointer is at the start of a frame.
        Returns simply the current offset minus ``payloadoffset``.
        """
        return self.offset - self.payloadoffset

    def find_frame(self, maximum=None, forward=True):
        """Look for the first occurrence of a frame, from the current position.

        The search is for the following pattern:

        * 32*tracks bits set at offset bytes
        * 32*tracks bits set at offset+2500*tracks bytes
        * 1*tracks bits unset before offset+2500*tracks bytes

        Only the currently opened file will be searched.

        Parameters
        ----------
        maximum : int or None
            Maximum number of bytes forward to search through.
            Default is the framesize (20000 * ntrack // 8).
        forward : bool
            Whether to search forwards or backwards.

        Returns
        -------
        offset : int
        """
        nset = np.ones(32 * self.ntrack // 8, dtype=np.int16)
        nunset = np.ones(self.ntrack // 8, dtype=np.int16)
        b = self.ntrack * 2500
        a = b - self.ntrack // 8
        if maximum is None:
            maximum = self.framesize
        # Loop over chunks to try to find the frame marker.
        step = b // 25
        file_pos = self.fh_raw.tell()
        if forward:
            iterate = range(file_pos, file_pos + maximum, step)
        else:
            iterate = range(file_pos - b - step - len(nset),
                            file_pos - b - step - len(nset) - maximum, -step)
        for frame in iterate:
            self.fh_raw.seek(frame)
            data = np.fromstring(self.fh_raw.read(b+step+len(nset)),
                                 dtype=np.uint8)
            if len(data) < b + step + len(nset):
                break
            databits1 = nbits[data[:step+len(nset)]]
            lownotset = np.convolve(databits1 < 6, nset, 'valid')
            databits2 = nbits[data[b:]]
            highnotset = np.convolve(databits2 < 6, nset, 'valid')
            databits3 = nbits[data[a:a+step+len(nunset)]]
            highnotunset = np.convolve(databits3 > 1, nunset, 'valid')
            wrong = lownotset + highnotset + highnotunset
            try:
                extra = np.where(wrong == 0)[0][0 if forward else -1]
            except IndexError:
                continue
            else:
                self.fh_raw.seek(file_pos)
                return frame + extra

        self.fh_raw.seek(file_pos)
        return None

    def extract_nibbles(self, numnibbles):
        """Extract encoded nibbles.

        Count bits in each track, assume set if more than half are.
        Then use those to form a 4-bit number, most significant bit first.

        Parameters
        ----------
        numnibbles: int
            number of nibbles to read.

        Returns
        -------
        nibbles : array of int
            containing numbers between 0 and 15 as encoded by the nibbles.
        """
        n = self.ntrack // 8
        data = self.read(4*n*numnibbles).reshape(numnibbles, 4, n)
        # Count the number of tracks with their bit set.
        c = nbits[data].sum(-1)
        # Let majority decide whether bit is set or unset.
        nibbles = np.where(c > n/2, np.array([8, 4, 2, 1]), 0).sum(-1)
        return nibbles

    def _frame_time(self):
        """Calculate time for the frame at the current position.

        This private routine returns mjd, sec, nsec, like the C code.
        Use the public routine ``frame_time`` to get an
        `~astropy.time.Time` instance.
        """
        lastdig = np.array([0, 1250000, 2500000, 3750000, 0, 5000000,
                            6250000, 7500000, 8750000, 0, 0,
                            0, 0, 0, 0, 0], dtype=int)
        # Assume we are at a frame start, offset ahead and read nibbles.
        old_offset = self.offset
        self._seek(self.frame + 4 * self.ntrack)
        nibs = self.extract_nibbles(13)
        self._seek(old_offset)
        # Interpret nibbles.
        if hasattr(self, 'decade'):
            nibs[0] += self.decade
        mjd = (51543 + 365*nibs[0] + int((nibs[0]+3.)/4.) +  # year
               nibs[1]*100 + nibs[2]*10 + nibs[3])  # day
        sec = (nibs[4]*36000 + nibs[5]*3600 +  # hour
               nibs[6]*600 + nibs[7]*60 +  # minute
               nibs[8]*10 + nibs[9])  # second
        ns = nibs[10]*100000000 + nibs[11]*10000000 + lastdig[nibs[12]]
        return mjd, sec, ns

    def frame_time(self):
        """Read the time for the current frame.

        Assumes the file pointer is at the start of a frame.
        """
        mjd, sec, ns = self._frame_time()
        return Time(mjd * u.day, sec * u.s + ns * u.ns, format='mjd',
                    scale='utc', precision=9)

    def validate(self):
        """Validate the current frame.

        Checks that the frame pointer points to a frame marker of all set bits,
        and that the frame contains a time that is consistent with the offset
        in the file.  Raises a warning if either test fails.

        Returns
        -------
        valid : bool
            ``True`` if the checks passed.
        """
        old_offset = self.offset
        # Check we are at the start of a frame, by offseting to where
        # the frame marker should be and verifying it is present.
        self.seek(self.frame)
        data = self.read(self.ntrack).view(np.uint32)
        self.seek(old_offset)
        if (data != 0xffffffff).sum() > 0:
            warnings.warn("Mark IV validate failed: not at frame marker.")
            return False
        # Also check that time stored in frame header is offset from the
        # start time by the amount expected.
        time_offset = self.frame_time() - self.time0
        if abs(time_offset - self.tell(unit=u.s)) > 1. * u.ns:
            warnings.warn("Mark IV validate failed: frame time inconstent "
                          "with frame location.")
            return False
        else:
            return True


# Mark4 defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['mark4'] = {
    'PRIMARY': {'TELESCOP':'Mark4',
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


#  2bit/fanout4 use the following in decoding 32 and 64 track data:
if sys.byteorder == 'big':
    def reorder32(x):
        x = x.view(np.uint32)
        return (((x & 0x55AA55AA)) |
                ((x & 0xAA00AA00) >> 9) |
                ((x & 0x00550055) << 9)).view(np.uint8)

    def reorder64(x):
        x = x.view(np.uint64)
        return (((x & 0x55AA55AA55AA55AA)) |
                ((x & 0xAA00AA00AA00AA00) >> 9) |
                ((x & 0x0055005500550055) << 9)).view(np.uint8)
else:
    def reorder32(x):
        x = x.view(np.uint32)
        return (((x & 0xAA55AA55)) |
                ((x & 0x55005500) >> 7) |
                ((x & 0x00AA00AA) << 7)).view(np.uint8)

    # can speed this up from 140 to 132 us by predefining bit patterns as
    # array scalars.  Inplace calculations do not seem to help much.
    def reorder64(x):
        x = x.view(np.uint64)
        return (((x & 0xAA55AA55AA55AA55)) |
                ((x & 0x5500550055005500) >> 7) |
                ((x & 0x00AA00AA00AA00AA) << 7)).view(np.uint8)
    # check on 2015-JAN-19: C code: 738811025863578102 -> 738829572664316278
    # 118, 209, 53, 244, 148, 217, 64, 10
    # reorder64(np.array([738811025863578102], dtype=np.int64)).view(np.int64)
    # # array([738829572664316278])
    # reorder64(np.array([-1329753610], dtype=np.int64))
    # # array([118, 209,  53, 244, 148, 217,  64,  10], dtype=uint8)
    # m4.decode_2bit_64track_fanout4_decimation1(
    #     np.array([738811025863578102], dtype=np.int64),
    #     blank=False).astype(int).T
    # -1  1  3  1  array([[-1,  1,  3,  1],
    #  1  1  3 -3         [ 1,  1,  3, -3],
    #  1 -3  1  3         [ 1, -3,  1,  3],
    # -3  1  3  3         [-3,  1,  3,  3],
    # -3  1  1 -1         [-3,  1,  1, -1],
    # -3 -3 -3  1         [-3, -3, -3,  1],
    #  1 -1  1  3         [ 1, -1,  1,  3],
    # -1 -1 -3 -3         [-1, -1, -3, -3]])


def init_luts():
    """Set up the look-up tables for levels as a function of input byte."""
    lut2level = np.array([1.0, -1.0], dtype=np.float32)
    lut4level = np.array([-OPTIMAL_2BIT_HIGH, 1.0, -1.0, OPTIMAL_2BIT_HIGH],
                         dtype=np.float32)
    b = np.arange(256)[:, np.newaxis]
    # lut1bit
    i = np.arange(8)
    # For all 1-bit modes
    lut1bit = lut2level[(b >> i) & 1]
    i = np.arange(4)
    # fanout 1 @ 8/16t, fanout 4 @ 32/64t !
    s = i*2
    m = s+1
    lut2bit1 = lut4level[(b >> s & 1) +
                         (b >> m & 1) * 2]
    # fanout 2 @ 8/16t, fanout 1 @ 32/64t
    s = i + (i//2)*2  # 0, 1, 4, 5
    m = s + 2         # 2, 3, 6, 7
    lut2bit2 = lut4level[(b >> s & 1) +
                         (b >> m & 1) * 2]
    # fanout 4 @ 8/16t, fanout 2 @ 32/64t
    s = i    # 0, 1, 2, 3
    m = s+4  # 4, 5, 6, 7
    lut2bit3 = lut4level[(b >> s & 1) +
                         (b >> m & 1) * 2]
    return lut1bit, lut2bit1, lut2bit2, lut2bit3

lut1bit, lut2bit1, lut2bit2, lut2bit3 = init_luts()

# Look-up table for the number of bits in a byte.
nbits = ((np.arange(256)[:, np.newaxis] >> np.arange(8) & 1)
         .sum(1).astype(np.int16))


def decode_2bit_64track_fanout4(frame, channels=None):
    """Decode the frame, assuming 64 tracks using 2 bits, fan-out 4.

    Optionally select some VLBI channels (by default, all 8 are returned).
    """
    # Bitwise reordering of tracks, to align sign and magnitude bits,
    # reshaping to get VLBI channels in sequential, but wrong order.
    frame = reorder64(frame).reshape(-1, 8)
    # Correct ordering, at the same time possibly selecting specific channels.
    reorder = np.array([0, 2, 1, 3, 4, 6, 5, 7])
    frame = frame[:, reorder if channels is None else reorder[channels]]
    # The look-up table splits each data byte into 4 measurements.
    # Using transpose ensures channels are first, then time samples, then
    # those 4 measurements, so the reshape orders the samples correctly.
    # Another transpose ensures samples are the first dimension.
    return lut2bit1[frame.T].reshape(frame.shape[1], -1).T

# Decoders keyed by (nbit, ntrack, fanout).
DECODERS = {(2, 64, 4): decode_2bit_64track_fanout4}
