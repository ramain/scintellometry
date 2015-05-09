# Helper functions for VLBI readers (VDIF, Mark5B).
import struct


OPTIMAL_2BIT_HIGH = 3.3359
eight_word_struct = struct.Struct('<8I')
four_word_struct = struct.Struct('<4I')


def make_parser(word_index, bit_index, bit_length):
    if bit_length == 1:
        return lambda x: bool((x[word_index] >> bit_index) & 1)
    elif bit_length == 32:
        assert bit_index == 0
        return lambda x: x[word_index]
    else:
        mask = (1 << bit_length) - 1  # e.g., bit_length=8 -> 0xff
        if bit_index == 0:
            return lambda x: x[word_index] & mask
        else:
            return lambda x: (x[word_index] >> bit_index) & mask


def bcd_decode(bcd):
    result = 0
    factor = 1
    while bcd > 0:
        result += (bcd & 0xf) * factor
        factor *= 10
        bcd >>= 4
    return result


def get_frame_rate(fh, header_class, thread_id=None):
    """Returns the number of frames

    Can be for a specific thread_id (by default just the first thread in
    the first header).
    """
    fh.seek(0)
    header = header_class.fromfile(fh)
    assert header['frame_nr'] == 0
    sec0 = header.seconds
    if thread_id is None and 'thread_id' in header:
        thread_id = header['thread_id']
    k = 0
    while(header.seconds == sec0):
        fh.seek(header.payloadsize, 1)
        header = header_class.fromfile(fh)
        if thread_id is None or header['thread_id'] == thread_id:
            k += 1

    if header.seconds != sec0 + 1:
        raise ValueError("Time in file has changed by more than 1 second.")

    return k
