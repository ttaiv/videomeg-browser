"""Functions shared between Helsinki VideoMEG project audio and video file readers."""

# License: BSD-3-Clause
# Copyright (c) 2014 BioMag Laboratory, Helsinki University Central Hospital

import struct


class UnknownVersionError(Exception):
    """Error due to unknown file version."""

    pass


def read_attrib(data_file, ver):
    """Read data block attributes.

    If cannot read the attributes (EOF?), return -1 in ts.
    """
    if ver == 0 or ver == 1:
        attrib = data_file.read(12)
        if len(attrib) == 12:
            ts, sz = struct.unpack("QI", attrib)
        else:
            ts = -1
            sz = -1
        total_sz = sz + 12

    elif ver == 2 or ver == 3:
        attrib = data_file.read(20)
        if len(attrib) == 20:
            ts, block_id, sz = struct.unpack("QQI", attrib)
        else:
            ts = -1
            sz = -1
        total_sz = sz + 20

    else:
        raise UnknownVersionError(ver)

    return ts, sz, total_sz
