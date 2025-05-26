# -*- coding: utf-8 -*-
"""
    Copyright (C) 2014 BioMag Laboratory, Helsinki University Central Hospital

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy

# Parameters for detecting timestamps. Should match the parameters used for
# generating the timing sequence
_BASELINE = 5        # seconds
_TRAIN_INTRVL = 10   # seconds
_TRAIN_STEP = 0.015  # seconds
_NBITS = 43          # including the parity bit
    
def _read_timestamp(dtrigs, cur, step, nbits):
    """
    Read and decode one timestamp. Return the timestamp on success or -1
    otherwise.
    """
        
    ts = 0
    parity = False

    if cur + nbits >= len(dtrigs):
        print('end of input reached before all the bits read')
        return(-1)
    
    # Read the bits
    for i in range(nbits):
        
        # check the interval between the two triggers
        if (dtrigs[cur+i+1] < step*1.5) or (dtrigs[cur+i+1] > step*4.5):
            print('invalid interval between two triggers')
            return(-1)

        # check whether the next bit is 0 or 1
        if dtrigs[cur+i+1] > step*3:
            parity = not parity
        
            if i < nbits-1:    # don't read the parity bit into the timestamp
                ts = ts + 2**i

    if parity:
        print('parity check failed')
        return(-1)
    else:
        return(ts)


def _comp_tstamps_1bit(inp, sfreq):
    """ Extract timestamps from a "normal" (not composite) trigger channel
    
        inp - vector of samples for the trigger channel
        sfreq - sampling frequency
    Return the vector of the same length as inp, containing timestamps for
    each entry of inp. For detecting timestamps use parameters in the beginning
    of the file. Assume that the input values are either 0 or 1.
    
    TODO: this function does not handle the boundary case for the first train
    of pulses correctly. This is because there is no trigger before the train
    and there will be no dtrigs value before the first trigger of the train.
    Thus the first pulse train will always be ignored. It would be neat to fix
    this.
    """
    
    THRESH = 0.5    
    
    # input should be a 1-d vector
    assert(inp.ndim == 1)
    
    # find all triggers (threshold crossings)
    trigs = numpy.where((inp[:-1] < THRESH) & (inp[1:] > THRESH))[0] + 1

    # iterate over all timestamp candidates
    samps = []
    tss = []
    dtrigs = numpy.diff(trigs)
    
    for i in numpy.where(dtrigs > _BASELINE * sfreq)[0]:
        ts = _read_timestamp(dtrigs, i, _TRAIN_STEP*sfreq, _NBITS)

        if ts != -1:
            samps.append(trigs[i+1])
            tss.append(ts)

    # do some sanity checking
    if len(tss) < 2:
        raise Exception('Less than 2 timestamps found')
        
    if len(tss) * _TRAIN_INTRVL * sfreq < len(inp) * 0.1:
        raise Exception('Too few timestamps detected')

    # fit timestamps to samples with linear regression
    p = numpy.polyfit(samps, tss, 1)
    data_tstamps = numpy.polyval(p, numpy.arange(len(inp)))
    errs = numpy.abs(numpy.polyval(p, samps) - tss)
    
    print('comp_tstamps: regression fit errors (abs): mean %f, median %f, max %f' % (errs.mean(), numpy.median(errs), errs.max()))

    return(data_tstamps)


def comp_tstamps(inp, sfreq):
    """ Extract timestamps from a trigger channel
    
        inp - vector of samples for the trigger channel
        sfreq - sampling frequency
    Check individual bits of the inp to see whether any of them contains timing
    information. Return the vector of the same length as inp, containing
    timestamps for each entry of inp.
    """
    
    if inp.min() < 0:
        raise Exception('Negative values in composite? channel')
    
    # Try different bits until succeeding or running out of the bits
    while inp.max() > 0:
        try:
            return(_comp_tstamps_1bit(inp % 2, sfreq))
        except Exception:
            inp = inp // 2
            
    raise Exception('No timing information found')
        
