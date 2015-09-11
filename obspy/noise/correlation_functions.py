#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np
from scipy.signal import hilbert

from obspy.noise.header import clibnoise
from obspy.signal import cross_correlation


def phase_xcorr(data1, data2, max_lag, nu=1, min_lag=0, **kwargs):
    """
    Phase cross correlation (Schimmel 1999), obtained with a
    variable window length

    data1, data2: Numpy arrays containing the trace data
    max_lag: maximum lag in number of samples, integer
    """
    # Initialize pcc array:
    pxc = np.zeros((2 * max_lag + 1, ), dtype=np.float64)

    # Obtain analytic signal
    data1 = hilbert(data1)
    data2 = hilbert(data2)

    # Normalization
    data1 = data1 / (np.abs(data1))
    data2 = data2 / (np.abs(data2))

    clibnoise.phase_xcorr_loop(data1, data2, len(data1), pxc, float(nu),
                               int(max_lag), int(min_lag))

    if min_lag:
        pxc = np.ma.array(pxc)
        pxc[-min_lag:min_lag] = True

    return pxc, []


def classic_xcorr(data1, data2, max_lag, **kwargs):
    """
    Classical (geometrically normalized) cross-correlation as contained in
    (uses :func:`obspy.signal.cross_correlation.xcorr`).
    """

    xcorr = cross_correlation.xcorr(data1.data, data2.data, max_lag, True)[2]

    return xcorr, []


def cross_covar(data1, data2, max_lag, normalize_traces=False,
                **kwargs):
    """
    An alternative to the classical cross-correlation, this function uses
    :func:`numpy.correlate`, operating in the frequency domain and applying
    no normalization.
    The function :func:`numpy.correlate` performs fft and ifft which works
    best on data cut into (or zeropadded to) windows of length of power of 2.
    """
    # Remove mean and normalize; this should have no effect on the
    # energy-normalized correlation result, but may avoid precision issues if
    # trace values are very small
    if normalize_traces is True:
        data1 -= np.mean(data1)
        data2 -= np.mean(data2)
        data1 /= np.max(np.abs(data1))
        data2 /= np.max(np.abs(data2))

    # Make the data more convenient for C function np.correlate
    data1 = np.ascontiguousarray(data1, np.float32)
    data2 = np.ascontiguousarray(data2, np.float32)

    # Obtain the signal energy; the cross correlation is obtained from the
    # cross covariance by normalizing with this value
    ren1 = np.correlate(data1, data1, mode='valid')[0]
    ren2 = np.correlate(data2, data2, mode='valid')[0]

    # Obtain the window rms; a diagnostic parameter
    rms1 = np.sqrt(ren1 / len(data1))
    rms2 = np.sqrt(ren2 / len(data2))

    # A further diagnostic parameter to 'see' impulsive events: range of
    # standard deviations
    nsmp = len(data1) / 4
    std1 = np.zeros(4)
    std2 = np.zeros(4)
    for i in range(4):
        std1[i] = np.std(data1[i * nsmp:(i + 1) * nsmp])
        std2[i] = np.std(data2[i * nsmp:(i + 1) * nsmp])

    rng1 = np.max(std1) / np.mean(std1)
    rng2 = np.max(std2) / np.mean(std2)

    # Obtain correlation via np.correlate (goes through frequency domain)
    ccv = np.correlate(data1, data2, mode='full')

    # Cut out the desired samples from the middle...
    i1 = (len(ccv) - (2 * max_lag + 1)) / 2
    i2 = i1 + 2 * max_lag + 1

    params = (ren1, ren2, rms1, rms2, rng1, rng2)

    return ccv[i1:i2], params
