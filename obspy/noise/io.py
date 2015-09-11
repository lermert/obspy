#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import obspy
from obspy.core.trace import Stats

from .correlation import Correlation, CorrelationStream


def to_corr(data):
    if isinstance(data, obspy.Trace):
        data = obspy.Stream(traces=[data])
    elif not isinstance(data, obspy.Stream):
        try:
            import pyasdf
            return asdf_to_corr(data)
        except ImportError:
            msg = 'Data must be Stream or Trace.'
            raise TypeError(msg)

        if isinstance(data, pyasdf.ASDFDataSet):
            pass
        else:
            msg = 'Data must be Stream, Trace, or an ASDF data set.'
            raise TypeError(msg)

    corrstream = CorrelationStream()

    for tr in data:
        st = tr.stats

        stats_a = Stats()
        stats_a.network = st.network
        stats_a.station = st.station
        stats_a.location = st.location
        stats_a.channel = st.channel
        stats_a.sampling_rate = st.sampling_rate
        stats_a.starttime = st.starttime

        stats_b = Stats()
        stats_b.sampling_rate = st.sampling_rate

        stats_b.network = str(st.sac['kuser0']).strip()
        if stats_b.network.startswith('-12345') is True:
            stats_b.network = ''

        stats_b.station = str(st.sac['kevnm']).strip()
        if stats_b.station.startswith('-12345') is True:
            stats_b.station = ''

        stats_b.location = str(st.sac['kuser1']).strip()
        if stats_b.location.startswith('-12345') is True:
            stats_b.location = ''

        stats_b.channel = str(st.sac['kuser2']).strip()
        if stats_b.channel.startswith('-12345') is True:
            stats_b.channel = ''

        nwins = st.sac['user0'] if st.sac['user0'] > 0 else 1
        corrtype = st.sac.kt8

        if str(st.sac['dist']).startswith('-12345') is True:
            dist = 0.0
        else:
            dist = st.sac['dist']

        corr = Correlation(stats_a, stats_b, tr.data, max_lag=st.sac.e,
                           correlation_type=corrtype, dist=dist,
                           min_lag=None, n_stack=nwins)

        corrstream += corr
    return corrstream


def asdf_to_corr(ds):
    if "CrossCorrelation" not in ds.auxiliary_data.list():
        raise ValueError("ASDF file contains no adjoint sources.")

    cc = ds.auxiliary_data.CrossCorrelation

    for corr in cc.list():
        corr = cc[corr]

        corr_obj = Correlation()
        corr_obj


