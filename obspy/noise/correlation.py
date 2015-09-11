#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import fnmatch
import warnings

import numpy as np
import matplotlib.pyplot as plt

from obspy.core import Trace
from obspy.core.util.base import _getFunctionFromEntryPoint
from obspy.noise.correlation_functions import phase_xcorr, classic_xcorr
# ============================================================================
# Open questions / To be included / Suggested improvements:

# -whether to include the station-station distance in the correlation object
# (not necessary but very convenient: however, requires knowledge of station
# coordinates, which are not in MSEED header --> metadata required; or default
# to some strange value?)

# - how to handle stacking across different locations

# -what's the contiguous array problem with classical cross correlation?

# - more preprocessing steps... --> Then the correlations will have to be
# 'informed' about what preprocessing has been done, i. e. include in stats

# - phase weighted stack...
# ============================================================================

# ============================================================================
# Correlation (1 window)
# ============================================================================


class Correlation(object):
    def __init__(self, stats_a=None, stats_b=None,
                 correlation=None, max_lag=0, correlation_type=None,
                 min_lag=None, n_stack=0, dist=0.0, locked=False,
                 correlation_options=None):

        if correlation is None:
            correlation = np.array([])

        _data_sanity_checks(correlation)

        self.stats_a = stats_a
        self.stats_b = stats_b
        self.correlation = correlation
        self.max_lag = max_lag
        self.min_lag = min_lag
        self.correlation_type = correlation_type
        self.n_stack = n_stack
        self.dist = dist
        # A locking parameter is useful for all nonlinear stacks
        # (of which none are implemented yet)
        self.__locked = locked
        self.correlation_options = correlation_options \
            if correlation_options else {}
        # Not sure if it makes more sense that 'add' creates a stack or a
        # correlation stream.  Should introduce creating a stream if IDs don't
        # match (should now just raise error) The addition procedure is not yet
        # well-solved I find. Hoping for better ideas here.

    def __add__(self, other):
        # Add a correlation to a stack.

        if self.__locked:
            msg = "Stack already locked."
            raise ValueError(msg)

        if not isinstance(other, Correlation):
            msg = ('Can only add single Correlation object to '
                   'correlation stack so far.')
            raise TypeError(msg)

        # Should the stats be recreated or modified or checked.
        # is there a better way to do it?
        if self.n_stack == 0:
            self.correlation = other.correlation.copy()
            self.n_stack = other.n_stack
            self.stats_a = other.stats_a.copy()
            self.stats_b = other.stats_b.copy()
            self.max_lag = other.max_lag
            self.correlation_type = other.correlation_type
            self.min_lag = other.min_lag
            self.__locked = other.__locked

        else:

            if len(self.correlation) == len(other.correlation) and \
                    self.max_lag == other.max_lag and \
                    self.stats_a.station == other.stats_a.station and \
                    self.stats_b.station == other.stats_b.station:
                self.correlation += other.correlation
                # print("Old stack lengths %g and %g"
                # %(self.n_stack,other.n_stack))
                self.n_stack += other.n_stack
                # print("New stack length now %g" %self.n_stack)
            else:
                msg = "Lag, sampling rate or station names not consistent."
                raise ValueError(msg)

        return self

    def __iadd__(self, other):

        self = self + other
        return self

    def __str__(self):

        if self.correlation_type == 'ccc':
            corrtype = 'cross-correlation '
        elif self.correlation_type == 'pcc':
            corrtype = 'phase-correlation '
        else:
            corrtype = 'No correlation '
        Fs = " | %(sampling_rate)5.2f Hz, "
        lag = "Max. lag %g seconds"
        wins = '  |  %g windows'
        out = corrtype + self.id + Fs % (self.stats_a) + \
            lag % (self.max_lag) + wins % (self.n_stack)
        return out

    def getId(self):
        if len(self.correlation) == 0:
            out = "..."
        else:
            id = "%(network)s.%(station)s.%(location)s.%(channel)s"
            out = id % (self.stats_a) + '--' + id % (self.stats_b)
        return str(out)

    id = property(getId)

    def plot(self):

        if self.min_lag == 0 or self.min_lag is None:
            # id = "%(network)s.%(station)s.%(location)s.%(channel)s"

            plt.figure(figsize=(12, 3.5))
            lag = np.linspace(-self.max_lag, self.max_lag,
                              len(self.correlation))
            plt.plot(lag, self.correlation, 'k')
            plt.ylim(
                [-2 * np.max(self.correlation), 2 * np.max(self.correlation)])
            plt.xlim([-self.max_lag, self.max_lag])
            plt.xlabel('Lag (seconds)')
            plt.ylabel(self.correlation_type)
            plt.title('Correlation of ' + self.id)
            plt.grid()
            plt.show()

    def filter(self, type, **options):
        """
        Equivalent to filter for Trace object.
        Filter the data of the current correlation.

        :type type: str
        :param type: String that specifies which filter is applied (e.g.
            ``"bandpass"``). See the `Supported Filter`_ section below for
            further details.
        :param options: Necessary keyword arguments for the respective filter
            that will be passed on. (e.g. ``freqmin=1.0``, ``freqmax=20.0`` for
            ``"bandpass"``)

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: _`Supported Filter`

        ``'bandpass'``
            Butterworth-Bandpass (uses :func:`obspy.signal.filter.bandpass`).

        ``'bandstop'``
            Butterworth-Bandstop (uses :func:`obspy.signal.filter.bandstop`).

        ``'lowpass'``
            Butterworth-Lowpass (uses :func:`obspy.signal.filter.lowpass`).

        ``'highpass'``
            Butterworth-Highpass (uses :func:`obspy.signal.filter.highpass`).

        ``'lowpassCheby2'``
            Cheby2-Lowpass (uses :func:`obspy.signal.filter.lowpassCheby2`).

        ``'lowpassFIR'`` (experimental)
            FIR-Lowpass (uses :func:`obspy.signal.filter.lowpassFIR`).

        ``'remezFIR'`` (experimental)
            Minimax optimal bandpass using Remez algorithm (uses
            :func:`obspy.signal.filter.remezFIR`).

        .. rubric:: Example
        """
        type = type.lower()
        # retrieve function call from entry points
        func = _getFunctionFromEntryPoint('filter', type)
        # filtering
        # the options dictionary is passed as kwargs to the function that is
        # mapped according to the filter_functions dictionary
        self.correlation = func(self.correlation,
                                df=self.stats_a.sampling_rate,
                                **options)
        return self

    def taper(self, max_percentage, type='hann', max_length=None,
              side='both', **kwargs):
        """
        Equivalent to taper for Trace object.
        Taper the correlation.

        Optional (and sometimes necessary) options to the tapering function can
        be provided as kwargs. See respective function definitions in
        `Supported Methods`_ section below.

        :type type: str
        :param type: Type of taper to use for detrending. Defaults to
            ``'cosine'``.  See the `Supported Methods`_ section below for
            further details.
        :type max_percentage: None, float
        :param max_percentage: Decimal percentage of taper at one end (ranging
            from 0. to 0.5).
        :type max_length: None, float
        :param max_length: Length of taper at one end in seconds.
        :type side: str
        :param side: Specify if both sides should be tapered (default, "both")
            or if only the left half ("left") or right half ("right") should be
            tapered.

        .. note::

            To get the same results as the default taper in SAC, use
            `max_percentage=0.05` and leave `type` as `hann`.

        .. note::

            If both `max_percentage` and `max_length` are set to a float, the
            shorter tape length is used. If both `max_percentage` and
            `max_length` are set to `None`, the whole trace will be tapered.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: _`Supported Methods`

        ``'cosine'``
            Cosine taper, for additional options like taper percentage see:
            :func:`obspy.signal.invsim.cosTaper`.
        ``'barthann'``
            Modified Bartlett-Hann window. (uses:
            :func:`scipy.signal.barthann`)
        ``'bartlett'``
            Bartlett window. (uses: :func:`scipy.signal.bartlett`)
        ``'blackman'``
            Blackman window. (uses: :func:`scipy.signal.blackman`)
        ``'blackmanharris'``
            Minimum 4-term Blackman-Harris window. (uses:
            :func:`scipy.signal.blackmanharris`)
        ``'bohman'``
            Bohman window. (uses: :func:`scipy.signal.bohman`)
        ``'boxcar'``
            Boxcar window. (uses: :func:`scipy.signal.boxcar`)
        ``'chebwin'``
            Dolph-Chebyshev window. (uses: :func:`scipy.signal.chebwin`)
        ``'flattop'``
            Flat top window. (uses: :func:`scipy.signal.flattop`)
        ``'gaussian'``
            Gaussian window with standard-deviation std. (uses:
            :func:`scipy.signal.gaussian`)
        ``'general_gaussian'``
            Generalized Gaussian window. (uses:
            :func:`scipy.signal.general_gaussian`)
        ``'hamming'``
            Hamming window. (uses: :func:`scipy.signal.hamming`)
        ``'hann'``
            Hann window. (uses: :func:`scipy.signal.hann`)
        ``'kaiser'``
            Kaiser window with shape parameter beta. (uses:
            :func:`scipy.signal.kaiser`)
        ``'nuttall'``
            Minimum 4-term Blackman-Harris window according to Nuttall.
            (uses: :func:`scipy.signal.nuttall`)
        ``'parzen'``
            Parzen window. (uses: :func:`scipy.signal.parzen`)
        ``'slepian'``
            Slepian window. (uses: :func:`scipy.signal.slepian`)
        ``'triang'``
            Triangular window. (uses: :func:`scipy.signal.triang`)
        """
        type = type.lower()
        side = side.lower()
        side_valid = ['both', 'left', 'right']
        npts = self.stats_a.npts
        if side not in side_valid:
            raise ValueError("'side' has to be one of: %s" % side_valid)
        # retrieve function call from entry points
        func = _getFunctionFromEntryPoint('taper', type)
        # store all constraints for maximum taper length
        max_half_lenghts = []
        if max_percentage is not None:
            max_half_lenghts.append(int(max_percentage * npts))
        if max_length is not None:
            max_half_lenghts.append(int(max_length * self.stats.sampling_rate))
        if np.all([2 * mhl > npts for mhl in max_half_lenghts]):
            msg = "The requested taper is longer than the trace. " \
                  "The taper will be shortened to trace length."
            warnings.warn(msg)
        # add full trace length to constraints
        max_half_lenghts.append(int(npts / 2))
        # select shortest acceptable window half-length
        wlen = min(max_half_lenghts)
        # obspy.signal.cosTaper has a default value for taper percentage,
        # we need to override is as we control percentage completely via npts
        # of taper function and insert ones in the middle afterwards
        if type == "cosine":
            kwargs['p'] = 1.0
        # tapering. tapering functions are expected to accept the number of
        # samples as first argument and return an array of values between 0 and
        # 1 with the same length as the data
        if 2 * wlen == npts:
            taper_sides = func(2 * wlen, **kwargs)
        else:
            taper_sides = func(2 * wlen + 1, **kwargs)
        if side == 'left':
            taper = np.hstack((taper_sides[:wlen], np.ones(npts - wlen)))
        elif side == 'right':
            taper = np.hstack((np.ones(npts - wlen), taper_sides[len(
                taper_sides) - wlen:]))
        else:
            taper = np.hstack((taper_sides[:wlen], np.ones(npts - 2 * wlen),
                               taper_sides[len(taper_sides) - wlen:]))
        self.correlation = self.correlation * taper
        return self

    def save(self, filename, format):

        if format == 'SAC':
            tr = Trace(data=self.correlation)
            tr.stats = self.stats_a

            tr.stats.sac['kt8'] = self.correlation_type \
                if self.correlation_type else ''
            tr.stats.sac['user0'] = self.n_stack
            tr.stats.sac['b'] = -self.max_lag
            tr.stats.sac['e'] = self.max_lag
            tr.stats.sac['kevnm'] = self.stats_b.station
            tr.stats.sac['dist'] = self.dist
            tr.stats.sac['kuser0'] = self.stats_b.network
            tr.stats.sac['kuser1'] = self.stats_b.location
            tr.stats.sac['kuser2'] = self.stats_b.channel
            tr.write(filename, format='SAC')
        elif format == 'asdf':
            print('Not implemented yet.')
        else:
            msg = 'Invalid format for saving: formats are SAC, asdf'
            raise ValueError(msg)

# ============================================================================
# Correlation stream ('container for correlations')
# one can add any correlation to this
# Further functions needed:
# -plot (correlation traces and 'record section')
# -(linear, nonlin) stack
# -save
# ============================================================================


class CorrelationStream(object):
    def __init__(self, correlations=None):

        # Use a 2-D array too? So far a 1-D list object used
        self.__correlations = []

        if isinstance(correlations, Correlation):
            self.__correlations.append(correlations)

        if isinstance(correlations, list):
            self.__correlations.extend(correlations)

    def __add__(self, other):
        if isinstance(other, Correlation):
            other = CorrelationStream(correlations=other)
        if not isinstance(other, CorrelationStream):
            msg = 'Can only add Correlation or CorrelationStream objects.'
            raise TypeError(msg)
        correlations = self.__correlations + other.__correlations
        return self.__class__(correlations=correlations)

    def __iadd__(self, other):
        if isinstance(other, Correlation):
            other = CorrelationStream(correlations=other)
        if not isinstance(other, CorrelationStream):
            raise TypeError
        self.__correlations += other.__correlations
        return self

    def __len__(self):

        return len(self.__correlations)

    def __getitem__(self, index):
        """
        __getitem__ method
        :return: Correlation objects
        """
        if isinstance(index, slice):
            return self.__class__(
                correlations=self.__correlations.__getitem__(index))
        else:
            return self.__correlations.__getitem__(index)

    def __str__(self, extended=False):
        out = 'Contains %g correlation(s):\n' % len(self.__correlations)
        # Don't print for all, it s too long!
        if len(self.__correlations) <= 10 or extended is True:
            out = out + "\n".join([_i.__str__() for _i in self])
        else:
            out = out + "\n" + self.__correlations[0].__str__() + "\n" + \
                '...\n(%i other correlations)\n...\n' % (
                len(self.__correlations) - 2) + \
                self.__correlations[-1].__str__() + \
                '\n\n[Use "print(' + \
                'CorrelationStack.__str__(extended=True))" to print ' + \
                'all correlations]'
        return out

# A proper solution is needed here. This is just temporary..

    def stack(self, station1=None, station2=None, location1=None,
              location2=None, channel1=None, channel2=None, n=None,
              noloczeroloc=False):
        if n is None:
            n = len(self.__correlations) + 1
        stack = Correlation()
        # perform a selection
        for corr in self.select(station1, station2, location1,
                                location2, channel1, channel2,
                                noloczeroloc).__correlations[0:n]:
            try:
                # stacks retaining the stats and updating the n_stack
                stack += corr

            except ValueError:
                print("At least one does not match: max. lag, delta, station1,"
                      " station2")
                continue
        return stack

    def select(self, station1=None, station2=None, location1=None,
               location2=None, channel1=None, channel2=None, min_dayrms=None,
               max_dayrms=None, min_enratio=None, max_enratio=None,
               noloczeroloc=False):
        # Should include something like start, endtime
        # Location, channel can be a concatenation of strings, like '0010'
        correlations = []

        if noloczeroloc is True:
            location1 = [location1, '']
            location2 = [location2, '']

        for corr in self.__correlations:

            # skip correlation if any given criterion is not matched
            if station1 is not None and station2 is None:
                if not fnmatch.fnmatch(corr.stats_a.station.upper(),
                                       station1.upper())\
                   and not fnmatch.fnmatch(corr.stats_b.station.upper(),
                                           station1.upper()):
                    continue

            if station1 is not None and station2 is not None:
                if not fnmatch.fnmatch(corr.stats_a.station.upper(),
                                       station1.upper()):
                    continue
            if station2 is not None:
                if not fnmatch.fnmatch(corr.stats_b.station.upper(),
                                       station2.upper()):
                    continue

            if location1 is not None:
                if corr.stats_a.location not in location1:
                    continue

            if location2 is not None:
                if corr.stats_b.location not in location2:
                    continue

            if channel1 is not None:
                if corr.stats_a.channel not in channel1:
                    continue

            if channel2 is not None:
                if corr.stats_b.channel not in channel2:
                    continue

            if min_dayrms is not None:
                if corr.stats_a.sac['user3'] < min_dayrms:
                    continue
                if corr.stats_a.sac['user4'] < min_dayrms:
                    continue

            if max_dayrms is not None:
                if corr.stats_a.sac['user3'] > max_dayrms:
                    continue
                if corr.stats_a.sac['user4'] > max_dayrms:
                    continue

            if max_enratio is not None:
                if corr.stats_a.sac['user7'] > max_enratio:
                    continue
                if corr.stats_a.sac['user8'] > max_enratio:
                    continue

            if min_enratio is not None:
                if corr.stats_a.sac['user7'] < min_enratio:
                    continue
                if corr.stats_a.sac['user8'] < min_enratio:
                    continue

            correlations.append(corr)
        return self.__class__(correlations=correlations)

    def sort(self, keys=['stadist', 'network', 'station', 'location',
                         'channel'], reverse=False):
        """
        Sort the traces in the CorrelationStream object.

        The correlations will be sorted according to the keys list.
        It will be sorted by the items relating to trace A (i.e. the station
        name, etc. of the first of the two correlated traces). It will be
        sorted by the first item first, then by the second and so on.
        It will always be sorted from low to high and from A to Z.

        :type keys: list, optional
        :param keys: List containing the values according to which the traces
             will be sorted. They will be sorted by the first item first and
             then by the second item and so on.
             Always available items: 'dist','network', 'station', 'channel',
             'location',  'sampling_rate'
             Defaults to ['dist','network', 'station', 'location',
             'channel'].
        :type reverse: bool
        :param reverse: Reverts sorting order to descending.
        """
        keys = [str(k) for k in keys]
        # check if list
        msg = "keys must be a list of strings. Always available items to " + \
            "sort after: \n'network', 'station', 'channel', 'location' "
        if not isinstance(keys, list):
            raise TypeError(msg)

        # Normally, the distance between the stations is handled as a variable
        # of the Correlation. If needed for sorting, add it
        if 'stadist' in keys:
            for corr in self.__correlations:
                corr.stats_a['stadist'] = corr.dist

        # Loop over all other keys in reversed order.
        for _i in keys[::-1]:
            self.__correlations.sort(key=lambda x: x.stats_a[_i],
                                     reverse=reverse)
        return self

    def taper(self, *args, **kwargs):
        """
        Directly equivalent to Stream object taper.
        Taper all Traces in CorrelationStream.

        For details see the corresponding :meth:`~obspy.core.trace.Trace.taper`
        method of :class:`~obspy.core.trace.Trace`.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
        """
        for tr in self:
            tr.taper(*args, **kwargs)
        return self

    def filter(self, type, **options):
        """
        Directly equivalent to Stream object filter.
        Filter the data of all correlations in the CorrelationStream.

        :type type: str
        :param type: String that specifies which filter is applied (e.g.
            ``"bandpass"``). See the `Supported Filter`_ section below for
            further details.
        :param options: Necessary keyword arguments for the respective filter
            that will be passed on. (e.g. ``freqmin=1.0``, ``freqmax=20.0`` for
            ``"bandpass"``)

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: _`Supported Filter`

        ``'bandpass'``
            Butterworth-Bandpass (uses :func:`obspy.signal.filter.bandpass`).

        ``'bandstop'``
            Butterworth-Bandstop (uses :func:`obspy.signal.filter.bandstop`).

        ``'lowpass'``
            Butterworth-Lowpass (uses :func:`obspy.signal.filter.lowpass`).

        ``'highpass'``
            Butterworth-Highpass (uses :func:`obspy.signal.filter.highpass`).

        ``'lowpassCheby2'``
            Cheby2-Lowpass (uses :func:`obspy.signal.filter.lowpassCheby2`).

        ``'lowpassFIR'`` (experimental)
            FIR-Lowpass (uses :func:`obspy.signal.filter.lowpassFIR`).

        ``'remezFIR'`` (experimental)
            Minimax optimal bandpass using Remez algorithm (uses
            :func:`obspy.signal.filter.remezFIR`).

        .. rubric:: Example


        """
        for correlation in self:
            correlation.filter(type, **options)
        return self

    def plot(self, maxlag=None):
        """
        Rudimentary plotting routine for correlations that plots a
        section sorted by interstation distance. May be
        slow for large sets of correlations.

        """
        if len(self.__correlations) == 1:
            self.__correlations[0].plot()
        else:
            self.sort()
            if maxlag is None:
                maxlag = self.__correlations[0].max_lag
            scaling = 0.01 * self.__correlations[-1].dist / 1000.

            for corr in self.__correlations:
                if True in np.isnan(corr.correlation):
                    continue
                if True in np.isinf(corr.correlation):
                    continue
                lag = np.linspace(-corr.max_lag, corr.max_lag,
                                  len(corr.correlation))
                plt.plot(lag, scaling * corr.correlation /
                         np.max(np.abs(corr.correlation)) + corr.dist/1000.,
                         'k')
                plt.xlabel('Lag (s)')
                plt.tick_params(
                    axis='y',
                    which='both',
                    right='off',
                    left='off', )
                plt.xlim([-maxlag, maxlag])
                plt.ylabel(
                    "Normalized correlations\n by interstation distance in km")
            plt.show()

    def __iter__(self):
        """
            Return a robust iterator for CorrelationStream.correlations.

            Doing this it is safe to remove traces from streams inside of
            for-loops using stream's :meth:`~obspy.core.stream.Stream.remove`
            method. Actually this creates a new iterator every time a trace is
            removed inside the for-loop.

            .. rubric:: Example

            >>> from obspy import Stream
            >>> st = Stream()
            >>> for component in ["1", "Z", "2", "3", "Z", "N", "E", "4", "5"]:
            ...     channel = "EH" + component
            ...     tr = Trace(header={'station': 'TEST', 'channel': channel})
            ...     st.append(tr)  # doctest: +ELLIPSIS
            <...Stream object at 0x...>
            >>> print(st)  # doctest: +ELLIPSIS
            9 Trace(s) in Stream:
            .TEST..EH1 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EH2 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EH3 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EHN | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EHE | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EH4 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EH5 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples

            >>> for tr in st:
            ...     if tr.stats.channel[-1] not in ["Z", "N", "E"]:
            ...         st.remove(tr)  # doctest: +ELLIPSIS
            <...Stream object at 0x...>
            >>> print(st)  # doctest: +ELLIPSIS
            4 Trace(s) in Stream:
            .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EHN | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            .TEST..EHE | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
            """
        return list(self.__correlations).__iter__()

# ============================================================================
# get correlation from two trace windows
# ============================================================================


def correlate_trace(trace_a, trace_b, max_lag, correlation_type, **kwargs):
    # XXX: Add more checks...
    if trace_a.stats.npts != trace_b.stats.npts:
        msg = "Not the same amount of samples."
        raise ValueError(msg)


    correlation_type = correlation_type.lower()
    # retrieve function call from entry points
    func = _getFunctionFromEntryPoint('cross_correlation', type)

    mlag = max_lag / trace_a.stats.delta
    mlag = int(mlag)

    corr = func(trace_a.data, trace_b.data, mlag, **kwargs)

    # if correlation_type == "pcc":
    #     corr = phase_xcorr(trace_a.data, trace_b.data, mlag,
    #                        kwargs.get("nu", 1), kwargs.get("min_lag", 0))[0]
    # elif correlation_type == "ccc":
    #     corr = classic_xcorr(trace_a.data, trace_b.data, mlag)[0]
    # elif correlation_type == "ccv":
    #    corr_result = cross_covar(data1, data2, max_lag_samples,\
    #    kwargs.get("normalize_traces", False))
    #    corr = corr_result[0]
    #    trace_a.stats['enr'] = corr_result[1][0]
    #    trace_a.stats['rms'] = corr_result[1][2]
    #    trace_a.stats['rng'] = corr_result[1][4]
    #    trace_b.stats['enr'] = corr_result[1][1]
    #    trace_b.stats['rms'] = corr_result[1][3]
    #    trace_b.stats['rng'] = corr_result[1][5]
    #

    corr = Correlation(trace_a.stats.copy(),
                       trace_b.stats.copy(),
                       max_lag=max_lag,
                       correlation=corr,
                       correlation_type=correlation_type,
                       n_stack=1,
                       correlation_options=kwargs)

    return CorrelationStream(correlations=corr)


def _data_sanity_checks(value):
    """
    Check if a given input is suitable to be used for Trace.data. Raises the
    corresponding exception if it is not, otherwise silently passes.
    """
    if not isinstance(value, np.ndarray):
        msg = "Trace.data must be a NumPy array."
        raise ValueError(msg)
    if value.ndim != 1:
        msg = ("NumPy array for Trace.data has bad shape ('%s'). Only 1-d "
               "arrays are allowed for initialization.") % str(value.shape)
        raise ValueError(msg)
