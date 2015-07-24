# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle, Longitude, Latitude
from astropy.time import Time, TimeDelta


def haversin(theta):
    """haversin(theta) = sin**2(theta/2.)"""
    return np.sin(theta/2.)**2


def archaversin(hs):
    """inverse haversin: 2 arcsin(sqrt(abs(hs)))"""
    return 2.*np.arcsin(np.sqrt(np.abs(hs)))


def time2gmst(time):
        """
        Converts a Time object to Greenwich Mean Sidereal Time.
        Uses an algorithm from the book "Practical Astronomy with your
        Calculator" by Peter Duffet-Smith (Page 17)

        Implementation copied from
        http://code.google.com/p/sdsspy/source/browse/sdsspy/sandbox/convert.py?r=206632952777140bcbf7df8f62541c61ea558be2

        Parameters
        ----------
        time : astropy.time.Time

        Returns
        -------
        Greenwich Mean Sidereal Time (float, hours)
        """
        mjd_int = np.rint(time.mjd)
        S = mjd_int - 51544.5
        T = S / 36525.0
        T0 = 6.697374558 + (2400.051336 * T) + (0.000025862 * T**2)
        UT = (time.mjd-mjd_int)*24.
        T0 += UT*1.002737909
        return Longitude(T0, u.hourangle)


def gmst2time(gmst, time):
        """
        Converts a Greenwich Mean Sidereal Time to UTC time, for a given date.

        Parameters
        ----------
        gmst: ~float
            Greenwich Mean Siderial Time (hours)
        time : astropy.time.Time
            UT date+time

        Returns
        -------
        astropy.time.Time object with closest UT day+time at which
            siderial time is correct.
        """
        dgmst = Longitude(gmst - time2gmst(time))
        return time+TimeDelta(dgmst.to(u.hourangle).value*0.9972695663/24.,
                              format='jd')


class Observatory(dict):
    """Observatory: initialize with East longitude, latitude, name
    (long,lat Quantities with angle units)"""
    def __init__(self, l, b, name=None):
        self['l'] = Longitude(l, wrap_angle=180.*u.degree)
        self['b'] = Latitude(b)
        self['name'] = name

    def ha2za(self, ha, d):
        """Calculate elevation for given hour angle and declination
        (Quantities with angle units)

        Use law of haversines: http://en.wikipedia.org/wiki/Law_of_haversines
        haversin(theta) = haversin(dec2-dec1)
                        + cos(dec2)*cos(dec1)*haversin(ra2-ra1)
        with haversin(theta) = sin**2(theta/2.)"""
        hsth = (haversin(d-self['b']) +
                np.cos(d) * np.cos(self['b']) * haversin(ha))
        return archaversin(hsth).to(u.degree)

    def ha2az(self, ha, d):
        y = (np.sin(d) * np.cos(self['l']) -
             np.cos(d) * np.cos(ha) * np.sin(self['b']))  # due N comp.
        z = -(np.cos(d) * np.sin(ha))  # due east comp.
        return np.arctan2(z, y).to(u.degree)

    def za2ha(self, za, d):
        """Calculate hour angle for given elevation and declination
        (Quantities with angle units)"""
        if abs(d-self['b']) > self.zamax:
            return Angle(0., u.degree)
        if abs(180.*u.degree-d-self['b']) < self.zamax:
            return Angle(180., u.degree)
        hsha = ((haversin(za) - haversin(d-self['b'])) /
                (np.cos(d) * np.cos(self['b'])))
        return archaversin(hsha).to(u.degree)


class BinaryPulsar(SkyCoord):
    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        super(BinaryPulsar, self).__init__(*args)
        self.name = name
        self.tasc = Time(55000., format='mjd', scale='tdb')
        self.porb = 200e6 * u.yr

    def set_ephemeris(self, tasc, porb):
        self.tasc = tasc
        self.porb = porb

    def cycle(self, time):
        return (time-self.tasc).jd/self.porb.to(u.day).value

    def phase(self, time):
        return np.mod(self.cycle(time), 1.)


def print_phases(psr, ist_date1='2013-06-16', ist_date2='2013-07-02'):
    ist_utc = 5.5/24.
    mjd1 = Time(ist_date1, scale='utc').mjd-ist_utc
    mjd2 = Time(ist_date2, scale='utc').mjd-ist_utc
    print('       IST =', ' '.join(['{:02d}'.format(h) for h in range(24)]))
    for mjd in np.arange(mjd1, mjd2+1.e-5):
        time = Time(mjd, format='mjd', scale='utc', precision=0)
        time0 = time + TimeDelta(ist_utc, format='jd')
        assert time0.iso[11:19] == '00:00:00'
        phaselist = []
        for h in range(0,24):
            phaselist.append(psr.phase(time))
            time += TimeDelta(3600., format='sec')
        print(time0.iso[:10], ':',
              ' '.join(['{:02d}'.format(int(round(phase*100.)) % 100)
                        for phase in phaselist]))

# from e-mail jroy@ncra.tifr.res.in 18-jul-2013
# tempo1 coordinates 190534.8100   -740323.62; this is C02
gmrt = Observatory('74d03m23.62s', '19d05m34.81s', 'GMRT')
# from www.ncra.tifr.res.in/ncra/gmrt/gtac/GMRT_status_doc_Dec_2012.pdf‎
gmrt.zamax = 73.*u.deg

gbt = Observatory('-79d50m23s', '38d25m59s', 'GBT')
gbt.zamax = 80.*u.deg  # guess
aro = Observatory('-78d04m22.95s', '45d57m19.81s', 'ARO')
aro.zamax = 82.9*u.deg  # includes feed being offset by 1 degree and
# being very broad at 200 cm, gaining another degree
lofar = Observatory('06d52m08.18s', '52d54m31.55s', 'LOFAR')
lofar.zamax = 70.*u.degree  # guess, gives factor 2 loss in collecting area
wsrt = Observatory('06d36m12s', '52d54m53.s', 'WSRT')
wsrt.hamax = 6.*u.hourangle  # from http://www.astron.nl/radio-observatory/astronomers/wsrt-guide-observations/3-telescope-parameters-and-array-configuration
effelsberg = Observatory('06d52m58s', '50d31m29s', 'EB')
effelsberg.zamax = 85.*u.deg  # guess
jodrell = Observatory('-02d18m25.74s', '53d14m13.2s', 'JB')
jodrell.zamax = 80.*u.deg  # guess
ao = Observatory('-66d45m10s', '18d20m39s', 'AO')
ao.zamax = 19.5*u.deg

j1012 = BinaryPulsar('10h12m33.43s +53d07m02.6s', name='J1012')
j1012.set_ephemeris(tasc=Time(50700.08162891, format='mjd', scale='tdb'),
                    porb=0.60467271355 * u.day)
b1957 = BinaryPulsar('19h59m36.76988s +20d48m15.1222s', name='B1957')
b1957.set_ephemeris(
    tasc=Time(51260.194925280940172, format='mjd', scale='tdb'),
    porb=0.38196748020990333082 * u.day)

j1810 = BinaryPulsar('18h10m37.28s +17d44m37.38s', name='J1810')
j1810.set_ephemeris(tasc=Time(55130.04813, format='mjd', scale='tdb'),
                    porb=3.5561 * u.hr)

b1937 = BinaryPulsar('19h39m38.558720s +21d34m59.13745s', name='B1937')
b1937.set_ephemeris(tasc=Time(55000., format='mjd', scale='tdb'),
                    porb=200e6 * u.yr)

# b1749 = BinaryPulsar('17h52m58.6896s -28d06m37.3s', name='B1749')

b1946 = BinaryPulsar('19h48m25.0067s +35d40m11.057s', name='B1946')

b1508 = BinaryPulsar('15h09m25.6298s +55d31m32.394s', name='B1508')

b0329 = BinaryPulsar('03h32m59.368s +54d34m43.57s', name='B0329')

b1919 = BinaryPulsar('19h21m44.815s +21d53m02.25s', name='B1919')

crab = BinaryPulsar('05h34m31.973s +22d00m52.06s', name='CRAB')

b2116 = BinaryPulsar('21h13m24.307s +46d44m08.70s', name='B2116')

b0809 = BinaryPulsar('08h14m59.500s +74d29m05.70s', name='B0809')

b0834 = BinaryPulsar('08h37m05.642s +06d10m14.56s', name='B0834')

b1133 = BinaryPulsar('11h36m03.180s +15d51m09.62s', name='B1133')

b1237 = BinaryPulsar('12h39m40.4614s +24d53m49.29s', name='B1237')

b1702 = BinaryPulsar('17h05m36.099s -19d06m38.6s', name='B1702')


if __name__ == '__main__':
    print('Source Obs.             HA  LocSidTime UnivSidTime')
    for src in crab, b1937:
        gmststart = -100. * u.hourangle
        gmststop = +100. * u.hourangle
        for obs in gmrt, jodrell, aro:  # gbt, ao, wsrt:
            hamax = getattr(obs, 'hamax', None)
            if hamax is None:
                hamax = obs.za2ha(obs.zamax, src.dec)
            if hamax < 12. * u.hourangle:
                lstmin = src.ra - hamax
                gmstmin = -obs['l'] + lstmin
                gmststart = max(gmststart, gmstmin)
                lstmax = src.ra + hamax
                gmstmax = -obs['l'] + lstmax
                gmststop = min(gmststop, gmstmax)
                print('{:6s} {:6s}({}) ±{:4.1f}: '
                      '{:04.1f}-{:04.1f} = {:04.1f}-{:04.1f}'.format(
                          src.name, obs['name'],
                          'ha<{:1.0f}h'.format(obs.hamax.to(u.hourangle).value)
                          if hasattr(obs, 'hamax')
                          else 'za<{:2d}'.format(
                                  int(round(obs.zamax.to(u.deg).value))),
                          hamax.to(u.hourangle).value,
                          np.mod(lstmin.to(u.hourangle).value, 24.),
                          np.mod(lstmax.to(u.hourangle).value, 24.),
                          np.mod(gmstmin.to(u.hourangle).value, 24.),
                          np.mod(gmstmax.to(u.hourangle).value, 24.)))
            else:
                print('{:6s} {:6s}(za<{:2d}) ±12.0: ++++-++++ = ++++'
                      '-++++'.format(src.name, obs['name'],
                                     int(round(obs.zamax.to(u.deg).value))))

        if gmststart >= gmststop:
            print('{:6s} all      ---:             ---- ----\n'.format(
                src.name, np.mod(gmststart.to(u.hourangle).value, 24.),
                np.mod(gmststop.to(u.hourangle).value, 24.)))
        else:
            print('{:6s} all            {:4.1f}:             {:04.1f}'
                  '-{:04.1f}'.format(
                      src.name, (gmststop-gmststart).to(u.hourangle).value,
                      np.mod(gmststart.to(u.hourangle).value, 24.),
                      np.mod(gmststop.to(u.hourangle).value, 24.)))

            # get corresponding orbital phases for a range of dates
            ist_date1 = '2015-04-23 12:00:00'
            ist_date2 = '2015-04-27 12:00:00'
            ist_utc = 5.5/24.
            mjd1 = Time(ist_date1, scale='utc').mjd-ist_utc
            mjd2 = Time(ist_date2, scale='utc').mjd-ist_utc
            for mjd in np.arange(mjd1, mjd2+1.e-5):
                time = Time(mjd, format='mjd', scale='utc', precision=0)
                ut_start = gmst2time(gmststart, time)
                ut_stop = gmst2time(gmststop, time)
                if ut_stop < ut_start:
                    ut_stop += TimeDelta(1., format='jd')
                ph_start, ph_stop = src.phase(ut_start), src.phase(ut_stop)
                ist_start = ut_start + TimeDelta(ist_utc, format='jd')
                ist_stop = ut_stop + TimeDelta(ist_utc, format='jd')
                print('{}-{}: {:4.2f}-{:4.2f}'.format(ist_start.iso,
                                                      ist_stop.iso[11:],
                                                      ph_start, ph_stop))


# 0834+06 before 1957+20
#
# 1133+16 before J1012+5207
#
#
# Need scintellation data for B1957, J1012
#
# LOFAR how high makes it useful? (elevation > 30?)
