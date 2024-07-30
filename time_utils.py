from copy import deepcopy
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
import astropy.units as u
import numpy as np


class TimeConverter:
    """
    Code to change times in any format to times in any other format
    """

    def __init__(
        self,
        input_format,
        input_timestamps,
        output_format,
        ra,
        dec,
        exptime,
        observatory,
    ):
        """
        Parameters
        ----------
        input_format : str
            The format of the input times, e.g. 'jd_utc', 'isot_utc', 'mjd_utc'
        input_timestamps : str
            The timestamps of the input times, e.g. 'start', 'mid', 'end'
        output_format : str
            The format of the output times, e.g. 'jd_utc', 'mjd_utc', 'bjd_tdb'
        ra : str
            The Right Ascension of the target in hourangle
            e.g. 16:00:00
        dec : str
            The Declination of the target in degrees
            e.g. +20:00:00
        exptime : float
            The exposure time in seconds
        observatory : str
            The name of the observatory, e.g. 'Roque de los Muchachos'
        """
        self.input_format = input_format
        self.input_timestamps = input_timestamps
        self.output_format = output_format
        self.ra = ra
        self.dec = dec
        self.exptime = exptime
        self.observatory = observatory
        return

    @staticmethod
    def getLightTravelTimes(ra, dec, time_to_correct):
        """
        Get the light travel times to the helio- and
        barycentres

        Parameters
        ----------
        ra : str
            The Right Ascension of the target in hourangle
            e.g. 16:00:00
        dec : str
            The Declination of the target in degrees
            e.g. +20:00:00
        time_to_correct : astropy.Time object
            The time of observation to correct. The astropy.Time
            object must have been initialised with an EarthLocation

        Returns
        -------
        ltt_bary : float
            The light travel time to the barycentre
        ltt_helio : float
            The light travel time to the heliocentre

        Raises
        ------
        None
        """
        target = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
        ltt_bary = time_to_correct.light_travel_time(target)
        ltt_helio = time_to_correct.light_travel_time(target, "heliocentric")
        return ltt_bary, ltt_helio

    def convert_time(self, input_times: list):
        # check for unnecessary conversion
        if self.input_format == self.output_format and self.input_timestamps == "mid":
            print("No conversion needed, times are in requested format")
            return

        # get the location of the observatory
        location = EarthLocation.of_site(self.observatory)

        # read in the input times - assumes first column if >1 col
        tinp = input_times

        # set up the astropy time inputs and convert them to JD-UTC-MID
        if self.input_format == "jd_utc":
            time_inp = Time(tinp, format="jd", scale="utc", location=location)
        elif self.input_format == "isot_utc":
            time_inp = Time(tinp, format="isot", scale="utc", location=location)
        elif self.input_format == "mjd_utc":
            time_inp = Time(tinp, format="mjd", scale="utc", location=location)
        elif self.input_format == "hjd_utc":
            time_inp = Time(tinp, format="jd", scale="utc", location=location)
            _, ltt_helio = TimeConverter.getLightTravelTimes(
                self.ra, self.dec, time_inp
            )
            time_inp = Time(
                time_inp.utc - ltt_helio, format="jd", scale="utc", location=location
            )
        elif self.input_format == "bjd_tdb":
            time_inp = Time(tinp, format="jd", scale="tdb", location=location)
            ltt_bary, _ = TimeConverter.getLightTravelTimes(self.ra, self.dec, time_inp)
            time_inp = Time(
                time_inp.tdb - ltt_bary, format="jd", scale="tdb", location=location
            ).utc
        else:
            raise ValueError("Unknown input time format, exiting...")

        # first correct the times to the mid-point, if required
        # correction is assuming to be in units of half_exptime
        correction = (self.exptime / 2.0) / 60.0 / 60.0 / 24.0
        if self.input_timestamps == "mid":
            correction = 0.0 * u.day
        elif self.input_timestamps == "start":
            correction *= 1.0 * u.day
        elif self.input_timestamps == "end":
            correction *= -1.0 * u.day
        time_inp = time_inp + correction

        # now convert to the output format requested
        if self.output_format == "jd_utc":
            new_time = (time_inp.utc).jd
        elif self.output_format == "mjd_utc":
            new_time = (time_inp.utc).mjd
        elif self.output_format == "mjd_tdb":
            new_time = (time_inp.tdb).mjd
        elif self.output_format == "hjd_utc":
            _, ltt_helio = TimeConverter.getLightTravelTimes(
                self.ra, self.dec, time_inp
            )
            new_time = (time_inp.utc + ltt_helio).jd
        elif self.output_format == "bjd_tdb":
            ltt_bary, _ = TimeConverter.getLightTravelTimes(self.ra, self.dec, time_inp)
            new_time = (time_inp.tdb + ltt_bary).jd
        else:
            ValueError("Unknown output time format, exiting...")

        return new_time
