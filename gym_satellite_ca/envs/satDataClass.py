import numpy as np
import math
import random


class SatelliteData:
    """ Class that describes the properties and attributes of the specific data of a satellite """

    # class constants
    ACCEPTED_ANGLE_TYPES = {
        "RADIAN_TYPE": "RADIAN",
        "DEGREE_TYPE": "DEGREE"
    }

    def __init__(self, sma: float, ecc: float, inc: float, argp: float, raan: float, tran: float = 0.0,
                 angle_type: str = ACCEPTED_ANGLE_TYPES["DEGREE_TYPE"], mass: float = 1.0,
                 area: float = 1.0, reflection_idx: float = 2.0, thruster_max_force: float = 1.0,
                 thruster_isp: float = 10.0):
        """ Constructor method for the satellite data class.

        :param sma: Semi-major axis value of the satellite's orbit [meters]
        :type sma: float
        :param ecc: Eccentricity angle value of the satellite's orbit
        :type ecc: float
        :param inc: Inclination angle value of the satellite's orbit
        :type inc: float
        :param argp: Argument of the perigee angle value of the satellite's orbit
        :type argp: float
        :param raan: Right ascension of the ascending node angle value of the satellite's orbit
        :type raan: float
        :param tran: True anomaly angle value of the satellite's orbit
        :type tran: float
        :param angle_type: The angle type, indicating whether the angles describing the orbit are in radians or degrees
        :type angle_type: str
        :param mass: The mass of the spacecraft
        :type mass: float
        :param area: The surface area of the spacecraft
        :type area: float
        :param reflection_idx: The reflection index of the spacecraft
        :type reflection_idx: float
        :param thruster_max_force: The maximum force that the thruster of the satellite can produce [N]
        :type thruster_max_force: float
        :param thruster_isp: The specific impulse of the thruster of the satellite [s]
        :type thruster_isp: float
        """
        self._sma = sma
        self._ecc = ecc
        self._inc = inc
        self._argp = argp
        self._raan = raan
        self._tran = tran

        if angle_type not in self.ACCEPTED_ANGLE_TYPES.values():
            raise ValueError(
                f"The angle_type attribute is not properly set. It should be one of the following: "
                f"{self.ACCEPTED_ANGLE_TYPES.values()}")
        self._angle_type = angle_type

        self._mass = mass
        self._area = area
        self._reflection_idx = reflection_idx

        self._thruster_max_force = thruster_max_force
        self._thruster_isp = thruster_isp

    def set_random_tran(self):
        if self.angle_type == self.ACCEPTED_ANGLE_TYPES["DEGREE_TYPE"]:
            self.tran = 360.0 * random.random()
        if self.angle_type == self.ACCEPTED_ANGLE_TYPES["RADIAN_TYPE"]:
            self.tran = 2.0 * np.pi * random.random()
        return self.tran

    def change_angles_to_degrees(self):
        if self.angle_type == self.ACCEPTED_ANGLE_TYPES["DEGREE_TYPE"]:
            return None

        self.ecc = self.rad_to_deg(self.ecc)
        self.inc = self.rad_to_deg(self.inc)
        self.argp = self.rad_to_deg(self.argp)
        self.raan = self.rad_to_deg(self.raan)
        self.tran = self.rad_to_deg(self.tran)

        self.angle_type = self.ACCEPTED_ANGLE_TYPES["DEGREE_TYPE"]

    def change_angles_to_radians(self):
        if self.angle_type == self.ACCEPTED_ANGLE_TYPES["RADIAN_TYPE"]:
            return None

        self.ecc = self.deg_to_rad(self.ecc)
        self.inc = self.deg_to_rad(self.inc)
        self.argp = self.deg_to_rad(self.argp)
        self.raan = self.deg_to_rad(self.raan)
        self.tran = self.deg_to_rad(self.tran)

        self.angle_type = self.ACCEPTED_ANGLE_TYPES["RADIAN_TYPE"]

    def to_dict(self):
        return {
            "sma": self.sma,
            "ecc": self.ecc,
            "inc": self.inc,
            "argp": self.argp,
            "raan": self.raan,
            "tran": self.tran,
            "mass": self.mass,
            "area": self.area,
            "reflection_idx": self.reflection_idx,
            "thruster_max_force": self.thruster_max_force,
            "thruster_isp": self.thruster_isp,
            "angle_type": self.angle_type
        }

    @property
    def sma(self):
        return self._sma

    @sma.setter
    def sma(self, x):
        self._sma = x

    @property
    def ecc(self):
        return self._ecc

    @ecc.setter
    def ecc(self, x):
        self._ecc = x

    @property
    def inc(self):
        return self._inc

    @inc.setter
    def inc(self, x):
        self._inc = x

    @property
    def argp(self):
        return self._argp

    @argp.setter
    def argp(self, x):
        self._argp = x

    @property
    def raan(self):
        return self._raan

    @raan.setter
    def raan(self, x):
        self._raan = x

    @property
    def tran(self):
        return self._tran

    @tran.setter
    def tran(self, x):
        self._tran = x

    @property
    def angle_type(self):
        return self._angle_type

    @angle_type.setter
    def angle_type(self, x):
        if x in self.ACCEPTED_ANGLE_TYPES.values():
            self._angle_type = x
        else:
            print(f"{x} is not an accepted angle type value: {self.ACCEPTED_ANGLE_TYPES.values()}")

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, x):
        self._mass = x

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, x):
        self._area = x

    @property
    def reflection_idx(self):
        return self._reflection_idx

    @reflection_idx.setter
    def reflection_idx(self, x):
        self._reflection_idx = x

    @property
    def thruster_max_force(self):
        return self._thruster_max_force

    @thruster_max_force.setter
    def thruster_max_force(self, x):
        self._thruster_max_force = x

    @property
    def thruster_isp(self):
        return self._thruster_isp

    @thruster_isp.setter
    def thruster_isp(self, x):
        self._thruster_isp = x

    @staticmethod
    def rad_to_deg(rad_value: float) -> float:
        return (180.0 * rad_value) / math.pi

    @staticmethod
    def deg_to_rad(deg_value: float) -> float:
        return (math.pi * deg_value) / 180.0
