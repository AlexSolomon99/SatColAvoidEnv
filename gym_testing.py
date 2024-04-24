import os
import numpy as np
import math
import random
import datetime

import orekit
vm = orekit.initVM()

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.attitudes import FrameAlignedProvider
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.maneuvers import ConstantThrustManeuver
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, IsotropicRadiationClassicalConvention
from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, CartesianOrbit
from org.orekit.orbits import Orbit
from org.orekit.orbits import OrbitType
from org.orekit.orbits import PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate
from org.orekit.time import TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.utils import IERSConventions
from org.orekit.utils import PVCoordinates
from org.orekit.ssa.metrics import ProbabilityOfCollision
from org.hipparchus.linear import RealMatrix
from org.orekit.propagation import StateCovariance
from org.orekit.frames import FramesFactory
from org.hipparchus.linear import MatrixUtils
from org.orekit.ssa.collision.shorttermencounter.probability.twod import Patera2005

from orekit.pyhelpers import datetime_to_absolutedate

random.seed(42)

from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir
download_orekit_data_curdir()
setup_orekit_curdir()

from org.orekit.frames import FramesFactory
gcrf = FramesFactory.getGCRF()

from org.orekit.time import TimeScalesFactory
utc = TimeScalesFactory.getUTC()


def deg_to_rad(deg: float)-> float:
    return (math.pi * deg) / 180.0


def get_orbital_period(sma: float):
    return 2.0 * np.pi * np.sqrt(np.divide(np.power(sma, 3), Constants.WGS84_EARTH_MU))


def create_propagator(orbit: Orbit, sc_mass: float, sc_area: float, sc_reflection: float, sc_frame: FramesFactory,
                      ref_time: AbsoluteDate, earth_order: float, earth_degree: float, use_perturbations: bool = True):
    # create the propagator
    orbit_type = orbit.getType()
    integrator = DormandPrince853IntegratorBuilder(1.0, 1000., 1.0).buildIntegrator(orbit, orbit_type)
    spacecraft_state = SpacecraftState(orbit, sc_mass)

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(orbit_type)
    propagator.setInitialState(spacecraft_state)

    # Earth gravity field
    if not use_perturbations:
        point_gravity = NewtonianAttraction(Constants.WGS84_EARTH_MU)
        propagator.addForceModel(point_gravity)
    else:
        earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                Constants.WGS84_EARTH_FLATTENING,
                                gcrf)
        harmonics_gravity_provider = GravityFieldFactory.getNormalizedProvider(earth_degree, earth_order)
        propagator.addForceModel(
            HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), harmonics_gravity_provider))

        # Sun and Moon attraction
        propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
        propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))

        # Solar radiation pressure
        propagator.addForceModel(
            SolarRadiationPressure(CelestialBodyFactory.getSun(),
                                earth,
                                IsotropicRadiationSingleCoefficient(sc_area,
                                                                    sc_reflection)))

    rotation = FramesFactory.getEME2000().getTransformTo(sc_frame, ref_time).getRotation()
    attitude = FrameAlignedProvider(rotation)
    propagator.setAttitudeProvider(attitude)

    return propagator

def propagate_(propagator, time):
    propag_response_state = propagator.propagate(time)
    return propag_response_state


# 1. Set the orbit definition parameters for the primary object
# keplerian elements of ISS, without the true anomally (which indictes the position on the orbit)
sma = 6795.e3
ecc = 0.00048
inc = 51.6413  # deg
argp = 21.0174  # deg
raan = 60  # deg

inc_rad = deg_to_rad(inc)
argp_rad = deg_to_rad(argp)
raan_rad = deg_to_rad(raan)

sc_mass = 100.0  # kg
sc_area = 1.0 # m^2
sc_reflection = 2.0 # Perfect reflection - maybe it is not needed to use the solar radiation model

# 2. Randomly select a point on the orbit (by setting the true anomally randomly)
tran = 2.0 * np.pi * random.random()

# define the orbit of the primary object with the parameters obtained in the GCRF ref frame, at the defined ref time
# initial SV
init_sv = np.array([sma, ecc, inc_rad, argp_rad, raan_rad, tran])
ref_sc_frame = gcrf
ref_time = AbsoluteDate(2023, 6, 16, 0, 0, 0.0, TimeScalesFactory.getUTC())

# 3. Create the orbit object of the primary satellite
kep = init_sv.tolist()
primary_orbit_kepl = KeplerianOrbit(kep[0], kep[1], kep[2], kep[3], kep[4], kep[5],
                                    PositionAngleType.MEAN, ref_sc_frame, ref_time, Constants.WGS84_EARTH_MU)
primary_orbit_cart = CartesianOrbit(primary_orbit_kepl)
primary_sc_state = SpacecraftState(primary_orbit_cart, sc_mass)

# 4. Get the position and velocity of the initial state of the primary object
init_primary_pos = np.array(primary_sc_state.getPVCoordinates().getPosition().toArray())
init_primary_vel = np.array(primary_sc_state.getPVCoordinates().getVelocity().toArray())

# 5. Get the position and velocity of the initial state of the secondary object
# 5.1 Position - Get the position difference between the secondary and primary objects sampled from a normal distribution,
# mean 0, std 1. Multiply it by 10, so the obtained value is in the order of 10 and add it to each component of the pos vector
# of the primary.
# 5.2 Velocity - Get the velocity difference between the secondary and primary objects sampled from a normal distribution,
# mean 0, std 1. Multiply it by 10, so the obtained value is in the order of 10 and add it to each component of the
# inverse of the vel vector of the primary. The reason for choosing the inverse is to have a head-on-collision.
init_secondary_pos = 10.0 * np.random.standard_normal(3) + init_primary_pos
init_secondary_vel = 10.0 * np.random.standard_normal(3) + (-1.0 * init_primary_vel)

# 6. Create the orbit object of the secondary sat
sec_pos_vec, sec_vel_vec = Vector3D(init_secondary_pos.tolist()), Vector3D(init_secondary_vel.tolist())
sec_pv_coord = PVCoordinates(sec_pos_vec, sec_vel_vec)
secondary_orbit_cart = CartesianOrbit(sec_pv_coord, gcrf, ref_time, Constants.WGS84_EARTH_MU)
secondary_sc_state = SpacecraftState(secondary_orbit_cart, 10.0)

print(init_primary_pos)
