from typing import List

from org.orekit.attitudes import FrameAlignedProvider
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.frames import Frame
from org.orekit.orbits import Orbit
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, CartesianOrbit
from org.orekit.orbits import PositionAngleType

import numpy as np
import copy

from gym_satellite_ca.envs import satDataClass


class PropagationUtilities:

    def __init__(self, satellite: satDataClass.SatelliteData, ref_time: AbsoluteDate, ref_frame: Frame):
        self._satellite = satellite
        self._ref_time = ref_time
        self._ref_frame = ref_frame

    @staticmethod
    def create_propagator(orbit: Orbit, sc_mass: float, sc_area: float, sc_reflection: float, sc_frame: Frame,
                          ref_time: AbsoluteDate, earth_order: int, earth_degree: int,
                          use_perturbations: bool = True):
        # create the propagator
        orbit_type = orbit.getType()
        integrator = DormandPrince853IntegratorBuilder(1.0, 1000., 1.0).buildIntegrator(orbit, orbit_type)
        spacecraft_state = SpacecraftState(orbit, sc_mass)

        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(orbit_type)
        propagator.setInitialState(spacecraft_state)

        # set Earth gravity field
        if not use_perturbations:
            point_gravity = NewtonianAttraction(Constants.WGS84_EARTH_MU)
            propagator.addForceModel(point_gravity)
        else:
            earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                     Constants.WGS84_EARTH_FLATTENING,
                                     sc_frame)
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

    @staticmethod
    def propagate_(propagator: NumericalPropagator, propagation_time: AbsoluteDate) -> SpacecraftState:
        propag_response_state = propagator.propagate(propagation_time)
        return propag_response_state

    @staticmethod
    def get_time_discretisation(step_duration: float, time_upper_bound: float) -> np.array:
        num_points = int(time_upper_bound / step_duration)
        time_discretisation_positive = np.linspace(start=0, stop=time_upper_bound, num=num_points)
        time_discretisation_neggative = np.linspace(start=-time_upper_bound, stop=-step_duration, num=num_points)
        time_discretisation = np.append(time_discretisation_neggative, time_discretisation_positive)

        return time_discretisation

    def propagate_sc_states(self, propagator: NumericalPropagator, initial_sc_states: np.array,
                            time_discretisation: np.array, propag_start_idx: int = 0) -> np.array:
        if propag_start_idx == 0:
            orbital_positions = []
        else:
            orbital_positions = copy.deepcopy(initial_sc_states[:propag_start_idx].tolist())

        # get the positions of the satellite at the required time
        for current_time in time_discretisation[propag_start_idx:]:
            primary_state = self.propagate_(propagator=propagator, propagation_time=current_time)

            orbital_positions.append(np.array(primary_state.getPosition().toArray()))
            # TODO: check which is correct
            # orbital_positions = np.append(orbital_positions, np.array([primary_state.getPosition().toArray()]), axis=0)

        return np.array(orbital_positions)

    def get_absolute_time_discretisation(self, time_discretisation: list) -> List[AbsoluteDate]:
        abs_time_disc = []
        for time_step in time_discretisation:
            abs_time_disc.append(self.ref_time.shiftedBy(float(time_step)))

        return abs_time_disc

    def get_orbit_state_from_sat(self) -> (CartesianOrbit, SpacecraftState):
        # define the orbit of the current object
        primary_orbit_kepl = KeplerianOrbit(self.satellite.sma, self.satellite.ecc, self.satellite.inc,
                                            self.satellite.argp, self.satellite.raan, self.satellite.tran,
                                            PositionAngleType.MEAN, self.ref_frame, self.ref_time,
                                            Constants.WGS84_EARTH_MU)
        primary_orbit_cart = CartesianOrbit(primary_orbit_kepl)
        primary_sc_state = SpacecraftState(primary_orbit_cart, self.satellite.mass)

        return primary_orbit_cart, primary_sc_state

    @property
    def satellite(self):
        return self._satellite

    @satellite.setter
    def satellite(self, x):
        self._satellite = x

    @property
    def ref_time(self):
        return self._ref_time

    @ref_time.setter
    def ref_time(self, x):
        self._ref_time = x

    @property
    def ref_frame(self):
        return self._ref_frame

    @ref_frame.setter
    def ref_frame(self, x):
        self._ref_frame = x
