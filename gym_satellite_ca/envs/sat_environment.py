import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import orekit

vm = orekit.initVM()

from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.forces.maneuvers import ConstantThrustManeuver
from org.orekit.frames import Frame
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation import SpacecraftState
from org.orekit.time import AbsoluteDate
from org.orekit.time import TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.utils import PVCoordinates
from org.orekit.frames import FramesFactory
from org.orekit.propagation.numerical import NumericalPropagator
from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir

from gym_satellite_ca.envs import satDataClass
from gym_satellite_ca.envs import propagUtils
from gym_satellite_ca.envs import rewardUtils

download_orekit_data_curdir()
setup_orekit_curdir()

# set up default parameters for the environment
UTC = TimeScalesFactory.getUTC()
DEFAULT_REF_TIME = AbsoluteDate(2023, 6, 16, 0, 0, 0.0, UTC)
DEFAULT_REF_FRAME = FramesFactory.getGCRF()


class CollisionAvoidanceEnv(gym.Env):
    metadata = {"render_modes": [None]}

    # Constants Definition
    # Propagation time constants
    PRIMARY_ORBIT_PROPAGATION_PERIOD = 4.0  # days
    SECONDARY_ORBIT_PROPAGATION_PERIOD = 40.0  # minutes
    PROPAGATION_TIME_STEP = 60.0 * 10  # seconds
    INTEGRATOR_MIN_STEP = 1.0
    INTEGRATOR_MAX_STEP = 200.0
    INTEGRATOR_ERR_THRESHOLD = 0.1  # meters

    # Secondary Satellite Data Constants
    SECONDARY_SC_MASS = 10.0  # kg
    SECONDARY_SC_AREA = 1.0  # m^2
    SECONDARY_REFLECTION_IDX = 2.0

    # Orbit constraints
    MAX_ALTITUDE_DIFF_ALLOWED = 5000.0  # meters
    # INITIAL_ORBIT_RADIUS_BOUND - the max distance allowed between the satellite in the final
    # orbit and the same satellite in the initial orbit such that the final orbit can be considered equivalent to the
    # initial one
    INITIAL_ORBIT_RADIUS_BOUND = 300.0  # meters

    # Collision constants
    COLLISION_MIN_DISTANCE = 500.0  # meters

    def __init__(self, satellite: satDataClass.SatelliteData, ref_time: AbsoluteDate = DEFAULT_REF_TIME,
                 ref_frame: Frame = DEFAULT_REF_FRAME, use_perturbations: bool = False,
                 earth_degree: int = 16, earth_order: int = 16):
        super(gym.Env, self).__init__()
        self._satellite = satellite
        self._ref_time = ref_time
        self._ref_frame = ref_frame
        self._use_perturbations = use_perturbations
        self._earth_degree = earth_degree
        self._earth_order = earth_order

        # instantiate the propagation and reward utilities classes
        self._propag_utils = propagUtils.PropagationUtilities(satellite=satellite,
                                                              ref_time=ref_time,
                                                              ref_frame=ref_frame)
        self._reward_utils = rewardUtils.RewardUtils()

        # initialise the initial orbits and states for the primary and secondary satellites
        self._primary_initial_orbit = None
        self._primary_initial_state = None
        self._initial_primary_sc_state_sequence = None
        self._primary_current_state = None
        self._secondary_initial_orbit = None
        self._secondary_initial_state = None

        # initialise time step and time bound
        self._time_discretisation_primary = None
        self._time_discretisation_secondary = None
        self._absolute_time_discretisation_primary = None
        self._absolute_time_discretisation_secondary = None
        self._time_step_idx_last_orbit = None

        # set the time step and time bound initialised above
        self.set_time_discretisation_variables()

        # instantiate the propagators for the primary and secondary satellites
        self._primary_propagator = None
        self._secondary_propagator = None

        # set the action space
        self.action_space = spaces.Box(low=-1.0 * self.satellite.thruster_max_force,
                                       high=1.0 * self.satellite.thruster_max_force,
                                       shape=(3,), dtype=np.float64)

        # set the observation space
        self.observation_space = spaces.Dict(
            {
                "primary_current_pv": spaces.Box(low=-np.inf,
                                                 high=np.inf,
                                                 shape=(6,),
                                                 dtype=np.float64),
                "primary_sc_state_seq": spaces.Box(low=-np.inf,
                                                   high=np.inf,
                                                   shape=(len(self.time_discretisation_primary), 6),
                                                   dtype=np.float64),
                "secondary_sc_state_seq": spaces.Box(low=-np.inf,
                                                     high=np.inf,
                                                     shape=(len(self.time_discretisation_secondary), 6),
                                                     dtype=np.float64),
                "tca_time_lapse": spaces.Box(low=self.time_discretisation_primary[0] - 10.0,
                                             high=abs(self.time_discretisation_primary[0]) + 10.0,
                                             shape=(1,), dtype=np.float64),
                "primary_sc_mass": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64)
            }
        )

        # instantiate the components of the observation space
        self._primary_current_pv = None
        self._primary_sc_state_sequence = None
        self._secondary_sc_state_sequence = None
        self._tca_time_lapse = None
        self._satellite_mass = None

        # instantiate the historical recordings
        self._hist_actions = []
        self._hist_primary_states = []
        self._time_step_idx = 0

        # reset the environment
        self.window = None
        self.clock = None
        self.close()
        self.reset()

    def _get_obs(self):
        return {"primary_current_pv": self.primary_current_pv,
                "primary_sc_state_seq": self.primary_sc_state_sequence,
                "secondary_sc_state_seq": self.secondary_sc_state_sequence,
                "tca_time_lapse": np.array([self.tca_time_lapse]),
                "primary_sc_mass": np.array([self.satellite_mass])
                }

    def _get_info(self):
        return {
            "None": None
        }

    def _get_reward(self) -> float:
        """
        Method used to compute the reward of the agent at the end of the episode.

        The purpose of the agent, which manoeuvers the primary satellite, is to satisfy the following conditions:

            1.   Increase the minimum distance between the primary satellite and the secondary satellite above a set
            threshold, for the entire period of intersection between the 2.

            2.   Keep the primary satellite as close as possible to the initial orbit for the duration of the episode.
            The point is to keep the satellite operational for the most amount of time possible.

            3.   Use as less fuel as possible.

            4.   Modify the orbit of the primary satellite without exceeding its "orbital boundaries". By this it is
            referred to the necessity of the primary satellite to not exit Earth's orbit or crash into it. These are
            high margins of error - some more restrictive margins are to be put in place.

        Conditions 1. and 4. must be met in order for the agent to receive any positive reward. If the
        collision occurs or the satellite crashes into Earth, the agent will only receive negative rewards and any other
        reward obtained from meeting the other 2 conditions will be nullified. If conditions 1. and 4. are met, the
        rewards from meeting conditions 2. and 3. can be taken into account.

        Conditions 2. and 3. are soft conditions. It is not expected that the agent will keep the satellite on the
        initial orbit for the entire duration of the episode or that it will not use fuel. Therefore, the more time
        the satellite will keep the initial orbit and the less fuel it consumes, the higher the positive
        reward obtained.

        :return: A floating point number representing the reward the agent receives at the end of the episode.
        """
        # instantiate the step reward
        reward_ = 0.0

        # get the reward for avoiding the collision (punishment for not avoiding it)
        # check if the current state of the primary corresponds to a state in which the position of the secondary
        # satellite is known and compute the distance between the states
        current_absolute_time = self.time_discretisation_primary[self.time_step_idx]
        if current_absolute_time in self.time_discretisation_secondary:
            idx_of_time_secondary = np.where(self.time_discretisation_secondary == current_absolute_time)[0][0]

            state_from_primary_orbit = self.primary_current_pv[:3]
            state_from_secondary_orbit = self.secondary_sc_state_sequence[idx_of_time_secondary, :3]
            diff_state_pvs = self.reward_utils.compute_dist_between_states(state_from_primary_orbit,
                                                                           state_from_secondary_orbit)

            if diff_state_pvs < self.COLLISION_MIN_DISTANCE:
                reward_ -= 1.0 * (self.COLLISION_MIN_DISTANCE - diff_state_pvs)

        # get reward for keeping the initial orbit
        state_from_init_orbit = self.primary_sc_state_sequence[self.time_step_idx][:3]
        current_state = self.primary_current_pv[:3]
        diff_state_cvi = self.reward_utils.compute_dist_between_states(current_state, state_from_init_orbit)

        if diff_state_cvi <= self.INITIAL_ORBIT_RADIUS_BOUND:
            reward_ += 1.0

        # get the negative reward for not returning to the initial orbit by the end of the event
        if self.time_step_idx >= self.time_step_idx_last_orbit:
            if diff_state_cvi > self.INITIAL_ORBIT_RADIUS_BOUND:
                reward_ -= 10.0 + (diff_state_cvi - self.INITIAL_ORBIT_RADIUS_BOUND)

        # get the negative reward for going outside the orbital bound, which is an extreme deviation from the
        # initial orbit
        if diff_state_cvi > self.MAX_ALTITUDE_DIFF_ALLOWED:
            reward_ -= (diff_state_cvi - self.MAX_ALTITUDE_DIFF_ALLOWED)

        # get the negative reward for using fuel
        fuel_used_perc = ((self.primary_initial_state.getMass() - self.satellite_mass) /
                          self.primary_initial_state.getMass())
        reward_ -= 100.0 * fuel_used_perc

        return reward_

    def _is_done(self) -> bool:
        return self._time_step_idx >= len(self.primary_sc_state_sequence) - 1

    def step(self, action):
        # The force in each direction cannot be greater than the maximum force of the thruster
        assert all(abs(a) <= 1.0 for a in action), \
            f"The action in each direction can't be greater than 1.0 - cannot use more than the maximum thrust."

        # get the current and new time, with the time step increased
        current_time = self.absolute_time_discretisation_primary[self.time_step_idx]
        self.time_step_idx += 1
        new_time = self.absolute_time_discretisation_primary[self.time_step_idx]

        # Assume there are 3 pairs of thrusters, each of them can be used independently
        for i in range(3):
            if abs(action[i]) > 0.0:
                direction = Vector3D(list((1.0 if action[i] > 0 else -1.0) if i == j else 0.0 for j in range(3)))
                force = self.satellite.thruster_max_force * abs(action[i])
                manoeuvre = ConstantThrustManeuver(current_time, self.PROPAGATION_TIME_STEP,
                                                   force, self.satellite.thruster_isp, direction)
                self.primary_propagator.addForceModel(manoeuvre)

        # get the current state of the primary, at the new time
        self.primary_propagator.resetInitialState(self.primary_current_state)
        self.primary_current_state = self.propag_utils.propagate_(propagator=self.primary_propagator,
                                                                  start_date=current_time,
                                                                  target_date=new_time)
        self.primary_current_pv = self.propag_utils.get_pv_from_state(self.primary_current_state)

        self.satellite_mass = self.primary_current_state.getMass()
        self.tca_time_lapse = self.ref_time.offsetFrom(new_time, UTC)

        # add to the historical recordings
        self.hist_actions.append(action)
        self.hist_primary_states.append(copy.deepcopy(self.primary_current_pv))

        # check if the current state indicates the termination of the episode
        terminated = self._is_done()
        truncated = False

        # set the observation and the additional information
        observation = self._get_obs()
        info = self._get_info()

        # compute the reward
        reward = self._get_reward()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # reset the reference time and the position of the satellite in the orbit
        self.satellite.set_random_tran()
        ref_time_offset_days = self.np_random.uniform(-10.0, 10.0)
        self.ref_time = self.ref_time.shiftedBy(ref_time_offset_days * 24.0 * 3600.0)

        # re-instantiate the propagation and reward utilities classes
        self.propag_utils = propagUtils.PropagationUtilities(satellite=self.satellite,
                                                             ref_time=self.ref_time,
                                                             ref_frame=self.ref_frame)
        self.reward_utils = rewardUtils.RewardUtils()

        # reset the time discretisation variables
        self.set_time_discretisation_variables()

        # set the initial orbits and states for the primary and secondary objects
        self.primary_initial_orbit, primary_tca_state = self.propag_utils.get_orbit_state_from_sat()
        self.secondary_initial_orbit, secondary_tca_state = self.set_orbit_state_for_secondary_object(
            primary_tca_state=primary_tca_state,
            secondary_mass=self.SECONDARY_SC_MASS
        )

        # set the propagators for the primary and secondary satellites
        self.primary_propagator = self.propag_utils.create_propagator(orbit=self.primary_initial_orbit,
                                                                      sc_mass=self.satellite.mass,
                                                                      sc_area=self.satellite.area,
                                                                      sc_reflection=self.satellite.reflection_idx,
                                                                      sc_frame=self.ref_frame,
                                                                      ref_time=self.ref_time,
                                                                      earth_order=self.earth_order,
                                                                      earth_degree=self.earth_degree,
                                                                      use_perturbations=self.use_perturbations,
                                                                      int_min_step=self.INTEGRATOR_MIN_STEP,
                                                                      int_max_step=self.INTEGRATOR_MAX_STEP,
                                                                      int_err_threshold=self.INTEGRATOR_ERR_THRESHOLD)

        self.secondary_propagator = self.propag_utils.create_propagator(orbit=self.secondary_initial_orbit,
                                                                        sc_mass=self.SECONDARY_SC_MASS,
                                                                        sc_area=self.SECONDARY_SC_AREA,
                                                                        sc_reflection=self.SECONDARY_REFLECTION_IDX,
                                                                        sc_frame=self.ref_frame,
                                                                        ref_time=self.ref_time,
                                                                        earth_order=self.earth_order,
                                                                        earth_degree=self.earth_degree,
                                                                        use_perturbations=self.use_perturbations,
                                                                        int_min_step=self.INTEGRATOR_MIN_STEP,
                                                                        int_max_step=self.INTEGRATOR_MAX_STEP,
                                                                        int_err_threshold=self.INTEGRATOR_ERR_THRESHOLD)

        # set the initial states of the propagators
        primary_propagator_initial_date = self.primary_propagator.getInitialState().getDate()
        secondary_propagator_initial_date = self.secondary_propagator.getInitialState().getDate()
        self.primary_initial_state = self.propag_utils.propagate_(propagator=self.primary_propagator,
                                                                  start_date=primary_propagator_initial_date,
                                                                  target_date=self.absolute_time_discretisation_primary[
                                                                      0])
        self.secondary_initial_state = self.propag_utils.propagate_(propagator=self.secondary_propagator,
                                                                    start_date=secondary_propagator_initial_date,
                                                                    target_date=
                                                                    self.absolute_time_discretisation_secondary[0])

        # reset the initial state of the propagator
        self.primary_propagator.resetInitialState(self.primary_initial_state)
        self.secondary_propagator.resetInitialState(self.secondary_initial_state)

        # compute the spacecraft's sequences
        primary_sat_states = self.propag_utils.propagate_sc_states(propagator=self.primary_propagator,
                                                                   initial_state_for_reset=self.primary_initial_state,
                                                                   time_discretisation=self.absolute_time_discretisation_primary)
        secondary_sat_states = self.propag_utils.propagate_sc_states(propagator=self.secondary_propagator,
                                                                     initial_state_for_reset=self.secondary_initial_state,
                                                                     time_discretisation=self.absolute_time_discretisation_secondary)

        # set the initial state sequence for the primary
        self.initial_primary_sc_state_sequence = copy.deepcopy(primary_sat_states)

        # set the current state of the primary satellite
        self.primary_current_state = self.primary_initial_state

        # set the components of the initial observation
        self.primary_current_pv = copy.deepcopy(primary_sat_states[0])
        self.primary_sc_state_sequence = copy.deepcopy(primary_sat_states)
        self.secondary_sc_state_sequence = copy.deepcopy(secondary_sat_states)
        self.tca_time_lapse = self.ref_time.offsetFrom(self.absolute_time_discretisation_primary[0], UTC)
        self.satellite_mass = self.primary_initial_state.getMass()

        # reset the records
        self.hist_actions = []
        self.hist_primary_states = [self.primary_current_pv]
        self.time_step_idx = 0

        # get the orbital period and the indexes in the time discretisation corresponding to it
        orbital_period = self.primary_initial_orbit.getKeplerianPeriod()
        num_time_steps_for_period = int(orbital_period // self.PROPAGATION_TIME_STEP)
        self.time_step_idx_last_orbit = len(self.time_discretisation_primary) - 1 - num_time_steps_for_period

        # set the observation and the additional information
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self):
        if self.render_mode is None:
            return None

    def close(self):
        return None

    def set_orbit_state_for_secondary_object(self,
                                             primary_tca_state: SpacecraftState,
                                             secondary_mass: float = SECONDARY_SC_MASS,
                                             uncertainty_multiplier: float = 10.0):
        # Get the position and velocity of the initial state of the primary object
        init_primary_pos = np.array(primary_tca_state.getPVCoordinates().getPosition().toArray())
        init_primary_vel = np.array(primary_tca_state.getPVCoordinates().getVelocity().toArray())

        # Get the position and velocity of the initial state of the secondary object
        # 1. Position - Get the position difference between the secondary and primary objects sampled from a normal
        # distribution, mean 0, std 1. Multiply it by 10, so the obtained value is in the order of 10 and add it to
        # each component of the pos vector of the primary.
        # 2. Velocity - Get the velocity difference between the secondary and primary objects sampled from a
        # normal distribution, mean 0, std 1. Multiply it by 10, so the obtained value is in the order of 10 and
        # add it to each component of the inverse of the vel vector of the primary. The reason for choosing the
        # inverse is to have a head-on-collision.
        init_secondary_pos = uncertainty_multiplier * self.np_random.standard_normal(3) + init_primary_pos
        init_secondary_vel = uncertainty_multiplier * self.np_random.standard_normal(3) + (-1.0 * init_primary_vel)

        # Create the orbit object of the secondary sat
        sec_pos_vec, sec_vel_vec = Vector3D(init_secondary_pos.tolist()), Vector3D(init_secondary_vel.tolist())
        sec_pv_coord = PVCoordinates(sec_pos_vec, sec_vel_vec)
        secondary_orbit_cart = CartesianOrbit(sec_pv_coord, self.ref_frame,
                                              self.ref_time, Constants.WGS84_EARTH_MU)
        secondary_sc_state = SpacecraftState(secondary_orbit_cart, secondary_mass)

        return secondary_orbit_cart, secondary_sc_state

    def set_time_discretisation_variables(self):
        # the orbit of the primary will be propagated for a number of days, discretised with the
        # time step (PROPAGATION_TIME_STEP), before and after the TCA
        # the orbit of the secondary will be propagated for a quarter of an orbital period approximately, which is
        # around 22.0 minutes, with the same time step, before and after the TCA
        primary_propagation_time_sec = self.PRIMARY_ORBIT_PROPAGATION_PERIOD * 24.0 * 60.0 * 60.0
        secondary_propagation_time_sec = self.SECONDARY_ORBIT_PROPAGATION_PERIOD * 60.0

        self.time_discretisation_primary = self.propag_utils.get_time_discretisation(
            step_duration=self.PROPAGATION_TIME_STEP,
            time_upper_bound=primary_propagation_time_sec)
        self.time_discretisation_secondary = self.propag_utils.get_time_discretisation(
            step_duration=self.PROPAGATION_TIME_STEP,
            time_upper_bound=secondary_propagation_time_sec)

        self.absolute_time_discretisation_primary = self.propag_utils.get_absolute_time_discretisation(
            self.time_discretisation_primary)
        self.absolute_time_discretisation_secondary = self.propag_utils.get_absolute_time_discretisation(
            self.time_discretisation_secondary)

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

    @property
    def use_perturbations(self):
        return self._use_perturbations

    @use_perturbations.setter
    def use_perturbations(self, x):
        self._use_perturbations = x

    @property
    def earth_degree(self):
        return self._earth_degree

    @earth_degree.setter
    def earth_degree(self, x):
        self._earth_degree = x

    @property
    def earth_order(self):
        return self._earth_order

    @earth_order.setter
    def earth_order(self, x):
        self._earth_order = x

    @property
    def primary_current_pv(self):
        return self._primary_current_pv

    @primary_current_pv.setter
    def primary_current_pv(self, x):
        self._primary_current_pv = x

    @property
    def primary_current_state(self):
        return self._primary_current_state

    @primary_current_state.setter
    def primary_current_state(self, x):
        self._primary_current_state = x

    @property
    def primary_initial_orbit(self) -> CartesianOrbit:
        return self._primary_initial_orbit

    @primary_initial_orbit.setter
    def primary_initial_orbit(self, x):
        self._primary_initial_orbit = x

    @property
    def primary_initial_state(self):
        return self._primary_initial_state

    @primary_initial_state.setter
    def primary_initial_state(self, x):
        self._primary_initial_state = x

    @property
    def secondary_initial_orbit(self):
        return self._secondary_initial_orbit

    @secondary_initial_orbit.setter
    def secondary_initial_orbit(self, x):
        self._secondary_initial_orbit = x

    @property
    def secondary_initial_state(self):
        return self._secondary_initial_state

    @secondary_initial_state.setter
    def secondary_initial_state(self, x):
        self._secondary_initial_state = x

    @property
    def time_discretisation_primary(self):
        return self._time_discretisation_primary

    @time_discretisation_primary.setter
    def time_discretisation_primary(self, x):
        self._time_discretisation_primary = x

    @property
    def time_discretisation_secondary(self):
        return self._time_discretisation_secondary

    @time_discretisation_secondary.setter
    def time_discretisation_secondary(self, x):
        self._time_discretisation_secondary = x

    @property
    def primary_propagator(self) -> NumericalPropagator:
        return self._primary_propagator

    @primary_propagator.setter
    def primary_propagator(self, x):
        self._primary_propagator = x

    @property
    def secondary_propagator(self):
        return self._secondary_propagator

    @secondary_propagator.setter
    def secondary_propagator(self, x):
        self._secondary_propagator = x

    @property
    def primary_sc_state_sequence(self):
        return self._primary_sc_state_sequence

    @primary_sc_state_sequence.setter
    def primary_sc_state_sequence(self, x):
        self._primary_sc_state_sequence = x

    @property
    def initial_primary_sc_state_sequence(self):
        return self._initial_primary_sc_state_sequence

    @initial_primary_sc_state_sequence.setter
    def initial_primary_sc_state_sequence(self, x):
        self._initial_primary_sc_state_sequence = x

    @property
    def secondary_sc_state_sequence(self):
        return self._secondary_sc_state_sequence

    @secondary_sc_state_sequence.setter
    def secondary_sc_state_sequence(self, x):
        self._secondary_sc_state_sequence = x

    @property
    def tca_time_lapse(self):
        return self._tca_time_lapse

    @tca_time_lapse.setter
    def tca_time_lapse(self, x):
        self._tca_time_lapse = x

    @property
    def satellite_mass(self):
        return self._satellite_mass

    @satellite_mass.setter
    def satellite_mass(self, x):
        self._satellite_mass = x

    @property
    def hist_actions(self):
        return self._hist_actions

    @hist_actions.setter
    def hist_actions(self, x):
        self._hist_actions = x

    @property
    def hist_primary_states(self):
        return self._hist_primary_states

    @hist_primary_states.setter
    def hist_primary_states(self, x):
        self._hist_primary_states = x

    @property
    def time_step_idx(self):
        return self._time_step_idx

    @time_step_idx.setter
    def time_step_idx(self, x):
        self._time_step_idx = x

    @property
    def time_step_idx_last_orbit(self):
        return self._time_step_idx_last_orbit

    @time_step_idx_last_orbit.setter
    def time_step_idx_last_orbit(self, x):
        self._time_step_idx_last_orbit = x

    @property
    def absolute_time_discretisation_primary(self):
        return self._absolute_time_discretisation_primary

    @absolute_time_discretisation_primary.setter
    def absolute_time_discretisation_primary(self, x):
        self._absolute_time_discretisation_primary = x

    @property
    def absolute_time_discretisation_secondary(self):
        return self._absolute_time_discretisation_secondary

    @absolute_time_discretisation_secondary.setter
    def absolute_time_discretisation_secondary(self, x):
        self._absolute_time_discretisation_secondary = x

    @property
    def propag_utils(self):
        return self._propag_utils

    @propag_utils.setter
    def propag_utils(self, x):
        self._propag_utils = x

    @property
    def reward_utils(self):
        return self._reward_utils

    @reward_utils.setter
    def reward_utils(self, x):
        self._reward_utils = x
