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
from org.orekit.frames import FramesFactory, LOFType
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
DEFAULT_REF_FRAME = FramesFactory.getEME2000()
LOCAL_ORBITAL_FRAME = LOFType.LVLH_CCSDS
DEFAULT_RESET_OPTIONS = {
    "propagator": "numerical",
    "generate_sat": False
}


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
    MIN_ORBIT_DISTANCE_TO_EARTH = 6500000.0  # meters
    MAX_ORBIT_DISTANCE_TO_EARTH = 8000000.0  # meters
    # INITIAL_ORBIT_RADIUS_BOUND - the max distance allowed between the satellite in the final
    # orbit and the same satellite in the initial orbit such that the final orbit can be considered equivalent to the
    # initial one
    INITIAL_ORBIT_RADIUS_BOUND = 300.0  # meters

    # Collision constants
    COLLISION_MIN_DISTANCE = 2000.0  # meters

    def __init__(self,
                 satellite: satDataClass.SatelliteData,
                 ref_time: AbsoluteDate = DEFAULT_REF_TIME,
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
        self._primary_init_kepl_orbit = None
        self._primary_initial_orbit = None
        self._primary_initial_state = None
        self._initial_primary_sc_state_sequence = None
        self._primary_initial_kepl_elements = None
        self._primary_current_state = None
        self._secondary_initial_orbit = None
        self._secondary_initial_state = None
        self._collision_diffs = None
        self._min_collision_diff = None

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
        self._primary_keplerian_propagator = None
        self._secondary_propagator = None

        # set the action space
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(3,), dtype=np.float64)

        # set the observation space
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(9,),
                                            dtype=np.float64)
        self.normalise_observations = False

        # instantiate the components of the observation space
        self._primary_current_pv = None
        self._primary_sc_state_sequence = None
        self._secondary_sc_state_sequence = None
        self._tca_time_lapse = None
        self._satellite_mass = None
        self.initial_time_lapse = 0.0

        # instantiate the historical recordings
        self._hist_actions = []
        self._hist_primary_states = []
        self._hist_kepl_elements = []
        self._hist_primary_at_collision_states = []
        self._hist_min_coll_dist = []
        self._time_step_idx = 0

        # instantiate conditions checkers
        self._collision_avoided = True
        self._returned_to_init_orbit = True
        self._drifted_out_of_bounds = False
        self._fuel_used_perc = 0.0

        # set up some environment variables
        self._truncated = False
        self.window = None
        self.clock = None
        self.close()

    def _get_obs(self):
        if self.normalise_observations:
            data_dict = self.normalise_obs()
        else:
            data_dict = {"primary_current_pv": self.primary_current_pv,
                         "min_collision_diff": np.array([self.min_collision_diff]),
                         "tca_time_lapse": np.array([self.tca_time_lapse]),
                         "primary_sc_mass": np.array([self.satellite_mass])
                         }
        observations = []
        for key in data_dict.keys():
            observations.extend(data_dict[key].tolist())
        return np.array(observations)

    def normalise_obs(self):
        norm_primary_current_pv = self.normalise_kepl_elements(kepl_elements=self.primary_current_pv)
        norm_min_coll_dist = self.normalise_min_collision_distance(min_collision_distance=self.min_collision_diff)
        norm_tca_time_lapse = self.normalise_tca_time_laps(tca_time_lapse=self.tca_time_lapse)
        norm_satellite_mass = self.normalise_satellite_mass(satellite_mass=self.satellite_mass)
        norm_obs_dict = {
            "primary_current_pv": norm_primary_current_pv,
            "min_collision_diff": norm_min_coll_dist,
            "tca_time_lapse": norm_tca_time_lapse,
            "primary_sc_mass": norm_satellite_mass
        }
        return norm_obs_dict

    def get_intermediary_info(self):
        return {
            "time_step_idx": self.time_step_idx,
            "collision_avoided": self.collision_avoided,
            "returned_to_init_orbit": self.returned_to_init_orbit,
            "drifted_out_of_bounds": self.drifted_out_of_bounds,
            "fuel_used_perc": self.fuel_used_perc
        }

    def get_final_info(self):
        return {
            "init_kepl_elements": self.primary_initial_kepl_elements,
            "historical_actions": self.hist_actions,
            "historical_primary_sequence": self.hist_primary_states,
            "hist_primary_at_collision_states": self.hist_primary_at_collision_states,
            "min_collision_distances": self.hist_min_coll_dist,
            "collision_distance": self.COLLISION_MIN_DISTANCE,
            "initial_orbit_radius_bound": self.INITIAL_ORBIT_RADIUS_BOUND,
            "max_altitude_diff_allowed": self.MAX_ALTITUDE_DIFF_ALLOWED,
            "time_step_idx": self.time_step_idx,
            "collision_avoided": self.collision_avoided,
            "returned_to_init_orbit": self.returned_to_init_orbit,
            "fuel_used_perc": self.fuel_used_perc,
            "collision_idx": int(len(self.time_discretisation_primary) // 2)
        }

    def _get_info(self):
        if self._is_done() or self._is_truncated():
            return self.get_final_info()
        return self.get_intermediary_info()

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

        # get the reward for avoiding the collision (punishment for not avoiding it)
        # check if the current state of the primary corresponds to a state in which the position of the secondary
        # satellite is known and compute the distance between the states
        # check if the collision has been avoided
        collision_reward_contrib = - max(
            (self.COLLISION_MIN_DISTANCE - self.min_collision_diff) / self.COLLISION_MIN_DISTANCE, 0)

        current_absolute_time = self.time_discretisation_primary[self.time_step_idx]
        if current_absolute_time in self.time_discretisation_secondary:
            if collision_reward_contrib < 0:
                self.collision_avoided = False

        # get the negative reward for not returning to the initial orbit by the end of the event
        return_init_orbit_contrib = 0
        if self.time_step_idx >= self.time_step_idx_last_orbit:
            return_init_orbit_contrib = self.reward_utils.compute_reward_for_orbit_return(
                current_kepl_elem=self.primary_current_pv,
                initial_kepl_elem=self.primary_initial_kepl_elements)

            if return_init_orbit_contrib == 0:
                self.truncated = True
                self.returned_to_init_orbit = True
                return 1.0

        # get the negative reward for using fuel
        fuel_used_perc = ((self.primary_initial_state.getMass() - self.satellite_mass) /
                          self.primary_initial_state.getMass())
        self.fuel_used_perc = fuel_used_perc
        fuel_usage_contrib = - self.reward_utils.FUEL_USED_NORM_TERM * fuel_used_perc

        # compute the overall reward
        reward_ = collision_reward_contrib + return_init_orbit_contrib + fuel_usage_contrib

        return reward_

    def _is_done(self) -> bool:
        return self._time_step_idx >= len(self.absolute_time_discretisation_primary) - 1

    def _is_truncated(self) -> bool:
        return self.truncated

    def step(self, action):
        # The force in each direction cannot be greater than the maximum force of the thruster
        for action_idx in range(len(action)):
            if action[action_idx] <= 0:
                action[action_idx] = max(-1.0, action[action_idx])
            else:
                action[action_idx] = min(1.0, action[action_idx])

        # get the current and new time, with the time step increased
        current_time = self.absolute_time_discretisation_primary[self.time_step_idx]
        self.time_step_idx += 1
        new_time = self.absolute_time_discretisation_primary[self.time_step_idx]
        mass_before_action = self.primary_current_state.getMass()

        # Assume there are 3 pairs of thrusters, each of them can be used independently
        for i in range(3):
            if abs(action[i]) > 0.3:
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

        # get the estimated collision differences
        if new_time.offsetFrom(self.absolute_time_discretisation_secondary[-1], UTC) < 0:
            self.primary_keplerian_propagator.resetInitialState(self.primary_current_state)
            self.secondary_propagator.resetInitialState(self.secondary_initial_state)
            primary_sat_states = self.propag_utils.propagate_sc_states(propagator=self.primary_keplerian_propagator,
                                                                       initial_state_for_reset=self.primary_current_state,
                                                                       time_discretisation=self.absolute_time_discretisation_secondary,
                                                                       UTC=UTC)
            secondary_sat_states = self.propag_utils.propagate_sc_states(propagator=self.secondary_propagator,
                                                                         initial_state_for_reset=self.secondary_initial_state,
                                                                         time_discretisation=self.absolute_time_discretisation_secondary,
                                                                         UTC=UTC)
            if len(secondary_sat_states) > len(primary_sat_states):
                secondary_sat_states = secondary_sat_states[-len(primary_sat_states):]
            self.collision_diffs = self.propag_utils.get_pv_diff_between_sequences_of_states(
                primary_sc_states=primary_sat_states,
                secondary_sc_states=secondary_sat_states)
        else:
            self.collision_diffs = [self.COLLISION_MIN_DISTANCE + 1]

        # set the observations from this step
        mass_after_action = self.primary_current_state.getMass()
        self.fuel_used_perc = (mass_before_action - mass_after_action) * 1e6

        self.primary_current_pv = copy.deepcopy(self.propag_utils.get_kepl_elem_from_state(self.primary_current_state))
        self.min_collision_diff = min(self.collision_diffs)
        self.satellite_mass = self.primary_current_state.getMass()
        self.tca_time_lapse = self.ref_time.offsetFrom(new_time, UTC)

        # add to the historical recordings
        self.hist_actions.append(action)
        self.hist_primary_states.append(copy.deepcopy(self.primary_current_pv))
        self.hist_min_coll_dist.append(self.min_collision_diff)

        # compute the reward
        reward = self._get_reward()

        # set the observation and the additional information
        observation = self._get_obs()
        info = self._get_info()

        # check if the current state indicates the termination of the episode
        terminated = self._is_done()
        truncated = self._is_truncated()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        if options is None:
            options = DEFAULT_RESET_OPTIONS
        super().reset(seed=seed)

        # regenerate satellite object if required
        if options["generate_sat"]:
            # generate satellite object
            self.satellite = satDataClass.generate_random_sat_class(sma_min=6785000.0, sma_max=6850000.0,
                                                                    ecc_min=0.01, ecc_max=0.2,
                                                                    inc_min=20, inc_max=40,
                                                                    argp_min=-180, argp_max=180,
                                                                    raan_min=-180, raan_max=180,
                                                                    tran_min=-180, tran_max=180)

        # reset the info states
        self.truncated = False
        self.returned_to_init_orbit = False
        self.collision_avoided = True
        self.fuel_used_perc = 0.0

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
        self.primary_init_kepl_orbit, self.primary_initial_orbit, primary_tca_state = (
            self.propag_utils.get_orbit_state_from_sat())
        self.secondary_initial_orbit, secondary_tca_state = self.set_orbit_state_for_secondary_object(
            primary_tca_state=primary_tca_state,
            secondary_mass=self.SECONDARY_SC_MASS
        )

        # set the propagators for the primary and secondary satellites
        self.primary_propagator = self.propag_utils.create_propagator(orbit=self.primary_init_kepl_orbit,
                                                                      sc_mass=self.satellite.mass,
                                                                      sc_area=self.satellite.area,
                                                                      sc_reflection=self.satellite.reflection_idx,
                                                                      sc_frame=self.ref_frame,
                                                                      earth_order=self.earth_order,
                                                                      earth_degree=self.earth_degree,
                                                                      use_perturbations=self.use_perturbations,
                                                                      int_min_step=self.INTEGRATOR_MIN_STEP,
                                                                      int_max_step=self.INTEGRATOR_MAX_STEP,
                                                                      int_err_threshold=self.INTEGRATOR_ERR_THRESHOLD)
        self.primary_keplerian_propagator = self.propag_utils.create_keplerian_propagator(
            orbit=self.primary_initial_orbit,
            sc_mass=self.satellite.mass)

        self.secondary_propagator = self.propag_utils.create_keplerian_propagator(
            orbit=self.secondary_initial_orbit,
            sc_mass=self.SECONDARY_SC_MASS)

        # set the initial states of the propagators
        primary_propagator_initial_date = self.primary_propagator.getInitialState().getDate()
        primary_keplerian_propagator_initial_date = self.primary_keplerian_propagator.getInitialState().getDate()
        secondary_propagator_initial_date = self.secondary_propagator.getInitialState().getDate()

        self.primary_initial_state = self.propag_utils.propagate_(propagator=self.primary_propagator,
                                                                  start_date=primary_propagator_initial_date,
                                                                  target_date=self.absolute_time_discretisation_primary[
                                                                      0])
        primary_collision_state = self.propag_utils.propagate_(propagator=self.primary_keplerian_propagator,
                                                               start_date=primary_keplerian_propagator_initial_date,
                                                               target_date=self.absolute_time_discretisation_secondary[
                                                                   0])
        self.secondary_initial_state = self.propag_utils.propagate_(propagator=self.secondary_propagator,
                                                                    start_date=secondary_propagator_initial_date,
                                                                    target_date=
                                                                    self.absolute_time_discretisation_secondary[0])

        # reset the initial state of the propagator
        self.primary_propagator.resetInitialState(self.primary_initial_state)
        self.secondary_propagator.resetInitialState(self.secondary_initial_state)

        # compute the spacecraft's states at the times of collision
        primary_sat_states = self.propag_utils.propagate_sc_states(propagator=self.primary_keplerian_propagator,
                                                                   initial_state_for_reset=primary_collision_state,
                                                                   time_discretisation=self.absolute_time_discretisation_secondary,
                                                                   UTC=UTC)
        secondary_sat_states = self.propag_utils.propagate_sc_states(propagator=self.secondary_propagator,
                                                                     initial_state_for_reset=self.secondary_initial_state,
                                                                     time_discretisation=self.absolute_time_discretisation_secondary,
                                                                     UTC=UTC)

        # compute the deltas of the states at the times of collision
        self.collision_diffs = self.propag_utils.get_pv_diff_between_sequences_of_states(
            primary_sc_states=primary_sat_states,
            secondary_sc_states=secondary_sat_states)

        # set the current state of the primary satellite
        self.primary_current_state = self.primary_initial_state
        self.primary_initial_kepl_elements = copy.deepcopy(
            self.propag_utils.get_kepl_elem_from_state(self.primary_initial_state))

        # set the components of the initial observation
        self.primary_current_pv = copy.deepcopy(self.propag_utils.get_kepl_elem_from_state(self.primary_initial_state))
        self.min_collision_diff = min(self.collision_diffs)
        self.tca_time_lapse = self.ref_time.offsetFrom(self.absolute_time_discretisation_primary[0], UTC)
        self.satellite_mass = self.primary_initial_state.getMass()

        # reset the records
        self.hist_actions = []
        self.hist_primary_states = [self.primary_current_pv]
        self.hist_primary_at_collision_states = []
        self.time_step_idx = 0
        self.initial_time_lapse = self.ref_time.offsetFrom(self.absolute_time_discretisation_primary[0], UTC)

        # get the orbital period and the indexes in the time discretisation corresponding to it
        orbital_period = self.primary_initial_orbit.getKeplerianPeriod()
        num_time_steps_for_period = int(orbital_period // self.PROPAGATION_TIME_STEP)
        self.time_step_idx_last_orbit = int(
            (len(self.time_discretisation_primary) - 1) // 2) + num_time_steps_for_period

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

    def normalise_kepl_elements(self, kepl_elements):
        sma, ecc, inc, par, ran, tan = copy.deepcopy(kepl_elements)
        init_sma, init_ecc, init_inc, init_par, init_ran, init_tan = copy.deepcopy(self.primary_initial_kepl_elements)
        sma_norm = self.reward_utils.min_max_norm(x=sma,
                                                  min_x=init_sma - 300,
                                                  max_x=init_sma + 300)
        ecc_norm = self.reward_utils.min_max_norm(x=ecc,
                                                  min_x=init_ecc - 1e-4,
                                                  max_x=init_ecc + 1e-4)
        inc_norm = self.reward_utils.min_max_norm(x=inc,
                                                  min_x=init_inc - 1e-5,
                                                  max_x=init_inc + 1e-5)
        par_norm = self.reward_utils.min_max_norm(x=par,
                                                  min_x=init_par - 1e-5,
                                                  max_x=init_par + 1e-5)
        ran_norm = self.reward_utils.min_max_norm(x=ran,
                                                  min_x=init_ran - 1e-5,
                                                  max_x=init_ran + 1e-5)
        tan_norm = self.reward_utils.min_max_norm(x=tan,
                                                  min_x=-np.pi,
                                                  max_x=np.pi)
        return np.array([sma_norm, ecc_norm, inc_norm, par_norm, ran_norm, tan_norm])

    def normalise_min_collision_distance(self, min_collision_distance):
        # the formula was tested empirically to bring the values between -1 and 1
        return np.array([np.log(min_collision_distance + 1.0 / self.COLLISION_MIN_DISTANCE) / 5.0])

    def normalise_tca_time_laps(self, tca_time_lapse):
        return np.array([tca_time_lapse/self.initial_time_lapse])

    def normalise_satellite_mass(self, satellite_mass):
        # the minimum mass is obtained by running a full episode with all the engines running at all times
        # this should be recalculated if the thrust value or thrust isp change
        satellite_mass_norm = self.reward_utils.min_max_norm(x=satellite_mass,
                                                             min_x=99.99471837987426,
                                                             max_x=self.primary_initial_state.getMass())
        return np.array([satellite_mass_norm])


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
    def primary_init_kepl_orbit(self):
        return self._primary_init_kepl_orbit

    @primary_init_kepl_orbit.setter
    def primary_init_kepl_orbit(self, x):
        self._primary_init_kepl_orbit = x

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
    def primary_keplerian_propagator(self) -> NumericalPropagator:
        return self._primary_keplerian_propagator

    @primary_keplerian_propagator.setter
    def primary_keplerian_propagator(self, x):
        self._primary_keplerian_propagator = x

    @property
    def collision_diffs(self):
        return self._collision_diffs

    @collision_diffs.setter
    def collision_diffs(self, x):
        self._collision_diffs = x

    @property
    def min_collision_diff(self):
        return self._min_collision_diff

    @min_collision_diff.setter
    def min_collision_diff(self, x):
        self._min_collision_diff = x

    @property
    def primary_initial_kepl_elements(self):
        return self._primary_initial_kepl_elements

    @primary_initial_kepl_elements.setter
    def primary_initial_kepl_elements(self, x):
        self._primary_initial_kepl_elements = x

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
    def hist_kepl_elements(self):
        return self._hist_kepl_elements

    @hist_kepl_elements.setter
    def hist_kepl_elements(self, x):
        self._hist_kepl_elements = x

    @property
    def hist_primary_at_collision_states(self):
        return self._hist_primary_at_collision_states

    @hist_primary_at_collision_states.setter
    def hist_primary_at_collision_states(self, x):
        self._hist_primary_at_collision_states = x

    @property
    def hist_min_coll_dist(self):
        return self._hist_min_coll_dist

    @hist_min_coll_dist.setter
    def hist_min_coll_dist(self, x):
        self._hist_min_coll_dist = x

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
    def collision_avoided(self):
        return self._collision_avoided

    @collision_avoided.setter
    def collision_avoided(self, x):
        self._collision_avoided = x

    @property
    def returned_to_init_orbit(self):
        return self._returned_to_init_orbit

    @returned_to_init_orbit.setter
    def returned_to_init_orbit(self, x):
        self._returned_to_init_orbit = x

    @property
    def drifted_out_of_bounds(self):
        return self._drifted_out_of_bounds

    @drifted_out_of_bounds.setter
    def drifted_out_of_bounds(self, x):
        self._drifted_out_of_bounds = x

    @property
    def fuel_used_perc(self):
        return self._fuel_used_perc

    @fuel_used_perc.setter
    def fuel_used_perc(self, x):
        self._fuel_used_perc = x

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

    @property
    def truncated(self):
        return self._truncated

    @truncated.setter
    def truncated(self, x):
        self._truncated = x
