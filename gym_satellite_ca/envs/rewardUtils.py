import numpy as np
import math

from org.orekit.orbits import KeplerianOrbit


class RewardUtils:

    # Constants definition
    COLLISION_AVOIDANCE_NORM_TERM = 1e-2
    RETURN_FAILURE_NORM_TERM = 1e-3
    MAX_PUNISHMENT_RETURN = 5.0
    BOUNDARY_EXIT_NORM_TERM = 1e-3
    MAX_PUNISHMENT_OUT_BOUND = 10
    FUEL_USED_NORM_TERM = 10000.0

    # Orbital differences allowed between the initial orbit and final orbit
    # MAX_SMA_DIFF = 200.0  # meters
    # MAX_ECC_DIFF = 0.001
    # MAX_INC_DIFF = 1.0 * np.pi / 180.0  # rad
    # MAX_PAR_DIFF = 10.0 * np.pi / 180.0  # rad
    # MAX_RAN_DIFF = 1.0 * np.pi / 180.0  # rad
    # try some extreme values
    MAX_SMA_DIFF = 10.0  # meters
    MAX_ECC_DIFF = 1e-8
    MAX_INC_DIFF = 1e-8
    MAX_PAR_DIFF = 10.0 * np.pi / 180.0  # rad
    MAX_RAN_DIFF = 1e-8

    # reward weights
    # wa = 1e-5
    # we = 20
    # wi = 15
    # wpa = 10
    # wran = 15
    # extreme reward weights
    wa = 2 * 1e-3
    we = 1e-2
    wi = 1e-2
    wpa = 0.1
    wran = 1e-3

    def __init__(self):
        pass

    @staticmethod
    def get_diff_between_kepl_elem(kepl_orbit_1: KeplerianOrbit, kepl_orbit_2: KeplerianOrbit):

        # get the differences between keplerian elements
        sma_diff = kepl_orbit_1.getA() - kepl_orbit_2.getA()
        ecc_diff = kepl_orbit_1.getE() - kepl_orbit_2.getE()
        inc_diff = kepl_orbit_1.getI() - kepl_orbit_2.getI()
        par_diff = kepl_orbit_1.getPerigeeArgument() - kepl_orbit_2.getPerigeeArgument()
        ran_diff = kepl_orbit_1.getRightAscensionOfAscendingNode() - kepl_orbit_2.getRightAscensionOfAscendingNode()

        return sma_diff, ecc_diff, inc_diff, par_diff, ran_diff

    def compare_kepl_orbits(self, kepl_orbit_1: KeplerianOrbit, kepl_orbit_2: KeplerianOrbit):
        sma_diff, ecc_diff, inc_diff, par_diff, ran_diff = self.get_diff_between_kepl_elem(
            kepl_orbit_1, kepl_orbit_2
        )
        reward = 0

        if abs(sma_diff) > self.MAX_SMA_DIFF:
            reward -= self.wa * abs(sma_diff)
        if abs(ecc_diff) > self.MAX_ECC_DIFF:
            reward -= self.we * abs(ecc_diff)
        if abs(inc_diff) > self.MAX_INC_DIFF:
            reward -= self.wi * abs(inc_diff)
        if abs(par_diff) > self.MAX_PAR_DIFF:
            reward -= self.wpa * abs(par_diff)
        if abs(ran_diff) > self.MAX_RAN_DIFF:
            reward -= self.wran * abs(ran_diff)

        return reward

    def compute_reward_for_orbit_return(self, current_kepl_elem: np.array, initial_kepl_elem: np.array):
        local_reward = 0
        a_d, e_d, i_d, pa_d, ra_d, _ = current_kepl_elem - initial_kepl_elem
        if abs(a_d) > self.MAX_SMA_DIFF:
            local_reward -= self.wa * abs(a_d)
        if abs(e_d) > self.MAX_ECC_DIFF:
            local_reward -= self.we * abs(e_d)
        if abs(i_d) > self.MAX_INC_DIFF:
            local_reward -= self.wi * abs(i_d)
        if abs(pa_d) > self.MAX_PAR_DIFF:
            local_reward -= self.wpa * abs(pa_d)
        if abs(ra_d) > self.MAX_RAN_DIFF:
            local_reward -= self.wran * abs(ra_d)

        return local_reward

    @staticmethod
    def compute_sequence_of_distances_between_state_seq(primary_sc_state_seq: np.array,
                                                        secondary_sc_state_seq: np.array) -> np.array:
        seq_of_distances = []
        for idx, orb_pos_primary in enumerate(primary_sc_state_seq):
            seq_of_distances.append(np.linalg.norm(orb_pos_primary - secondary_sc_state_seq[idx]))

        return np.array(seq_of_distances)

    @staticmethod
    def compute_dist_between_states(primary_sc_state: np.array,
                                    secondary_sc_state: np.array) -> float:
        return np.linalg.norm(primary_sc_state - secondary_sc_state)

    @staticmethod
    def min_max_norm(x, min_x, max_x):
        return (x - min_x)/(max_x - min_x)
