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
    FUEL_USED_NORM_TERM = 10.0

    # Orbital differences allowed between the initial orbit and final orbit
    MAX_SMA_DIFF = 800.0  # meters
    MAX_ECC_DIFF = 0.001
    MAX_INC_DIFF = 2.0  # deg
    MAX_PAR_DIFF = 5.0  # deg
    MAX_RAN_DIFF = 5.0  # deg

    def __init__(self):
        pass

    @staticmethod
    def get_diff_between_kepl_elem(kepl_orbit_1: KeplerianOrbit, kepl_orbit_2: KeplerianOrbit):

        # get the differences between keplerian elements
        sma_diff = abs(kepl_orbit_1.getA() - kepl_orbit_2.getA())
        ecc_diff = abs(kepl_orbit_1.getE() - kepl_orbit_2.getE())
        inc_diff = abs(kepl_orbit_1.getI() - kepl_orbit_2.getI())
        par_diff = abs(kepl_orbit_1.getPerigeeArgument() - kepl_orbit_2.getPerigeeArgument())
        ran_diff = abs(kepl_orbit_1.getRightAscensionOfAscendingNode() - kepl_orbit_2.getRightAscensionOfAscendingNode())

        return sma_diff, ecc_diff, inc_diff, par_diff, ran_diff

    def compare_kepl_orbits(self, kepl_orbit_1: KeplerianOrbit, kepl_orbit_2: KeplerianOrbit):
        sma_diff, ecc_diff, inc_diff, par_diff, ran_diff = self.get_diff_between_kepl_elem(
            kepl_orbit_1, kepl_orbit_2
        )
        reward = 0

        if sma_diff > self.MAX_SMA_DIFF:
            reward -= 1.0
        if ecc_diff > self.MAX_ECC_DIFF:
            reward -= 1.0
        if inc_diff > self.MAX_INC_DIFF:
            reward -= 1.0
        if par_diff > self.MAX_PAR_DIFF:
            reward -= 1.0
        if ran_diff > self.MAX_RAN_DIFF:
            reward -= 1.0

        return reward

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
