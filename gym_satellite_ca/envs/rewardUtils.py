import numpy as np


class RewardUtils:

    def __init__(self):
        pass

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
