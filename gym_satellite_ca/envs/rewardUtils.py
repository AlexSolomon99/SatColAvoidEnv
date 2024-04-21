import numpy as np


class RewardUtils:

    def __init__(self):
        pass

    def compute_sequence_of_distances_between_states(self,
                                                     primary_sc_state_seq: np.array,
                                                     secondary_sc_state_seq: np.array) -> np.array:
        seq_of_distances = []
        for idx, orb_pos_primary in enumerate(primary_sc_state_seq):
            seq_of_distances.append(np.linalg.norm(orb_pos_primary - secondary_sc_state_seq[idx]))

        return np.array(seq_of_distances)
