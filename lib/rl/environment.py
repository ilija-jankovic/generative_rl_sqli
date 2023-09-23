import numpy as np
from .dqn import DQN

class Environment():
    dqn: DQN

    def __init__(self, dqn: DQN):
        self.dqn = dqn

    @staticmethod
    def get_payload(state: np.ndarray):
        chrs = state.tolist()
        chrs = [chr(int(i)) for i in chrs if i != 0.0]
        return ''.join(chrs)

    def _mutate_state(self, state: np.ndarray, action: str):
        # Append character(s) to the state if the state is not
        # completely filled (0.0 represents an empty character slot).
        for i in range(len(state)):
            if(state[i] != 0.0):
                continue

            for j in range(len(action)):
                state_index = i + j
                if(state_index >= len(state)):
                    break

                state[state_index] = ord(action[j])
            
            break

        return state