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
