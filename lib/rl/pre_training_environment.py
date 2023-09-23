import numpy as np
from typing import List

from .dqn import DQN
from .environment import Environment

import os


class PreTrainingEnvironment(Environment):
    __injections: List[str]

    def __load_injections(self):
        dirname = os.path.dirname(__file__)
        parsed_injections_path = os.path.join(dirname, '../../parsed_injections.txt')

        with open(parsed_injections_path, 'r') as f:
            self.__injections = f.read().split('\n')

    def __init__(self, dqn: DQN):
        super().__init__(dqn)
        self.__load_injections()
    
    def perform_action(self, action: str, state: np.ndarray):
        new_state = self._mutate_state(state, action)
        payload_fragment = self.get_payload(new_state)

        for injection in self.__injections:
            if payload_fragment in injection:
                print(payload_fragment)
                return new_state, 1, False

        return self.dqn.create_empty_state(), -1, True 