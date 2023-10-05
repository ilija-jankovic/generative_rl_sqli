import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Environment(ABC):
    dictionary: List[str]
    action_size: int
    state_size: int
    
    __attempted_actions: List[np.ndarray] = []

    def __init__(self, dictionary: List[str], action_size: int, state_size: int):
        self.dictionary = dictionary
        self.action_size = action_size
        self.state_size = state_size

    def _get_token_index(self, action_class: float):
        # >= 1.0 condition ensures the max action class value rounds
        # down to the maximum allowed index.
        if action_class < -1.0 or action_class >= 1.0:
            return -1
        
        denormalised = (action_class + 1.0) * len(self.dictionary) / 2
        return int(denormalised)

    def _get_token(self, action_class: float):
        index = self._get_token_index(action_class)
        return '' if index == -1 else self.dictionary[index]

    def get_payload(self, action: np.ndarray):
        chrs = [self._get_token(cls) for cls in action]
        return ''.join(chrs)

    def _record_action(self, action: np.ndarray):
        self.__attempted_actions.append(action)

    def action_attempted(self, action: np.ndarray):
        return any(np.array_equal(action, attempted_action)
                   for attempted_action in self.__attempted_actions)
    
    @abstractmethod
    def _perform_action(self, action: np.ndarray):
        pass

    def perform_action(self, action: np.ndarray):
        if self.action_attempted(action):
            return self.create_empty_state(), -1.0, True
        
        self._record_action(action)
        return self._perform_action(action)

    def create_empty_state(self):
        return np.array([-1] * self.state_size, dtype='float32')