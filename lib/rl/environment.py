import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Environment(ABC):
    dictionary: List[str]
    action_size: int
    state_size: int
    
    __attempted_payloads: List[str] = []

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

    def _record_payload(self, action: np.ndarray):
        payload = self.get_payload(action)
        self.__attempted_payloads.append(payload)
        
        return payload

    def payload_attempted(self, action: np.ndarray):
        payload = self.get_payload(action)
        return payload in self.__attempted_payloads
    
    @abstractmethod
    def perform_termination_action(self, action: np.ndarray):
        pass

    def create_empty_state(self):
        return np.array([-1] * self.state_size, dtype='float32')