import numpy as np
from abc import ABC
from typing import List
from .dqn import DQN

class Environment(ABC):
    actions: List[str]
    dqn: DQN

    def __init__(self, dqn: DQN, actions: List[str]):
        self.dqn = dqn
        self.actions = actions

    def _get_available_action_slot_index(self, state: np.ndarray):
        '''
        Returns `-1` if no empty action slot remaining.
        '''
        for i in range(len(state)):
            if state[i] == -1:
                return i
        return -1

    def get_payload(self, state: np.ndarray):
        payload = ''

        slot_index = self._get_available_action_slot_index(state)
        end = len(state) if slot_index == -1 else slot_index
        for i in range(end):
            payload += self.actions[int(state[i])]
        
        return payload

    def _mutate_state(self, state: np.ndarray, action_index: int):
        slot_index = self._get_available_action_slot_index(state)
        if slot_index == -1:
            return state

        state[slot_index] = action_index
        return state