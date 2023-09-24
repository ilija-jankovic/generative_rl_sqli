import numpy as np
import os
from typing import List
from .dqn import DQN
from .environment import Environment

class PreTrainingEnvironment(Environment):
    __encoded_injections: List[List[int]]
    __columns: List[str]
    __tables: List[str]

    def __parse_encoded_tokens(self, data: str):
        self.__encoded_injections = []
        for payload in data.split('\n'):
            self.__encoded_injections.append([])
            for token in payload.split(','):
                self.__encoded_injections[-1].append(int(token))

    def __load_injections(self):
        dirname = os.path.dirname(__file__)

        encoded_injections_path = os.path.join(dirname, '../../parsed_injections_indexed.txt')
        columns_path = os.path.join(dirname, '../../columns.txt')
        tables_path = os.path.join(dirname, '../../tables.txt')

        with open(encoded_injections_path, 'r') as f:
            data = f.read()
            self.__parse_encoded_tokens(data)

        f.close()

        with open(columns_path, 'r') as f:
            self.__columns = f.read().split('\n')
        f.close()
        
        with open(tables_path, 'r') as f:
            self.__tables = f.read().split('\n')
        f.close()

    def __init__(self, dqn: DQN, actions: List[str]):
        super().__init__(dqn, actions)
        self.__load_injections()
    
    def perform_action(self, action_index: int, state: np.ndarray):
        slot_index = self._get_available_action_slot_index(state)
        if slot_index == -1:
            return self.dqn.create_empty_state(), 0, True
    
        new_state = self._mutate_state(state, action_index)

        longer_than_all_injections = True
        for encoded_injection in self.__encoded_injections:
            if(slot_index >= len(encoded_injection) - 1):
                continue

            longer_than_all_injections = False

            expected_action_index = encoded_injection[slot_index]
            if action_index == expected_action_index:
                print(self.get_payload(new_state))
                return new_state, 1, False
            
            if expected_action_index != -1:
                continue
            
            # If the expected index is -1, match any table or column name.
            decoded_action = self.actions[action_index]
            if decoded_action in self.__columns or decoded_action in self.__tables:
                print(self.get_payload(new_state))
                return new_state, 1, False

        return self.dqn.create_empty_state(), 0 if longer_than_all_injections else -1, True 