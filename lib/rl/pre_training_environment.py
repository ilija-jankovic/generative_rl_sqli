import numpy as np
import os
from typing import List
from .dqn import DQN
from .environment import Environment

class PreTrainingEnvironment(Environment):
    __encoded_injections: List[List[int]]
    __sql_syntax: List[str]
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
        sql_syntax_path = os.path.join(dirname, '../../sql_list.txt')
        columns_path = os.path.join(dirname, '../../columns.txt')
        tables_path = os.path.join(dirname, '../../tables.txt')

        with open(encoded_injections_path, 'r') as f:
            data = f.read()
            self.__parse_encoded_tokens(data)
        f.close()

        with open(sql_syntax_path, 'r') as f:
            self.__sql_syntax = f.read().split('\n')
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

    def __is_action_valid(self, action_index: int, injection_action_index: int):
        # If the expected index is -1, match any non-SQL syntax (such as column/table names,
        # or numbers/ASCII characters).
        if injection_action_index == -1:
            decoded_action = self.actions[action_index]
            return decoded_action not in self.__sql_syntax

        return action_index == injection_action_index
    
    def perform_termination_action(self, state: np.ndarray):
        if self._get_available_action_slot_index(state) == -1:
            return self.dqn.create_empty_state(), 0, True

        # TODO: Sort injections by ascending order of size and reward
        # based on the proportion of the injection matching.

        highest_reward = 0

        for encoded_injection in self.__encoded_injections:
            reward = 0

            for state_index in range(len(encoded_injection)):
                action_index = int(state[state_index])
                if action_index == -1:
                    break
                
                injection_action_index = int(state[state_index])
                
                if(self.__is_action_valid(action_index, injection_action_index)):
                    reward += 1
            
            highest_reward = max(reward, highest_reward)

        print(self.get_payload(state))
        state = self.dqn.create_empty_state()

        return state, highest_reward, True 