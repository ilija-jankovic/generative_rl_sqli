import numpy as np
import os
from typing import List
from .dqn import DQN
from .environment import Environment

class PreTrainingEnvironment(Environment):
    # Expected to be sorted from smallest to largest.
    __encoded_injections: List[List[int]]

    __sql_syntax: List[str]
    __columns: List[str]
    __tables: List[str]

    def __init__(self, dqn: DQN, actions: List[str], state_length: int):
        super().__init__(dqn, actions)
        self.__state_length = state_length
        self.__load_injections(state_length)

    def __parse_encoded_tokens(self, data: str):
        self.__encoded_injections = []

        for payload in data.split('\n'):
            self.__encoded_injections.append([])

            for token in payload.split(','):
                self.__encoded_injections[-1].append(int(token))

    def __remove_oversized_injections(self, state_length: int):
        self.__encoded_injections = list(filter(
            lambda injection:len(injection) <= self.__state_length, self.__encoded_injections))
    
    def __load_injections(self, state_length: int):
        dirname = os.path.dirname(__file__)

        encoded_injections_path = os.path.join(dirname, '../../parsed_injections_indexed.txt')
        sql_syntax_path = os.path.join(dirname, '../../sql_list.txt')
        columns_path = os.path.join(dirname, '../../columns.txt')
        tables_path = os.path.join(dirname, '../../tables.txt')

        with open(encoded_injections_path, 'r') as f:
            data = f.read()

            self.__parse_encoded_tokens(data)
            self.__remove_oversized_injections(state_length)
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

    def __is_action_valid(self, action_index: int, injection_action_index):
        if action_index == -1 or injection_action_index == None:
            return False
        
        # If the expected index is -1, match any non-SQL syntax (such as column/table names,
        # or numbers/ASCII characters).
        if injection_action_index == -1:
            decoded_action = self.actions[action_index]
            return decoded_action not in self.__sql_syntax

        return action_index == injection_action_index
    
    def perform_termination_action(self, state: np.ndarray):
        filled_state_length = self._get_filled_state_length(state)
        if filled_state_length == len(state):
            return self.dqn.create_empty_state(), 0, True

        # TODO: Sort injections by ascending order of size and reward
        # based on the proportion of the injection matching.

        # Initialise to lowest possible normalised reward.
        highest_norm_reward = -1.0

        for encoded_injection in self.__encoded_injections:
            injection_length = len(encoded_injection)
            comparison_count = max(injection_length, filled_state_length)

            reward = 0

            for state_index in range(comparison_count):
                action_index = int(state[state_index])

                injection_action_index = None if \
                    state_index >= injection_length else \
                        int(encoded_injection[state_index])
                
                action_valid = self.__is_action_valid(action_index, injection_action_index)
                reward += 1 if action_valid else -1
            
            reward /= comparison_count
            highest_norm_reward = max(reward, highest_norm_reward)

        print(self.get_payload(state))
        state = self.dqn.create_empty_state()

        return state, highest_norm_reward, True 