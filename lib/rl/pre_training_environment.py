import numpy as np
import os
from typing import List
from .environment import Environment

class PreTrainingEnvironment(Environment):
    # Expected to be sorted from smallest to largest.
    __encoded_injections: List[List[int]]

    __sql_syntax: List[str]
    __columns: List[str]
    __tables: List[str]

    def __init__(self, dictionary: List[str], action_size: int, state_size: int):
        super().__init__(dictionary, action_size, state_size)
        self.__load_injections()

    def __parse_encoded_tokens(self, data: str):
        self.__encoded_injections = []

        for payload in data.split('\n'):
            self.__encoded_injections.append([])

            for token in payload.split(','):
                self.__encoded_injections[-1].append(int(token))

    def __remove_oversized_injections(self):
        self.__encoded_injections = list(filter(
            lambda injection:len(injection) <= self.action_size, self.__encoded_injections))
    
    def __load_injections(self):
        dirname = os.path.dirname(__file__)

        encoded_injections_path = os.path.join(dirname, '../../parsed_injections_indexed.txt')
        sql_syntax_path = os.path.join(dirname, '../../sql_list.txt')
        columns_path = os.path.join(dirname, '../../columns.txt')
        tables_path = os.path.join(dirname, '../../tables.txt')

        with open(encoded_injections_path, 'r') as f:
            data = f.read()

            self.__parse_encoded_tokens(data)
            self.__remove_oversized_injections()
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

    def __is_action_token_valid(self, action_dict_index: int, injection_dict_index: int):
        # If the expected index is -1, match any non-SQL syntax (such as column/table names,
        # or numbers/ASCII characters).
        if injection_dict_index == -1:
            decoded_action = self.dictionary[action_dict_index]
            return decoded_action not in self.__sql_syntax

        return action_dict_index == injection_dict_index
    
    def _perform_action(self, action: np.ndarray):
        action_length = len(action)

        # Initialise to lowest possible normalised reward.
        highest_norm_reward = -1.0

        for encoded_injection in self.__encoded_injections:
            injection_length = len(encoded_injection)
            comparison_count = max(injection_length, action_length)

            reward = 0

            for action_index in range(comparison_count):
                if action_index >= action_length or action_index >= injection_length:
                    reward -= 1
                    continue

                action_dict_index = self._get_token_index(action[action_index])

                if action_dict_index == -1:
                    reward -= comparison_count - action_index + 1
                    break

                injection_dict_index = int(encoded_injection[action_index])
                
                action_valid = self.__is_action_token_valid(action_dict_index, injection_dict_index)
                reward += 1 if action_valid else -1
            
            reward /= comparison_count
            highest_norm_reward = max(reward, highest_norm_reward)

        print(self.get_payload(action))
        
        state = action

        return state, highest_norm_reward, True 