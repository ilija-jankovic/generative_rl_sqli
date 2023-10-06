import numpy as np
import os
from typing import List
from .environment import Environment

class PreTrainingEnvironment(Environment):
    encoded_injections: List[List[int]]

    sql_syntax: List[str]
    columns: List[str]
    tables: List[str]

    def __init__(self, dictionary: List[str], action_size: int, state_size: int,
                 encoded_injections: List[List[int]], sql_syntax: List[str],
                     columns: List[str], tables: List[str]):
        super().__init__(dictionary, action_size, state_size)
        
        self.encoded_injections = encoded_injections
        self.sql_syntax = sql_syntax
        self.columns = columns
        self.tables = tables

        self.__remove_oversized_injections()

    def __remove_oversized_injections(self):
        self.encoded_injections = list(filter(
            lambda injection:len(injection) <= self.action_size, self.encoded_injections))

    def __is_action_token_valid(self, action_dict_index: int, injection_dict_index: int):
        # If the expected index is -1, match any non-SQL syntax (such as column/table names,
        # or numbers/ASCII characters).
        if injection_dict_index == -1:
            decoded_action = self.dictionary[action_dict_index]
            return decoded_action not in self.sql_syntax

        return action_dict_index == injection_dict_index
    
    def _perform_action(self, action: np.ndarray):
        # Initialise to lowest possible normalised reward.
        highest_norm_reward = -1.0

        action_token_indicies = [self._get_token_index(action[i]) for i in range(len(action))]
        action_token_indicies = list(filter(lambda index: index != -1, action_token_indicies))

        token_indicies_length = len(action_token_indicies)

        for encoded_injection in self.encoded_injections:
            injection_length = len(encoded_injection)
            comparison_count = max(injection_length, token_indicies_length)

            reward = 0

            for action_index in range(comparison_count):
                if action_index >= token_indicies_length or action_index >= injection_length:
                    reward -= 1
                    continue

                action_dict_index = action_token_indicies[action_index]
                injection_dict_index = int(encoded_injection[action_index])
                
                action_valid = self.__is_action_token_valid(action_dict_index, injection_dict_index)
                reward += 1 if action_valid else -1
            
            reward /= comparison_count
            highest_norm_reward = max(reward, highest_norm_reward)

        print(f'Payload attempted (reward: {highest_norm_reward}):')
        print(self.get_payload(action))
        
        state = action

        return state, highest_norm_reward, True 