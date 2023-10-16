import numpy as np
from typing import List
import re
from sqltree import sqltree
from requests import Response
from typing import Callable

from .episode_state import EpisodeState

class Environment():
    dictionary: List[str]
    action_size: int
    state_size: int

    encoded_injections: List[List[int]]

    sql_syntax: List[str]
    columns: List[str]
    tables: List[str]

    send_request_callback: Callable[[str], Response]
    
    __attempted_payloads: List[str] = []
    __found_tokens: List[str] = []
    __episode = EpisodeState(100)

    def __init__(
            self,
            dictionary: List[str],
            action_size: int,
            state_size: int,
            encoded_injections: List[List[int]],
            sql_syntax: List[str],
            columns: List[str],
            tables: List[str],
            send_request_callback: Callable[[str], Response]
        ):
        self.dictionary = dictionary
        self.action_size = action_size
        self.state_size = state_size

        self.encoded_injections = encoded_injections

        self.sql_syntax = sql_syntax
        self.columns = columns
        self.tables = tables
        
        self.send_request_callback = send_request_callback

        self.__remove_oversized_injections()

    def __is_valid_sql_payload(self, state: np.ndarray):
        try:
            payload = self.__get_payload(state)
            sqltree(f'SELECT * FROM test WHERE test = \'{payload}\'')
            return True
        except:
            return False

    def __get_token_index(self, action_class: float):
        '''
        Returns an integer in the range `[-1, len(self.dictionary) - 1]`.
        '''
        # >= 1.0 condition ensures the max action class value rounds
        # down to the maximum allowed index.
        if action_class < -1.0 or action_class >= 1.0:
            return -1
        
        denormalised = (action_class + 1.0) * (len(self.dictionary) + 1.0) / 2.0 - 1.0
        return int(denormalised)

    def __get_token(self, action_class: float):
        index = self.__get_token_index(action_class)
        return '' if index == -1 else self.dictionary[index]

    def __get_payload(self, action: np.ndarray):
        chrs = [self.__get_token(cls) for cls in action]
        return ''.join(chrs)

    def __record_payload(self, payload: str):
        self.__attempted_payloads.append(payload)

    def __payload_attempted(self, payload: str):
        return payload in self.__attempted_payloads

    def create_empty_state(self):
        return np.array([-1] * self.state_size, dtype='float32')
    
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
    
    def __get_static_reward(self, action: np.ndarray):
        # Initialise to lowest possible normalised reward.
        highest_norm_reward = -1.0

        action_token_indicies = [self.__get_token_index(action[i]) for i in range(len(action))]
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

        return highest_norm_reward
    
    def __attempt_injection(self, payload: str):
        res = self.send_request_callback(payload)

        unique_tokens = set()
        for token in re.split('[^a-zA-Z]+', res.text):
            unique_tokens.add(token)

        reward = 0

        new_tokens = []
        for token in unique_tokens:
            if(token not in self.__found_tokens):
                self.__found_tokens.append(token)
                new_tokens.append(token)
                reward += 1

        if(len(new_tokens) > 0):
            print(payload)
            print('\nNEW DATA')
            print('\nFound:', new_tokens, '\n')

        return reward

    def __reset_token_cache(self):
        self.__found_tokens.clear()
    
    def __update_episode(self, extend: bool):
        '''
        Returns whether the episode has ended.
        '''

        if extend:
            self.__episode.extend_episode()

        self.__episode.next_frame()

        episode_ended = self.__episode.has_episode_ended()
        if episode_ended:
            self.__episode.next_episode()

            # Remove found tokens from demonstrations to allow DDPG to learn
            # with more reward opportunity.
            self.__reset_token_cache()

        return episode_ended

    
    def perform_action(self, action: np.ndarray):
        payload = self.__get_payload(action)
        
        if self.__payload_attempted(payload):
            episode_ended = self.__update_episode(extend=False)
            return self.create_empty_state(), -1.0, episode_ended
        
        self.__record_payload(payload)

        static_reward = self.__get_static_reward(action)
        dynamic_reward = self.__attempt_injection(payload)

        reward = static_reward + dynamic_reward

        print(f'Payload attempted (reward: {reward}):')
        print(self.__get_payload(action))
        
        state = action

        #columns = [self.__get_token_index(action[i]) for i in action]

        episode_ended = self.__update_episode(extend=reward > 1.0)

        return state, reward, episode_ended
