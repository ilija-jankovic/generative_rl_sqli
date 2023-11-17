import numpy as np
from typing import List
import re
from sqltree import sqltree
from requests import Response
from typing import Callable
from numpy import dot
from numpy.linalg import norm

from .episode_state import EpisodeState

class Environment():
    dictionary: List[str]

    action_size: int
    state_size: int

    embeddings: List[List[float]]

    # Calculated dynamically from embeddings.
    __embedding_size : int

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
            embeddings: List[List[int]],
            columns: List[str],
            tables: List[str],
            send_request_callback: Callable[[str], Response]
        ):
        assert(action_size > 0)
        assert(state_size > 0)
        assert(len(embeddings) == len(dictionary))
        
        for embedding in embeddings[1:]:
            if len(embedding) != len(embeddings[0]):
                raise Exception('All embeddings must be of the same length')
        
        self.__embedding_size = len(embeddings[0])
        assert(action_size % self.__embedding_size)
            
        self.dictionary = dictionary

        self.action_size = action_size
        self.state_size = state_size
        
        self.embeddings = embeddings

        self.columns = columns
        self.tables = tables
        
        self.send_request_callback = send_request_callback

    def __inject_random_payloads(self):
        self.__inject_payload('')
        self.__inject_payload('random string')

    def __reset_token_cache(self):
        self.__found_tokens.clear()

        # Ensures data from non-useful injections is not rewarded.
        self.__inject_random_payloads()

    def __is_valid_sql_payload(self, payload: str):
        ast_contraint_index = float('inf')

        try:
            ast_contraint_index = payload.index('UNION')
        except:
            pass

        try:
            ast_contraint_index = min(payload.index('SELECT'), ast_contraint_index)
        except:
            if ast_contraint_index == float('inf'):
                return True

        try:
            sqltree(payload[ast_contraint_index: -1])
            return True
        except:
            return False
        
    def __payload_has_unnecessary_tokens(self, payload: str):
        comment_index = float('inf')

        try:
            comment_index = payload.index('--')
        except:
            pass

        try:
            comment_index = min(payload.index('#'), comment_index)
        except:
            if comment_index == float('inf'):
                return False

        return len(payload) > comment_index + 1

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
    
    def __get_closest_embedding(self, action_slice: List[float]):
        '''
        Gets most similar embedding vector by max cosine similarity with
        `action_slice`.
        '''
        return max(self.embeddings, lambda embedding:
                   dot(action_slice, embedding)/(norm(action_slice)*norm(embedding)))

    def __get_payload(self, action: np.ndarray):
        action_slices = [action[i:i + self.__embedding_size]
                         for i in range(0, self.action_size, self.__embedding_size)]

        payload = ''
        for action_slice in action_slices:
            closest_embedding = self.__get_closest_embedding(action_slice)
            closest_dict_index = self.embeddings.index(closest_embedding)
            payload += self.dictionary[closest_dict_index]

        return payload

    def __record_payload(self, payload: str):
        self.__attempted_payloads.append(payload)

    def __payload_attempted(self, payload: str):
        return payload in self.__attempted_payloads

    def create_empty_state(self):
        return np.array([-1] * self.state_size, dtype='float32')

    def __filter_payload_from_text(self, text: str, payload: str):
        return text.replace(payload, '')
    
    def __tokenize_text(self, text: str):
        unique_tokens = set()
        for token in re.split('[^a-zA-Z]+', text):
            unique_tokens.add(token)

        return unique_tokens
    
    def __filter_non_matching_text(self, text1: str, text2: str):
        tokens1 = list(self.__tokenize_text(text1))
        tokens2 = list(self.__tokenize_text(text2))

        combined = tokens1 + tokens2

        for token in combined:
            if token in tokens1 and token in tokens2:
                yield token

    def __filter_found_tokens(self, tokens: List[str]):
        for token in tokens:
            if token not in self.__found_tokens:
                yield token

    def __inject_payload(self, payload: str):
        '''
        Returns new tokens found after filtering responses.
        '''
        res1 = self.send_request_callback(payload)
        res2 = self.send_request_callback(payload)

        resText1 = self.__filter_payload_from_text(res1.text, payload)
        resText2 = self.__filter_payload_from_text(res2.text, payload)

        unique_tokens = list(self.__filter_non_matching_text(resText1, resText2))
        new_tokens = list(self.__filter_found_tokens(unique_tokens))

        self.__found_tokens += new_tokens

        return res2, new_tokens
    
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
    
    # TODO: Add table and column names from response to state definition.
    def __create_state(self, action: np.ndarray, data: str, new_tokens: List[str]):
        res_size = self.state_size - self.action_size

        res_section_size = res_size // 2

        res_data = [ord(char) for char in data[:res_section_size]]
        res_new_tokens_joined = [ord(char) for char in ','.join(new_tokens)[:res_section_size]]

        res_data = res_data[:len(res_data)] + [-1.0]*(res_section_size - len(res_data))
        res_new_tokens_joined = res_new_tokens_joined[:len(res_new_tokens_joined)] + [-1.0]*(res_section_size - len(res_new_tokens_joined))

        return np.array(action.tolist() + res_data + res_new_tokens_joined)
    
    def perform_action(self, action: np.ndarray):
        #
        #
        # !IMPORTANT!
        #
        # TODO: Do not run payload if it contains any sql_blacklist.txt tokens.
        # Severely negatively reward such actions.
        #
        #

        payload = self.__get_payload(action)

        self.__record_payload(payload)

        #static_reward = self.__get_static_reward(action)
        #dynamic_reward = self.__attempt_injection(payload)

        #reward = static_reward + dynamic_reward
        response, new_tokens = self.__inject_payload(payload)

        reward = len(new_tokens)
        reward = -0.1 if reward <= 0.0 else reward
        if reward > 0.0:
            print(f'Successful payload (reward: {reward}):')
            print(self.__get_payload(action))
        
        state = self.__create_state(action, response.text, new_tokens)

        # NOTE: extend=reward > 1.0 (strictly greater) only if static
        # rewards count, as we should only end an episode if data is found.
        episode_ended = self.__update_episode(extend=reward >= 1.0)

        return state, reward, episode_ended
