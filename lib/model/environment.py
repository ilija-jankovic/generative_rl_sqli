import random
import numpy as np
from typing import List
import re
from requests import Response
from typing import Callable
import tensorflow as tf

from .payload_builder import PayloadBuilder
from .episode_state import EpisodeState

class Environment():
    dictionary: List[str]
    payload_builder: PayloadBuilder

    action_size: int
    state_size: int

    columns: List[str]
    tables: List[str]

    send_request_callback: Callable[[str], Response]

    embeddings: List[List[float]]

    # Calculated dynamically from embeddings.
    embedding_size : int
    
    __attempted_payloads: List[str] = []
    __found_tokens: List[str] = []
    __episode: EpisodeState

    def __init__(
            self,
            payload_builder: PayloadBuilder,
            embeddings: List[List[int]], 
            action_size: int,
            state_size: int,
            batch_size: int,
            columns: List[str],
            tables: List[str],
            send_request_callback: Callable[[str], Response]
        ):
        assert(action_size > 0)

        assert(state_size > 0)
        assert(state_size % 2 == 0)
        
        dictionary = payload_builder.dictionary

        assert(len(embeddings) == len(dictionary))

        for embedding in embeddings[1:]:
            if len(embedding) != len(embeddings[0]):
                raise Exception('All embeddings must be of the same length')
            
        self.dictionary = dictionary
        self.payload_builder = payload_builder

        self.action_size = action_size
        self.state_size = state_size

        self.embeddings = embeddings
        self.embedding_size = len(embeddings[0])

        self.columns = columns
        self.tables = tables
        
        self.send_request_callback = send_request_callback
        self.__episode = EpisodeState(batch_size * 5)

        self.__inject_random_payloads()

    def __inject_random_payloads(self):
        self.__inject_payload('')
        self.__inject_payload('random string')

        if len(self.payload_builder.prefix) > 0 or len(self.payload_builder.suffix) > 0:
            # Simulate empty action.
            self.__inject_payload(self.payload_builder.prefix + self.payload_builder.suffix)

    def __reset_token_cache(self):
        self.__found_tokens.clear()

    def get_payload(self, action: tf.Tensor):
        return self.payload_builder.convert_action_to_payload(action)

    def __record_payload(self, payload: str):
        self.__attempted_payloads.append(payload)

    def __payload_attempted(self, payload: str):
        return payload in self.__attempted_payloads

    def create_empty_state(self, index: float):
        '''
        Creates a state filled with index.

        Used to start off each branch of a batch in different directions.
        '''
        return tf.fill((self.state_size, self.embedding_size), index)

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

    def __inject_payload(self, payload: str, record_tokens: bool = True):
        '''
        Returns new tokens found after filtering responses.
        '''
        res1 = self.send_request_callback(payload)
        res2 = self.send_request_callback(payload)

        resText1 = self.__filter_payload_from_text(res1.text, payload)
        resText2 = self.__filter_payload_from_text(res2.text, payload)

        unique_tokens = list(self.__filter_non_matching_text(resText1, resText2))
        new_tokens = list(set(unique_tokens) - set(self.__found_tokens))

        if record_tokens:
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

            # Remove found tokens to allow DDPG to learn
            # with more reward opportunity.
            self.__reset_token_cache()

            # Ensures data from non-useful injections is not rewarded.
            self.__inject_random_payloads()

        return episode_ended
    
    # TODO: Add table and column names from response to state definition.
    def __create_state(self, action: np.ndarray, data: str, new_tokens: List[str]):
        res_size = self.state_size - self.action_size

        embeddings = [self.embeddings[i.numpy()] if i.numpy() > 0.0 and i.numpy() < len(self.dictionary) else [0.0] * self.embedding_size for i in action]

        res_section_size = res_size // 2

        res_data = [self.embeddings[self.dictionary.index(char)] if char in self.dictionary else [0.0] * self.embedding_size for char in data[:res_section_size]]
        res_new_tokens = [self.embeddings[self.dictionary.index(char)] if char in self.dictionary else [0.0] * self.embedding_size for char in new_tokens[:res_section_size]]

        res_data += [[0.0] * self.embedding_size] * (res_section_size - len(res_data))
        res_new_tokens += [[0.0] * self.embedding_size] * (res_section_size - len(res_new_tokens))

        return tf.convert_to_tensor(embeddings + res_data + res_new_tokens, dtype=tf.float32)
    
    def perform_action(self, action: np.ndarray):
        '''
        If `ignore_episode` is `True`, this method always returns `False` for episode ended,
        and resets token cache on every invocation.
        '''
        #
        #
        # !IMPORTANT!
        #
        # TODO: Do not run payload if it contains any sql_blacklist.txt tokens.
        # Severely negatively reward such actions.
        #
        #

        payload = self.get_payload(action)

        self.__record_payload(payload)

        response, new_tokens = self.__inject_payload(payload)

        new_tokens_count = len(new_tokens)
        
        if new_tokens_count > 0:
            reward = new_tokens_count
            print(f'Successful payload (reward: {reward}):')
            print(payload)
        else:
            reward = -1.0
        
        state = self.__create_state(action, response.text, new_tokens)

        episode_ended = self.__update_episode(extend=reward >= 1.0)

        return state, reward, episode_ended
