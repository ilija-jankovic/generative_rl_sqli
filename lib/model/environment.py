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
    dictionary_uppercase: List[str]
    payload_builder: PayloadBuilder

    action_size: int
    state_size: int

    send_request_callback: Callable[[str], Response]

    embeddings: List[List[float]]

    # Calculated dynamically from embeddings.
    embedding_size : int

    double_requests: bool

    __attempted_payloads: List[str]
    __found_tokens: List[str]

    # Most recent tokens at front.
    __new_tokens: List[str]

    __episode: EpisodeState

    def __init__(
            self,
            payload_builder: PayloadBuilder,
            embeddings: List[List[int]], 
            action_size: int,
            state_size: int,
            frames_per_episode: int,
            double_requests: bool,
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
        self.dictionary_uppercase = [token.upper() for token in dictionary]
        self.payload_builder = payload_builder

        self.action_size = action_size
        self.state_size = state_size

        self.embeddings = embeddings
        self.embedding_size = len(embeddings[0])

        self.double_requests = double_requests

        self.__attempted_payloads = []
        self.__found_tokens = []
        self.__new_tokens = []
        
        self.send_request_callback = send_request_callback
        self.__episode = EpisodeState(frames_per_episode)

        self.__inject_initial_payloads()

    def __inject_initial_payloads(self):
        self.__send_request('1', is_expected=True)
        self.__send_request('2', is_expected=True)
        self.__send_request('3', is_expected=True)
        self.__send_request('4', is_expected=True)
        self.__send_request('5', is_expected=True)

    def __reset_token_cache(self):
        self.__found_tokens.clear()
        self.__new_tokens.clear()

    def __reset_payload_cache(self):
        self.__attempted_payloads.clear()

    def get_payload(self, action: tf.Tensor):
        return self.payload_builder.convert_action_to_payload(action)

    def __record_payload(self, payload: str):
        self.__attempted_payloads.append(payload)

    def __payload_attempted(self, payload: str):
        return payload in self.__attempted_payloads

    def create_empty_state(self):
        '''
        Creates a state filled with index.

        Used to start off each branch of a batch in different directions.
        '''

        # Floating point type as the state is expected to be concatenated
        # with other inputs which are floating point in the policy model.
        #
        # The combined tensor must be of the same type.
        return tf.zeros((self.state_size,), dtype=tf.float32)

    def __filter_payload_from_text(self, text: str, payload: str):
        return text.replace(payload, '')
    
    def __tokenize_text(self, text: str):
        '''
        Tokenizes by matching all visible ASCII characters.
        '''
        return re.findall(r'[!-~]+', text)
    
    def __filter_non_matching_text(self, text1: str, text2: str):
        tokens1 = self.__tokenize_text(text1)
        tokens2 = self.__tokenize_text(text2)

        combined = tokens1 + tokens2

        for token in combined:
            if token in tokens1 and token in tokens2:
                yield token

    def __send_request(self, data: str, is_expected: bool = False):
        '''
        Returns new tokens found after filtering responses.
        '''
        if self.double_requests:
            res1 = self.send_request_callback(data)
            res2 = self.send_request_callback(data)

            # Do not filter data from response if expected as full caching of public response
            # data is desired.
            resText1 = res1.text if is_expected else \
                self.__filter_payload_from_text(res1.text, data)
            resText2 = res2.text if is_expected else \
                self.__filter_payload_from_text(res2.text, data)

            # Only uppercase considered as the dictionary and SQL is case insensitive.
            resText1 = resText1.upper()
            resText2 = resText2.upper()

            resTokens = self.__filter_non_matching_text(resText1, resText2)
        else:
            res2 = self.send_request_callback(data)

            # Do not filter data from response if expected as full caching of public response
            # data is desired.
            resText = res2.text if is_expected else \
                self.__filter_payload_from_text(res2.text, data)
            
            # Only uppercase considered as the dictionary and SQL is case insensitive.
            resText = resText.upper()

            resTokens = self.__tokenize_text(resText)

        new_tokens: List[str] = []

        # Avoid sets for token processing, as their order is non-deterministic.
        # This is undesirable for tests, as well as consistency for the agent.
        for token in resTokens:
            if token not in self.__found_tokens:
                self.__found_tokens.append(token)

                if(not is_expected):
                    new_tokens.insert(0, token)

        self.__new_tokens = new_tokens + self.__new_tokens

        return res2, new_tokens
    
    def __update_episode(self):
        '''
        Returns whether the episode has ended.
        '''

        self.__episode.next_frame()

        episode_ended = self.__episode.has_episode_ended()
        if episode_ended:
            self.__episode.next_episode()

            # Remove found tokens to allow DDPG to learn
            # with more reward opportunity.
            self.__reset_token_cache()
            self.__reset_payload_cache()

            # Ensures data from non-useful injections is not rewarded.
            self.__inject_initial_payloads()

        return episode_ended

    def __string_to_indices(self, data: str, max_size: int):
        dictionary_length = len(self.dictionary_uppercase)
        data = data.upper()

        indexed_data: List[int] = []

        # Prioritise dictionary indices.
        #
        # Fall back to shifted ASCII indices.
        while len(data) > 0 and len(indexed_data) < max_size:
            appended = False

            for i, token in enumerate(self.dictionary_uppercase):
                if data.startswith(token):
                    indexed_data.append(i)

                    # Remove token from prefix.
                    data = data[len(token):]
                    appended = True

                    break

            if appended:
                continue
            
            # Append ASCII code shifted by max dictionary index.
            indexed_data.append(ord(data[0]) + dictionary_length)

            data = data[1:]

        return indexed_data
    
    def __create_state(self, data: str):
        total_new_tokens_count = len(self.__new_tokens)

        new_tokens = ''.join(self.__new_tokens)

        max_new_tokens_size = self.state_size // 2 - 2
        new_token_indices = self.__string_to_indices(new_tokens, max_size=max_new_tokens_size)
        
        state = [total_new_tokens_count, -1, *new_token_indices, -1]
        max_data_tokens_size = self.state_size - len(state)
        
        data_indices = self.__string_to_indices(data, max_size=max_data_tokens_size)

        state.extend(data_indices)

        # Pad state until self.state_size is reached.
        if(len(state) < self.state_size):
            state.extend([-1] * (self.state_size - len(state)))

        # Floating point type as the state is expected to be concatenated
        # with other inputs which are floating point in the policy model.
        #
        # The combined tensor must be of the same type.
        return tf.convert_to_tensor(state, dtype=tf.float32)


    def perform_action(self, action: tf.Tensor):
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

        response, new_tokens = self.__send_request(payload)

        new_tokens_count = len(new_tokens)
        
        if not self.__payload_attempted(payload) and new_tokens_count > 0:
            reward = new_tokens_count

            self.__episode.extend_episode()

            print(f'Successful payload (unscaled reward: {reward}):')
            print(payload)
        else:
            reward = 0.0

        self.__record_payload(payload)
        done = self.__update_episode()

        state = self.create_empty_state() \
            if done \
            else self.__create_state(response.text)

        return state, reward, done
