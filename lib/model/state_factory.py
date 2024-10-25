from typing import List
import numpy as np
import tensorflow as tf

from . import any_string_tokeniser


class StateFactory:
    
    dictionary: List[str]

    __sorted_dictionary: List[str]

    __tokenised_responses: List[List[int]]
    '''
    Indexed response tokens offset by an increment of one.
    '''
    
    @staticmethod
    def create_empty_state(
        state_size: int,
    ):
        return tf.zeros([state_size,], dtype=tf.float64)


    def __sort_dictionary(self, dictionary: List[str]):
        '''
        Since tokens may be a subset of each other, longer ones must be prioritied
        during tokenisation.

        The sorted dictionary is expected to have two layers of sorting: first by
        negative length, then by alphabetical order (for the case of multiple tokens
        of the same length existing).
        '''

        # Second condition prioritises alphabetically, as stated by Johannes from:
        # https://stackoverflow.com/a/44835987
        return sorted(
            dictionary,
            key=lambda token: (-len(token), token),
        )


    def __init__(self, dictionary: List[str]) -> None:
        self.dictionary = dictionary
        
        self.__sorted_dictionary = self.__sort_dictionary(dictionary)
        self.__tokenised_responses = []
    
    
    def add_response(self, state_size: int, response: str):
        assert(state_size > 2)

        indices = any_string_tokeniser.tokens_to_indices(
            tokens=response,
            index_list_length=state_size - 2,
            dictionary=self.dictionary,
            sorted_dictionary=self.__sorted_dictionary,
        )

        # Add one to ensure 0-index does not interfere with
        # padding during averaging.
        indices = [index + 1 for index in indices]

        self.__tokenised_responses.append(indices)


    def create_state(
        self,
        state_size: int,
        total_private_tokens_count: int,
    ):
        max_response_length = max([
            len(response) for response in self.__tokenised_responses
        ])

        zero_padded_responses = [
            response + [0] * (max_response_length - len(response)) \
                for response in self.__tokenised_responses
        ]
        
        average_response = np.average(zero_padded_responses, axis=0)

        state = [float(total_private_tokens_count), -1.0]
        state.extend(average_response)
        state.extend([-1.0] * (state_size - max_response_length - 2))

        assert(len(state) == state_size)
        
        return tf.convert_to_tensor(state, dtype=tf.float64)
