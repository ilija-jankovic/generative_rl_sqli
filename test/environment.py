import unittest

from requests import Response
import tensorflow as tf
import unittest.test

from lib.model.environment import Environment
from lib.model.payload_builder import PayloadBuilder

ACTION_SIZE = 10

DICTIONARY = [
    'publicToken',
    'privateToken',
    'publicToken2',
]

dictionary_size = len(DICTIONARY)

class TestEnvironment(unittest.TestCase):

    __env: Environment
    __response: Response

    def __set_response_body(self, body: str):
        self.__response._content = str.encode(body)

    def setUp(self):
        self.__response = Response()
        self.__set_response_body('publicToken publicToken2 123')

        self.__env = Environment(
            payload_builder=PayloadBuilder(
                dictionary=DICTIONARY,
                prefix='',
                suffix='',
            ),
            embeddings=[[0.0], [0.1], [0.2]],
            action_size=ACTION_SIZE,
            state_size=10,
            frames_per_episode=5,
            double_requests=False,
            send_request_callback=lambda _: self.__response
        )

    def __perform_dummy_action(self):
        state, _, __ = self.__env.perform_action(
            tf.zeros([ACTION_SIZE,], dtype=tf.int16)
        )

        return state

    def test_one_new_token_count(self):
        '''
        State counts one new token after private token returned.
        '''

        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.assertEqual(state[0], 1)

    def test_three_new_tokens_count(self):
        '''
        State counts three new tokens after three unique private tokens
        returned.
        '''
        
        self.__set_response_body('privateToken privateToken2 privateToken3')
        state = self.__perform_dummy_action()

        self.assertEqual(state[0], 3)

    def test_three_consecutive_tokens_count(self):
        '''
        State counts three new tokens after three unique private tokens
        consecutively returned.
        '''

        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.__set_response_body('privateToken2')
        state = self.__perform_dummy_action()

        self.__set_response_body('privateToken3')
        state = self.__perform_dummy_action()

        self.assertEqual(state[0], 3)

    def test_one_new_token_buffer(self):
        '''
        New private token 'privateToken' is tokenized to index 1 in the
        dictionary.
        '''

        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.assertEqual(state[2], 1)

    def test_three_new_tokens_buffer(self):
        '''
        Three new private tokens are stored in reverse order with ASCII
        offsets for indices of non-dictionary tokens.
        '''

        self.__set_response_body('privateToken a b')
        state = self.__perform_dummy_action()

        self.assertEqual(state[2], ord('B') + dictionary_size)
        self.assertEqual(state[3], ord('A') + dictionary_size)
        self.assertEqual(state[4], 1)

    def test_three_consecutive_new_tokens_buffer(self):
        '''
        Three new private tokens from consecutive requests are stored in
        reverse order with ASCII offsets for indices of non-dictionary
        tokens.
        '''

        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.__set_response_body('a')
        state = self.__perform_dummy_action()

        self.__set_response_body('b')
        state = self.__perform_dummy_action()

        self.assertEqual(state[2], ord('B') + dictionary_size)
        self.assertEqual(state[3], ord('A') + dictionary_size)
        self.assertEqual(state[4], 1)


if __name__ == '__main__':
    unittest.main()