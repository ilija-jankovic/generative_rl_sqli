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
        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.assertEqual(
            state[0], 1,
            'State counts one new token after private token returned.'
        )

    def test_three_new_tokens_count(self):
        self.__set_response_body('privateToken privateToken2 privateToken3')
        state = self.__perform_dummy_action()

        self.assertEqual(
            state[0], 3,
            'State counts three new tokens after three unique private tokens ' +
            'returned.'
        )

    def test_three_consecutive_tokens_count(self):
        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.__set_response_body('privateToken2')
        state = self.__perform_dummy_action()

        self.__set_response_body('privateToken3')
        state = self.__perform_dummy_action()

        self.assertEqual(
            state[0], 3,
            'State counts three new tokens after three unique private tokens ' +
            'consecutively returned.'
        )

    def test_one_new_token_buffer(self):
        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.assertEqual(
            state[2], 1,
            'New private token \'privateToken\' is tokenized to index 1 in ' +
            'the dictionary.'
        )

    def test_three_new_tokens_buffer(self):
        self.__set_response_body('privateToken a b')
        state = self.__perform_dummy_action()

        self.assertEqual(
            state[2], 1,
            'Third newest private token \'privateToken\' is tokenized to index 1 in ' +
            'the dictionary.'
        )

        self.assertEqual(
            state[3], ord('A') + dictionary_size,
            'Second newest private token \'a\' is tokenized to index ASCII code of \'A\' ' +
            'plus dictionary size.'
        )

        self.assertEqual(
            state[4], ord('B') + dictionary_size,
            'Newest private token \'b\' is tokenized to index ASCII code of \'B\' ' +
            'plus dictionary size.'
        )


if __name__ == '__main__':
    unittest.main()