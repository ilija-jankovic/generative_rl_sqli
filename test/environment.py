import unittest

from requests import Response
import tensorflow as tf
import unittest.test

from lib.model.environment import Environment
from lib.model.payload_builder import PayloadBuilder

ACTION_SIZE = 10

class TestEnvironment(unittest.TestCase):

    __env: Environment
    __response: Response

    def setUp(self):
        super().setUp()

        self.__response = Response()
        self.__response._content = str.encode('publicToken publicToken2 123')

        self.__env = Environment(
            payload_builder=PayloadBuilder(
                dictionary=[
                    'publicToken',
                    'privateToken',
                    'publicToken2',
                ],
                prefix='',
                suffix='',
            ),
            embeddings=[[0.0], [0.1], [0.2]],
            action_size=ACTION_SIZE,
            state_size=8,
            frames_per_episode=5,
            double_requests=False,
            send_request_callback=lambda _: self.__response
        )

    def test_new_tokens_count(self):
        self.__response._content = str.encode('privateToken')

        state, _, __ = self.__env.perform_action(
            tf.zeros([ACTION_SIZE,], dtype=tf.int16)
        )

        self.assertEqual(
            state[0], 1,
            'State counts one new token after private token returned.'
        )

if __name__ == '__main__':
    unittest.main()