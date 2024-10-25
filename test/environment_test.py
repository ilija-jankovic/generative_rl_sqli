import unittest
import tensorflow as tf
from lib.model.environment import Environment
from lib.network.attacker import attack


ACTION_SIZE = 10

DICTIONARY = [
    'privateToken',
    'privateToken2',
    'privateToken3',
    'publicToken',
    'publicToken2',
]

dictionary_size = len(DICTIONARY)

class TestEnvironment(unittest.TestCase):

    __env: Environment
    __response_body: str

    def __set_response_body(self, body: str):
        self.__response_body = attack(
            send_request_callback=lambda: body,
            payload='',
            perform_double_requests=False,
        )

    def setUp(self):
        self.__set_response_body('publicToken publicToken2 123')

        self.__env = Environment(
            dictionary=DICTIONARY,
            action_size=ACTION_SIZE,
            state_size=10,
            frames_per_episode=5,
            attack_callback=lambda _: self.__response_body,
        )

    def __perform_dummy_action(self):
        state, _ = self.__env.perform_action(
            action=tf.zeros([ACTION_SIZE,], dtype=tf.int16),
            reporter=None,
            timestep=None,
        )

        return state

    def test_state_is_predefined_size(self):
        '''
        State is the predefined size of 10.
        '''
        
        self.__set_response_body('')
        state = self.__perform_dummy_action()
        
        self.assertEqual(len(state), 10)
        
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
        New private token 'privateToken' is tokenized to index 0 in the
        dictionary.
        '''

        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.assertEqual(state[2], 0)

    def test_one_new_ascii_token_buffer(self):
        '''
        New private token 'a' is tokenized to index ASCII code of 'a' plus
        dictionary size.
        '''

        self.__set_response_body('a')
        state = self.__perform_dummy_action()

        self.assertEqual(state[2], ord('a') + dictionary_size)

    def test_three_new_tokens_buffer(self):
        '''
        Three new private tokens are stored in reverse order.
        '''

        self.__set_response_body('privateToken privateToken2 privateToken3')
        state = self.__perform_dummy_action()

        self.assertEqual(state[2], 2)
        self.assertEqual(state[3], 1)
        self.assertEqual(state[4], 0)

    def test_three_new_with_ascii_tokens_buffer(self):
        '''
        Three new private tokens are stored in reverse order with ASCII
        offsets for indices of non-dictionary tokens.
        '''

        self.__set_response_body('privateToken a b')
        state = self.__perform_dummy_action()

        self.assertEqual(state[2], ord('b') + dictionary_size)
        self.assertEqual(state[3], ord('a') + dictionary_size)
        self.assertEqual(state[4], 0)

    def test_three_consecutive_new_tokens_buffer(self):
        '''
        Three new private tokens from consecutive requests are stored in
        reverse order.
        '''

        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.__set_response_body('privateToken2')
        state = self.__perform_dummy_action()

        self.__set_response_body('privateToken3')
        state = self.__perform_dummy_action()

        self.assertEqual(state[2], 2)
        self.assertEqual(state[3], 1)
        self.assertEqual(state[4], 0)

    def test_three_consecutive_new_with_ascii_tokens_buffer(self):
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

        self.assertEqual(state[2], ord('b') + dictionary_size)
        self.assertEqual(state[3], ord('a') + dictionary_size)
        self.assertEqual(state[4], 0)
        
    def test_state_section_markers(self):
        '''
        State sections are joined with -1s.
        '''
        
        self.__set_response_body('test test2 test3')
        state = self.__perform_dummy_action()
        
        self.assertEqual(
            state[1],
            -1,
            msg='New token count section ends at index 1.',
        )
        
        self.assertEqual(
            state[5],
            -1,
            msg='New tokens buffer section ends at middle of ' +
                'remaining space, rounded down at index 5.'
        )

    def test_state_is_sectioned(self):
        '''
        Single-token response begins at start of response section.
        '''

        self.__set_response_body('privateToken')
        state = self.__perform_dummy_action()

        self.assertEqual(state[6], 0)



if __name__ == '__main__':
    unittest.main()