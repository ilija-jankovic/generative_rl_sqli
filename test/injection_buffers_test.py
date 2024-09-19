import unittest

from lib.model.injection_buffers import InjectionBuffers
from lib.model.payload import Payload


class TestInjectionBuffers(unittest.TestCase):

    __buffers: InjectionBuffers


    def setUp(self) -> None:
        self.__buffers = InjectionBuffers()


    def test_new_tokens_buffer_initialised_as_empty(self):
        '''
        New tokens buffer is initialised as empty.
        '''
        
        self.assertListEqual(self.__buffers.new_tokens, [])


    def test_record_empty_token_list(self):
        '''
        Recording an empty list of tokens does not add to
        'new tokens' buffer.
        '''

        self.__buffers.record_tokens(
            tokens=[],
            is_expected=False,
        )
        
        self.assertListEqual(self.__buffers.new_tokens, [])
        

    def test_record_single_token(self):
        '''
        Recording a single token adds it to 'new tokens' buffer.
        '''

        self.__buffers.record_tokens(
            tokens=['test'],
            is_expected=False,
        )
        
        self.assertListEqual(self.__buffers.new_tokens, ['test',])
        

    def test_record_repeated_tokens(self):
        '''
        Recording a list of repeating tokens adds the token to
        'new tokens' buffer once.
        '''

        self.__buffers.record_tokens(
            tokens=['test', 'test', 'test',],
            is_expected=False,
        )
        
        self.assertListEqual(self.__buffers.new_tokens, ['test',])


    def test_record_three_unique_tokens(self):
        '''
        Recording three unique tokens adds them to 'new tokens'
        buffer in reverse chronological order.
        '''

        self.__buffers.record_tokens(
            tokens=['test1', 'test2', 'test3',],
            is_expected=False,
        )
        
        self.assertListEqual(
            self.__buffers.new_tokens,
            ['test3', 'test2', 'test1',]
        )
        
        
    def test_record_three_unique_tokens_with_repetitions(self):
        '''
        Recording three unique tokens with repetitions adds them
        uniquely to 'new tokens' buffer in reverse chronological
        order.
        '''

        self.__buffers.record_tokens(
            tokens=[
                'test1',
                'test2',
                'test2',
                'test3',
                'test2',
                'test3',
            ],
            is_expected=False,
        )
        
        self.assertListEqual(
            self.__buffers.new_tokens,
            ['test3', 'test2', 'test1',]
        )


    def test_record_single_token_return(self):
        '''
        Recording a single token returns it.
        '''

        new_tokens = self.__buffers.record_tokens(
            tokens=['test'],
            is_expected=False,
        )
        
        self.assertListEqual(new_tokens, ['test',])
        

    def test_record_repeated_tokens_return(self):
        '''
        Recording a list of repeating tokens returns the token.
        '''

        new_tokens = self.__buffers.record_tokens(
            tokens=['test', 'test', 'test',],
            is_expected=False,
        )
        
        self.assertListEqual(new_tokens, ['test',])


    def test_record_three_unique_tokens_return(self):
        '''
        Recording three unique tokens returns them in reverse
        chronological order.
        '''

        new_tokens = self.__buffers.record_tokens(
            tokens=['test1', 'test2', 'test3',],
            is_expected=False,
        )
        
        self.assertListEqual(
            new_tokens,
            ['test3', 'test2', 'test1',]
        )
        
        
    def test_record_three_unique_tokens_with_repetitions_return(self):
        '''
        Recording three unique tokens with repetitions returns them
        uniquely in reverse chronological order.
        '''

        new_tokens = self.__buffers.record_tokens(
            tokens=[
                'test1',
                'test2',
                'test2',
                'test3',
                'test2',
                'test3',
            ],
            is_expected=False,
        )
        
        self.assertListEqual(
            new_tokens,
            ['test3', 'test2', 'test1',]
        )


    def test_recording_single_expected_token(self):
        '''
        Recording an expected single token does not add it to 'new
        tokens' buffer.
        '''

        self.__buffers.record_tokens(
            tokens=['test',],
            is_expected=True,
        )
        
        self.assertListEqual(self.__buffers.new_tokens, [])
        
        
    def test_recording_three_expected_tokens(self):
        '''
        Recording three expected tokens does not add them to 'new
        tokens' buffer.
        '''

        self.__buffers.record_tokens(
            tokens=['test1', 'test2', 'test3',],
            is_expected=True,
        )
        
        self.assertListEqual(self.__buffers.new_tokens, [])
        
        
    def test_recording_single_expected_token_return(self):
        '''
        Recording an expected single token does not return it.
        '''

        new_tokens = self.__buffers.record_tokens(
            tokens=['test',],
            is_expected=True,
        )
        
        self.assertListEqual(new_tokens, [])
        
        
    def test_recording_three_expected_tokens_return(self):
        '''
        Recording three expected tokens does not return them.
        '''

        new_tokens = self.__buffers.record_tokens(
            tokens=['test1', 'test2', 'test3',],
            is_expected=True,
        )
        
        self.assertListEqual(new_tokens, [])
        
        
    def test_recording_expected_unexpected_difference(self):
        '''
        Recording expected tokens then other tokens sets 'new 
        tokens' buffer to the difference of other tokens minus
        expected tokens.
        '''

        self.__buffers.record_tokens(
            tokens=[
                'test1',
                'test2',
                'test3',
                'test4',
                'test5',
            ],
            is_expected=True,
        )
        
        self.__buffers.record_tokens(
            tokens=[
                'test2',
                'test4',
                'test6',
                'test7',
            ],
            is_expected=False,
        )
        
        self.assertListEqual(
            self.__buffers.new_tokens,
            ['test7', 'test6',],
        )
        
        
    def test_recording_expected_unexpected_difference_return(self):
        '''
        Recording expected tokens then other tokens returns the
        difference of other tokens minus expected tokens.
        '''

        self.__buffers.record_tokens(
            tokens=[
                'test1',
                'test2',
                'test3',
                'test4',
                'test5',
            ],
            is_expected=True,
        )
        
        new_tokens = self.__buffers.record_tokens(
            tokens=[
                'test2',
                'test4',
                'test6',
                'test7',
            ],
            is_expected=False,
        )
        
        self.assertListEqual(new_tokens, ['test7', 'test6',])

        
    def test_multiple_recordings_latest_difference_return(self):
        '''
        Recording multiple times lastly returns the difference
        between the last tokens and the combined set of previous
        recorded tokens.
        '''

        self.__buffers.record_tokens(
            tokens=[
                'test1',
                'test2',
                'test3',
                'test4',
                'test5',
            ],
            is_expected=False,
        )
        
        self.__buffers.record_tokens(
            tokens=[
                'test2',
                'test4',
                'test6',
                'test7',
            ],
            is_expected=False,
        )
        
        new_tokens = self.__buffers.record_tokens(
            tokens=[
                'test3',
                'test8',
                'test5',
                'test6',
                'test9',
            ],
            is_expected=False,
        )
        
        self.assertListEqual(new_tokens, ['test9', 'test8',])


    def test_no_payload_recorded_not_attempted(self):
        '''
        An unrecorded payload was not attempted.
        '''
        
        was_attempted = self.__buffers.was_payload_attempted(
            payload=Payload('test'),
        )
        
        self.assertEqual(was_attempted, False)


    def test_recorded_payload_attempted(self):
        '''
        A recorded payload was attempted.
        '''
        
        self.__buffers.record_payload(payload=Payload('test'))
        
        was_attempted = self.__buffers.was_payload_attempted(
            payload=Payload('test'),
        )
        
        self.assertEqual(was_attempted, True)


    def test_record_payload_another_not_attempted(self):
        '''
        A recorded payload does not mark another payload as attempted.
        '''
        
        self.__buffers.record_payload(payload=Payload('test1'))
        
        was_attempted = self.__buffers.was_payload_attempted(
            payload=Payload('test2'),
        )
        
        self.assertEqual(was_attempted, False)


    def test_record_three_payloads_one_attempted_others_not(self):
        '''
        Three recorded payloads mark one of them as attempted with
        other payloads marked as not attempted.
        '''
        
        self.__buffers.record_payload(payload=Payload('test1'))
        self.__buffers.record_payload(payload=Payload('test2'))
        self.__buffers.record_payload(payload=Payload('test3'))
        
        was_attempted_1 = self.__buffers.was_payload_attempted(
            payload=Payload('test3'),
        )

        was_attempted_2 = self.__buffers.was_payload_attempted(
            payload=Payload('test4'),
        )
        
        was_attempted_3 = self.__buffers.was_payload_attempted(
            payload=Payload('test5'),
        )

        self.assertEqual(was_attempted_1, True)
        self.assertEqual(was_attempted_2, False)
        self.assertEqual(was_attempted_3, False)


    def test_record_three_payloads_all_three_attempted(self):
        '''
        Three recorded payloads are marked as attempted.
        '''
        
        self.__buffers.record_payload(payload=Payload('test1'))
        self.__buffers.record_payload(payload=Payload('test2'))
        self.__buffers.record_payload(payload=Payload('test3'))
        
        was_attempted_1 = self.__buffers.was_payload_attempted(
            payload=Payload('test1'),
        )

        was_attempted_2 = self.__buffers.was_payload_attempted(
            payload=Payload('test2'),
        )
        
        was_attempted_3 = self.__buffers.was_payload_attempted(
            payload=Payload('test3'),
        )
        
        self.assertEqual(was_attempted_1, True)
        self.assertEqual(was_attempted_2, True)
        self.assertEqual(was_attempted_3, True)



if __name__ == '__main__':
    unittest.main()