import math
from typing import List
from typing import Callable
import tensorflow as tf
from .payload import Payload
from .payload_factory import create_payload_from_action
from .ppo_reporter import PPOReporter
from .ppo_payload_statistics import PPOPayloadStatistics
from .episode_state import EpisodeState


AttackCallback = Callable[[str], str]
'''
Use for injecting a payload and returning a list of filtered
response tokens.
'''


class Environment:
    dictionary: List[str]
    dictionary_sorted: List[str]
    '''
    Permutated version of `dictionary` which handles subset cases of tokens.
    '''

    action_size: int
    state_size: int
    attack_callback: AttackCallback
    '''
    Payload injection to filtered response mechanism.
    '''

    __attempted_payloads: List[Payload]
    __found_tokens: List[str]

    # Most recent tokens at front.
    __new_tokens: List[str]

    __episode: EpisodeState

    @property
    def episode(self):
        return self.__episode.episode

    def __init_tokenizer_dictionary(self):
        '''
        Since tokens may be a subset of each other, longer ones must be prioritied
        during this tokenization.

        `dictionary_sorted` is expected to have two layers of sorting: first by
        negative length, then by alphabetical order (for the case of multiple tokens
        of the same length existing).
        '''

        # Second condition prioritises alphabetically, as stated by Johannes from:
        # https://stackoverflow.com/a/44835987
        self.dictionary_sorted = sorted(
            self.dictionary,
            key=lambda token: (-len(token), token)
        )
        

    def __inject_payload(self, payload_text: str, is_expected: bool):
        response_tokens = self.attack_callback(payload_text)

        new_tokens: List[str] = []

        # Avoid sets for token processing, as their order is non-deterministic.
        # This is undesirable for tests, as well as consistency for the agent.
        for token in response_tokens:
            if token not in self.__found_tokens:
                self.__found_tokens.append(token)

                if(not is_expected):
                    new_tokens.insert(0, token)

        self.__new_tokens = new_tokens + self.__new_tokens

        return response_tokens, new_tokens
    

    def __init__(
        self,
        dictionary: List[str],
        action_size: int,
        state_size: int,
        frames_per_episode: int,
        attack_callback: AttackCallback
    ):
        assert(action_size > 0)
        assert(state_size > 0)
        assert(state_size % 2 == 0)

        self.dictionary = dictionary
            
        self.__init_tokenizer_dictionary()

        assert(
            len(set(self.dictionary)) ==
            len(set(self.dictionary_sorted))
        )

        self.action_size = action_size
        self.state_size = state_size

        self.__attempted_payloads = []
        self.__found_tokens = []
        self.__new_tokens = []
        
        self.attack_callback = attack_callback
        self.__episode = EpisodeState(frames_per_episode)

        self.__inject_initial_payloads()

    def __inject_initial_payloads(self):
        self.__inject_payload('1', is_expected=True)

    def __reset_token_cache(self):
        self.__found_tokens.clear()
        self.__new_tokens.clear()

    def __reset_payload_cache(self):
        self.__attempted_payloads.clear()

    def __record_payload(self, payload: Payload):
        self.__attempted_payloads.append(payload)

    def __payload_attempted(self, payload: Payload):
        return payload in self.__attempted_payloads

    def create_empty_state(self):
        '''
        Creates a state filled with index.

        Used to start off each branch of a batch in different directions.
        '''

        return tf.zeros([self.state_size,], dtype=tf.float32)
    
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
        dictionary_length = len(self.dictionary)

        indexed_data: List[int] = []

        # Prioritise dictionary indices.
        #
        # Fall back to shifted ASCII indices.
        while len(data) > 0 and len(indexed_data) < max_size:
            appended = False

            for token in self.dictionary_sorted:
                if data.startswith(token):
                    index = self.dictionary.index(token)
                    indexed_data.append(index)

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
    
    def __create_state(self, responseTokens: List[str]):
        total_new_tokens_count = len(self.__new_tokens)

        new_tokens_joined = ''.join(self.__new_tokens)
        response_tokens_joined = ''.join(responseTokens)
        
        max_new_tokens_size = self.state_size // 2 - 2
        new_token_indices = self.__string_to_indices(
            data=new_tokens_joined,
            max_size=max_new_tokens_size,
        )
        
        state = [total_new_tokens_count, -1, *new_token_indices, -1]
        max_data_tokens_size = self.state_size - len(state)
        
        data_indices = self.__string_to_indices(
            data=response_tokens_joined, 
            max_size=max_data_tokens_size,
        )

        state.extend(data_indices)

        # Pad state until self.state_size is reached.
        if(len(state) < self.state_size):
            state.extend([-1] * (self.state_size - len(state)))
        
        return tf.convert_to_tensor(state, dtype=tf.float32)

    def perform_action(
        self,
        action: tf.Tensor,
        timestep: int,
        reporter: PPOReporter | None,
    ):
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

        payload = create_payload_from_action(
            action=action,
            dictionary=self.dictionary,
        )
        
        response, new_tokens = self.__inject_payload(
            payload_text=str(payload),
            is_expected=False,
        )

        new_tokens_count = len(new_tokens)
        
        if not self.__payload_attempted(payload) and new_tokens_count > 0:
            
            # Successful payloads further along the episode are more greatly rewarded.
            #
            # They are less likely to be the result of simple injections, as these were
            # already likely rewarded.
            reward_weight = 1.0 + self.__episode.frames_since_last_episode / self.__episode.initial_frames
            reward = new_tokens_count * reward_weight

            # Map reward to [-1, 1].
            #
            # Approximately linearly scaled down for low reward values, then tapers off to
            # upper/lower bound.
            reward = math.tanh(reward / 20.0)
            
            self.__episode.extend_episode()
            
            if reporter != None:
                stats = PPOPayloadStatistics(
                    timestep=timestep,
                    reward=reward,
                    payload=payload,
                )
                
                if not reporter.is_payload_recorded(payload):
                    reporter.record_payload_statistic(stats)
        else:
            reward = 0.0 if payload.is_syntax_correct else -1.0

        self.__record_payload(payload)
        done = self.__update_episode()

        state = self.create_empty_state() \
            if done \
            else self.__create_state(response)

        return state, reward

    def perform_demonstration_action(self, action: tf.Tensor):
        return self.perform_action(
            action=action,
            timestep=None,
            reporter=None,
        )