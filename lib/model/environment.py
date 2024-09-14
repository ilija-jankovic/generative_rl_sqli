import math
from typing import List
from typing import Callable
import tensorflow as tf
from .payload import Payload
from . import payload_factory
from . import state_factory
from .ppo_reporter import PPOReporter
from .ppo_payload_statistics import PPOPayloadStatistics
from .episode_state import EpisodeState


AttackCallback = Callable[[str], List[str]]
'''
Use for injecting a payload and returning a list of filtered
response tokens.
'''


class Environment:
    dictionary: List[str]
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
        attack_callback: AttackCallback,
        frames_per_episode: int,
    ):
        assert(action_size > 0)
        assert(state_size > 0)
        assert(state_size % 2 == 0)

        self.dictionary = dictionary
        self.action_size = action_size
        self.state_size = state_size
        self.attack_callback = attack_callback
        
        self.__episode = EpisodeState(frames_per_episode)
        self.__attempted_payloads = []
        self.__found_tokens = []
        self.__new_tokens = []
        
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


    def perform_action(
        self,
        action: tf.Tensor,
        timestep: int,
        reporter: PPOReporter | None,
    ):
        # TODO: Do not run payload if it contains any sql_blacklist.txt tokens.
        # Severely negatively reward such actions.

        payload = payload_factory.create_payload_from_action(
            action=action,
            dictionary=self.dictionary,
        )
        
        response_tokens, new_tokens = self.__inject_payload(
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

        state = state_factory.create_empty_state(state_size=self.state_size) \
            if done \
            else state_factory.create_state_from_tokens(
                state_size=self.state_size,
                tokens=response_tokens,
                new_tokens_buffer=self.__new_tokens,
                dictionary=self.dictionary,
            )

        return state, reward

    def perform_demonstration_action(self, action: tf.Tensor):
        return self.perform_action(
            action=action,
            timestep=None,
            reporter=None,
        )