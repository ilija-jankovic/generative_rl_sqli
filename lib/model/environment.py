import math
from typing import List, Set
from typing import Callable
import tensorflow as tf

from .payload import Payload

from .injection_buffers import InjectionBuffers
from . import payload_factory
from . import state_factory
from .ppo_reporter import PPOReporter
from .ppo_payload_statistics import PPOPayloadStatistics
from .episode_state import EpisodeState


AttackCallback = Callable[[Payload], str]
'''
Use for injecting a payload and returning a list of filtered
response tokens. The set of payload's individual tokens are
filtered from reponse tokens.
'''


class Environment:
    dictionary: List[str]
    action_size: int
    state_size: int

    attack_callback: AttackCallback
    '''
    Payload injection to filtered response mechanism.
    '''

    __episode: EpisodeState
    __injection_buffers: InjectionBuffers


    @property
    def episode(self):
        return self.__episode.episode


    def __init__(
        self,
        dictionary: List[str],
        action_size: int,
        state_size: int,
        attack_callback: AttackCallback,
        expected_responses: Set[str],
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
        self.__injection_buffers = InjectionBuffers(
            expected_responses=expected_responses,
        )


    def __inject_payload(self, payload: Payload):
        response = self.attack_callback(payload)

        min_levenshtein_norm = self.__injection_buffers.record_response(
            response,
        )

        return response, min_levenshtein_norm

        
    def __calculate_reward(self, min_levenshtein_norm: float):
        reward = min_levenshtein_norm
        
        extension_threshold = 0.5
        
        if reward >= extension_threshold:
            self.__episode.extend_episode(proportion=reward)
        
        return reward
    
    
    def __try_report_payload_statistics(
        self,
        payload: Payload,
        reporter: PPOReporter | None,
        timestep: int | None,
        reward: float,
    ):
        '''
        Reports payload statistics if a reporter is provided and the
        payload was not already recorded by the reporter.
        
        Either both `reporter` and `timestep` must be defined or
        both must be `None`.
        
        If one is defined and the other is not, an assertion error
        is raised.
        '''

        assert(
            reporter == None and timestep == None or \
            reporter != None and timestep != None
        )
        assert(timestep == None or timestep > 0)

        if reporter == None or reporter.is_payload_recorded(payload):
            return
        
        stats = PPOPayloadStatistics(
            timestep=timestep,
            reward=reward,
            payload=payload,
        )
        
        reporter.record_payload_statistic(stats)
        

    def __calculate_unsuccessful_reward(self, payload: Payload):
        '''
        Negatively rewards payloads which syntax is incorrect.

        This quickly enforces AST adherence.
        
        A zero reward (no punishment) is given if the syntax is
        correct but the payload was not successful.
        '''

        # TODO: Ensure backend query is defined by pen-tester.
        return 0.0 #if payload.is_syntax_correct else -0.01


    def __update_episode(self):
        '''
        Returns whether the episode has ended.
        '''

        self.__episode.next_frame()

        episode_ended = self.__episode.has_episode_ended()

        if episode_ended:
            self.__episode.next_episode()
            self.__injection_buffers.clear()

        return episode_ended
    
    
    def __create_next_state(
        self,
        response: str,
        is_episode_done: bool,
    ):
        return state_factory.create_empty_state(
            state_size=self.state_size,
        ) if is_episode_done \
            else state_factory.create_state_from_response(
                state_size=self.state_size,
                response=response,
                dictionary=self.dictionary,
            )


    def perform_action(
        self,
        action: tf.Tensor,
        reporter: PPOReporter | None,
        timestep: int | None,
    ):
        '''
        Either both `reporter` and `timestep` must be defined or
        both must be `None`.
        
        If one is defined and the other is not, an assertion error
        is raised.
        '''
        
        assert(
            reporter == None and timestep == None or \
            reporter != None and timestep != None
        )
        assert(timestep == None or timestep > 0)
        
        # TODO: Do not run payload if it contains any sql_blacklist.txt tokens.
        # Severely negatively reward such actions.

        payload = payload_factory.create_payload_from_action(
            action=action,
            dictionary=self.dictionary,
        )
        
        response, min_levenshtein_norm = self.__inject_payload(payload)

        reward = self.__calculate_reward(
            min_levenshtein_norm=min_levenshtein_norm,
        )
        
        if reward > 0.0:
            self.__try_report_payload_statistics(
                payload=payload,
                reporter=reporter,
                timestep=timestep,
                reward=reward,
            )

        self.__injection_buffers.record_payload(payload)

        is_episode_done = self.__update_episode()

        state = self.__create_next_state(
            response=response,
            is_episode_done=is_episode_done,
        )

        return state, reward

    def perform_demonstration_action(self, action: tf.Tensor):
        return self.perform_action(
            action=action,
            reporter=None,
            timestep=None,
        )