import math
from typing import List
from typing import Callable
import tensorflow as tf

from .payload import Payload

from .injection_buffers import InjectionBuffers
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

    __episode: EpisodeState
    __injection_buffers: InjectionBuffers


    @property
    def episode(self):
        return self.__episode.episode
        

    def __inject_payload(self, payload_text: str, is_expected: bool):
        response_tokens = self.attack_callback(payload_text)

        new_tokens = self.__injection_buffers.record_tokens(
            response_tokens,
            is_expected=is_expected,
        )

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
        self.__injection_buffers = InjectionBuffers()
        
        self.__inject_initial_payloads()


    def __inject_initial_payloads(self):
        self.__inject_payload('1', is_expected=True)
        
        
    def __is_successful_payload(
        self,
        payload: Payload,
        new_tokens_count: int,
    ):
        assert(new_tokens_count >= 0)
        
        # Do not mark repeated payloads as successful, even in the case
        # of a non-deterministic network response.
        #
        # A pen-tester is unlikely to repeat the same injection multiple
        # times unless checking a previous result.
        return new_tokens_count > 0 and \
            not self.__injection_buffers.was_payload_attempted(payload)
        
        
    def __calculate_successful_reward(self, new_tokens_count: int):
        assert(new_tokens_count > 0)
        
        # Successful payloads further along the episode are more greatly
        # rewarded.
        #
        # They are less likely to be the result of simple injections, as
        # these were already likely rewarded.
        reward_weight = 1.0 + self.__episode.frames_since_last_episode / self.__episode.initial_frames
        reward = new_tokens_count * reward_weight

        # Map reward to (0.0, 1.0].
        #
        # Approximately linearly scaled down for low reward values, then
        # tapers off to upper/lower bound.
        reward = math.tanh(reward / 20.0)
        
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
        return 0.0 if payload.is_syntax_correct else -0.01


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
        response_tokens: List[str],
        is_episode_done: bool,
    ):
        return state_factory.create_empty_state(
            state_size=self.state_size,
        ) if is_episode_done \
            else state_factory.create_state_from_tokens(
                state_size=self.state_size,
                tokens=response_tokens,
                new_tokens_buffer=self.__injection_buffers.new_tokens,
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
        
        response_tokens, new_tokens = self.__inject_payload(
            payload_text=str(payload),
            is_expected=False,
        )

        new_tokens_count = len(new_tokens)
        
        if self.__is_successful_payload(
            payload=payload,
            new_tokens_count=new_tokens_count
        ):
            reward = self.__calculate_successful_reward(
                new_tokens_count=new_tokens_count,
            )
            
            self.__try_report_payload_statistics(
                payload=payload,
                reporter=reporter,
                timestep=timestep,
                reward=reward,
            )
        else:
            reward = self.__calculate_unsuccessful_reward(
                payload=payload,
            )

        self.__injection_buffers.record_payload(payload)

        is_episode_done = self.__update_episode()

        state = self.__create_next_state(
            response_tokens=response_tokens,
            is_episode_done=is_episode_done,
        )

        return state, reward

    def perform_demonstration_action(self, action: tf.Tensor):
        return self.perform_action(
            action=action,
            reporter=None,
            timestep=None,
        )