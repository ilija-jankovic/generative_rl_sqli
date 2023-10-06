from typing import List
from .pre_training_environment import PreTrainingEnvironment
import random as random
import numpy as np

class InitialTransitionsFactory:
    env: PreTrainingEnvironment

    def __init__(self, env: PreTrainingEnvironment):
        self.env = env

    def __get_random_non_sql_token_index(self):
        '''
        Will always return a non-empty token.
        '''
        return random.randint(len(self.env.sql_syntax), len(self.env.dictionary) - 1)
    
    def __normalise(self, token_index: int):
        return 2.0 * (token_index + 1.0) / (len(self.env.dictionary) + 1.0) - 1.0
    
    def gather_transitions(self, num_transitions: int):
        state_next = self.env.create_empty_state()

        for _ in range(num_transitions):
            state = state_next

            injection = random.choice(self.env.encoded_injections).copy()

            # Fill variable tokens with non-SQL tokens.
            injection = [self.__get_random_non_sql_token_index() if i == -1 else i for i in injection]
            
            # Ensure actions are always the correct size by padding with
            # empty tokens until the action size is reached.
            injection += [-1] * (self.env.action_size - len(injection))
            
            # Normalise to [-1.0, 1.0].
            injection = [self.__normalise(i) for i in injection]

            action = np.array(injection)
            state_next, reward, _ = self.env.perform_action(action)

            yield (state, action, reward, state_next)
