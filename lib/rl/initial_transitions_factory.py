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

    def __get_corrupted_indicies(self, injection: List[int], avg_corruption_percentage: float):
        injections_length = len(injection)

        avg_corrupted_tokens: int = round(avg_corruption_percentage * injections_length)
        num_corrupted_tokens = random.randint(0, avg_corrupted_tokens * 2)
        num_corrupted_tokens = min(num_corrupted_tokens, injections_length)

        indicies = [i for i in range(injections_length)]
        return random.sample(indicies, num_corrupted_tokens)

    def __get_rand_token_index(self):
        '''
        Returns a uniformly random integer in `[-1, len(self.env.dictionary) -1]`.

        `-1` represents an empty token.
        '''
        return random.randint(0, len(self.env.dictionary)) - 1
    
    def __normalise(self, token_index: int):
        return 2.0 * (token_index + 1.0) / (len(self.env.dictionary) + 1.0) - 1.0
    
    def gather_transitions(self, num_transitions: int):
        state_next = self.env.create_empty_state()

        for _ in range(num_transitions):
            state = state_next

            injection = random.choice(self.env.encoded_injections).copy()

            print(''.join(['[VARIABLE]' if i == -1 else self.env.dictionary[i] for i in injection]))

            injection = [self.__get_random_non_sql_token_index() if i == -1 else i for i in injection]
            '''
            corrupted_indicies = self.__get_corrupted_indicies(injection, 0.2)
            
            for i in corrupted_indicies:
                injection[i] = self.__get_rand_token_index()
                '''
            
            # Ensure actions are always the correct size by padding with
            # null tokens until the environment's action size is reached.
            injection += [-1] * (self.env.action_size - len(injection))
            
            # Normalise to [-1.0, 1.0].
            injection = [self.__normalise(i) for i in injection]

            action = np.array(injection)
            state_next, reward, _ = self.env.perform_action(action)

            yield (state, action, reward, state_next)
