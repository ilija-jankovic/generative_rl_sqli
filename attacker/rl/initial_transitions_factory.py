from .environment import Environment
import random as random
import numpy as np

class InitialTransitionsFactory:
    env: Environment

    def __init__(self, env: Environment):
        self.env = env

    def __normalise(self, token_index: int):
        return 2.0 * (token_index + 1.0) / (len(self.env.dictionary) + 1.0) - 1.0
    
    def gather_transitions(self, num_transitions: int):
        state_next = self.env.create_empty_state()

        for _ in range(num_transitions):
            state = state_next

            injection = random.choice(self.env.encoded_payloads).copy()
            
            # Ensure actions are always the correct size by padding with
            # empty tokens until the action size is reached.
            injection += [-1] * (self.env.action_size - len(injection))
            
            # Normalise to [-1.0, 1.0].
            injection = [self.__normalise(i) for i in injection]

            action = np.array(injection)
            state_next, reward, _ = self.env.perform_action(action)

            yield (state, action, reward, state_next)
