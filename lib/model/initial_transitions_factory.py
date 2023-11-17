import itertools
from typing import List
from .environment import Environment
import random as random
import numpy as np

class InitialTransitionsFactory:
    env: Environment
    encoded_payloads: List[List[int]]

    def __init__(self, env: Environment, encoded_payloads: List[List[int]]):
        self.env = env
        self.encoded_payloads = list(filter(
            lambda encoded_payload: len(encoded_payload) <= env.action_size // env.embedding_size, encoded_payloads))
    
    def gather_transitions(self, num_transitions: int):
        state_next = self.env.create_empty_state()

        for _ in range(num_transitions):
            state = state_next

            encoded_payload = random.choice(self.encoded_payloads)

            action = list(itertools.chain(*[self.env.embeddings[i] for i in encoded_payload]))
            action = np.array(action)

            state_next, reward, _ = self.env.perform_action(action)

            yield (state, action, reward, state_next)
