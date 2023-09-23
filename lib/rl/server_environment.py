import numpy as np
import re
from typing import List
from sqltree import sqltree
from .dqn import DQN
from .environment import Environment
from requests import Response
from typing import Callable

class ServerEnvironment(Environment):
    send_request_callback: Callable[[str], Response]

    __found_tokens: List[str] = []
    __attempted_payloads: List[str] = []

    def __init__(self, dqn: DQN, send_request_callback: Callable[[str], Response]):
        super().__init__(dqn)
        self.send_request_callback = send_request_callback
        
    def __is_valid_sql_payload(self, state: np.ndarray):
        try:
            payload = self.get_payload(state)
            sqltree(f'SELECT * FROM test WHERE test = \'{payload}\'')
            return True
        except:
            return False
        
    def perform_termination_action(self, state: np.ndarray):
        payload = self.get_payload(state)
        self.__attempted_payloads.append(payload)

        if(self.__is_valid_sql_payload(payload)):
            state = self.dqn.create_empty_state()
            return state, -1, True

        print(payload)
        res = self.send_request_callback(payload)

        unique_tokens = set()
        for token in re.split('[^a-zA-Z]+', res.text):
            unique_tokens.add(token)

        reward = 0

        new_tokens = []
        for token in unique_tokens:
            if(token not in self.__found_tokens):
                self.__found_tokens.append(token)
                new_tokens.append(token)
                reward += 1

        if(len(new_tokens) > 0):
            print(payload)
            print('\nNEW DATA')
            print('\nFound:', new_tokens, '\n')

        state = self.dqn.create_empty_state()

        return state, reward, True

    def perform_mutation_action(self, action: str, state: np.ndarray):
        return self._mutate_state(state, action), 0, False
    
    def payload_attempted(self, payload: str):
        return payload in self.__attempted_payloads