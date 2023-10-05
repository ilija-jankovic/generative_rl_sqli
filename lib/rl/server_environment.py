import numpy as np
import re
from typing import List
from sqltree import sqltree
from .environment import Environment
from requests import Response
from typing import Callable

class ServerEnvironment(Environment):
    send_request_callback: Callable[[str], Response]

    __found_tokens: List[str] = []

    def __init__(self, dictionary: List[str], action_size: int, state_size: int,
        send_request_callback: Callable[[str], Response]):
        super().__init__(dictionary, action_size, state_size)
        self.send_request_callback = send_request_callback
        
    def __is_valid_sql_payload(self, state: np.ndarray):
        try:
            payload = self.get_payload(state)
            sqltree(f'SELECT * FROM test WHERE test = \'{payload}\'')
            return True
        except:
            return False
        
    def perform_termination_action(self, state: np.ndarray):
        payload = self._record_payload(state)

        #if(self.__is_valid_sql_payload(payload)):
        #    state = self.dqn.create_empty_state()
        #    return state, -1, True

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

        state = self.create_empty_state()

        return state, reward, True

    