

from typing import List
from .payload import Payload


class InjectionBuffers:
    
    __found_tokens: List[str]

    __new_tokens: List[str]
    '''
    Buffer of unique newfound tokens in reverse chronological
    order.
    '''
    
    __attempted_payloads: List[Payload]

    
    @property
    def new_tokens(self):
        return self.__new_tokens

    
    def __init__(self) -> None:
        self.__found_tokens = []
        self.__new_tokens = []
        self.__attempted_payloads = []
        
        
    def record_tokens(
        self,
        tokens: List[str],
        is_expected: bool,
    ):
        '''
        If unknown tokens are found in `tokens`, these tokens
        are appended to the 'found tokens' buffer.
        
        These newfound tokens are prepended to the 'new tokens'
        buffer if `is_expected` is `False`.
        
        Returns the newfound tokens if added to the respective
        buffer.
        '''
        new_tokens: List[str] = []
        
        for token in tokens:
            
            # Avoid sets for token processing, as their order
            # is non-deterministic. This is undesirable for
            # tests, as well as consistency for the agent.
            #
            # We therefore check whether a token is in the
            # 'found tokens' buffer for ensuring uniqueness.
            if token not in self.__found_tokens:
                self.__found_tokens.append(token)

                if(not is_expected):
                    new_tokens.insert(0, token)

        # Prepend new tokens for most recent representation of
        # new tokens with capped buffer size, supporting FIFO
        # buffer reading operations.
        self.__new_tokens = new_tokens + self.__new_tokens

        return new_tokens


    def record_payload(self, payload: Payload):
        self.__attempted_payloads.append(payload)


    def was_payload_attempted(self, payload: Payload):
        return payload in self.__attempted_payloads
    

    def clear(self):
        self.__found_tokens.clear()
        self.__new_tokens.clear()
        self.__attempted_payloads.clear()