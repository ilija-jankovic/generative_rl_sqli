import re
from Levenshtein import ratio as levenshteinRatio

from typing import List, Set
from .payload import Payload


class InjectionBuffers:

    expected_responses: Set[str]
    
    __responses: Set[str]
    __response_tokens: Set[str]
    __attempted_payloads: List[Payload]
    __private_tokens_count: int
    
    
    @property
    def private_tokens_count(self):
        return self.__private_tokens_count


    def __record_response_tokens(self, response: str):
        for token in re.findall(r'[!-~]+', response):
            self.__response_tokens.add(token)
    
    
    def __reset_response_tokens(self):
        self.__response_tokens = set()
        
        for response in self.expected_responses:
            self.__record_response_tokens(response=response)


    def __init__(self, expected_responses: Set[str]) -> None:
        self.expected_responses = expected_responses
        
        self.__responses = expected_responses.copy()
        self.__reset_response_tokens()
 
        self.__attempted_payloads = []
        self.__private_tokens_count = 0

        
    def record_response(
        self,
        response: str,
    ):
        min_distance_norm = 1.0

        for recorded_response in self.__responses:
            distance_norm = 1.0 - levenshteinRatio(
                response,
                recorded_response,
            )
            
            if distance_norm < min_distance_norm:
                min_distance_norm = distance_norm
        
        new_tokens_count = 0
        
        for token in re.findall(r'[!-~]+', response):
            if token in self.__response_tokens:
                continue
            
            new_tokens_count += 1
        
        self.__responses.add(response)
        self.__record_response_tokens(response=response)
        
        self.__private_tokens_count += new_tokens_count

        return min_distance_norm * new_tokens_count


    def record_payload(self, payload: Payload):
        self.__attempted_payloads.append(payload)


    def was_payload_attempted(self, payload: Payload):
        return payload in self.__attempted_payloads
    

    def clear(self):
        self.__responses = self.expected_responses.copy()
        self.__reset_response_tokens()
        self.__attempted_payloads.clear()
        self.__private_tokens_count = 0