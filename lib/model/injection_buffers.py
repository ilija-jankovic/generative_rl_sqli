from Levenshtein import ratio as levenshteinRatio

from typing import List, Set
from .payload import Payload


class InjectionBuffers:

    expected_responses: Set[str]
    
    __responses: Set[str]
    __attempted_payloads: List[Payload]

    
    def __init__(self, expected_responses: Set[str]) -> None:
        self.expected_responses = expected_responses
        
        self.__responses = expected_responses.copy()
        self.__attempted_payloads = []

        
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
        
        self.__responses.add(response)

        return min_distance_norm


    def record_payload(self, payload: Payload):
        self.__attempted_payloads.append(payload)


    def was_payload_attempted(self, payload: Payload):
        return payload in self.__attempted_payloads
    

    def clear(self):
        self.__responses = self.expected_responses.copy()
        self.__attempted_payloads.clear()