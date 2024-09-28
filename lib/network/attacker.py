import re
from typing import Callable, List
from bs4 import BeautifulSoup

from ..model.payload import Payload


SendRequestCallback = Callable[[], str]


def __filter_payload_from_text(
    text: str,
    payload: Payload,
):
    for token in payload.tokens:
        text = text.replace(token, '')
        
    return text


def __strip_lxml(text: str):
    '''
    Strip HTML and XML tags and recover text.

    Text instances are separated with a space.
    '''

    return BeautifulSoup(text, "lxml").get_text(separator='\0')


def __find_visible_text(text: str) -> List[str]:
    '''
    Tokenizes by matching all visible ASCII characters.
    '''

    return re.findall(r'[!-~]+', text)


def __send_single_request(
    send_request_callback: SendRequestCallback,
    payload: Payload,
):
    response = send_request_callback()

    response_text = __filter_payload_from_text(response, payload)
    response_text = __strip_lxml(response_text)

    return __find_visible_text(response_text)


def __filter_non_matching_text(text1: str, text2: str):
    tokens1 = __find_visible_text(text1)
    tokens2 = __find_visible_text(text2)

    combined = tokens1 + tokens2

    for token in combined:
        if token in tokens1 and token in tokens2:
            yield token
            

def __send_double_requests(
    send_request_callback: SendRequestCallback,
    payload: Payload,
):
    response1 = send_request_callback()
    response2 = send_request_callback()

    response_text1 = __filter_payload_from_text(response1, payload)
    response_text2 = __filter_payload_from_text(response2, payload)
    
    response_text1 = __strip_lxml(response_text1)
    response_text2 = __strip_lxml(response_text2)

    return list(__filter_non_matching_text(response_text1, response_text2))


def attack(
    send_request_callback: SendRequestCallback,
    payload: Payload,
    perform_double_requests: bool,
):
    return \
        __send_double_requests(
            send_request_callback=send_request_callback,
            payload=payload,
        ) if perform_double_requests else \
        __send_single_request(
            send_request_callback=send_request_callback,
            payload=payload,
        )
