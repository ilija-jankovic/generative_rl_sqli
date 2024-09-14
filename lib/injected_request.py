

import requests

from .configuration import Configuration


def send_request(payload: str, config: Configuration) -> str:
    '''
    ===DEFINE YOUR INJECTED REQUEST SENDING LOGIC HERE===
    
    Must return a string.
    '''
    
    respose = requests.get(
        config.open_url + payload,
        headers=config.headers_with_cookie,
    )
    
    return respose.text