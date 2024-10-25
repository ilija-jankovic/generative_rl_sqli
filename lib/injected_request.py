

import requests

from .configuration import Configuration


def send_request(payload: str, config: Configuration) -> str:
    '''
    ===DEFINE YOUR INJECTED REQUEST SENDING LOGIC HERE===
    
    Must return a string.
    '''
    
    respose = requests.post(
        config.open_url + payload,
        headers=config.headers_with_cookie,
        json={
            'username': payload,
        }
    )
    
    return respose.text