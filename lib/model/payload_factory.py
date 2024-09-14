
from typing import List
import tensorflow as tf
from .payload import Payload


def __convert_action_to_payload(
    action: tf.Tensor,
    dictionary: List[str]
):
    tokens = [dictionary[i] for i in action]

    try:
        empty_token_index = tokens.index('')
    except:
        empty_token_index = None

    # Empty token counts as termination token for the agent. Slice tokens list 
    # up to first empty token.
    return ''.join(tokens if empty_token_index is None else tokens[:empty_token_index])


def create_payload_from_action(action: tf.Tensor, dictionary: List[str]):
    payload = __convert_action_to_payload(
        action=action,
        dictionary=dictionary,
    )
    
    return Payload(payload)