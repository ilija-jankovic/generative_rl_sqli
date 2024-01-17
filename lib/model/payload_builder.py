import math
from typing import List

import tensorflow as tf


class PayloadBuilder:
    dictionary: List[str]
    prefix: str
    suffix: str

    def __init__(self, dictionary: List[str], prefix: str, suffix: str) -> None:
        self.dictionary = dictionary
        self.prefix = prefix
        self.suffix = suffix

    def convert_action_to_payload(self, action: tf.Tensor):
        dictionary_length = len(self.dictionary)

        tokens = [self.dictionary[int(i)] if i >= 0.0 and i < dictionary_length else '' for i in action.numpy()]

        try:
            empty_token_index = tokens.index('')
        except:
            empty_token_index = None

        # Empty token counts as termination token for the agent. Slice tokens list 
        # up to first empty token.
        payload = ''.join(tokens if empty_token_index is None else tokens[:empty_token_index])

        return self.prefix + payload + self.suffix
