from typing import List
import tensorflow as tf


def __sort_dictionary(dictionary: List[str]):
    '''
    Since tokens may be a subset of each other, longer ones must be prioritied
    during this tokenization.

    The sorted dictionary is expected to have two layers of sorting: first by
    negative length, then by alphabetical order (for the case of multiple tokens
    of the same length existing).
    '''

    # Second condition prioritises alphabetically, as stated by Johannes from:
    # https://stackoverflow.com/a/44835987
    return sorted(
        dictionary,
        key=lambda token: (-len(token), token),
    )


def __string_to_indices(
    data: str,
    max_size: int,
    dictionary: List[str],
    sorted_dictionary: List[str]
):
    dictionary_length = len(dictionary)

    indexed_data: List[int] = []

    # Prioritise dictionary indices.
    #
    # Fall back to shifted ASCII indices.
    while len(data) > 0 and len(indexed_data) < max_size:
        appended = False

        for token in sorted_dictionary:
            if data.startswith(token):
                index = dictionary.index(token)
                indexed_data.append(index)

                # Remove token from prefix.
                data = data[len(token):]
                appended = True

                break

        if appended:
            continue
        
        # Append ASCII code shifted by max dictionary index.
        indexed_data.append(ord(data[0]) + dictionary_length)

        data = data[1:]

    return indexed_data


def create_state_from_tokens(
    state_size: int,
    tokens: List[str],
    new_tokens_buffer: List[str],
    dictionary: List[str],
):

    joined_tokens = ''.join(tokens)
    joined_new_tokens = ''.join(new_tokens_buffer)
    
    sorted_dictionary = __sort_dictionary(dictionary)
    
    max_new_tokens_size = state_size // 2 - 2
    new_token_indices = __string_to_indices(
        data=joined_new_tokens,
        max_size=max_new_tokens_size,
        dictionary=dictionary,
        sorted_dictionary=sorted_dictionary,
    )
    
    total_new_tokens_count = len(new_tokens_buffer)

    state = [total_new_tokens_count, -1, *new_token_indices, -1]
    max_data_tokens_size = state_size - len(state)
    
    data_indices = __string_to_indices(
        data=joined_tokens, 
        max_size=max_data_tokens_size,
        dictionary=dictionary,
        sorted_dictionary=sorted_dictionary,
    )

    state.extend(data_indices)

    # Pad state until self.state_size is reached.
    if(len(state) < state_size):
        state.extend([-1] * (state_size - len(state)))
    
    return tf.convert_to_tensor(state, dtype=tf.float32)


def create_empty_state(state_size: int):
    return tf.zeros([state_size,], dtype=tf.float32)