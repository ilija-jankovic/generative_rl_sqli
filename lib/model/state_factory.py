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
    tokens: str,
    max_size: int,
    dictionary: List[str],
    sorted_dictionary: List[str],
):
    dictionary_length = len(dictionary)

    indices: List[int] = []

    # Prioritise dictionary indices.
    #
    # Fall back to shifted ASCII indices.
    while len(tokens) > 0 and len(indices) < max_size:
        appended = False

        for token in sorted_dictionary:
            if tokens.startswith(token):
                index = dictionary.index(token)
                indices.append(index)

                # Remove token from prefix.
                tokens = tokens[len(token):]
                appended = True

                break

        if appended:
            continue
        
        # Append ASCII code shifted by max dictionary index.
        indices.append(ord(tokens[0]) + dictionary_length)

        tokens = tokens[1:]

    return indices


def __pad_index_list(indices: List[int], padded_length: int):
    indices_length = len(indices)
    
    assert(padded_length >= indices_length)
    
    return indices + [-1] * (padded_length - indices_length)


def __tokens_to_indices(
    tokens: List[str],
    index_list_length: int,
    dictionary: List[str],
    sorted_dictionary: List[str],
) -> List[int]:
    indices = __string_to_indices(
        tokens=tokens,
        max_size=index_list_length,
        dictionary=dictionary,
        sorted_dictionary=sorted_dictionary,
    )
    
    indices = __pad_index_list(
        indices=indices,
        padded_length=index_list_length,
    )
    
    return indices


def create_state_from_tokens(
    state_size: int,
    new_tokens_buffer: List[str],
    tokens: List[str],
    dictionary: List[str],
):
    total_new_tokens_count = len(new_tokens_buffer)

    joined_new_tokens = ''.join(new_tokens_buffer)
    joined_tokens = ''.join(tokens)

    sorted_dictionary = __sort_dictionary(dictionary)
    
    buffer_section_size = (state_size - 3) // 2
    
    new_tokens_indices = __tokens_to_indices(
        tokens=joined_new_tokens,
        index_list_length=buffer_section_size,
        dictionary=dictionary,
        sorted_dictionary=sorted_dictionary,
    )
    
    token_indices = __tokens_to_indices(
        tokens=joined_tokens, 
        index_list_length=buffer_section_size,
        dictionary=dictionary,
        sorted_dictionary=sorted_dictionary,
    )
    
    state = [total_new_tokens_count, -1] + \
        new_tokens_indices + [-1] + token_indices

    # Account for even state sizes, as the two buffers evenly
    # expand into the remaining space - state size minus three
    # slots.
    if state_size % 2 == 0:
        state.append(-1)

    assert(len(state) == state_size)
    
    return tf.convert_to_tensor(state, dtype=tf.float32)


def create_empty_state(state_size: int):
    return tf.zeros([state_size,], dtype=tf.float32)