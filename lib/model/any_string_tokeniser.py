from typing import List


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


def tokens_to_indices(
    tokens: str,
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