from typing import List, Tuple
import tqdm

class TokenParser:

    tokens: List[str]
    token_blacklist: List[str]
    data: List[str]

    def __init__(self,
            tokens: List[str],
            token_blacklist: List[str]):
        '''
        `tokens` must be in descending order.
        '''

        # Sort in reverse alphabetically so that we match longer tokens
        # first if multiple tokens contain the same prefix.
        #
        # For example, for tokens 'foo' and 'foo bar', we want 'foo
        # bar' to get tokenized first, else only 'foo' of 'foo bar'
        # will get tokenized.
        self.tokens = sorted(tokens, reverse=True)
        if self.tokens != tokens:
            raise Exception('Tokens must be sorted in descending order.')

        self.token_blacklist = token_blacklist

    def __contains_blacklisted_token(self, datum: str):
        for token in self.token_blacklist:
            if token in datum:
                return True
        
        for token in self.tokens:
            datum = datum.replace(token, '')

        return len(datum) > 0

    def __remove_blacklisted_tokens(self, data: List[str]):
        return list(filter(
            lambda datum: not self.__contains_blacklisted_token(datum),
            data
        ))
    
    def __tokenize(self, data: List[str]):
        '''
        Assumes `data` does not contain any blacklisted tokens.
        '''
        tokens_per_row = len(max(data, key=len))

        indexed_data: List[List[int]] = []

        for datum in tqdm.tqdm(data):
            index_map_list: List[Tuple[int, int]] = []

            for token in self.tokens:
                indices = [i for i,
                           item in enumerate(datum) if item == token]

                token_index = self.tokens.index(token)
                for index in indices:
                    index_map_list.append((index, token_index))

            index_map_list.sort(key = lambda map: map[0])

            indexed_datum = [map[1] for map in index_map_list]

            # Pad with padding token (which is expected to be at the bottom of the tokens list
            # due to it being an empty string after sorting).
            indexed_datum += [len(self.tokens) - 1] * (tokens_per_row - len(indexed_datum))

            indexed_data.append(indexed_datum)

        return indexed_data
    
    def __filter_unformatted_data(self, data: List[str], prefix: str, suffix: str):
        data = list(filter(lambda datum: datum.startswith(prefix) and datum.endswith(suffix), data))

        prefix_length = len(prefix)
        suffix_length = len(suffix)

        if prefix_length > 0:
            data = list(map(lambda datum: datum[prefix_length:], data))

        if suffix_length > 0:
            data = list(map(lambda datum: datum[:-suffix_length], data))

        return data
    
    def parse(self, data: List[str], required_prefix: str = '', required_suffix: str = ''):
        '''
        Returns data parsed to token indices in range `[0, ..., len(tokens) - 1]`.

        If `required_prefix` and/or `required_suffix` are provided, data is filtered
        based on these and these are trimmed off.
        '''
        data = self.__remove_blacklisted_tokens(data)

        if len(required_prefix) > 0 or len(required_suffix) > 0:
            data = self.__filter_unformatted_data(data, required_prefix, required_suffix)

        return self.__tokenize(data)
