from typing import List, Tuple
import tqdm

class TokenParser:

    tokens: List[str]
    token_blacklist: List[str]
    data: List[str]
    tokens_per_row: int

    def __init__(self,
            tokens: List[str],
            token_blacklist: List[str],
            tokens_per_row: int):
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

        self.tokens_per_row = tokens_per_row

    def __contains_blacklisted_token(self, datum: str):
        for token in self.token_blacklist:
            if token in datum:
                return True
        
        # Ignore padding token at end of list.
        for token in self.tokens[:-1]:
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
        indexed_data: List[List[int]] = []

        for datum in tqdm.tqdm(data):
            index_map_list: List[Tuple[int, int]] = []

            # Ignore padding token at end of list.
            for token in self.tokens[:-1]:
                indices = [i for i,
                           item in enumerate(datum) if item == token]

                token_index = self.tokens.index(token)
                for index in indices:
                    index_map_list.append((index, token_index))

            index_map_list.sort(key = lambda map: map[0])

            indexed_datum = [map[1] for map in index_map_list]
            indexed_datum += [0] * (self.tokens_per_row - len(indexed_datum))

            indexed_data.append(indexed_datum)

        return indexed_data
    
    def parse(self, data: List[str]):
        '''
        Returns data parsed to token indices in range `[0, ..., len(tokens) - 1]`.
        '''
        data = self.__remove_blacklisted_tokens(data)
        return self.__tokenize(data)
