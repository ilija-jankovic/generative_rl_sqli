from typing import List, Tuple


class TokenParser:

    tokens: List[str]
    token_blacklist: List[str]
    data: List[str]

    def __init__(self, tokens: List[str], token_blacklist: List[str],
                data: List[str]):
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
        self.data = data

    def __contains_blacklisted_token(self, datum: str):
        for token in self.token_blacklist:
            if token in datum:
                return True
            
        for token in self.tokens:
            if token not in datum:
                return True

        return False

    def __remove_blacklisted_tokens(self):
        return list(filter(
            lambda datum: not self.__contains_blacklisted_token(datum),
            self.data
        ))
    
    def __tokenize(self, data: List[str]):
        '''
        Assumes `data` does not contain any blacklisted tokens.
        '''
        indexed_data: List[List[int]] = []

        for datum in data:
            index_map_list: List[Tuple[int, int]] = []

            for token in self.tokens:
                indices = [i for i,
                           item in enumerate(datum) if item == token]

                for index in indices:
                    index_map_list.append((index, token))

            index_map_list.sort(key = lambda map: map[0])
            indexed_data.append([map[1] for map in index_map_list])

        return indexed_data
    
    def parse(self):
        '''
        Returns data parsed to token indices in range `[0, ..., len(tokens) - 1]`.
        '''
        data = self.__remove_blacklisted_tokens(self.data)
        return self.__tokenize(data)
