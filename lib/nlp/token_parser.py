from typing import List, Tuple
import tqdm

class TokenParser:

    tokens: List[str]
    token_blacklist: List[str]
    data: List[str]


    def __init__(
        self,
        tokens: List[str],
        token_blacklist: List[str],
    ):
        '''
        `tokens` must be in descending order.
        '''

        # Sort in reverse alphabetically so that we match longer tokens
        # first if multiple tokens contain the same prefix.
        #
        # For example, for tokens 'foo' and 'foo bar', we want 'foo
        # bar' to get tokenized first, else only 'foo' of 'foo bar'
        # will get tokenized.
        self.tokens = sorted(tokens, key=len, reverse=True)
        if self.tokens != tokens:
            raise Exception('Tokens must be sorted in descending order.')
        
        if self.tokens[-1] != '':
            raise Exception('Sorted tokens list must contain empty token at the end.')

        self.token_blacklist = token_blacklist


    def __contains_blacklisted_token(self, datum: str):
        for token in self.token_blacklist:
            if token in datum:
                return True
        
        # Blacklist data with sequences not contained in tokens.
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
        indexed_data: List[List[int]] = []

        for datum in tqdm.tqdm(data):
            datum_replaced = datum

            index_map_list: List[Tuple[int, int]] = []

            # Disregard empty token at the end of the token list.
            for token_index, token in enumerate(self.tokens[:-1]):
                token_length = len(token)
                filler = '\0' * token_length

                index = datum_replaced.find(token)

                # Looping .find solution by AkiRoss from:
                # https://stackoverflow.com/questions/4664850/how-to-find-all-occurrences-of-a-substring
                while index != -1:
                    token_end = index + token_length

                    datum_replaced = datum_replaced[0:index] + filler + datum_replaced[token_end:]

                    index_map_list.append((index, token_index))

                    index = datum_replaced.find(token)

                parsed = True
                for chr in datum_replaced:
                    if chr != '\0':
                        parsed = False
                        break
                
                if parsed:
                    break

            index_map_list.sort(key = lambda map: map[0])

            indexed_datum = [map[1] for map in index_map_list]
            indexed_data.append(indexed_datum)

        tokens_per_row = len(max(indexed_data, key=len))
        
        # Pad with padding token (which is expected to be at the bottom of the tokens list
        # due to it being an empty string after sorting).
        for indexed_datum in indexed_data:
            indexed_datum += [len(self.tokens) - 1] * (tokens_per_row - len(indexed_datum))

        return indexed_data

    
    def parse(self, data: List[str]):
        '''
        Returns data parsed to token indices in range `[0, ..., len(tokens) - 1]`.
        '''
        data = self.__remove_blacklisted_tokens(data)

        return self.__tokenize(data)
