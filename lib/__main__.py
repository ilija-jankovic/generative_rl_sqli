#!/usr/local/bin/python
import sys
import numpy as np
import os
import random
import requests

from .hyperparameters import ACTION_SIZE, BATCH_SIZE, EMBEDDING_DIM, STATE_SIZE
from .model.ppo_actor_critic import PPOActorCritic
from .model.ppo import PPO, T
from .sql_parser.contextual_template_populator import ContextualTemplatePopulator
from .sql_parser.wikisql_parser import WikiSQLParser
from .sqlmap_runner import SqlmapRunner
from .sql_parser.token_parser import TokenParser
from .sql_parser import sql_data_service
from .sql_parser.schema_parser import get_column_tokens_from_schema, get_table_tokens_from_schema
from .nlp.token_embedder import TokenEmbedder 
from .model.environment import Environment
from .model.payload_builder import PayloadBuilder
from .util import match_list_lengths

import tensorflow as tf

# GPU private mode allows for optimisation. Thread count of two recommended but we set
# it to three.Environment variable documentation from:
# https://www.tensorflow.org/guide/gpu_performance_analysis#2_gpu_host_thread_contention
#
# TensorFlow environment variable setting solution by Prerak Mody from Performance Summary
# section in:
# https://towardsdatascience.com/profiling-in-tensorflow-2-x-576f5700124a
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '3'

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Multi-GPU Keras environment variable from official Keras tutorial:
# https://keras.io/guides/distributed_training_with_tensorflow/
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Print GPU devices visible to TensorFlow based on official guide:
# https://www.tensorflow.org/api_docs/python/tf/config/experimental/get_device_details
print('Visible GPU devices:')

gpu_devices = tf.config.list_physical_devices('GPU')
for device in gpu_devices:
  details = tf.config.experimental.get_device_details(device)
  print(details.get('device_name', 'Unknown GPU'))

tf.config.optimizer.set_jit(True)

# TODO: Use argparse instead.
args = sys.argv[1:]

run_sqlmap = '--no-run-sqlmap' not in args
use_cache = '--from-cache' in args
double_requests = '--no-double-requests' not in args
profile = '--profile' in args

embedding_data_rows = None

try:
    for arg in args:
        if arg.startswith('--embedding-dataset-count='):
            embedding_data_rows = int(arg.split('--embedding-dataset-count=')[1])
            break
except:
    pass

#
#
# !
# ! TODO: IMPORTANT: Add legal disclaimer.
# !
#
#

OPEN_URL = 'http://localhost:5000/items?id='

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'}
COOKIE = 'pma_lang=en; PHPSESSID=850f0124b1e274eccf6e9e13d8131e6c; {flag}=92262bf907af914b95a0fc33c3f33bf6'

visible_chars = [chr(i) for i in range(32, 127)]

sqlmap = SqlmapRunner(OPEN_URL, vulernable_param='id', default_vulnerable_param_value='1')

if run_sqlmap:
    print('Running sqlmap...')
    sqlmap.run(HEADERS, COOKIE)
    print('Attempted payloads gathered from sqlmap.')
else:
    print('Skipping running sqlmap...')

schema = sql_data_service.load_schema()

columns = get_column_tokens_from_schema(schema)

# NON-SYSTEM TABLE NAMES CAN BE CASE SENSITIVE (MySQL...)!!!!!
#
# Column names don't appear to be. This could be because they are not
# part of the file naming system (unlike table names depending on DBMS).
tables = get_table_tokens_from_schema(schema)

sql_tokens = sql_data_service.load_sql_tokens()
token_blacklist = sql_data_service.load_sql_blacklist()

payload_templates = sql_data_service.load_contextual_payload_templates()

# Get from cache for consistent tests.
payloads = ContextualTemplatePopulator(schema).generate_randomised_examples(templates=payload_templates)
sql_data_service.save_contextual_payloads(payloads)

payloads += sql_data_service.load_payload_files(domain_name='localhost')

dictionary = sql_tokens + tables + columns + visible_chars + ['']

# Remove duplicate schema column names and ASCII characters from dictionary.
dictionary = list(set(dictionary))

dictionary.sort(key=len, reverse=True)

print('DICTIONARY:')
print('\n'.join(dictionary))

token_parser = TokenParser(dictionary, token_blacklist)

if use_cache:
    print('Using cached embeddings...')
    embeddings = sql_data_service.load_embeddings()
else:
    queries = sql_data_service.load_wikisql_queries()

    parser = WikiSQLParser(schema)
    queries = parser.generate_randomised_examples(queries)

    sql_data_service.save_parsed_wikisql_queries(queries)

    embedding_training_data = payloads + queries
    
    # Ensure equally distributed categories of payloads and queries
    # before uniform shuffling. Both categories are important for training,
    # so equal importance is assumed.
    #
    # Note that this will duplicate members in the smaller list, allowing
    # for repeated data.
    match_list_lengths(payloads, queries)

    # Ensure same distribution uniformly random selection across every run.
    random.Random(19549708).shuffle(embedding_training_data)
    
    print('\n'.join(embedding_training_data[:100]))
    print('Printed first 100 slice of embedding training data.')
    
    embedding_training_data = embedding_training_data if embedding_data_rows is None else embedding_training_data[:embedding_data_rows]

    print('Encoding SQL fragment(s) for embeddings...')
    embedding_training_data = token_parser.parse(embedding_training_data)
    print(f'{len(embedding_training_data)} fragment(s) encoded.')

    # TODO: Retrieve cached embeddings (if already generated) if dictionary and
    # embedding examples are unchanged.
    print('Running token embedder...')
    embeddings = TokenEmbedder(EMBEDDING_DIM).learn_embeddings(embedding_training_data, len(dictionary))
    print('Embeddings learned.')

    sql_data_service.save_embeddings(embeddings.numpy().tolist())

headers = HEADERS.copy()
headers.update({'cookie': COOKIE})

payload_builder = PayloadBuilder(dictionary, '', '')

environments = [
    Environment(
        payload_builder=payload_builder,
        action_size=ACTION_SIZE,
        state_size=STATE_SIZE,
        frames_per_episode=T,
        embeddings=embeddings,
        double_requests=double_requests,
        send_request_callback= lambda payload:
            requests.get(OPEN_URL + payload, headers=headers))
            for _ in range(BATCH_SIZE + 1)
]

demonstration_environment = environments[0]
environments = environments[1:]

state: np.ndarray

def print_decoded_injections(encoded_payloads):
    '''
    Use to verify that the predefined encoded injection list is correctly
    encoded.
    '''
    print('DEMONSTRATION PAYLOADS:')
    for injection in encoded_payloads:
        decoded = [dictionary[i] for i in injection]
        print(''.join(decoded))

def main():
    dictionary_length = len(dictionary)

    with open(f'{os.path.dirname(__file__)}/../injections_demonstration.txt', 'r') as f:
        payloads = f.read().splitlines()
    f.close()

    encoded_demonstration = token_parser.parse(payloads)

    for encoded_payload in encoded_demonstration:
        assert(len(encoded_payload) <= ACTION_SIZE)

    # Pad with empty token.
    encoded_demonstration = [[payload[i] if i < len(payload) else dictionary_length - 1 for i in range(ACTION_SIZE)] for payload in encoded_demonstration]

    print_decoded_injections(encoded_demonstration)
    
    demonstration_actions = tf.convert_to_tensor(encoded_demonstration)

    actor_critic = PPOActorCritic(
        dictionary_length=len(dictionary),
        action_size=ACTION_SIZE,
        state_size=STATE_SIZE,
        embedding_size=EMBEDDING_DIM,
        batch_size=BATCH_SIZE,
        embeddings=embeddings,
    )
    
    ppo = PPO(
        actor_critic,
        environments=environments,
    )

    ppo.run(
        demonstration_environment=demonstration_environment,
        demonstration_actions=demonstration_actions
    )


if __name__ == '__main__':
    main()