import os
from typing import List

from dqn import DQN
from models.epsilon_model import EpsilonModel
from models.rl_hyperparameters_model import RLHyperparametersModel
import numpy as np
import requests
import re

url = 'http://localhost.proxyman.io:3000/rest/user/login'

# Scrape site solution by T0ny lombardi from:
# https://stackoverflow.com/questions/9265172/scrape-an-entire-website
# os.system('wget -m -k -K -E -l 7 -t 6 -w 5 http://localhost:3000 -P ./scraped')

visible_chars = [chr(i) for i in range(32, 127)]

with open('parsed-scrape.txt', 'r') as f:
    data = f.read()
f.close()

#found_words = list(filter(lambda s: s != '', data.split(' ')))
found_words = []

with open('sql_list.txt', 'r') as f:
    data = f.read()
f.close()

sql_list = data.split('\n')

visible_chars *= max(1, int(len(found_words)/len(visible_chars)))

sql_list *= max(1, int(len(visible_chars)/len(sql_list)))

non_terminating_actions = visible_chars + sql_list + found_words

terminating_actions = ['TERMINATE' for _ in range(len(non_terminating_actions))]

actions: List[str] = terminating_actions + non_terminating_actions

terminated = False
state: np.ndarray

def __get_payload():
    chrs = state.tolist()
    chrs = [chr(int(i)) for i in chrs if i != 0.0]
    return ''.join(chrs)

def __perform_action(action_index: int):
    global state

    action = actions[action_index]

    if(action != 'TERMINATE'):
        for i in range(len(state)):
            if(state[i] != 0.0):
                continue

            for j in range(len(action)):
                state_index = i + j
                if(state_index >= len(state)):
                    break

                state[state_index] = ord(action[j])
            return state, 0, False
            
        return state, -1, False

    payload = __get_payload()
    res = requests.post(url, data={
        'email': payload
    })

    unique_tokens = set()
    for token in re.split('[^a-zA-Z]+', res.text):
        unique_tokens.add(token)

    reward = 0

    for token in unique_tokens:
        if(token not in actions):
            actions.append(token)
            print(token)
            reward += 1

    if(reward > 0):
        print('\nPayload: ' + payload)

    #dqn.add_to_available_actions_count(reward)

    state = dqn.create_empty_state()
    return state, reward, True

dqn = DQN(
    RLHyperparametersModel(
        gamma=0.99,
        learning_rate=0.001,
        batch_size=1,
        training_episodes=5000,
        test_episodes=100,
        max_steps_per_episode=100,
        feature_count=10000,
        action_count=len(actions) + 10000
    ),
    EpsilonModel(
        start=1.0,
        min=0.05,
        max=1.0,
        num_random_frames=2000
    ),
    available_actions_count=len(actions),
    perform_action_callback=__perform_action
)

state = dqn.create_empty_state()

model, model_target = dqn.create_model()
dqn.run(model, model_target)