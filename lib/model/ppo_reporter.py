
from datetime import datetime
import os
from typing import List, Set

from .ppo_running_statistics import PPORunningStatistics
from .total_episodic_statistics import TotalEpisodicStatistics
from .ppo_payload_statistics import PPOPayloadStatistics

from .. import hyperparameters

class PPOReporter:

    __RUNNING_COLUMNS: List[str] = [
        'Since Beginning Seconds',
        'Timestep',
        'Mean Batch Reward',
        'Mean Actor Loss',
        'Mean Critic Loss',
        'Exploration Seconds',
        'Learning Seconds',
        'Training Step Seconds',
    ]
    
    __EPISODE_COLUMNS: List[str] = [
        'Since Beginning Seconds',
        'Epsiode',
        'Mean Cumulative Episodic Reward',
        'Mean Accuracy',
    ]

    __PAYLOAD_COLUMNS: List[str] = [
        'Since Beginning Seconds',
        'Timestep',
        'Reward',
        'Syntax Estimated Correct',
        
        # Payload at end as a backup in case a cell break character is not escaped.
        'Payload',
    ]
    
    __recorded_payloads: Set[str]
    
    __startedAt: datetime = None
    __statistics_filename: str
    __episodes_filename: str
    __payloads_filename: str
    
    def __init__(self):
        self.__recorded_payloads = set()
    
    def is_payload_recorded(self, payload: str):
        return payload in self.__recorded_payloads

    def start(self):
        dirname = os.path.dirname(__file__)
        
        self.__recorded_payloads.clear()

        # Replacement removes invalid token in filename.
        self.__startedAt = datetime.now()
        now = str(self.__startedAt.isoformat(sep=' ', timespec='seconds')).replace(':', '-')

        directory = f'{dirname}/../../reports/{now}'

        self.__statistics_filename = f'{directory}/{now} PPO Running Statistics Report.csv'
        self.__episodes_filename = f'{directory}/{now} PPO Episodes Report.csv'
        self.__payloads_filename = f'{directory}/{now} PPO Successful Payloads Report.csv'
        
        os.makedirs(os.path.dirname(self.__statistics_filename), exist_ok=True)

        params = {
            'STATE_SIZE': hyperparameters.STATE_SIZE,
            'ACTION_SIZE': hyperparameters.ACTION_SIZE,
            'EMBEDDING_DIM': hyperparameters.EMBEDDING_DIM,
            'T': hyperparameters.T,
            'INITIAL_EPISODE_LENGTH': hyperparameters.INITIAL_EPISODE_LENGTH,
            'PRETRAIN_ACTOR_TYPE': hyperparameters.PRETRAIN_ACTOR_TYPE.name,
            'PRETRAINING_STEPS': hyperparameters.PRETRAINING_STEPS,
            'PRETRAINING_LEARNING_RATE': hyperparameters.PRETRAINING_LEARNING_RATE,
            'MAX_EPISODE_EXTENSION': hyperparameters.MAX_EPISODE_EXTENSION,
            'EPOCHS': hyperparameters.EPOCHS,
            'BATCH_SIZE': hyperparameters.BATCH_SIZE,
            'MINIBATCH_SIZE': hyperparameters.MINIBATCH_SIZE,
            'PPO_PROBABILITY_RATIO_CLIP_THRESHOLD': hyperparameters.PPO_PROBABILITY_RATIO_CLIP_THRESHOLD,
            'PPO_SUCCESSFUL_BATCH_SIZE': hyperparameters.PPO_SUCCESSFUL_BATCH_SIZE,
            'PPO_SUCCESSFUL_BUFFER_SIZE': hyperparameters.PPO_SUCCESSFUL_BUFFER_SIZE,
            'GAMMA': hyperparameters.GAMMA,
            'INITIAL_ACTOR_LEARNING_RATE': hyperparameters.INITIAL_ACTOR_LEARNING_RATE,
            'INITIAL_CRITIC_LEARNING_RATE': hyperparameters.INITIAL_CRITIC_LEARNING_RATE,
            'L2_WEIGHT': hyperparameters.L2_WEIGHT,
            'ACTOR_LSTM_UNITS': hyperparameters.ACTOR_LSTM_UNITS,
            'ACTOR_DENSE_UNITS': hyperparameters.ACTOR_DENSE_UNITS,
            'ACTOR_SOFTMAX_TEMPERATURE': 2.0,
            'ADAM_BETA1': hyperparameters.ADAM_BETA1,
            'ADAM_BETA2': hyperparameters.ADAM_BETA2,
            'ADAM_EPSILON': hyperparameters.ADAM_EPSILON,
            'LR_SCHEDULE_DECAY_STEPS': hyperparameters.LR_SCHEDULE_DECAY_STEPS,
            'LR_SCHEDULE_DECAY_RATE': hyperparameters.LR_SCHEDULE_DECAY_RATE,
        }

        with open(self.__statistics_filename, 'w', encoding='utf-8') as f:
            f.write(f'Report Started,{now}\n\n')
            f.write('Constants\n')
            
            for name, value in params.items():
                f.write(f'{name},{value}\n')
                
            f.write(f'{",".join(self.__RUNNING_COLUMNS)}\n')
            
        f.close()

        with open(self.__episodes_filename, 'w', encoding='utf-8') as f:
            f.write(f'Report Started,{now}\n\n')
            
            f.write(f'{",".join(self.__EPISODE_COLUMNS)}\n')
            
        f.close()

        with open(self.__payloads_filename, 'w', encoding='utf-8') as f:
            f.write(f'Report Started,{now}\n\n')
            
            f.write(f'{",".join(self.__PAYLOAD_COLUMNS)}\n')
            
        f.close()
        
    def __get_seconds_since_start(self):
        assert(self.__startedAt != None)
        
        return (datetime.now() - self.__startedAt).total_seconds()

    def record_running_statistics(self, stats: PPORunningStatistics):
        if self.__startedAt == None:
            raise Exception('Must call start() before recording statistics.')
        
        stats_dict = {
            'Since Beginning Seconds': self.__get_seconds_since_start(),
            'Timestep': stats.timestep,
            'Mean Batch Reward': stats.mean_batch_reward,
            'Mean Actor Loss': stats.mean_actor_loss,
            'Mean Critic Loss': stats.mean_critic_loss,
            'Exploration Seconds': stats.exploration_seconds,
            'Learning Seconds': stats.learning_seconds,
            'Training Step Seconds': stats.training_step_seconds,
        }
        
        ordered_stats = [str(stats_dict[column]) for column in self.__RUNNING_COLUMNS]

        with open(self.__statistics_filename, 'a', encoding='utf-8') as f:
            f.write(f'{",".join(ordered_stats)}\n')
            
        f.close()
        
    def record_episodic_statistics(self, stats: TotalEpisodicStatistics):
        if self.__startedAt == None:
            raise Exception('Must call start() before recording statistics.')

        stats_dict = {
            'Since Beginning Seconds': self.__get_seconds_since_start(),
            'Epsiode': stats.episode,
            'Mean Cumulative Episodic Reward': stats.mean_cumulative_episodic_reward,
            'Mean Accuracy': stats.mean_accuracy,
        }
        
        ordered_stats = [str(stats_dict[column]) for column in self.__EPISODE_COLUMNS]

        with open(self.__episodes_filename, 'a', encoding='utf-8') as f:
            f.write(f'{",".join(ordered_stats)}\n')
            
        f.close()


    def record_payload_statistic(self, stats: PPOPayloadStatistics):
        if self.__startedAt == None:
            raise Exception('Must call start() before recording statistics.')
        
        payload_str = str(stats.payload)
        self.__recorded_payloads.add(payload_str)
        
        # - Double double-quotes escape them in Excel.
        # - Quotes around the entry escape commas.
        escaped_payload = payload_str.replace('"', '""')
        escaped_payload = f'\"{escaped_payload}\"'

        stats_dict = {
            'Since Beginning Seconds': self.__get_seconds_since_start(),
            'Timestep': stats.timestep,
            'Reward': stats.reward,
            'Syntax Estimated Correct': stats.payload.is_syntax_correct,
            'Payload': escaped_payload,
        }
        
        ordered_stats = [str(stats_dict[column]) for column in self.__PAYLOAD_COLUMNS]
        
        with open(self.__payloads_filename, 'a', encoding='utf-8') as f:
            f.write(f'{",".join(ordered_stats)}\n')
            
        f.close()
        
