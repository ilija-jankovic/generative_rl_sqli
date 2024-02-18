from datetime import datetime
import os

from .ddpg_running_statistic import DDPGRunningStatistic
from .ddpg_hyperparameters import DDPGHyperparameters
from .ddpg_payload_statistic import DDPGPayloadStatistic


class Reporter:

    __started: bool = False
    __statistics_filename: str
    __payloads_filename: str

    __params: DDPGHyperparameters

    def start(self, params: DDPGHyperparameters):
        self.__started = True
        self.__params = params
        
        dirname = os.path.dirname(__file__)

        # Replacement removes invalid token in filename.
        now = str(datetime.now().isoformat(sep=' ', timespec='seconds')).replace(':', '-')

        directory = f'{dirname}/../../reports/{now}'

        self.__statistics_filename = f'{directory}/{now} Statistics Report.csv'
        self.__payloads_filename = f'{directory}/{now} Successful Payloads Report.csv'
        
        os.makedirs(os.path.dirname(self.__statistics_filename), exist_ok=True)

        constant_stddev = params.constant_stddev

        column_names = [
            'γ',
            'τ',
            'Actor η',
            'Critic η',
            'Embedding Size',
            'Buffer Size',
            'Batch Size',
            'Standard Deviation' if constant_stddev else 'Starting Standard Deviation',
            'Constant Standard Deviation',
            'Alpha Scalar',
            'Starting ε',
            'ε Decay',
            'Min ε',
            'ψ',
            'Temperature',
            'n-Step Rollout',
            'n-Step Rollout Loss Weight',
            'L2 Regularisation Weight',
            'Priority Calculation Actor Loss Weight',
            'Action Size',
            'State Size',
            'Prefix',
            'Suffix',
        ]

        if constant_stddev:
            column_names.remove('Alpha Scalar')
            column_names.remove('Starting ε')
            column_names.remove('ε Decay')
            column_names.remove('Min ε')
        
        with open(self.__statistics_filename, 'w', encoding='utf-8') as f:
            f.write(f'Report started at {now}\n')
            f.write('\n')
            f.write('Constants\n')
            f.write(','.join(column_names) + '\n')
            f.write(f'{params.gamma},')
            f.write(f'{params.tau},')
            f.write(f'{params.actor_learning_rate},')
            f.write(f'{params.critic_learning_rate},')
            f.write(f'{params.embedding_size},')
            f.write(f'{params.buffer_size},')
            f.write(f'{params.batch_size},')
            f.write(f'{params.starting_stddev},')
            f.write(f'{params.constant_stddev},')

            if not constant_stddev:
                f.write(f'{params.alpha_scalar},')
                f.write(f'{params.epsilon_start},')
                f.write(f'{params.epsilon_decay},')
                f.write(f'{params.epsilon_min},')

            f.write(f'{params.psi},')
            f.write(f'{params.temperature},')
            f.write(f'{params.n_step_rollout},')
            f.write(f'{params.rollout_weight},')
            f.write(f'{params.l2_weight},')
            f.write(f'{params.priority_weight},')
            f.write(f'{params.action_size},')
            f.write(f'{params.state_size},')
            f.write(f'{params.prefix},')
            f.write(f'{params.suffix}\n\n')

            column_names = [
                'Time',
                'Episode',
                'Frame',
                'Is Demonstration',
                'Critic Loss',
                'Actor Loss',
                'Standard Deviation',
                'ε',
                'Average n-Step KL Divergence',
                'Distance Threshold',
                'Average n-Step Reward',
            ]

            if constant_stddev:
                column_names.remove('Standard Deviation')
                column_names.remove('ε')
                column_names.remove('Distance Threshold')

            f.write(','.join(column_names) + '\n')
        f.close()

        with open(self.__payloads_filename, 'w', encoding='utf-8') as f:
            f.write(f'Report started at {now}\n\n')
            f.write('Time,Epsiode,Frame,Is Demonstration,Reward,Successful Payload\n')
        f.close()

    def record_running_statistic(self, stat: DDPGRunningStatistic):
        if not self.__started:
            raise Exception('Must call start() before recording a statistic.')
        
        constant_stddev = self.__params.constant_stddev
        
        with open(self.__statistics_filename, 'a', encoding='utf-8') as f:
            f.write(f'{datetime.now()},')
            f.write(f'{stat.epsiode},')
            f.write(f'{stat.frame},')
            f.write(f'{stat.is_demonstration},')
            f.write(f'{stat.critic_loss},')
            f.write(f'{stat.actor_loss},')

            if not constant_stddev:
                f.write(f'{stat.stddev},')
                f.write(f'{stat.epsilon},')

            f.write(f'{stat.avg_n_step_kl_divergence},')

            if not constant_stddev:
                f.write(f'{stat.distance_threshold},')
                
            f.write(f'{stat.avg_n_step_reward}\n')
        f.close()

    def record_payload_statistic(self, stat: DDPGPayloadStatistic):
        if not self.__started:
            raise Exception('Must call start() before recording a statistic.')
        
        with open(self.__payloads_filename, 'a', encoding='utf-8') as f:
            f.write(f'{datetime.now()},')
            f.write(f'{stat.epsiode},')
            f.write(f'{stat.frame},')
            f.write(f'{stat.is_demonstration},')
            f.write(f'{stat.reward},')
            f.write(f'{stat.payload}\n')
        f.close()
        
