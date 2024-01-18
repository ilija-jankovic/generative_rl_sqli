from datetime import datetime
import os

from .ddpg_running_statistic import DDPGRunningStatistic
from .ddpg_hyperparameters import DDPGHyperparameters
from .ddpg_payload_statistic import DDPGPayloadStatistic


class Reporter:

    __started: bool = False
    __statistics_filename: str
    __payloads_filename: str

    def start(self, params: DDPGHyperparameters):
        self.__started = True
        
        dirname = os.path.dirname(__file__)

        # Replacement removes invalid token in filename.
        now = str(datetime.now().isoformat(sep=' ', timespec='seconds')).replace(':', '-')

        directory = f'{dirname}/../../reports/{now}'

        self.__statistics_filename = f'{directory}/{now} Statistics Report.csv'
        self.__payloads_filename = f'{directory}/{now} Successful Payloads Report.csv'
        
        os.makedirs(os.path.dirname(self.__statistics_filename), exist_ok=True)
        
        with open(self.__statistics_filename, 'w', encoding='utf-8') as f:
            f.write(f'Report started at {now}\n')
            f.write('\n')
            f.write('Constants\n')
            f.write('γ,τ,Actor η,Critic η,Embedding Size,Buffer Size,Batch Size,Starting ε,ε Decay,Min ε,ψ,Action Size,State Size,Prefix,Suffix\n')
            f.write(f'{params.gamma},')
            f.write(f'{params.tau},')
            f.write(f'{params.actor_learning_rate},')
            f.write(f'{params.critic_learning_rate},')
            f.write(f'{params.embedding_size},')
            f.write(f'{params.buffer_size},')
            f.write(f'{params.batch_size},')
            f.write(f'{params.epsilon_start},')
            f.write(f'{params.epsilon_decay},')
            f.write(f'{params.epsilon_min},')
            f.write(f'{params.psi},')
            f.write(f'{params.action_size},')
            f.write(f'{params.state_size},')
            f.write(f'{params.prefix},')
            f.write(f'{params.suffix}\n\n')
            f.write('Time,Episode,Frame,Is Demonstration,ε,Total Average Reward\n')
        f.close()

        with open(self.__payloads_filename, 'w', encoding='utf-8') as f:
            f.write(f'Report started at {now}\n\n')
            f.write('Time,Epsiode,Frame,Is Demonstration,Reward,Successful Payload\n')
        f.close()

    def record_running_statistic(self, stat: DDPGRunningStatistic):
        if not self.__started:
            raise Exception('Must call start() before recording a statistic.')
        
        with open(self.__statistics_filename, 'a', encoding='utf-8') as f:
            f.write(f'{datetime.now()},')
            f.write(f'{stat.epsiode},')
            f.write(f'{stat.frame},')
            f.write(f'{stat.is_demonstration},')
            f.write(f'{stat.epsilon},')
            f.write(f'{stat.total_avg_reward}\n')
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
        
