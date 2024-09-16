from typing import List


class RunningEpisodicStatistics:
    
    __episode: int
    __running_rewards: List[float | None]
    __running_successful_actions: List[int]
    __running_unsuccessful_actions: List[int]
    
    
    @property
    def episode(self):
        return self.__episode
    
    
    @property
    def is_populated(self):
        return None not in self.__running_rewards
    

    @property
    def mean_running_reward(self) -> float:
        assert(self.is_populated)

        return sum(self.__running_rewards) / len(self.__running_rewards)
    
    
    @property
    def mean_accuracy(self):
        assert(self.is_populated)
        
        batch_size = len(self.__running_successful_actions)
        
        accuracies = [
            self.__running_successful_actions[i] / (
                self.__running_successful_actions[i] + 
                self.__running_unsuccessful_actions[i]
            ) for i in range(batch_size)
        ]
        
        return sum(accuracies) / len(accuracies)


    def __init__(
        self,
        episode: int,
        batch_size: int,
    ):
        self.__episode = episode
        self.__running_rewards = [None] * batch_size
        self.__running_successful_actions = [0] * batch_size
        self.__running_unsuccessful_actions = [0] * batch_size


    def record_reward(self, reward: float, batch_index: int):
        if self.__running_rewards[batch_index] == None:
            self.__running_rewards[batch_index] = reward
        else:
            self.__running_rewards[batch_index] += reward
        
        if reward > 0.0:
            self.__running_successful_actions[batch_index] += 1
        else:
            self.__running_unsuccessful_actions[batch_index] += 1
