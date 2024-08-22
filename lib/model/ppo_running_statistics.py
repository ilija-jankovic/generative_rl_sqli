class PPORunningStatistics:

    timestep: int
    mean_batch_reward: float
    mean_actor_loss: float
    mean_critic_loss: float
    exploration_seconds: float
    learning_seconds: float
    training_step_seconds: float

    def __init__(
        self,
        timestep: int,
        mean_batch_reward: float,
        mean_actor_loss: float,
        mean_critic_loss: float,
        exploration_seconds: float,
        learning_seconds: float,
        training_step_seconds: float,
    ):
        self.timestep = timestep
        self.mean_batch_reward = mean_batch_reward
        self.mean_actor_loss = mean_actor_loss
        self.mean_critic_loss = mean_critic_loss
        self.exploration_seconds = exploration_seconds
        self.learning_seconds = learning_seconds
        self.training_step_seconds = training_step_seconds
