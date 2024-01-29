class DDPGHyperparameters:

    gamma: float

    # Parameter for updating targets.
    tau: float

    actor_learning_rate: float
    critic_learning_rate: float
    embedding_size: int
    buffer_size: int
    batch_size: int
    epsilon_start: float
    epsilon_decay: float
    epsilon_min: float
    psi: float
    action_size: int
    state_size: int
    prefix: str
    suffix: str

    def __init__(
            self,
            gamma: float,
            tau: float,
            actor_learning_rate: float,
            critic_learning_rate: float,
            embedding_size: int,
            batch_size: int,
            epsilon_start: float,
            epsilon_decay: float,
            epsilon_min: float,
            psi: float,
            action_size: int,
            state_size: int,
            prefix: str,
            suffix: str,
        ) -> None:
        self.gamma = gamma
        self.tau = tau
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.psi = psi
        self.action_size = action_size
        self.state_size = state_size
        self.prefix = prefix
        self.suffix = suffix