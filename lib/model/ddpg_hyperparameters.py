class DDPGHyperparameters:

    gamma: float

    # Parameter for updating targets.
    tau: float

    actor_learning_rate: float
    critic_learning_rate: float
    embedding_size: int
    buffer_size: int
    batch_size: int
    starting_stddev: float
    
    # NOTE: Deprecated parameter until application in reward function.
    psi: float

    temperature: float
    n_step_rollout: int
    learnings_per_batch: int
    priority_weight: float
    rollout_weight: float
    l2_weight: float
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
            starting_stddev: float,
            psi: float,
            temperature: float,
            n_step_rollout: int,
            learnings_per_batch: int,
            priority_weight: float,
            rollout_weight: float,
            l2_weight: float,
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
        self.starting_stddev = starting_stddev
        self.psi = psi
        self.temperature = temperature
        self.n_step_rollout = n_step_rollout
        self.learnings_per_batch = learnings_per_batch
        self.priority_weight = priority_weight
        self.rollout_weight =rollout_weight
        self.l2_weight = l2_weight
        self.action_size = action_size
        self.state_size = state_size
        self.prefix = prefix
        self.suffix = suffix