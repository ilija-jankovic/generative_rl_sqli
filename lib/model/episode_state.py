class EpisodeState:
    batches_in_initial_episode_frames: int
    batch_size: int

    __initial_episode_end_frames: int
    __frames: int
    __frames_since_last_episode: int = 0

    def __init__(self, batch_size: int, batches_in_initial_episode_frames: int):
        self.batch_size = batch_size
        self.__initial_episode_end_frames = batches_in_initial_episode_frames * batch_size
        
        self.next_episode()

    def next_episode(self):
        self.__frames = self.__initial_episode_end_frames
        self.__frames_since_last_episode = 0

    def next_frame(self):
        self.__frames_since_last_episode += self.batch_size

    def has_episode_ended(self):
        return self.__frames_since_last_episode >= self.__frames
    
    def extend_episode(self):
        self.__frames += self.__initial_episode_end_frames + self.__frames_since_last_episode - self.__frames 
