import math


class EpisodeState:
    initial_frames: int
    max_episode_extension: int

    __frames: int
    __frames_since_last_episode: int
    
    __episode: int

    @property
    def frames_since_last_episode(self):
        return self.__frames_since_last_episode

    @property
    def episode(self):
        return self.__episode

    def __init__(
        self,
        initial_frames: int,
        max_episode_extension: int,
    ):
        self.initial_frames = initial_frames
        self.max_episode_extension = max_episode_extension
        
        self.__frames_since_last_episode = 0
        self.__episode = 0
        
        self.next_episode()

    def next_episode(self):
        self.__frames = self.initial_frames
        self.__frames_since_last_episode = 0
        
        self.__episode += 1

    def next_frame(self):
        self.__frames_since_last_episode += 1

    def has_episode_ended(self):
        return self.__frames_since_last_episode >= self.__frames
    
    def extend_episode(self, proportion: float):
        '''
        Extends the episode by the ceiling of
        `initial_frames * proportion`.
        '''

        assert(proportion > 0.0)
        assert(proportion <= 1.0)

        self.__frames += math.ceil(self.max_episode_extension * proportion)
        