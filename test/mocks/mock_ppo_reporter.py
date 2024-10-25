from typing import List
from lib.model.ppo_reporter import PPOReporter
from lib.model.total_episodic_statistics import TotalEpisodicStatistics


class MockPPOReporter(PPOReporter):
    
    stats: List[TotalEpisodicStatistics]


    def __init__(self):
        self.stats = []


    def record_episodic_statistics(self, stats: TotalEpisodicStatistics):
        self.stats.append(stats)