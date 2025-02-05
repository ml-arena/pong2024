import numpy as np
from pettingzoo.atari import pong_v3
from typing import Dict, List, Tuple, Type
import time
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class Agent:
    """Base Agent class for Pong competition."""
    def __init__(self, env, player_name = None):
        self.env = env
        self.player_name = player_name
        
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        """Choose an action based on the current game state."""
        return self.env.action_space(self.player_name).sample()