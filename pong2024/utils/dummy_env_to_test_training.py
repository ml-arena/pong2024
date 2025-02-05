import numpy as np
from gymnasium.spaces import Discrete, Dict, Box
import gymnasium
from pettingzoo import AECEnv
from collections import defaultdict

class SimpleGame(AECEnv):
    """
    A pattern prediction game where:
    - Observation is (last_last_action, last_action)
    - Get reward +1 if:
        * obs = (0,0) and action = 1
        * obs = (1,1) and action = 0
    - Get reward -1 if:
        * obs = (0,0) and action = 0
        * obs = (1,1) and action = 1
    - No reward if obs = (0,1) or (1,0)
    """
    metadata = {"render_modes": ["human"], "name": "pattern_game"}

    def __init__(self):
        super().__init__()
        self.possible_agents = ["player_0", "player_1"]

    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(2)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.steps = 0
        
        # Initialize game state
        self.agent_selection = self.agents[0]
        self.actions = {agent: None for agent in self.agents}
        
        # Initialize observations with [0,0]
        self.observations = {agent: np.zeros(2, dtype=np.float32) for agent in self.agents}
        
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        
        return self.observe(self.agent_selection)

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        
        # Calculate reward based on the pattern rule
        obs = self.observations[agent]
        if obs[0] == obs[1]:  # If both observations are the same
            if obs[0] == 0:  # If observations are (0,0)
                reward = 1 if action == 1 else -1
            else:  # If observations are (1,1)
                reward = 1 if action == 0 else -1
        else:
            reward = 0  # No reward for mixed observations (0,1) or (1,0)
        
        self.rewards[agent] = reward
        
        # Update observation: shift last action and add new action
        self.observations[agent][0] = self.observations[agent][1]
        self.observations[agent][1] = action
        
        # Update step count
        if agent == self.agents[-1]:
            self.steps += 1
            
            # Check termination
            if self.steps >= 100:
                self.terminations = {agent: True for agent in self.agents}
            
        # Update agent selection
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._next_agent()
        self._accumulate_rewards()
        
        return self.observe(self.agent_selection)

    def observe(self, agent):
        """Return observation for the specified agent"""
        return self.observations[agent]

    def render(self):
        pass

    def _next_agent(self):
        current_idx = self.agents.index(self.agent_selection)
        next_idx = (current_idx + 1) % len(self.agents)
        return self.agents[next_idx]

class QLearningAgent:
    """Simple Q-Learning Agent"""
    def __init__(self, env, player_name=None, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.player_name = player_name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.last_state = None
        self.last_action = None
        self.last_reward = None  # Add tracking of last reward
        
    def state_to_key(self, observation):
        """Convert observation array to hashable tuple"""
        return tuple(observation.astype(np.int8))  # Convert to int for stability
        
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        state = self.state_to_key(observation)
        
        # Store reward for learning step
        self.last_reward = reward
        
        # Only learn when we have all components of a transition
        if self.last_state is not None and self.last_action is not None and not terminated:
            old_value = self.q_table[self.last_state][self.last_action]
            next_max = np.max(self.q_table[state])
            
            new_value = old_value + self.learning_rate * (
                reward + self.discount_factor * next_max - old_value
            )
            
            self.q_table[self.last_state][self.last_action] = new_value
        
        # Choose action
        if np.random.random() < self.epsilon:
            action = self.env.action_space(self.player_name).sample()
        else:
            q_values = self.q_table[state]
            # Handle equal values randomly
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            action = np.random.choice(max_actions)
        
        self.last_state = state
        self.last_action = action
        
        return action
    
    def learn(self):
        """Learning is handled in choose_action"""
        pass

class RandomAgent:
    """Random action agent for testing"""
    def __init__(self, env, player_name=None):
        self.env = env
        self.player_name = player_name
    
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        return self.env.action_space(self.player_name).sample()
    
    def learn(self):
        pass

# Example usage:
def make_env():
    return SimpleGame()

class heuristicAgent:
    """
    Perfect agent for the pattern prediction game.
    Strategy:
    - If obs is (0,0): play 1 for reward +1
    - If obs is (1,1): play 0 for reward +1
    - If obs is (0,1) or (1,0): play anything (no reward)
    """
    def __init__(self, env, player_name=None):
        self.env = env
        self.player_name = player_name
    
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        # If both observations are the same
        if observation[0] == observation[1]:
            if observation[0] == 0:
                return 1  # For (0,0), play 1
            else:
                return 0  # For (1,1), play 0
        # For mixed observations, play randomly
        return np.random.randint(2)
    
    def learn(self):
        pass


env = make_env()
opponent_agents = [RandomAgent]
opponent_probs = [1] 

results = train_sequential(
    make_env=make_env,
    main_agent_class=QLearningAgent,
    opponent_classes=opponent_agents,
    opponent_probs=opponent_probs,
    n_total_episodes=100,
    eval_frequency=10,
    max_cycles=5000 
)
