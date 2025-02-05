import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
import random
import time
from typing import Type, List, Dict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import numpy as np
from pettingzoo.atari import pong_v3
from typing import Dict, List, Tuple, Type
import time
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pong2024.eval import evaluate_agents

class Agent:
    """Base Agent class for Pong competition."""
    def __init__(self, env, player_name = None):
        self.env = env
        self.player_name = player_name
        
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        """Choose an action based on the current game state."""
        return self.env.action_space(self.player_name).sample()


import numpy as np
import random
import time
from typing import Type, List, Dict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

def train_parallel_environments(
    make_env,
    main_agent_class: Type[Agent],
    opponent_classes: List[Type[Agent]],
    n_envs: int = 16,
    n_total_episodes: int = 10000,
    eval_frequency: int = 100,
    max_cycles: int = 3000,
    opponent_probs: List[float] = None,
    seed: int = None
) -> Dict:
    """
    Train one main agent against multiple opponent agents simultaneously in parallel environments.
    
    Args:
        make_env: Function that creates a new environment instance
        main_agent_class: The primary agent class to train
        opponent_classes: List of opponent agent classes to train against
        n_envs: Number of parallel environments
        n_total_episodes: Total number of training episodes across all opponents
        eval_frequency: Number of episodes between metric computations
        max_cycles: Maximum number of cycles per episode
        opponent_probs: List of probabilities for selecting each opponent. Must sum to 1.
                       If None, uniform distribution is used.
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing training results and metrics
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Initialize opponent probabilities if not provided
    if opponent_probs is None:
        opponent_probs = [1.0 / len(opponent_classes)] * len(opponent_classes)
    assert len(opponent_probs) == len(opponent_classes), "Must provide probability for each opponent"
    assert abs(sum(opponent_probs) - 1.0) < 1e-6, "Probabilities must sum to 1"
    
    training_results = {
        'opponents': [{
            'opponent_id': idx,
            'opponent_class': opponent_class.__name__,
            'episodes': [],
            'metrics_history': [],
            'win_rate_history': [],
            'draw_rate_history': [],  # Track draw rates
            'lose_rate_history': []   # Track lose rates
        } for idx, opponent_class in enumerate(opponent_classes, 1)],
        'summary': {
            'total_episodes': 0,
            'total_training_time': 0
        }
    }
    
    # Create environments and agents
    envs = [make_env() for _ in range(n_envs)]
    main_agents = [main_agent_class(env) for env in envs]
    opponent_instances = {
        opponent_class.__name__: [opponent_class(env) for env in envs]
        for opponent_class in opponent_classes
    }
    
    start_time = time.time()
    metrics_window = []
    current_window_start = 0
    
    def run_environment(
        env_idx: int,
        opponent_class: Type[Agent],
        episode: int
    ) -> Dict:
        """Run a single episode in one environment"""
        env = envs[env_idx]
        main_agent = main_agents[env_idx]
        opponent = opponent_instances[opponent_class.__name__][env_idx]
        
        env.reset()
        
        # Randomly assign roles
        possible_players = list(env.possible_agents)
        random.shuffle(possible_players)
        main_agent.player_name = possible_players[0]
        opponent.player_name = possible_players[1]
        
        agent_mapping = {
            main_agent.player_name: (main_agent, "main_agent"),
            opponent.player_name: (opponent, "opponent")
        }
        
        episode_rewards = {"main_agent": 0, "opponent": 0}
        step_count = 0
        episode_active = True
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            agent, agent_name = agent_mapping[agent_id]
            episode_rewards[agent_name] += reward
            
            if termination or truncation:
                action = None
                episode_active = False
            else:
                action = agent.choose_action(
                    observation, reward, termination, truncation, info
                )
                if agent_name == "main_agent":
                    agent.learn()
            
            env.step(action)
            step_count += 1
            
            if not episode_active or step_count >= max_cycles:
                break
        
        # Determine game outcome (win, loss, or draw)
        main_score = episode_rewards["main_agent"]
        opponent_score = episode_rewards["opponent"]
        is_draw = main_score == opponent_score
        is_win = main_score > opponent_score
        is_lose = main_score < opponent_score
        
        return {
            "episode": episode + 1,
            "opponent_class": opponent_class.__name__,
            "main_agent_role": main_agent.player_name,
            "main_agent_score": main_score,
            "opponent_score": opponent_score,
            "steps": step_count,
            "win": is_win,
            "draw": is_draw,
            "lose": is_lose
        }
    
    def select_opponent() -> Type[Agent]:
        """Select opponent based on provided probabilities"""
        return np.random.choice(opponent_classes, p=opponent_probs)
    
    n_batches = (n_total_episodes + n_envs - 1) // n_envs
    
    with ThreadPoolExecutor(max_workers=n_envs) as executor:
        for batch in range(n_batches):
            batch_start = batch * n_envs
            batch_size = min(n_envs, n_total_episodes - batch_start)
            
            # Select different opponents for each environment in the batch
            batch_opponents = [select_opponent() for _ in range(batch_size)]
            
            # Submit batch of episodes to executor
            futures = [
                executor.submit(
                    run_environment,
                    env_idx,
                    opponent_class,
                    batch_start + env_idx
                )
                for env_idx, opponent_class in enumerate(batch_opponents)
            ]
            
            # Collect results
            batch_results = []
            for future in futures:
                episode_data = future.result()
                opponent_idx = next(
                    i for i, res in enumerate(training_results['opponents'])
                    if res['opponent_class'] == episode_data['opponent_class']
                )
                training_results['opponents'][opponent_idx]['episodes'].append(episode_data)
                batch_results.append(episode_data)
            
            metrics_window.extend(batch_results)
            
            # Compute and display metrics periodically
            current_episode = batch_start + batch_size
            if current_episode % eval_frequency == 0 or current_episode == n_total_episodes:
                # Get episodes for the current evaluation window
                window_episodes = metrics_window[current_window_start:]
                current_window_start = len(metrics_window)
                
                if window_episodes:
                    print(f"\nEpisode {current_episode}/{n_total_episodes}")
                    print("=" * 50)
                    
                    total_games = len(window_episodes)
                    total_wins = 0
                    total_draws = 0
                    
                    print("\nPer-Opponent Statistics:")
                    print("-" * 25)
                    
                    for opponent_results in training_results['opponents']:
                        recent_episodes = [
                            ep for ep in window_episodes
                            if ep['opponent_class'] == opponent_results['opponent_class']
                        ]
                        
                        if recent_episodes:
                            wins = sum(ep['win'] for ep in recent_episodes)
                            draws = sum(ep['draw'] for ep in recent_episodes)
                            losses = sum(ep['lose'] for ep in recent_episodes)
                            total_wins += wins
                            total_draws += draws
                            total_losses = sum(ep['lose'] for ep in recent_episodes)
                            
                            metrics = {
                                "episode": current_episode,
                                "games": len(recent_episodes),
                                "win_rate": wins / len(recent_episodes),
                                "draw_rate": draws / len(recent_episodes),
                                "lose_rate": losses / len(recent_episodes),
                                "avg_score": sum(ep['main_agent_score'] for ep in recent_episodes) / len(recent_episodes),
                                "avg_opponent_score": sum(ep['opponent_score'] for ep in recent_episodes) / len(recent_episodes),
                                "avg_steps": sum(ep['steps'] for ep in recent_episodes) / len(recent_episodes)
                            }
                            opponent_results['metrics_history'].append(metrics)
                            opponent_results['win_rate_history'].append(metrics['win_rate'])
                            opponent_results['draw_rate_history'].append(metrics['draw_rate'])
                            opponent_results['lose_rate_history'].append(metrics['lose_rate'])
                            
                            print(f"\n{opponent_results['opponent_class']}:")
                            print(f"Games Played: {metrics['games']} ({metrics['games']/total_games:.1%} of total)")
                            print(f"Win Rate: {metrics['win_rate']:.1%}")
                            print(f"Draw Rate: {metrics['draw_rate']:.1%}")
                            print(f"Lose Rate: {metrics['lose_rate']:.1%}")
                            print(f"Average Score: {metrics['avg_score']:.1f} vs {metrics['avg_opponent_score']:.1f}")
                    
                    print("\nOverall Statistics:")
                    print("-" * 20)
                    print(f"Total Games: {total_games}")
                    print(f"Overall Win Rate: {total_wins/total_games:.1%}")
                    print(f"Overall Draw Rate: {total_draws/total_games:.1%}")
                    print(f"Overall Lose Rate: {total_losses/total_games:.1%}")
    
    training_results['summary'].update({
        'total_episodes': n_total_episodes,
        'total_training_time': time.time() - start_time,
        'final_metrics': {
            opponent_results['opponent_class']: opponent_results['metrics_history'][-1]
            if opponent_results['metrics_history'] else None
            for opponent_results in training_results['opponents']
        }
    })
    
    # Print final summary
    print("\nTraining Complete!")
    print("=" * 50)
    print(f"Total Training Time: {training_results['summary']['total_training_time']:.1f} seconds")
    
    return training_results


def visualize_training_results(results: dict):
    """
    Create visualizations for the training results using metrics computed directly from training episodes.
    
    Args:
        results: Dictionary containing the training results from train_against_multiple_agents
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Learning Curves (Win Rates)
    ax1 = fig.add_subplot(gs[0, 0])
    
    for opponent_results in results['opponents']:
        metrics_data = pd.DataFrame(opponent_results['metrics_history'])
        ax1.plot(metrics_data['episode'], metrics_data['win_rate'], 
                marker='o', label=f"vs {opponent_results['opponent_class']}")
    
    ax1.set_title('Learning Progress - Win Rates')
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Win Rate')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Average Scores Over Time
    ax2 = fig.add_subplot(gs[0, 1])
    
    for opponent_results in results['opponents']:
        metrics_data = pd.DataFrame(opponent_results['metrics_history'])
        ax2.plot(metrics_data['episode'], metrics_data['avg_score'], 
                marker='o', label=f"vs {opponent_results['opponent_class']}")
    
    ax2.set_title('Learning Progress - Average Scores')
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Average Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Average Steps Over Time
    ax3 = fig.add_subplot(gs[1, 0])
    
    for opponent_results in results['opponents']:
        metrics_data = pd.DataFrame(opponent_results['metrics_history'])
        ax3.plot(metrics_data['episode'], metrics_data['avg_steps'], 
                marker='o', label=f"vs {opponent_results['opponent_class']}")
    
    ax3.set_title('Learning Progress - Average Steps per Episode')
    ax3.set_xlabel('Training Episodes')
    ax3.set_ylabel('Average Steps')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Final Performance Summary
    ax4 = fig.add_subplot(gs[1, 1])
    
    final_metrics = pd.DataFrame(results['summary']['final_metrics']).T
    final_metrics = final_metrics.reset_index()
    final_metrics.columns = ['Opponent'] + list(final_metrics.columns[1:])
    
    x = np.arange(len(final_metrics))
    width = 0.25
    
    # Plot three metrics side by side
    ax4.bar(x - width, final_metrics['win_rate'], width, label='Win Rate')
    ax4.bar(x, final_metrics['avg_score'] / final_metrics['avg_score'].max(), 
            width, label='Norm. Avg Score')
    ax4.bar(x + width, final_metrics['avg_steps'] / final_metrics['avg_steps'].max(), 
            width, label='Norm. Avg Steps')
    
    ax4.set_title('Final Performance Summary')
    ax4.set_xticks(x)
    ax4.set_xticklabels(final_metrics['Opponent'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Score distributions
    plt.figure(figsize=(12, 6))
    
    for opponent_results in results['opponents']:
        episodes_df = pd.DataFrame(opponent_results['episodes'])
        sns.kdeplot(data=episodes_df['main_agent_score'], 
                   label=f"vs {opponent_results['opponent_class']}")
    
    plt.title('Distribution of Main Agent Scores During Training')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    class YourAgent:
        """Base Agent class for Pong competition."""
        def __init__(self, env, player_name = None):
            self.env = env
            self.player_name = player_name
            
        def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
            """Choose an action based on the current game state."""
            return self.env.action_space(self.player_name).sample()
        def learn(self):
            pass
    class AlwaysLeftAgent:
        """Base Agent class for Pong competition."""
        def __init__(self, env, player_name = None):
            self.env = env
            self.player_name = player_name
            
        def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
            """Choose an action based on the current game state."""
            return 1
    class AlwaysRightAgent:
        """Base Agent class for Pong competition."""
        def __init__(self, env, player_name = None):
            self.env = env
            self.player_name = player_name
            
        def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
            """Choose an action based on the current game state."""
            return 1

    env = pong_v3.env()
    opponent_agents = [AlwaysLeftAgent, AlwaysRightAgent]

    # Train the agent
    results = train_against_multiple_agents(
        env=env,
        main_agent_class=YourAgent,
        opponent_classes=opponent_agents,
        n_episodes_per_opponent=100,
        eval_frequency=10
    )

    # Visualize the results
    visualize_training_results(results)