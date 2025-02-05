import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

def visualize_evaluation_results(results: dict):
    """
    Create visualizations for the evaluation results.
    """
    games_df = pd.DataFrame(results['games'])
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Win Rates Visualization
    win_rates = results['summary']['win_rates']
    win_rate_df = pd.DataFrame({
        'Agent': ['Agent 1', 'Agent 2'],
        'Win Rate': [win_rates['agent1'], win_rates['agent2']]
    })
    
    sns.barplot(data=win_rate_df, x='Agent', y='Win Rate', ax=ax1)
    ax1.set_title('Win Rates by Agent')
    ax1.set_ylabel('Win Rate')
    
    # Add percentage labels on bars
    for i, v in enumerate(win_rate_df['Win Rate']):
        ax1.text(i, v, f'{v:.1%}', ha='center', va='bottom')
    
    # 2. Score Distribution by Role
    role_data = []
    for _, game in games_df.iterrows():
        role_data.extend([
            {
                'Agent': 'Agent 1',
                'Role': game['agent1_role'],
                'Score': game['agent1_score']
            },
            {
                'Agent': 'Agent 2',
                'Role': game['agent2_role'],
                'Score': game['agent2_score']
            }
        ])
    
    role_df = pd.DataFrame(role_data)
    sns.boxplot(data=role_df, x='Agent', y='Score', hue='Role', ax=ax2)
    ax2.set_title('Score Distribution by Agent and Role')
    ax2.set_ylabel('Score')
    
    plt.tight_layout()
    plt.show()
    
    # Create additional plot for score differences over time
    plt.figure(figsize=(12, 6))
    
    # Calculate score differences
    games_df['score_diff'] = games_df['agent1_score'] - games_df['agent2_score']
    
    plt.plot(games_df['game_number'], games_df['score_diff'], 
             marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Add role annotations
    for _, game in games_df.iterrows():
        plt.annotate(f"A1:{game['agent1_role']}", 
                    (game['game_number'], game['score_diff']),
                    xytext=(0, 10 if game['score_diff'] >= 0 else -10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    alpha=0.7)
    
    plt.title('Score Difference Over Games (Agent 1 - Agent 2)')
    plt.xlabel('Game Number')
    plt.ylabel('Score Difference')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_agents(
    env,
    agent1_class: Type[Agent],
    agent2_class: Type[Agent],
    n_games: int = 10,
    max_cycles: int = 1000,
    seed: int = None
) -> Dict:
    """
    Evaluate two agent classes playing against each other in the Pong environment using AEC pattern.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Initialize agents
    agent1 = agent1_class(env)
    agent2 = agent2_class(env)
    
    # Initialize game history
    game_history = []
    
    for game in range(n_games):
        env.reset()
        
        # Randomly assign agents to player roles for this game
        possible_players = list(env.possible_agents)
        random.shuffle(possible_players)
        
        # Assign the shuffled roles
        agent1.player_name = possible_players[0]
        agent2.player_name = possible_players[1]
        
        # Map environment agents to our agent instances for this game
        agent_mapping = {
            agent1.player_name: (agent1, "agent1"),
            agent2.player_name: (agent2, "agent2")
        }
        
        step_count = 0
        game_rewards = {"agent1": 0, "agent2": 0}
        game_active = True
        
        # Game loop following AEC pattern
        while game_active and step_count < max_cycles:
            for agent_id in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                agent, agent_name = agent_mapping[agent_id]
                game_rewards[agent_name] += reward
                
                if termination or truncation:
                    action = None
                    game_active = False
                else:
                    action = agent.choose_action(
                        observation, reward, termination, truncation, info
                    )
                
                env.step(action)
                step_count += 1
                
                if not game_active:
                    break
            
            if not game_active:
                break
        
        # Record game results
        game_result = {
            "game_number": game + 1,
            "agent1_role": agent1.player_name,
            "agent2_role": agent2.player_name,
            "agent1_score": game_rewards["agent1"],
            "agent2_score": game_rewards["agent2"],
            "steps": step_count,
            "winner": "agent1" if game_rewards["agent1"] > game_rewards["agent2"] else 
                     "agent2" if game_rewards["agent2"] > game_rewards["agent1"] else "tie"
        }
        game_history.append(game_result)
    
    env.close()
    
    # Convert game history to DataFrame for easier analysis
    games_df = pd.DataFrame(game_history)
    
    # Calculate summary statistics
    results = {
        "games": games_df.to_dict('records'),
        "summary": {
            "n_games": n_games,
            "win_rates": {
                "agent1": (games_df['winner'] == 'agent1').mean(),
                "agent2": (games_df['winner'] == 'agent2').mean()
            },
            "average_scores": {
                "agent1": games_df['agent1_score'].mean(),
                "agent2": games_df['agent2_score'].mean()
            },
            "total_scores": {
                "agent1": games_df['agent1_score'].sum(),
                "agent2": games_df['agent2_score'].sum()
            },
            "average_game_length": games_df['steps'].mean()
        }
    }
    
    return results


if __name__ == "__main__":

    env = pong_v3.env()

    # Now we pass the agent classes instead of instances
    results = evaluate_agents(
        env,
        agent1_class=Agent,
        agent2_class=Agent,
        n_games=100,
        max_cycles=10000
    )

    # Print results
    print("\nEvaluation Summary:")
    print(f"Games played: {results['summary']['n_games']}")
    print(f"Agent 1 Win Rate: {results['summary']['win_rates']['agent1']:.1%}")
    print(f"Agent 2 Win Rate: {results['summary']['win_rates']['agent2']:.1%}")
    print(f"Average Game Length: {results['summary']['average_game_length']:.1f} steps")
        
    visualize_evaluation_results(results)