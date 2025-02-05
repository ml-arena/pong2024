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

from pong2024.eval import evaluate_agents

class Agent:
    """Base Agent class for Pong competition."""
    def __init__(self, env, player_name = None):
        self.env = env
        self.player_name = player_name
        
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        """Choose an action based on the current game state."""
        return self.env.action_space(self.player_name).sample()

def evaluate_against_multiple_agents(
    env,
    main_agent_class: Type[Agent],
    opponent_classes: List[Type[Agent]],
    n_games_per_matchup: int = 10,
    max_cycles: int = 1000,
    seed: int = None
) -> Dict:
    """
    Evaluate one main agent against multiple opponent agents.
    
    Args:
        env: The Pong environment
        main_agent_class: The primary agent class to evaluate
        opponent_classes: List of opponent agent classes to evaluate against
        n_games_per_matchup: Number of games to play per matchup
        max_cycles: Maximum number of cycles per game
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing results for all matchups
    """
    all_results = {
        'matchups': [],
        'summary': {
            'main_agent_overall_winrate': 0,
            'main_agent_average_score': 0,
            'opponent_performance': []
        }
    }
    
    total_wins = 0
    total_games = 0
    total_score = 0
    
    # Evaluate against each opponent
    for idx, opponent_class in enumerate(opponent_classes, 1):
        results = evaluate_agents(
            env=env,
            agent1_class=main_agent_class,
            agent2_class=opponent_class,
            n_games=n_games_per_matchup,
            max_cycles=max_cycles,
            seed=seed
        )
        
        # Store detailed matchup results
        matchup_summary = {
            'opponent_id': idx,
            'opponent_class': opponent_class.__name__,
            'n_games': n_games_per_matchup,
            'main_agent_winrate': results['summary']['win_rates']['agent1'],
            'main_agent_avg_score': results['summary']['average_scores']['agent1'],
            'opponent_avg_score': results['summary']['average_scores']['agent2'],
            'detailed_games': results['games']
        }
        
        all_results['matchups'].append(matchup_summary)
        
        # Update overall statistics
        total_wins += sum(1 for game in results['games'] if game['winner'] == 'agent1')
        total_games += n_games_per_matchup
        total_score += results['summary']['total_scores']['agent1']
        
        # Store opponent performance summary
        all_results['summary']['opponent_performance'].append({
            'opponent_id': idx,
            'opponent_class': opponent_class.__name__,
            'winrate_vs_main': results['summary']['win_rates']['agent2'],
            'avg_score': results['summary']['average_scores']['agent2']
        })
    
    # Calculate overall statistics
    all_results['summary']['main_agent_overall_winrate'] = total_wins / total_games
    all_results['summary']['main_agent_average_score'] = total_score / total_games
    
    return all_results

def visualize_multiple_matchups(results: dict):
    """
    Create visualizations for multiple agent matchup results.
    
    Args:
        results: Dictionary containing the evaluation results from evaluate_against_multiple_agents
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Main Agent Win Rates Against Each Opponent
    ax1 = fig.add_subplot(gs[0, 0])
    winrates = [matchup['main_agent_winrate'] for matchup in results['matchups']]
    opponent_names = [matchup['opponent_class'] for matchup in results['matchups']]
    
    sns.barplot(x=opponent_names, y=winrates, ax=ax1)
    ax1.set_title('Main Agent Win Rates vs Opponents')
    ax1.set_xlabel('Opponent')
    ax1.set_ylabel('Win Rate')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for i, v in enumerate(winrates):
        ax1.text(i, v, f'{v:.1%}', ha='center', va='bottom')
    
    # 2. Average Score Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    score_data = []
    for matchup in results['matchups']:
        score_data.extend([
            {'Opponent': matchup['opponent_class'], 'Role': 'Main Agent', 
             'Avg Score': matchup['main_agent_avg_score']},
            {'Opponent': matchup['opponent_class'], 'Role': 'Opponent', 
             'Avg Score': matchup['opponent_avg_score']}
        ])
    
    score_df = pd.DataFrame(score_data)
    sns.barplot(data=score_df, x='Opponent', y='Avg Score', hue='Role', ax=ax2)
    ax2.set_title('Average Scores by Matchup')
    ax2.set_xlabel('Opponent')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Performance Over Time
    ax3 = fig.add_subplot(gs[1, :])
    
    # Prepare data for line plot
    for matchup in results['matchups']:
        games = matchup['detailed_games']
        games_df = pd.DataFrame(games)
        
        # Calculate running win rate
        running_winrate = [
            sum(1 for g in games[:i+1] if g['winner'] == 'agent1') / (i+1)
            for i in range(len(games))
        ]
        
        ax3.plot(range(1, len(games) + 1), running_winrate, 
                marker='o', label=f'vs {matchup["opponent_class"]}')
    
    ax3.set_title('Main Agent Running Win Rate by Matchup')
    ax3.set_xlabel('Game Number')
    ax3.set_ylabel('Running Win Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add overall win rate reference line
    ax3.axhline(y=results['summary']['main_agent_overall_winrate'],
                color='r', linestyle='--', alpha=0.5,
                label=f'Overall Win Rate ({results["summary"]["main_agent_overall_winrate"]:.1%})')
    
    plt.tight_layout()
    plt.show()
    
    # Additional visualization: Performance distribution
    plt.figure(figsize=(12, 6))
    
    all_scores = []
    for matchup in results['matchups']:
        scores = [game['agent1_score'] for game in matchup['detailed_games']]
        all_scores.extend([(matchup['opponent_class'], score) for score in scores])
    
    score_dist_df = pd.DataFrame(all_scores, columns=['Opponent', 'Main Agent Score'])
    
    sns.violinplot(data=score_dist_df, x='Opponent', y='Main Agent Score')
    plt.title('Distribution of Main Agent Scores by Opponent')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    class YourMainAgent:
        """Base Agent class for Pong competition."""
        def __init__(self, env, player_name = None):
            self.env = env
            self.player_name = player_name
            
        def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
            """Choose an action based on the current game state."""
            return self.env.action_space(self.player_name).sample()
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
    opponent_agents = [AlwaysLeftAgent, AlwaysRightAgent, YourMainAgent]

    results = evaluate_against_multiple_agents(
        env=env,
        main_agent_class=YourMainAgent,
        opponent_classes=opponent_agents,
        n_games_per_matchup=50,
        max_cycles = 10000
    )
