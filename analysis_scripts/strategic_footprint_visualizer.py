#!/usr/bin/env python3
"""
Strategic Footprint Visualizer

Creates radar charts showing the strategic footprint for each LLM-temperature combination
per experiment. The strategic footprint consists of four conditional cooperation probabilities:
P(C|CC), P(C|DC), P(C|CD), and P(C|DD).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import glob
import re
from pathlib import Path
from collections import OrderedDict
import json

def get_memory_mode_from_experiment(experiment_name):
    """
    Determine memory mode from experiment name.
    """
    # Anonymous Memory Mode experiments
    anonymous_experiments = [
        "experiment_20250908_081108",  # 5%
        "experiment_20250905_152754",  # 10%
        "experiment_20250905_152805",  # 25%
        "experiment_20250905_152321"   # 75%
    ]

    # Opponent Tracking Mode experiments
    tracking_experiments = [
        "experiment_20250912_153315",  # 5%
        "experiment_20250911_112723",  # 10%
        "experiment_20250910_074608",  # 25%
        "experiment_20250910_074557"   # 75%
    ]

    if experiment_name in anonymous_experiments:
        return "Anonymous Memory"
    elif experiment_name in tracking_experiments:
        return "Tracking Memory"
    else:
        return "Unknown Memory"

def extract_llm_info(agent_name):
    """
    Extract LLM model and variant/temperature from agent name.
    OpenAI models: GPT5mini_T1, GPT5nano_T1, GPT4mini_T1 -> model family: OpenAI, variant: GPT5mini, GPT5nano, GPT4mini
    Mistral models: Mistral-Medium_T02, Ministral-Large_T02 -> model family: Mistral Medium/Large, variant: T02
    Claude models: Claude4-Sonnet_T02 -> model family: Claude4-Sonnet, variant: T02
    Gemini models: Gemini20Flash_T02 -> model family: Gemini-2.0-Flash, variant: T02
    """
    # OpenAI models pattern (GPT5mini, GPT5nano, GPT4.1mini - these are different models, not temperatures)
    openai_pattern = r'^(GPT\d+(?:\.\d+)?\w*)_T(\d+)'
    match = re.match(openai_pattern, agent_name)
    if match:
        model_variant = match.group(1)
        return "OpenAI", model_variant
    
    # Mistral models pattern (Mistral-Medium_T02, Ministral-Large_T02, etc.)
    mistral_pattern = r'^(Mistral-Medium|Ministral-Large)_T(\d+(?:\.\d+)?)'
    match = re.match(mistral_pattern, agent_name)
    if match:
        model_type = "Mistral Medium" if "Medium" in match.group(1) else "Mistral Large"
        temp_variant = f"T{match.group(2)}"
        return model_type, temp_variant
    
    # Claude models pattern
    claude_pattern = r'^Claude[\w-]*_T(\d+(?:\.\d+)?)'
    match = re.match(claude_pattern, agent_name)
    if match:
        temp_variant = f"T{match.group(1)}"
        return "Claude4-Sonnet", temp_variant
    
    # Gemini models pattern
    gemini_pattern = r'^Gemini[\w]*_T(\d+(?:\.\d+)?)'
    match = re.match(gemini_pattern, agent_name)
    if match:
        temp_variant = f"T{match.group(1)}"
        return "Gemini-2.0-Flash", temp_variant
    
    return None, None

def calculate_strategic_footprint(df, agent_name):
    """
    Calculate strategic footprint for a given agent from the tournament data.
    Returns P(C|CC), P(C|DC), P(C|CD), P(C|DD)
    """
    # Filter for games where this agent participated
    agent_games = df[(df['agent1'] == agent_name) | (df['agent2'] == agent_name)]
    
    if agent_games.empty:
        return None
    
    # Initialize counters for each condition
    conditions = {'CC': {'total': 0, 'coop': 0},
                 'DC': {'total': 0, 'coop': 0}, 
                 'CD': {'total': 0, 'coop': 0},
                 'DD': {'total': 0, 'coop': 0}}
    
    # Process each match
    for match_id in agent_games['match_id'].unique():
        match_data = agent_games[agent_games['match_id'] == match_id].sort_values('round')
        
        if len(match_data) < 2:
            continue
            
        # Determine which column contains our agent's moves
        if match_data.iloc[0]['agent1'] == agent_name:
            agent_col = 'agent1_move'
            opponent_col = 'agent2_move'
        else:
            agent_col = 'agent2_move'
            opponent_col = 'agent1_move'
        
        # Analyze rounds 2 onwards (need previous round for condition)
        for i in range(1, len(match_data)):
            prev_round = match_data.iloc[i-1]
            curr_round = match_data.iloc[i]
            
            # Get previous round outcome
            if match_data.iloc[0]['agent1'] == agent_name:
                prev_agent_move = prev_round['agent1_move']
                prev_opponent_move = prev_round['agent2_move']
                curr_agent_move = curr_round['agent1_move']
            else:
                prev_agent_move = prev_round['agent2_move']
                prev_opponent_move = prev_round['agent1_move']
                curr_agent_move = curr_round['agent2_move']
            
            # Determine the condition (previous round outcome)
            condition = prev_agent_move + prev_opponent_move
            
            if condition in conditions:
                conditions[condition]['total'] += 1
                if curr_agent_move == 'C':
                    conditions[condition]['coop'] += 1
    
    # Calculate probabilities
    # Note: DC = I defected, opponent cooperated (successful defection)
    #       CD = I cooperated, opponent defected (being exploited)
    footprint = {}
    for cond in ['CC', 'DC', 'CD', 'DD']:
        if conditions[cond]['total'] > 0:
            footprint[f'P(C|{cond})'] = conditions[cond]['coop'] / conditions[cond]['total']
        else:
            footprint[f'P(C|{cond})'] = 0
    
    return footprint

def calculate_dominant_outcome_previous_rounds(match_data, agent_name, current_round_idx):
    """
    Calculate the dominant (most frequent) outcome in all previous rounds of a match for a given agent.
    Returns the dominant outcome or None if there's a tie or no previous rounds.
    """
    if current_round_idx == 0:
        return None  # No previous rounds
        
    # Count outcomes in all previous rounds
    outcome_counts = {'CC': 0, 'CD': 0, 'DC': 0, 'DD': 0}
    
    # Determine which column contains our agent's moves
    if match_data.iloc[0]['agent1'] == agent_name:
        agent_col = 'agent1_move'
        opponent_col = 'agent2_move'
    else:
        agent_col = 'agent2_move'
        opponent_col = 'agent1_move'
    
    # Only count outcomes from rounds 0 to current_round_idx-1 (previous rounds)
    for i in range(current_round_idx):
        row = match_data.iloc[i]
        if match_data.iloc[0]['agent1'] == agent_name:
            agent_move = row['agent1_move']
            opponent_move = row['agent2_move']
        else:
            agent_move = row['agent2_move']
            opponent_move = row['agent1_move']
        
        outcome = agent_move + opponent_move
        if outcome in outcome_counts:
            outcome_counts[outcome] += 1
    
    # Find the dominant outcome among previous rounds
    if all(count == 0 for count in outcome_counts.values()):
        return None  # No previous rounds
        
    max_count = max(outcome_counts.values())
    dominant_outcomes = [outcome for outcome, count in outcome_counts.items() if count == max_count]
    
    # Return dominant outcome only if it's unique
    if len(dominant_outcomes) == 1:
        return dominant_outcomes[0]
    else:
        return None  # Tie - no single dominant outcome

def calculate_extended_strategic_footprint(df, agent_name):
    """
    Calculate extended strategic footprint for a given agent.
    Returns P(C|DominantOutcome=ω) for ω ∈ {CC, CD, DC, DD}
    where DominantOutcome is the mode of joint outcomes among all previous rounds in that specific match.
    """
    # Filter for games where this agent participated
    agent_games = df[(df['agent1'] == agent_name) | (df['agent2'] == agent_name)]
    
    if agent_games.empty:
        return None
    
    # Initialize counters for each dominant outcome condition
    conditions = {'CC': {'total': 0, 'coop': 0},
                 'CD': {'total': 0, 'coop': 0}, 
                 'DC': {'total': 0, 'coop': 0},
                 'DD': {'total': 0, 'coop': 0}}
    
    # Process each match
    for match_id in agent_games['match_id'].unique():
        match_data = agent_games[agent_games['match_id'] == match_id].sort_values('round')
        
        if len(match_data) < 2:
            continue  # Need at least 2 rounds (one previous, one current)
        
        # Analyze each round from round 1 onwards (need at least one previous round)
        for round_idx in range(1, len(match_data)):
            # Calculate dominant outcome of all previous rounds for this current round
            dominant_outcome_prev = calculate_dominant_outcome_previous_rounds(match_data, agent_name, round_idx)
            
            if dominant_outcome_prev is None:
                continue  # Skip if no unique dominant outcome in previous rounds
            
            # Get current round's move
            current_row = match_data.iloc[round_idx]
            if match_data.iloc[0]['agent1'] == agent_name:
                current_agent_move = current_row['agent1_move']
            else:
                current_agent_move = current_row['agent2_move']
            
            # Update counters based on dominant outcome of previous rounds
            conditions[dominant_outcome_prev]['total'] += 1
            if current_agent_move == 'C':
                conditions[dominant_outcome_prev]['coop'] += 1
    
    # Calculate probabilities with new key format for extended footprint
    # Note: DC = I defected, opponent cooperated (successful defection)
    #       CD = I cooperated, opponent defected (being exploited)
    extended_footprint = {}
    for cond in ['CC', 'DC', 'CD', 'DD']:
        if conditions[cond]['total'] > 0:
            extended_footprint[f'P(C|{cond}=dom)'] = conditions[cond]['coop'] / conditions[cond]['total']
        else:
            extended_footprint[f'P(C|{cond}=dom)'] = 0
    
    return extended_footprint

def format_variant_label(variant, model_family):
    """
    Format variant labels for display in legend.
    For temperature variants: T02 -> Temperature 0.2, T05 -> Temperature 0.5, T1 -> Temperature 1.0, T12 -> Temperature 1.2
    For OpenAI models: keep as is (GPT5mini, GPT5nano, GPT4mini)
    """
    if model_family == "OpenAI":
        return variant
    
    # Handle temperature formatting
    if variant.startswith('T'):
        temp_str = variant[1:]  # Remove 'T' prefix
        try:
            # Convert to float and handle different formats
            temp_val = float(temp_str)
            # T02, T05, T07, T08 -> 0.2, 0.5, 0.7, 0.8
            if temp_val <= 9 and len(temp_str) == 2:
                temp_val = temp_val / 10
            # T12 -> 1.2
            elif temp_val >= 10:
                temp_val = temp_val / 10
            # T1 -> 1.0
            return f"Temperature {temp_val}"
        except ValueError:
            return variant
    
    return variant

def extract_shadow_length(experiment_path):
    """
    Extract shadow length from evolutionary CSV files in the experiment directory.
    """
    csv_files = glob.glob(os.path.join(experiment_path, "evolutionary_shadow*_phase*.csv"))
    if csv_files:
        # Extract from filename like evolutionary_shadow75_phase1.csv
        filename = os.path.basename(csv_files[0])
        import re
        match = re.search(r'shadow(\d+)', filename)
        if match:
            shadow_val = int(match.group(1))
            return shadow_val / 100  # Convert 75 to 0.75, 25 to 0.25, etc.
    return None

def calculate_cooperation_rate(df, agent_name):
    """
    Calculate overall cooperation rate for an agent across all games.
    """
    agent_games = df[(df['agent1'] == agent_name) | (df['agent2'] == agent_name)]
    if agent_games.empty:
        return None
    
    total_moves = 0
    coop_moves = 0
    
    for _, row in agent_games.iterrows():
        if row['agent1'] == agent_name:
            move = row['agent1_move']
        else:
            move = row['agent2_move']
        
        total_moves += 1
        if move == 'C':
            coop_moves += 1
    
    return coop_moves / total_moves if total_moves > 0 else 0

def sort_variants_by_temperature(variants, model_family):
    """
    Sort variants by temperature (lowest to highest).
    OpenAI models follow specific order: GPT5mini, GPT5nano, GPT4mini.
    """
    if model_family == "OpenAI":
        # Define desired order for OpenAI models
        openai_order = ["GPT5mini", "GPT5nano", "GPT4mini"]
        sorted_variants = []
        for model in openai_order:
            if model in variants:
                sorted_variants.append(model)
        # Add any remaining variants not in the predefined order
        for variant in variants:
            if variant not in sorted_variants:
                sorted_variants.append(variant)
        return sorted_variants
    
    def get_temp_value(variant):
        if variant.startswith('T'):
            temp_str = variant[1:]
            try:
                temp_val = float(temp_str)
                # T02, T05, T07, T08 -> 0.2, 0.5, 0.7, 0.8
                if temp_val <= 9 and len(temp_str) == 2:
                    return temp_val / 10
                # T12 -> 1.2, T10 -> 1.0
                elif temp_val >= 10:
                    return temp_val / 10
                # T1 -> 1.0
                return temp_val
            except ValueError:
                return 0
        return 0
    
    return sorted(variants, key=get_temp_value)

def create_radar_chart(footprints, title, save_path, model_family, chart_type="strategic"):
    """
    Create a radar chart for strategic footprints.
    """
    # The four metrics for strategic/extended footprint
    # Radar chart positioning (starting from top, going clockwise):
    # - P(C|CC): Top (12 o'clock) - cooperation after mutual cooperation
    # - P(C|CD): Right (3 o'clock) - cooperation after being exploited  
    # - P(C|DD): Bottom (6 o'clock) - cooperation after mutual defection
    # - P(C|DC): Left (9 o'clock) - cooperation after successful defection
    # Note: DC = successful defection (I defect, opponent cooperates)
    #       CD = being exploited (I cooperate, opponent defects)
    if chart_type == "extended":
        categories = ['P(C|CC=dom)', 'P(C|CD=dom)', 'P(C|DD=dom)', 'P(C|DC=dom)']
    else:
        categories = ['P(C|CC)', 'P(C|CD)', 'P(C|DD)', 'P(C|DC)']
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize the plot with smaller size
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Sort variants by temperature and create ordered footprints
    sorted_variants = sort_variants_by_temperature(list(footprints.keys()), model_family)
    ordered_footprints = OrderedDict((variant, footprints[variant]) for variant in sorted_variants)
    
    # Colors for different agents
    if model_family == "OpenAI":
        # Custom colors for OpenAI models: GPT5mini (red), GPT5nano (orange), GPT4mini (grey)
        openai_colors = {"GPT5mini": "red", "GPT5nano": "orange", "GPT4mini": "grey"}
        colors = [openai_colors.get(variant, "black") for variant in ordered_footprints.keys()]
    else:
        colors = plt.cm.Set1(np.linspace(0, 1, len(ordered_footprints)))
    
    # Plot data for each agent
    for i, (variant, footprint) in enumerate(ordered_footprints.items()):
        values = [footprint.get(cat, 0) for cat in categories]
        values += values[:1]  # Complete the circle
        
        # Format label for legend
        display_label = format_variant_label(variant, model_family)
        
        ax.plot(angles, values, 'o-', linewidth=2, label=display_label, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax.grid(True)
    
    plt.title(title, size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_experiment(experiment_path):
    """
    Process a single experiment folder and generate radar charts.
    """
    experiment_name = os.path.basename(experiment_path)
    print(f"Processing experiment: {experiment_name}")
    
    # Extract shadow length for title
    shadow_length = extract_shadow_length(experiment_path)
    if shadow_length is None:
        print(f"Could not extract shadow length from {experiment_name}")
        return None
    
    # Find all evolutionary CSV files
    csv_files = glob.glob(os.path.join(experiment_path, "evolutionary_shadow*_phase*.csv"))
    
    if not csv_files:
        print(f"No evolutionary CSV files found in {experiment_path}")
        return
    
    # Combine all phases
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    if not all_data:
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Find all unique agents
    agents = set()
    agents.update(combined_df['agent1'].unique())
    agents.update(combined_df['agent2'].unique())
    
    # Group LLM agents by model family
    llm_agents_by_family = {}
    for agent in agents:
        model_family, variant = extract_llm_info(agent)
        if model_family and variant:
            if model_family not in llm_agents_by_family:
                llm_agents_by_family[model_family] = {}
            if variant not in llm_agents_by_family[model_family]:
                llm_agents_by_family[model_family][variant] = []
            llm_agents_by_family[model_family][variant].append(agent)
    
    if not llm_agents_by_family:
        print(f"No LLM agents found in {experiment_name}")
        return
    
    # Create output directory
    output_dir = os.path.join(experiment_path, "strategic_footprints")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each LLM family separately
    for model_family, variants in llm_agents_by_family.items():
        print(f"  Processing {model_family} family...")
        
        # Calculate strategic footprints for each variant in this family
        footprints = {}
        for variant, agent_list in variants.items():
            # If multiple agents with same variant, combine their data
            # Note: DC = successful defection, CD = being exploited
            combined_footprint = {'P(C|CC)': [], 'P(C|CD)': [], 'P(C|DD)': [], 'P(C|DC)': []}
            
            for agent in agent_list:
                footprint = calculate_strategic_footprint(combined_df, agent)
                if footprint:
                    for key in combined_footprint:
                        combined_footprint[key].append(footprint[key])
            
            # Average across all agents of same variant
            if any(combined_footprint.values()):
                avg_footprint = {}
                for key in combined_footprint:
                    if combined_footprint[key]:
                        avg_footprint[key] = np.mean(combined_footprint[key])
                    else:
                        avg_footprint[key] = 0
                footprints[variant] = avg_footprint
        
        if footprints:
            # Create radar chart for this model family with shadow length in title
            # Get memory mode for title
            memory_mode = get_memory_mode_from_experiment(experiment_name)
            chart_title = f"{memory_mode} - {model_family} Models (Shadow {shadow_length})"
            # Create safe filename by replacing spaces and special characters
            safe_model_name = model_family.lower().replace(' ', '_').replace('-', '_')
            save_path = os.path.join(output_dir, f"strategic_footprint_{safe_model_name}_{experiment_name}.png")
            create_radar_chart(footprints, chart_title, save_path, model_family)
            print(f"    Saved {model_family} radar chart: {save_path}")
            
            # Save numerical data as CSV
            footprint_df = pd.DataFrame(footprints).T
            csv_path = os.path.join(output_dir, f"strategic_footprint_data_{safe_model_name}_{experiment_name}.csv")
            footprint_df.to_csv(csv_path)
            print(f"    Saved {model_family} footprint data: {csv_path}")
        
        # Calculate and save extended strategic footprints
        extended_footprints = {}
        for variant, agent_list in variants.items():
            # If multiple agents with same variant, combine their data
            # Note: DC = successful defection, CD = being exploited
            combined_extended_footprint = {'P(C|CC=dom)': [], 'P(C|CD=dom)': [], 'P(C|DD=dom)': [], 'P(C|DC=dom)': []}
            
            for agent in agent_list:
                extended_footprint = calculate_extended_strategic_footprint(combined_df, agent)
                if extended_footprint:
                    for key in combined_extended_footprint:
                        combined_extended_footprint[key].append(extended_footprint[key])
            
            # Average across all agents of same variant
            if any(combined_extended_footprint.values()):
                avg_extended_footprint = {}
                for key in combined_extended_footprint:
                    if combined_extended_footprint[key]:
                        avg_extended_footprint[key] = np.mean(combined_extended_footprint[key])
                    else:
                        avg_extended_footprint[key] = 0
                extended_footprints[variant] = avg_extended_footprint
        
        if extended_footprints:
            # Create extended radar chart
            # Get memory mode for title
            memory_mode = get_memory_mode_from_experiment(experiment_name)
            extended_chart_title = f"{memory_mode} - {model_family} Models (Shadow {shadow_length})"
            extended_save_path = os.path.join(output_dir, f"extended_strategic_footprint_{safe_model_name}_{experiment_name}.png")
            create_radar_chart(extended_footprints, extended_chart_title, extended_save_path, model_family, "extended")
            print(f"    Saved {model_family} extended radar chart: {extended_save_path}")
            
            # Save extended numerical data as CSV
            extended_footprint_df = pd.DataFrame(extended_footprints).T
            extended_csv_path = os.path.join(output_dir, f"extended_strategic_footprint_data_{safe_model_name}_{experiment_name}.csv")
            extended_footprint_df.to_csv(extended_csv_path)
            print(f"    Saved {model_family} extended footprint data: {extended_csv_path}")
    
    return {"experiment_name": experiment_name, "shadow_length": shadow_length, "llm_agents_by_family": llm_agents_by_family, "combined_df": combined_df}

def generate_latex_cooperation_table(experiment_results):
    """
    Generate a LaTeX table with cooperation rates for all LLM agents and RL agents across experiments.
    """
    # Collect all cooperation rates
    cooperation_data = []
    
    for exp_result in experiment_results:
        if exp_result is None:
            continue
            
        experiment_name = exp_result["experiment_name"]
        shadow_length = exp_result["shadow_length"]
        llm_agents_by_family = exp_result["llm_agents_by_family"]
        combined_df = exp_result["combined_df"]
        
        # Skip shadow lengths 0.02 and 0.05 from the table
        if shadow_length in [0.02, 0.05]:
            continue
        
        # Add LLM agents
        for model_family, variants in llm_agents_by_family.items():
            for variant, agent_list in variants.items():
                # Calculate cooperation rate for this variant
                coop_rates = []
                for agent in agent_list:
                    coop_rate = calculate_cooperation_rate(combined_df, agent)
                    if coop_rate is not None:
                        coop_rates.append(coop_rate)
                
                if coop_rates:
                    avg_coop_rate = sum(coop_rates) / len(coop_rates)
                    cooperation_data.append({
                        'shadow_length': shadow_length,
                        'model_family': model_family,
                        'variant': variant,
                        'cooperation_rate': avg_coop_rate
                    })
        
        # Add RL agents
        rl_agents = ['QLearning', 'ThompsonSampling', 'GradientMetaLearner']
        for rl_agent in rl_agents:
            # Find all instances of this RL agent in the data
            agent_pattern = f"{rl_agent}_p"
            matching_agents = []
            
            # Get all unique agents from the combined_df
            all_agents = set()
            all_agents.update(combined_df['agent1'].unique())
            all_agents.update(combined_df['agent2'].unique())
            
            for agent in all_agents:
                if agent.startswith(agent_pattern):
                    matching_agents.append(agent)
            
            if matching_agents:
                # Calculate cooperation rates for all instances
                coop_rates = []
                for agent in matching_agents:
                    coop_rate = calculate_cooperation_rate(combined_df, agent)
                    if coop_rate is not None:
                        coop_rates.append(coop_rate)
                
                if coop_rates:
                    avg_coop_rate = sum(coop_rates) / len(coop_rates)
                    cooperation_data.append({
                        'shadow_length': shadow_length,
                        'model_family': 'RL Agent',
                        'variant': rl_agent,
                        'cooperation_rate': avg_coop_rate
                    })
    
    # Create LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Cooperation Rates by LLM Model and Shadow Length}")
    latex_content.append("\\label{tab:cooperation_rates}")
    latex_content.append("\\begin{tabular}{llll}")
    latex_content.append("\\toprule")
    latex_content.append("Shadow Length & Model & Temperature/Variant & Cooperation Rate \\\\")
    latex_content.append("\\midrule")
    
    # Sort by shadow length, then model family (RL Agent last), then variant
    def sort_key(x):
        # Put RL Agent last in model family ordering
        family_order = {'Claude4-Sonnet': 1, 'Gemini-2.0-Flash': 2, 'Mistral Large': 3, 'OpenAI': 4, 'RL Agent': 5}
        model_priority = family_order.get(x['model_family'], 999)
        variant_priority = get_sort_key_for_variant(x['variant'], x['model_family'])
        return (x['shadow_length'], model_priority, variant_priority)
    
    cooperation_data.sort(key=sort_key)
    
    for data in cooperation_data:
        shadow = data['shadow_length']
        model = data['model_family']
        variant = format_variant_label(data['variant'], data['model_family'])
        coop_rate = f"{data['cooperation_rate']:.3f}"
        
        latex_content.append(f"{shadow} & {model} & {variant} & {coop_rate} \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    return "\n".join(latex_content)

def get_sort_key_for_variant(variant, model_family):
    """
    Get sort key for variant to ensure proper ordering in table.
    """
    if model_family == "OpenAI":
        # Order: GPT5mini, GPT5nano, GPT4mini (to match chart legend order)
        order = {"GPT5mini": 1, "GPT5nano": 2, "GPT4mini": 3}
        return order.get(variant, 999)
    
    if model_family == "RL Agent":
        # Order: QLearning, ThompsonSampling, GradientMetaLearner
        order = {"QLearning": 1, "ThompsonSampling": 2, "GradientMetaLearner": 3}
        return order.get(variant, 999)
    
    # For temperature variants, use numeric value
    if variant.startswith('T'):
        temp_str = variant[1:]
        try:
            temp_val = float(temp_str)
            if temp_val <= 9 and len(temp_str) == 2:
                return temp_val / 10
            elif temp_val >= 10:
                return temp_val / 10
            return temp_val
        except ValueError:
            return 999
    
    return 999

def main():
    """
    Main function to process all experiments.
    """
    results_dir = "../results"
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return
    
    # Find all experiment directories
    experiment_dirs = [d for d in glob.glob(os.path.join(results_dir, "experiment_*")) 
                      if os.path.isdir(d)]
    
    if not experiment_dirs:
        print("No experiment directories found!")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Process experiments and collect results
    experiment_results = []
    for exp_dir in sorted(experiment_dirs):
        try:
            result = process_experiment(exp_dir)
            experiment_results.append(result)
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
            experiment_results.append(None)
    
    # Generate LaTeX cooperation table
    latex_table = generate_latex_cooperation_table(experiment_results)
    
    # Save LaTeX table
    latex_path = os.path.join(results_dir, "cooperation_rates_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\nSaved LaTeX cooperation table: {latex_path}")
    
    print("Strategic footprint visualization complete!")

if __name__ == "__main__":
    main()