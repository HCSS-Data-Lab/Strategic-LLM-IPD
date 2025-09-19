#!/usr/bin/env python3
"""
Generate strategic footprints for tracking memory 5% experiment only
"""

import sys
import os
sys.path.append('/mnt/c/Github/LLM-IPD-ARXIV2/analysis_scripts')

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
    if experiment_name == "experiment_20250908_081108":
        return "Anonymous Memory"
    elif experiment_name == "experiment_20250912_153315":
        return "Tracking Memory"
    else:
        return "Unknown Memory"

def extract_llm_info(agent_name):
    """Extract LLM model and variant/temperature from agent name."""

    # OpenAI pattern with decimal point support
    openai_pattern = r'^(GPT\d+(?:\.\d+)?\w*)_T(\d+)'
    if re.match(openai_pattern, agent_name):
        match = re.match(openai_pattern, agent_name)
        model = match.group(1)
        temp = int(match.group(2)) / 10.0
        return 'OpenAI', model, temp

    # Claude pattern
    claude_pattern = r'^Claude4Sonnet_T(\d+)'
    if re.match(claude_pattern, agent_name):
        match = re.match(claude_pattern, agent_name)
        temp = int(match.group(1)) / 10.0
        return 'Claude4-Sonnet', 'Claude4-Sonnet', temp

    # Mistral pattern
    mistral_pattern = r'^MistralMedium_T(\d+)'
    if re.match(mistral_pattern, agent_name):
        match = re.match(mistral_pattern, agent_name)
        temp = int(match.group(1)) / 10.0
        return 'Mistral Medium', 'Mistral-Medium', temp

    # Gemini pattern
    gemini_pattern = r'^Gemini20Flash_T(\d+)'
    if re.match(gemini_pattern, agent_name):
        match = re.match(gemini_pattern, agent_name)
        temp = int(match.group(1)) / 10.0
        return 'Gemini-2.0-Flash', 'Gemini-2.0-Flash', temp

    return None, None, None

def load_evolutionary_data(experiment_path):
    """Load and process evolutionary data from experiment directory."""
    csv_files = glob.glob(os.path.join(experiment_path, 'evolutionary_*.csv'))

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def calculate_strategic_footprint(df):
    """Calculate strategic footprint for each LLM variant."""
    footprint_data = {}

    # Group by agent1 (assuming all agents appear as agent1 at some point)
    for agent in df['agent1'].unique():
        family, model, temp = extract_llm_info(agent)
        if family is None:
            continue

        agent_rows = df[df['agent1'] == agent].copy()

        if len(agent_rows) == 0:
            continue

        # Calculate conditional cooperation probabilities
        cooperations = {'CC': [], 'CD': [], 'DC': [], 'DD': []}

        for _, row in agent_rows.iterrows():
            if pd.notna(row['agent1_last_move']) and pd.notna(row['agent2_last_move']):
                last_outcome = row['agent1_last_move'] + row['agent2_last_move']
                current_move = row['agent1_move']

                if last_outcome in cooperations:
                    cooperations[last_outcome].append(1 if current_move == 'C' else 0)

        # Calculate probabilities
        footprint = {}
        for outcome in ['CC', 'DC', 'CD', 'DD']:
            if cooperations[outcome]:
                footprint[f'P(C|{outcome})'] = np.mean(cooperations[outcome])
            else:
                footprint[f'P(C|{outcome})'] = 0.0

        footprint_data[agent] = {
            'family': family,
            'model': model,
            'temperature': temp,
            'footprint': footprint
        }

    return footprint_data

def calculate_extended_footprint(df):
    """Calculate extended strategic footprint based on match dominance patterns."""
    extended_data = {}

    # Group by match_id and agent1
    matches = df.groupby(['match_id', 'agent1'])

    for (match_id, agent), match_df in matches:
        family, model, temp = extract_llm_info(agent)
        if family is None:
            continue

        if agent not in extended_data:
            extended_data[agent] = {
                'family': family,
                'model': model,
                'temperature': temp,
                'match_patterns': {'CC': [], 'CD': [], 'DC': [], 'DD': []}
            }

        # Count outcome frequencies in this match
        outcome_counts = {'CC': 0, 'CD': 0, 'DC': 0, 'DD': 0}
        cooperation_rate = 0
        total_moves = 0

        for _, row in match_df.iterrows():
            if pd.notna(row['agent1_move']) and pd.notna(row['agent2_move']):
                outcome = row['agent1_move'] + row['agent2_move']
                if outcome in outcome_counts:
                    outcome_counts[outcome] += 1

                if row['agent1_move'] == 'C':
                    cooperation_rate += 1
                total_moves += 1

        if total_moves > 0:
            cooperation_rate /= total_moves

            # Find dominant outcome
            dominant_outcome = max(outcome_counts, key=outcome_counts.get)
            if outcome_counts[dominant_outcome] > 0:
                extended_data[agent]['match_patterns'][dominant_outcome].append(cooperation_rate)

    # Calculate average cooperation rates for each pattern
    extended_footprint = {}
    for agent, data in extended_data.items():
        footprint = {}
        for pattern in ['CC', 'CD', 'DC', 'DD']:
            if data['match_patterns'][pattern]:
                footprint[f'{pattern}-dominated'] = np.mean(data['match_patterns'][pattern])
            else:
                footprint[f'{pattern}-dominated'] = 0.0

        extended_footprint[agent] = {
            'family': data['family'],
            'model': data['model'],
            'temperature': data['temperature'],
            'footprint': footprint
        }

    return extended_footprint

def create_radar_chart(footprint_data, title, filename, extended=False):
    """Create radar chart for strategic footprint."""

    if extended:
        categories = ['CC-dominated', 'CD-dominated', 'DC-dominated', 'DD-dominated']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    else:
        categories = ['P(C|CC)', 'P(C|DC)', 'P(C|CD)', 'P(C|DD)']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set3(np.linspace(0, 1, len(footprint_data)))

    for i, (agent, data) in enumerate(footprint_data.items()):
        values = [data['footprint'][cat] for cat in categories]
        values += values[:1]  # Complete the circle

        model_temp = f"{data['model']} (T={data['temperature']:.1f})"
        ax.plot(angles, values, 'o-', linewidth=2, label=model_temp, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)

    plt.title(title, size=16, weight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_footprint_data(footprint_data, filename):
    """Save footprint data to CSV."""

    if not footprint_data:
        return

    # Get categories from first entry
    first_agent = next(iter(footprint_data.values()))
    categories = list(first_agent['footprint'].keys())

    # Create DataFrame
    rows = []
    for agent, data in footprint_data.items():
        row = [data['model']] + [data['footprint'][cat] for cat in categories]
        rows.append(row)

    df = pd.DataFrame(rows, columns=[''] + categories)
    df.to_csv(filename, index=False)

def process_experiment_20250912_153315():
    """Process the tracking memory 5% experiment."""

    experiment_path = '../results/experiment_20250912_153315'
    output_dir = f'{experiment_path}/strategic_footprints'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing experiment: experiment_20250912_153315")

    # Load evolutionary data
    df = load_evolutionary_data(experiment_path)
    if df.empty:
        print("No evolutionary data found!")
        return

    # Calculate strategic footprints
    footprint_data = calculate_strategic_footprint(df)
    extended_data = calculate_extended_footprint(df)

    # Group by LLM family
    families = {}
    extended_families = {}

    for agent, data in footprint_data.items():
        family = data['family']
        if family not in families:
            families[family] = {}
        families[family][agent] = data

    for agent, data in extended_data.items():
        family = data['family']
        if family not in extended_families:
            extended_families[family] = {}
        extended_families[family][agent] = data

    memory_mode = get_memory_mode_from_experiment("experiment_20250912_153315")

    # Create charts for each family
    for family, family_data in families.items():
        family_name = family.replace('-', '_').replace('.', '_').lower()

        # Regular strategic footprint
        chart_title = f"{memory_mode} - {family} Models (Shadow 5%)"
        chart_filename = f"{output_dir}/strategic_footprint_{family_name}_experiment_20250912_153315.png"
        create_radar_chart(family_data, chart_title, chart_filename, extended=False)
        print(f"    Saved {family} radar chart: {chart_filename}")

        # Save data
        data_filename = f"{output_dir}/strategic_footprint_data_{family_name}_experiment_20250912_153315.csv"
        save_footprint_data(family_data, data_filename)
        print(f"    Saved {family} footprint data: {data_filename}")

    # Extended strategic footprints
    for family, family_data in extended_families.items():
        family_name = family.replace('-', '_').replace('.', '_').lower()

        # Extended strategic footprint
        chart_title = f"Extended {memory_mode} - {family} Models (Shadow 5%)"
        chart_filename = f"{output_dir}/extended_strategic_footprint_{family_name}_experiment_20250912_153315.png"
        create_radar_chart(family_data, chart_title, chart_filename, extended=True)
        print(f"    Saved {family} extended radar chart: {chart_filename}")

        # Save data
        data_filename = f"{output_dir}/extended_strategic_footprint_data_{family_name}_experiment_20250912_153315.csv"
        save_footprint_data(family_data, data_filename)
        print(f"    Saved {family} extended footprint data: {data_filename}")

if __name__ == "__main__":
    process_experiment_20250912_153315()
    print("Tracking memory strategic footprints generation completed!")