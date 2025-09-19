#!/usr/bin/env python3
"""
Population Evolution Plots for IPD Experiments
Generates stacked area charts showing agent population evolution across phases
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from collections import defaultdict
import seaborn as sns

def load_experiment_data(analysis_file: str):
    """Load the experiment analysis data."""
    with open(analysis_file, 'r') as f:
        return json.load(f)

def get_strategy_category_and_color(strategy: str):
    """Get category and color for a strategy with logical grouping."""
    
    # Classical strategies - much better color distinction
    classical_strategies = {
        'TitForTat': '#1f77b4',           # blue
        'GrimTrigger': '#aec7e8',         # light blue
        'SuspiciousTitForTat': '#ff6600', # bright orange (very distinct from blue)
        'GenerousTitForTat': '#98df8a',   # light green
        'ForgivingGrimTrigger': '#ff7f0e', # orange
        'Gradual': '#ffbb78',             # light orange
        'SoftGrudger': '#2ca02c',         # green
        'Prober': '#ff9896',              # light red
        'Detective': '#9467bd',           # purple
        'Alternator': '#c5b0d5',          # light purple
        'Random': '#8c564b',              # brown
        'WinStayLoseShift': '#c49c94',    # light brown
        'Bayesian': '#e377c2'             # pink
    }
    
    # Adaptive strategies - greys
    adaptive_colors = {
        'QLearning': '#7f7f7f',           # grey
        'ThompsonSampling': '#c7c7c7',    # light grey
        'GradientMetaLearner': '#bcbd22'   # olive
    }
    
    # LLM strategies with temperature variants - improved color distinction
    # Anthropic (Claude) - distinct blues
    if 'Claude4-Sonnet' in strategy:
        if '_T02' in strategy:
            return 'Anthropic', '#08306b'  # very dark blue
        elif '_T05' in strategy:
            return 'Anthropic', '#2171b5'  # medium blue
        elif '_T08' in strategy:
            return 'Anthropic', '#6baed6'  # light blue

    # Mistral - distinct oranges/reds
    elif 'Mistral-Medium' in strategy:
        if '_T02' in strategy:
            return 'Mistral', '#a63603'    # very dark orange
        elif '_T07' in strategy:
            return 'Mistral', '#e6550d'    # medium orange
        elif '_T12' in strategy:
            return 'Mistral', '#fd8d3c'    # light orange

    # Gemini - distinct greens
    elif 'Gemini20Flash' in strategy:
        if '_T02' in strategy:
            return 'Gemini', '#00441b'     # very dark green
        elif '_T07' in strategy:
            return 'Gemini', '#238b45'     # medium green
        elif '_T12' in strategy:
            return 'Gemini', '#74c476'     # light green

    # OpenAI - distinct purples/magentas for better contrast
    elif 'GPT5mini' in strategy:
        return 'OpenAI', '#88419d'         # purple
    elif 'GPT5nano' in strategy:
        return 'OpenAI', '#8c6bb1'         # light purple
    elif 'GPT4.1mini' in strategy:
        return 'OpenAI', '#9ebcda'         # very light purple
    
    # Check classical strategies - use exact match to avoid substring issues
    for strat_name, color in classical_strategies.items():
        if strategy == strat_name:
            return 'Classical', color

    # Check adaptive strategies - use exact match to avoid substring issues
    for strat_name, color in adaptive_colors.items():
        if strategy == strat_name:
            return 'Adaptive', color
    
    # Default
    return 'Other', '#666666'

def create_population_evolution_plot(experiment_data, experiment_name, shadow_condition, memory_mode="Unknown"):
    """Create a stacked area plot for population evolution."""
    
    phases = experiment_data.get('phases', {})
    if not phases:
        return None
    
    # Sort phases by phase number
    sorted_phases = sorted(phases.items(), key=lambda x: x[1].get('phase_number', 0))
    
    # Collect population data across phases
    phase_numbers = []
    all_strategies = set()
    phase_populations = {}
    
    for phase_key, phase_data in sorted_phases:
        phase_num = phase_data.get('phase_number', 0)
        phase_numbers.append(phase_num)
        
        # Get agent counts by strategy
        agents_by_strategy = phase_data.get('agents_by_strategy', {})
        phase_populations[phase_num] = agents_by_strategy
        all_strategies.update(agents_by_strategy.keys())
    
    if not all_strategies:
        return None
    
    # Group strategies by category for better organization
    strategy_groups = defaultdict(list)
    strategy_colors = {}
    
    for strategy in all_strategies:
        category, color = get_strategy_category_and_color(strategy)
        strategy_groups[category].append(strategy)
        strategy_colors[strategy] = color
    
    # Order strategies: Classical -> Adaptive -> Anthropic -> Mistral -> Gemini -> OpenAI
    ordered_strategies = []
    category_order = ['Classical', 'Adaptive', 'Anthropic', 'Mistral', 'Gemini', 'OpenAI', 'Other']
    
    for category in category_order:
        if category in strategy_groups:
            # Sort strategies within each category
            strategies_in_category = sorted(strategy_groups[category])
            ordered_strategies.extend(strategies_in_category)
    
    # Create data matrix for stacking
    population_matrix = []
    for strategy in ordered_strategies:
        strategy_populations = []
        for phase_num in phase_numbers:
            count = phase_populations[phase_num].get(strategy, 0)
            strategy_populations.append(count)
        population_matrix.append(strategy_populations)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create stacked area plot
    colors = [strategy_colors[strategy] for strategy in ordered_strategies]
    
    ax.stackplot(phase_numbers, *population_matrix, 
                labels=ordered_strategies, colors=colors, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Evolutionary Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Agent Population', fontsize=12, fontweight='bold')
    ax.set_title(f'Population Evolution - Shadow {shadow_condition*100:.0f}% ({memory_mode})',
                fontsize=14, fontweight='bold')
    
    # Set x-axis ticks
    ax.set_xticks(phase_numbers)
    ax.set_xlim(min(phase_numbers), max(phase_numbers))
    
    # Create custom legend grouped by category
    legend_elements = []
    current_category = None
    
    for strategy in ordered_strategies:
        category, _ = get_strategy_category_and_color(strategy)
        
        # Add category header
        if category != current_category:
            if current_category is not None:  # Add spacing between categories
                legend_elements.append(mpatches.Patch(color='none', label=''))
            legend_elements.append(mpatches.Patch(color='black', label=f'{category}:', 
                                                linestyle='none', alpha=0))
            current_category = category
        
        # Format strategy name for legend
        display_name = format_strategy_name_for_legend(strategy)
        legend_elements.append(mpatches.Patch(color=strategy_colors[strategy], 
                                            label=f'  {display_name}'))
    
    # Place legend outside the plot area
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
             fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Tight layout to prevent legend cutoff
    plt.tight_layout()
    
    return fig

def format_strategy_name_for_legend(strategy: str) -> str:
    """Format strategy names for legend display."""

    # Handle temperature variants for LLM agents
    if '_T' in strategy and any(llm in strategy for llm in ['GPT', 'Claude', 'Mistral', 'Gemini', 'GPT4mini']):
        parts = strategy.split('_T')
        if len(parts) == 2:
            base_name = parts[0]
            temp = parts[1]
            # Convert temperature format to proper decimal
            if temp == '02':
                temp_val = "0.2"
            elif temp == '05':
                temp_val = "0.5"
            elif temp == '07':
                temp_val = "0.7"
            elif temp == '08':
                temp_val = "0.8"
            elif temp == '12':
                temp_val = "1.2"
            elif temp == '1':
                temp_val = "1.0"
            else:
                # Handle other formats
                if len(temp) == 2 and temp.startswith('0'):
                    temp_val = f"0.{temp[1]}"
                elif len(temp) == 2 and temp.startswith('1'):
                    temp_val = f"1.{temp[1]}"
                else:
                    temp_val = temp
            return f"{base_name} (T={temp_val})"
    
    # Handle other naming conventions
    replacements = {
        'TitForTat': 'Tit-for-Tat',
        'WinStayLoseShift': 'Win-Stay-Lose-Shift',
        'GenerousTitForTat': 'Generous TFT',
        'SuspiciousTitForTat': 'Suspicious TFT',
        'ForgivingGrimTrigger': 'Forgiving Grim',
        'GrimTrigger': 'Grim Trigger',
        'GradientMetaLearner': 'Gradient Meta',
        'ThompsonSampling': 'Thompson Sampling',
        'QLearning': 'Q-Learning',
        'SoftGrudger': 'Soft Grudger'
    }
    
    return replacements.get(strategy, strategy)

def get_memory_mode(experiment_name: str) -> str:
    """Get memory mode based on experiment name mapping"""
    # Anonymous Memory Mode
    anonymous_experiments = [
        "experiment_20250908_081108",  # 0.05
        "experiment_20250905_152754",  # 0.10
        "experiment_20250905_152805",  # 0.25
        "experiment_20250905_152321"   # 0.75
    ]

    # Opponent Tracking Mode
    tracking_experiments = [
        "experiment_20250912_153315",  # 0.05
        "experiment_20250911_112723",  # 0.10
        "experiment_20250910_074608",  # 0.25
        "experiment_20250910_074557"   # 0.75
    ]

    if experiment_name in anonymous_experiments:
        return "Anonymous Memory"
    elif experiment_name in tracking_experiments:
        return "Opponent Tracking"
    else:
        return "Unknown"

def main():
    """Generate population evolution plots for all experiments."""

    analysis_file = "/mnt/c/Github/LLM-IPD-ARXIV2/visualization_results/all_8experiments_composition_analysis.json"

    try:
        data = load_experiment_data(analysis_file)
    except FileNotFoundError:
        print(f"Error: Analysis file {analysis_file} not found")
        print("Run agent_composition_tracker.py first to generate the analysis")
        return
    
    experiments = data.get('experiments', {})
    
    if not experiments:
        print("No experiments found in analysis file")
        return
    
    # Generate plots for each experiment
    for experiment_name, experiment_data in sorted(experiments.items()):
        print(f"Generating plot for {experiment_name}...")

        # Get shadow condition from first phase
        phases = experiment_data.get('phases', {})
        if phases:
            first_phase = next(iter(phases.values()))
            shadow_condition = first_phase.get('shadow_condition', 0)
        else:
            shadow_condition = 0

        # Get memory mode
        memory_mode = get_memory_mode(experiment_name)

        # Create plot
        fig = create_population_evolution_plot(experiment_data, experiment_name, shadow_condition, memory_mode)
        
        if fig:
            # Create descriptive filename
            memory_short = "anonymous" if "Anonymous" in memory_mode else "tracking"
            shadow_percent = int(shadow_condition * 100)

            # Save plot in visualization_results directory
            output_filename = f"/mnt/c/Github/LLM-IPD-ARXIV2/visualization_results/population_evolution_shadow{shadow_percent}_{memory_short}.png"
            fig.savefig(output_filename, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved plot: {output_filename}")

            # Also save as PDF for publication
            pdf_filename = f"/mnt/c/Github/LLM-IPD-ARXIV2/visualization_results/population_evolution_shadow{shadow_percent}_{memory_short}.pdf"
            fig.savefig(pdf_filename, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved PDF: {pdf_filename}")

            plt.close(fig)
        else:
            print(f"No data available for {experiment_name}")
    
    print("Population evolution plots generated successfully!")

if __name__ == "__main__":
    main()