#!/usr/bin/env python3
"""
Combined Population Evolution Plots
Creates a side-by-side comparison of Shadow 25% and Shadow 10% experiments
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
    
    # Classical strategies - distinct colors
    classical_strategies = {
        'TitForTat': '#1f77b4',           # blue
        'GrimTrigger': '#aec7e8',         # light blue
        'SuspiciousTitForTat': '#2ca02c', # green
        'GenerousTitForTat': '#98df8a',   # light green
        'ForgivingGrimTrigger': '#ff7f0e', # orange
        'Gradual': '#ffbb78',             # light orange
        'SoftGrudger': '#d62728',         # red
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
    
    # LLM strategies with temperature variants
    # Anthropic (Claude) - blues with temperature gradients
    if 'Claude4-Sonnet' in strategy:
        if '_T02' in strategy:
            return 'Anthropic', '#08519c'  # dark blue
        elif '_T05' in strategy:
            return 'Anthropic', '#3182bd'  # medium blue
        elif '_T08' in strategy:
            return 'Anthropic', '#6baed6'  # light blue
    
    # Mistral - oranges with temperature gradients
    elif 'Mistral-Large' in strategy:
        if '_T02' in strategy:
            return 'Mistral', '#d94801'    # dark orange
        elif '_T07' in strategy:
            return 'Mistral', '#fd8d3c'    # medium orange
        elif '_T12' in strategy:
            return 'Mistral', '#fdbe85'    # light orange
    
    # Gemini - greens with temperature gradients
    elif 'Gemini25Pro' in strategy:
        if '_T02' in strategy:
            return 'Gemini', '#238b45'     # dark green
        elif '_T07' in strategy:
            return 'Gemini', '#66c2a4'     # medium green
        elif '_T12' in strategy:
            return 'Gemini', '#b2e0ab'     # light green
    
    # OpenAI - reds/purples with model distinctions
    elif 'GPT5nanomini' in strategy:
        return 'OpenAI', '#cb181d'         # red
    elif 'GPT5nano' in strategy:
        return 'OpenAI', '#fb6a4a'         # light red
    elif 'GPT4mini' in strategy:
        return 'OpenAI', '#fcae91'         # very light red
    
    # Check classical strategies
    for strat_name, color in classical_strategies.items():
        if strat_name in strategy:
            return 'Classical', color
    
    # Check adaptive strategies
    for strat_name, color in adaptive_colors.items():
        if strat_name in strategy:
            return 'Adaptive', color
    
    # Default
    return 'Other', '#666666'

def format_strategy_name_for_legend(strategy: str) -> str:
    """Format strategy names for legend display."""
    
    # Handle temperature variants for LLM agents
    if '_T' in strategy and any(llm in strategy for llm in ['GPT', 'Claude', 'Mistral', 'Gemini', 'GPT4mini']):
        parts = strategy.split('_T')
        if len(parts) == 2:
            base_name = parts[0]
            temp = parts[1]
            # Convert temperature format
            if temp.startswith('0') and len(temp) == 2:
                temp_val = f"0.{temp[1:]}"
            elif temp.isdigit() and len(temp) <= 2:
                temp_val = f"0.{temp}" if len(temp) == 1 else f"1.{temp[1:]}" if temp.startswith('1') else temp
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

def get_experiment_population_data(experiment_data):
    """Extract population data from experiment."""
    phases = experiment_data.get('phases', {})
    if not phases:
        return None, None, None
    
    # Sort phases by phase number
    sorted_phases = sorted(phases.items(), key=lambda x: x[1].get('phase_number', 0))
    
    # Collect population data across phases
    phase_numbers = []
    phase_populations = {}
    
    for phase_key, phase_data in sorted_phases:
        phase_num = phase_data.get('phase_number', 0)
        phase_numbers.append(phase_num)
        
        # Get agent counts by strategy
        agents_by_strategy = phase_data.get('agents_by_strategy', {})
        phase_populations[phase_num] = agents_by_strategy
    
    return phase_numbers, phase_populations, sorted_phases

def create_combined_population_plot():
    """Create a combined figure with Shadow 25% and Shadow 10% side by side."""
    
    analysis_file = "../results/all_experiments_composition_analysis.json"
    
    try:
        data = load_experiment_data(analysis_file)
    except FileNotFoundError:
        print(f"Error: Analysis file {analysis_file} not found")
        return None
    
    experiments = data.get('experiments', {})
    
    # Find the specific experiments
    exp_25_data = None
    exp_10_data = None
    
    for exp_name, exp_data in experiments.items():
        phases = exp_data.get('phases', {})
        if phases:
            first_phase = next(iter(phases.values()))
            shadow_condition = first_phase.get('shadow_condition', 0)
            
            if abs(shadow_condition - 0.25) < 0.01:  # Shadow 25%
                exp_25_data = exp_data
            elif abs(shadow_condition - 0.10) < 0.01:  # Shadow 10%
                exp_10_data = exp_data
    
    if not exp_25_data or not exp_10_data:
        print("Could not find both Shadow 25% and Shadow 10% experiments")
        return None
    
    # Get population data for both experiments
    phase_numbers_25, phase_populations_25, _ = get_experiment_population_data(exp_25_data)
    phase_numbers_10, phase_populations_10, _ = get_experiment_population_data(exp_10_data)
    
    # Collect all unique strategies from both experiments
    all_strategies = set()
    if phase_populations_25:
        for phase_pop in phase_populations_25.values():
            all_strategies.update(phase_pop.keys())
    if phase_populations_10:
        for phase_pop in phase_populations_10.values():
            all_strategies.update(phase_pop.keys())
    
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
            strategies_in_category = sorted(strategy_groups[category])
            ordered_strategies.extend(strategies_in_category)
    
    # Create the combined figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    colors = [strategy_colors[strategy] for strategy in ordered_strategies]
    
    # Plot Shadow 25% (left subplot)
    if phase_populations_25 and phase_numbers_25:
        population_matrix_25 = []
        for strategy in ordered_strategies:
            strategy_populations = []
            for phase_num in phase_numbers_25:
                count = phase_populations_25[phase_num].get(strategy, 0)
                strategy_populations.append(count)
            population_matrix_25.append(strategy_populations)
        
        ax1.stackplot(phase_numbers_25, *population_matrix_25, colors=colors, alpha=0.8)
        ax1.set_title('Shadow 25%', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Evolutionary Phase', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Agent Population', fontsize=12, fontweight='bold')
        ax1.set_xticks(phase_numbers_25)
        ax1.set_xlim(min(phase_numbers_25), max(phase_numbers_25))
        ax1.grid(True, alpha=0.3)
        ax1.set_axisbelow(True)
    
    # Plot Shadow 10% (right subplot)
    if phase_populations_10 and phase_numbers_10:
        population_matrix_10 = []
        for strategy in ordered_strategies:
            strategy_populations = []
            for phase_num in phase_numbers_10:
                count = phase_populations_10[phase_num].get(strategy, 0)
                strategy_populations.append(count)
            population_matrix_10.append(strategy_populations)
        
        ax2.stackplot(phase_numbers_10, *population_matrix_10, colors=colors, alpha=0.8)
        ax2.set_title('Shadow 10%', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Evolutionary Phase', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Agent Population', fontsize=12, fontweight='bold')
        ax2.set_xticks(phase_numbers_10)
        ax2.set_xlim(min(phase_numbers_10), max(phase_numbers_10))
        ax2.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)
    
    # Create shared legend
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
    
    # Place shared legend much closer to the figures
    fig.legend(handles=legend_elements, bbox_to_anchor=(0.72, 0.96), loc='upper left',
              fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Add main title positioned above the two plots only (not including legend)
    fig.suptitle('Population Evolution Across Evolutionary Phases', fontsize=16, fontweight='bold', 
                x=0.35, y=0.99)  # Center title over the two subplots
    
    # Adjust layout with much tighter spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, right=0.68, wspace=0.15)  # Much smaller right margin
    
    return fig

def main():
    """Generate the combined population evolution plot."""
    
    print("Generating combined Shadow 25% and Shadow 10% population evolution plot...")
    
    fig = create_combined_population_plot()
    
    if fig:
        # Save plot
        output_filename = "combined_population_evolution_25_10.png"
        fig.savefig(output_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved plot: {output_filename}")
        
        # Also save as PDF for publication
        pdf_filename = "combined_population_evolution_25_10.pdf"
        fig.savefig(pdf_filename, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved PDF: {pdf_filename}")
        
        plt.close(fig)
        print("Combined population evolution plot generated successfully!")
    else:
        print("Failed to generate combined plot")

if __name__ == "__main__":
    main()