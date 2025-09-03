#!/usr/bin/env python3
"""
Experiment Results Summary
Analyzes and summarizes mean scores per phase and across phases for each agent.
"""

import json
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

def load_analysis_data() -> dict:
    """Load the experiment analysis data."""
    with open('../results/all_experiments_composition_analysis.json', 'r') as f:
        return json.load(f)

def get_strategy_category(strategy: str) -> str:
    """Categorize strategies for better organization."""
    # Classical strategies
    classical_strategies = ['TitForTat', 'GrimTrigger', 'SuspiciousTitForTat', 'GenerousTitForTat',
                           'ForgivingGrimTrigger', 'Gradual', 'SoftGrudger', 'Prober', 'Detective',
                           'Alternator', 'Random', 'WinStayLoseShift', 'Bayesian']
    
    # Adaptive strategies
    adaptive_strategies = ['QLearning', 'ThompsonSampling', 'GradientMetaLearner']
    
    # Check for classical strategies
    for classic in classical_strategies:
        if classic in strategy:
            return 'Classical'
    
    # Check for adaptive strategies
    for adaptive in adaptive_strategies:
        if adaptive in strategy:
            return 'Adaptive'
    
    # LLM strategies
    if 'Claude' in strategy:
        return 'Anthropic'
    elif 'Mistral' in strategy:
        return 'Mistral'
    elif 'Gemini' in strategy:
        return 'Gemini'
    elif any(llm in strategy for llm in ['GPT', 'GPT4mini']):
        return 'OpenAI'
    
    return 'Other'

def clean_strategy_name(strategy: str) -> str:
    """Clean strategy names for display."""
    # Fix Ministral -> Mistral-Large (backward compatibility)
    if 'Ministral' in strategy:
        strategy = strategy.replace('Ministral', 'Mistral-Large')
    
    return strategy

def summarize_experiment_results():
    """Summarize experimental results with mean scores per phase and across phases, per experiment."""
    
    print("Loading experiment data...")
    data = load_analysis_data()
    
    experiments = data.get('experiments', {})
    
    # Organize data by experiment (shadow condition)
    experiments_by_shadow = {}
    
    for exp_name, exp_data in experiments.items():
        phases = exp_data.get('phases', {})
        
        if not phases:
            continue
            
        # Get experiment shadow condition
        first_phase = next(iter(phases.values()))
        shadow_condition = first_phase.get('shadow_condition', 'Unknown')
        shadow_key = f"Shadow {shadow_condition*100:.0f}%"
        
        print(f"\nAnalyzing {exp_name} ({shadow_key})...")
        
        # Initialize experiment data structure
        if shadow_key not in experiments_by_shadow:
            experiments_by_shadow[shadow_key] = {
                'strategy_phase_scores': defaultdict(lambda: defaultdict(list)),
                'strategy_overall_scores': defaultdict(list)
            }
        
        exp_data_struct = experiments_by_shadow[shadow_key]
        
        for phase_key, phase_data in phases.items():
            phase_number = phase_data.get('phase_number', 0)
            strategy_performance = phase_data.get('strategy_performance', {})
            
            for strategy, performance_data in strategy_performance.items():
                clean_strategy = clean_strategy_name(strategy)
                score_per_move = performance_data.get('score_per_move', 0)
                
                # Store scores by phase and overall for this experiment
                exp_data_struct['strategy_phase_scores'][clean_strategy][phase_number].append(score_per_move)
                exp_data_struct['strategy_overall_scores'][clean_strategy].append(score_per_move)
    
    print("\n" + "="*120)
    print("EXPERIMENTAL RESULTS SUMMARY - PER EXPERIMENT")
    print("="*120)
    print(f"Score represents total payoff per move in Prisoner's Dilemma")
    print(f"Experiments analyzed: {len(experiments_by_shadow)} shadow conditions")
    print("="*120)
    
    # Process each experiment separately
    all_experiment_results = {}
    
    for shadow_key in sorted(experiments_by_shadow.keys()):
        exp_data = experiments_by_shadow[shadow_key]
        strategy_phase_scores = exp_data['strategy_phase_scores']
        strategy_overall_scores = exp_data['strategy_overall_scores']
        
        print(f"\n{'='*120}")
        print(f"EXPERIMENT: {shadow_key}")
        print("="*120)
        
        # Calculate results for this experiment
        results = []
        
        for strategy in sorted(strategy_phase_scores.keys(), key=lambda s: (get_strategy_category(s), s)):
            row = {'Strategy': strategy, 'Category': get_strategy_category(strategy)}
            
            # Calculate mean scores for each phase
            for phase in range(1, 6):  # Phases 1-5
                phase_scores = strategy_phase_scores[strategy][phase]
                if phase_scores:
                    row[f'Phase {phase}'] = np.mean(phase_scores)
                else:
                    row[f'Phase {phase}'] = None
            
            # Calculate overall mean
            overall_scores = strategy_overall_scores[strategy]
            if overall_scores:
                row['Overall Mean'] = np.mean(overall_scores)
                row['Std Dev'] = np.std(overall_scores)
                row['Min'] = np.min(overall_scores)
                row['Max'] = np.max(overall_scores)
            else:
                row['Overall Mean'] = None
                row['Std Dev'] = None
                row['Min'] = None
                row['Max'] = None
            
            results.append(row)
        
        # Create DataFrame for this experiment
        df = pd.DataFrame(results)
        all_experiment_results[shadow_key] = df
        
        # Group by category for organized display
        categories = ['Classical', 'Adaptive', 'Anthropic', 'Mistral', 'Gemini', 'OpenAI', 'Other']
        
        for category in categories:
            category_data = df[df['Category'] == category]
            if category_data.empty:
                continue
                
            print(f"\n{category.upper()} STRATEGIES:")
            print("-" * 80)
            
            for _, row in category_data.iterrows():
                strategy = row['Strategy']
                print(f"\n{strategy}:")
                
                # Phase scores
                phase_scores = []
                for phase in range(1, 6):
                    score = row[f'Phase {phase}']
                    if score is not None:
                        phase_scores.append(f"Phase {phase}: {score:.3f}")
                    else:
                        phase_scores.append(f"Phase {phase}: N/A")
                
                print(f"  Per Phase: {' | '.join(phase_scores)}")
                
                # Overall statistics
                if row['Overall Mean'] is not None:
                    print(f"  Overall:   Mean: {row['Overall Mean']:.3f} | "
                          f"Std: {row['Std Dev']:.3f} | "
                          f"Min: {row['Min']:.3f} | "
                          f"Max: {row['Max']:.3f}")
                else:
                    print(f"  Overall:   No data available")
        
        # Summary statistics by category for this experiment
        print(f"\n{shadow_key} - CATEGORY SUMMARY:")
        print("-" * 80)
        
        category_summary = []
        for category in categories:
            category_data = df[df['Category'] == category]
            if category_data.empty:
                continue
                
            overall_means = category_data['Overall Mean'].dropna()
            if len(overall_means) > 0:
                category_summary.append({
                    'Category': category,
                    'Strategies': len(category_data),
                    'Mean Score': overall_means.mean(),
                    'Std Dev': overall_means.std(),
                    'Min': overall_means.min(),
                    'Max': overall_means.max()
                })
        
        summary_df = pd.DataFrame(category_summary)
        if not summary_df.empty:
            print(f"\n{'Category':<12} {'Strategies':<11} {'Mean Score':<12} {'Std Dev':<10} {'Min':<8} {'Max':<8}")
            print("-" * 70)
            for _, row in summary_df.iterrows():
                print(f"{row['Category']:<12} {row['Strategies']:<11} "
                      f"{row['Mean Score']:<12.3f} {row['Std Dev']:<10.3f} "
                      f"{row['Min']:<8.3f} {row['Max']:<8.3f}")
        
        # Save individual experiment results to CSV
        output_file = f'../experiment_results_{shadow_key.replace(" ", "_").replace("%", "")}.csv'
        df.to_csv(output_file, index=False, float_format='%.4f')
        print(f"\nResults for {shadow_key} saved to: {output_file}")
    
    return all_experiment_results

if __name__ == "__main__":
    summarize_experiment_results()