#!/usr/bin/env python3
"""
Agent Composition Tracker for IPD Experiments
Tracks and logs the composition of agents per experiment phase with detailed performance metrics
"""

import os
import json
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any, Optional
import glob
import csv

def parse_csv_file(csv_file: str) -> Dict[str, Any]:
    """
    Parse a CSV file to extract all relevant data from the simple CSV format.
    Expected format: timestamp,shadow_condition,match_id,round,agent1,agent2,agent1_move,agent2_move,agent1_total_score,agent2_total_score,agent1_reasoning,agent2_reasoning
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Dictionary containing parsed data
    """
    sections = {
        'metadata': {},
        'match_data': [],
        'raw_data': []
    }
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Store raw data
                sections['raw_data'].append(row)
                
                # Extract metadata from first row
                if not sections['metadata']:
                    sections['metadata']['shadow_condition'] = row.get('shadow_condition', '')
                    sections['metadata']['timestamp'] = row.get('timestamp', '')
                
                # Process match data
                try:
                    match_data = {
                        'timestamp': row.get('timestamp', ''),
                        'shadow_condition': float(row.get('shadow_condition', 0)),
                        'match_id': row.get('match_id', ''),
                        'round': int(row.get('round', 0)),
                        'agent1': row.get('agent1', ''),
                        'agent2': row.get('agent2', ''),
                        'move1': row.get('agent1_move', ''),
                        'move2': row.get('agent2_move', ''),
                        'score1': int(row.get('agent1_total_score', 0)),
                        'score2': int(row.get('agent2_total_score', 0)),
                        'reasoning1': row.get('agent1_reasoning', ''),
                        'reasoning2': row.get('agent2_reasoning', '')
                    }
                    sections['match_data'].append(match_data)
                except (ValueError, TypeError):
                    continue
                    
    except Exception as e:
        print(f"Error parsing CSV file {csv_file}: {e}")
        
    return sections

def categorize_agent(agent_name: str) -> Dict[str, str]:
    """
    Categorize an agent by type and extract strategy information.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Dictionary with agent type and strategy information
    """
    # Clean agent name by removing phase/instance suffixes
    clean_name = agent_name
    if '_p' in agent_name and 'i' in agent_name.split('_p')[-1]:
        # Remove _pXiY suffix
        parts = agent_name.split('_')
        clean_parts = []
        for part in parts:
            if part.startswith('p') and 'i' in part:
                break
            clean_parts.append(part)
        clean_name = '_'.join(clean_parts)
    
    # LLM agents - check for various LLM naming patterns
    if any(x in clean_name for x in ['GPT', 'Claude', 'Mistral', 'Ministral', 'Gemini', 'GPT4mini', 'GPT5nano']):
        if any(x in clean_name for x in ['GPT', 'GPT4mini', 'GPT5nano']):
            agent_type = 'LLM_OpenAI'
        elif 'Claude' in clean_name:
            agent_type = 'LLM_Anthropic'
        elif any(x in clean_name for x in ['Mistral', 'Ministral']):
            agent_type = 'LLM_Mistral'
        elif 'Gemini' in clean_name:
            agent_type = 'LLM_Gemini'
        else:
            agent_type = 'LLM_Unknown'
        
        # Fix the strategy name for Mistral variants
        if 'Ministral' in clean_name:
            # Convert "Ministral-Large_T02" to "Mistral-Large_T02"
            strategy = clean_name.replace('Ministral', 'Mistral')
        else:
            strategy = clean_name  # Keep full name with temperature
    else:
        # Classical/behavioral/adaptive agents
        strategy = clean_name.split('_')[0] if '_' in clean_name else clean_name
        
        if strategy in ['TitForTat', 'GrimTrigger', 'WinStayLoseShift', 'Random', 
                       'GenerousTitForTat', 'SuspiciousTitForTat', 'Prober', 'Gradual', 'Alternator']:
            agent_type = 'Classical'
        elif strategy in ['ForgivingGrimTrigger', 'Detective', 'SoftGrudger']:
            agent_type = 'Behavioral'
        elif strategy in ['QLearning', 'ThompsonSampling', 'GradientMetaLearner', 'Bayesian']:
            agent_type = 'Adaptive'
        else:
            agent_type = 'Unknown'
    
    return {
        'agent_type': agent_type,
        'strategy': strategy,
        'original_name': agent_name,
        'clean_name': clean_name
    }


def assess_agent_composition_per_phase(experiment_dir: str) -> Dict[str, Any]:
    """
    Assess the composition of agents per phase for a given experiment using only CSV files.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Dictionary containing composition analysis for each phase
    """
    composition_data = {
        'experiment_path': experiment_dir,
        'experiment_name': os.path.basename(experiment_dir),
        'timestamp': datetime.now().isoformat(),
        'phases': {},
        'experiment_metadata': {}
    }
    
    # Find all CSV files that contain phase results
    csv_pattern = os.path.join(experiment_dir, "evolutionary_shadow*_phase*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No phase CSV files found in {experiment_dir}")
        return composition_data
    
    # Process each CSV file
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        
        # Parse shadow condition and phase number from filename
        # Format: evolutionary_shadow{XX}_phase{Y}.csv
        parts = filename.replace('.csv', '').split('_')
        shadow_part = next((p for p in parts if p.startswith('shadow')), None)
        phase_part = next((p for p in parts if p.startswith('phase')), None)
        
        if not shadow_part or not phase_part:
            print(f"Could not parse shadow/phase from filename: {filename}")
            continue
            
        shadow_value = shadow_part.replace('shadow', '')
        phase_number = phase_part.replace('phase', '')
        
        try:
            shadow_float = int(shadow_value) / 100.0
            phase_int = int(phase_number)
        except ValueError:
            print(f"Could not parse numeric values from filename: {filename}")
            continue
        
        phase_key = f"shadow_{shadow_value}_phase_{phase_number}"
        
        # Parse the CSV file
        try:
            csv_data = parse_csv_file(csv_file)
            phase_analysis = analyze_phase_composition(csv_data, shadow_float, phase_int, csv_file)
            composition_data['phases'][phase_key] = phase_analysis
            
            # Store experiment metadata from first file
            if not composition_data['experiment_metadata'] and csv_data['metadata']:
                composition_data['experiment_metadata'] = csv_data['metadata']
                
        except Exception as e:
            print(f"Error analyzing {csv_file}: {e}")
            continue
    
    return composition_data

def analyze_phase_composition(csv_data: Dict, shadow_condition: float, phase_number: int, csv_file: str) -> Dict[str, Any]:
    """
    Analyze agent composition for a single phase from parsed CSV data.
    
    Args:
        csv_data: Parsed CSV data sections
        shadow_condition: Shadow condition (termination probability)
        phase_number: Phase number
        csv_file: Path to the CSV file
        
    Returns:
        Dictionary with detailed phase composition analysis
    """
    phase_data = {
        'shadow_condition': shadow_condition,
        'phase_number': phase_number,
        'csv_file': csv_file,
        'agents_by_type': defaultdict(int),
        'agents_by_strategy': defaultdict(int),
        'unique_agents': set(),
        'strategy_performance': {},
        'evolutionary_metrics': {},
        'match_statistics': {
            'total_matches': 0,
            'total_rounds': 0,
            'avg_rounds_per_match': 0.0,
            'unique_match_ids': 0
        },
        'detailed_agent_info': []
    }
    
    # Process match data to collect agents and statistics
    agents_seen = set()
    match_ids = set()
    strategy_stats = defaultdict(lambda: {
        'total_score': 0,
        'total_moves': 0,
        'matches_played': set(),
        'total_rounds': 0
    })
    
    for match_data in csv_data['match_data']:
        agent1 = match_data['agent1']
        agent2 = match_data['agent2']
        match_id = match_data['match_id']
        
        # Collect agents
        if agent1:
            agents_seen.add(agent1)
        if agent2:
            agents_seen.add(agent2)
            
        # Collect match IDs
        match_ids.add(match_id)
        
        # Calculate payoffs for this round
        move1, move2 = match_data['move1'], match_data['move2']
        if move1 == 'C' and move2 == 'C':
            payoff1, payoff2 = 3, 3
        elif move1 == 'C' and move2 == 'D':
            payoff1, payoff2 = 0, 5
        elif move1 == 'D' and move2 == 'C':
            payoff1, payoff2 = 5, 0
        else:  # Both defect
            payoff1, payoff2 = 1, 1
        
        # Update strategy statistics
        strategy1 = categorize_agent(agent1)['strategy']
        strategy2 = categorize_agent(agent2)['strategy']
        
        strategy_stats[strategy1]['total_score'] += payoff1
        strategy_stats[strategy1]['total_moves'] += 1
        strategy_stats[strategy1]['matches_played'].add(match_id)
        
        strategy_stats[strategy2]['total_score'] += payoff2
        strategy_stats[strategy2]['total_moves'] += 1
        strategy_stats[strategy2]['matches_played'].add(match_id)
    
    # Calculate strategy performance metrics
    for strategy, stats in strategy_stats.items():
        total_moves = stats['total_moves']
        total_score = stats['total_score']
        matches_played = len(stats['matches_played'])
        
        if total_moves > 0:
            score_per_move = total_score / total_moves
            score_per_match = total_score / matches_played if matches_played > 0 else 0
        else:
            score_per_move = 0
            score_per_match = 0
            
        phase_data['strategy_performance'][strategy] = {
            'score_per_move': score_per_move,
            'score_per_match': score_per_match,
            'total_score': total_score,
            'total_rounds': total_moves,
            'matches_played': matches_played
        }
    
    # Calculate evolutionary metrics
    if phase_data['strategy_performance']:
        strategy_fitness = {s: perf['score_per_move'] for s, perf in phase_data['strategy_performance'].items()}
        total_fitness = sum(strategy_fitness.values()) if strategy_fitness.values() else 0
        mean_fitness = total_fitness / len(strategy_fitness) if strategy_fitness else 0
        
        for strategy, fitness in strategy_fitness.items():
            relative_fitness = (fitness / mean_fitness) ** 2 if mean_fitness > 0 else 1.0
            raw_count = relative_fitness * 1  # Assuming equal initial population
            
            phase_data['evolutionary_metrics'][strategy] = {
                'fitness': fitness,
                'relative_fitness': relative_fitness,
                'raw_count': raw_count,
                'total_fitness_pool': total_fitness,
                'mean_fitness': mean_fitness
            }
    
    # Categorize agents and create detailed info
    for agent_name in agents_seen:
        if not agent_name or agent_name in ['Agent1', 'Agent2']:
            continue
            
        phase_data['unique_agents'].add(agent_name)
        
        # Categorize agent
        agent_info = categorize_agent(agent_name)
        agent_type = agent_info['agent_type']
        strategy = agent_info['strategy']
        
        phase_data['agents_by_type'][agent_type] += 1
        phase_data['agents_by_strategy'][strategy] += 1
        
        # Store detailed agent information
        agent_detail = {
            'agent_name': agent_name,
            'agent_type': agent_type,
            'strategy': strategy,
            'clean_name': agent_info['clean_name']
        }
        
        # Add performance data if available
        if strategy in phase_data['strategy_performance']:
            perf = phase_data['strategy_performance'][strategy]
            agent_detail.update({
                'score_per_move': perf['score_per_move'],
                'score_per_match': perf['score_per_match'],
                'total_score': perf['total_score'],
                'total_rounds': perf['total_rounds'],
                'matches_played': perf['matches_played']
            })
        
        # Add evolutionary metrics if available
        if strategy in phase_data['evolutionary_metrics']:
            evo = phase_data['evolutionary_metrics'][strategy]
            agent_detail.update({
                'evolutionary_fitness': evo['fitness'],
                'relative_fitness': evo['relative_fitness'],
                'raw_count': evo['raw_count']
            })
        
        phase_data['detailed_agent_info'].append(agent_detail)
    
    # Calculate match statistics
    total_rounds_from_matches = defaultdict(int)
    for match_data in csv_data['match_data']:
        total_rounds_from_matches[match_data['match_id']] += 1
    
    phase_data['match_statistics']['total_matches'] = len(match_ids)
    phase_data['match_statistics']['unique_match_ids'] = len(match_ids)
    phase_data['match_statistics']['total_rounds'] = sum(total_rounds_from_matches.values())
    if len(match_ids) > 0:
        phase_data['match_statistics']['avg_rounds_per_match'] = phase_data['match_statistics']['total_rounds'] / len(match_ids)
    
    # Convert defaultdicts and sets for JSON serialization
    phase_data['agents_by_type'] = dict(phase_data['agents_by_type'])
    phase_data['agents_by_strategy'] = dict(phase_data['agents_by_strategy'])
    phase_data['unique_agents'] = list(phase_data['unique_agents'])
    phase_data['total_unique_agents'] = len(phase_data['unique_agents'])
    
    return phase_data

def log_composition_to_experiment_folder(experiment_dir: str, composition_data: Dict[str, Any]) -> str:
    """
    Save the composition analysis to the experiment folder.
    
    Args:
        experiment_dir: Path to the experiment directory
        composition_data: Composition analysis data
        
    Returns:
        Path to the saved composition file
    """
    output_file = os.path.join(experiment_dir, "agent_composition_analysis.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(composition_data, f, indent=2, default=str)
        
        print(f"Agent composition analysis saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error saving composition analysis to {output_file}: {e}")
        return None

def generate_composition_summary_report(composition_data: Dict[str, Any]) -> str:
    """
    Generate a comprehensive human-readable summary report of agent composition.
    
    Args:
        composition_data: Composition analysis data
        
    Returns:
        Formatted summary report as string
    """
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE AGENT COMPOSITION ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Experiment: {composition_data['experiment_name']}")
    report.append(f"Analysis Date: {composition_data['timestamp']}")
    report.append(f"Total Phases Analyzed: {len(composition_data['phases'])}")
    
    # Add experiment metadata
    if composition_data['experiment_metadata']:
        report.append("\nExperiment Metadata:")
        for key, value in composition_data['experiment_metadata'].items():
            report.append(f"  {key}: {value}")
    
    report.append("")
    
    if not composition_data['phases']:
        report.append("No phase data found to analyze.")
        return "\n".join(report)
    
    # Sort phases for consistent reporting
    sorted_phases = sorted(composition_data['phases'].keys())
    
    for phase_key in sorted_phases:
        phase_data = composition_data['phases'][phase_key]
        
        report.append(f"Phase: {phase_key}")
        report.append("="*60)
        report.append(f"Shadow Condition: {phase_data['shadow_condition']*100:.0f}%")
        report.append(f"Phase Number: {phase_data['phase_number']}")
        report.append(f"Total Unique Agents: {phase_data.get('total_unique_agents', 0)}")
        
        # Match statistics
        match_stats = phase_data.get('match_statistics', {})
        report.append(f"Total Matches: {match_stats.get('total_matches', 0)}")
        report.append(f"Total Rounds: {match_stats.get('total_rounds', 0)}")
        report.append(f"Average Rounds per Match: {match_stats.get('avg_rounds_per_match', 0):.2f}")
        report.append("")
        
        # Agent composition by type
        if phase_data.get('agents_by_type'):
            report.append("Agents by Type:")
            total_agents = sum(phase_data['agents_by_type'].values())
            for agent_type, count in sorted(phase_data['agents_by_type'].items()):
                percentage = (count / total_agents) * 100 if total_agents > 0 else 0
                report.append(f"  {agent_type}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Agent composition by strategy  
        if phase_data.get('agents_by_strategy'):
            report.append("Agents by Strategy:")
            total_agents = sum(phase_data['agents_by_strategy'].values())
            for strategy, count in sorted(phase_data['agents_by_strategy'].items()):
                percentage = (count / total_agents) * 100 if total_agents > 0 else 0
                report.append(f"  {strategy}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Strategy performance metrics
        if phase_data.get('strategy_performance'):
            report.append("Strategy Performance Metrics:")
            report.append("Strategy               | Score/Move | Score/Match | Total Score | Rounds | Matches")
            report.append("-" * 85)
            
            # Sort by score per move (descending)
            sorted_strategies = sorted(phase_data['strategy_performance'].items(), 
                                     key=lambda x: x[1]['score_per_move'], reverse=True)
            
            for strategy, perf in sorted_strategies:
                report.append(f"{strategy:22s} | {perf['score_per_move']:10.3f} | "
                            f"{perf['score_per_match']:11.3f} | {perf['total_score']:11.1f} | "
                            f"{perf['total_rounds']:6d} | {perf['matches_played']:7d}")
            report.append("")
        
        # Evolutionary metrics
        if phase_data.get('evolutionary_metrics'):
            report.append("Evolutionary Fitness Metrics:")
            report.append("Strategy               | Fitness | Relative | Raw Count | Mean Fitness")
            report.append("-" * 70)
            
            # Sort by fitness (descending)
            sorted_evo = sorted(phase_data['evolutionary_metrics'].items(), 
                               key=lambda x: x[1]['fitness'], reverse=True)
            
            for strategy, evo in sorted_evo:
                report.append(f"{strategy:22s} | {evo['fitness']:7.3f} | "
                            f"{evo['relative_fitness']:8.3f} | {evo['raw_count']:9.3f} | "
                            f"{evo['mean_fitness']:11.3f}")
            report.append("")
        
        report.append("")
    
    return "\n".join(report)

def analyze_all_experiments(results_dir: str = "results") -> Dict[str, Any]:
    """
    Analyze agent composition for all experiments in the results directory.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        Dictionary containing analysis for all experiments
    """
    all_experiments = {
        'analysis_timestamp': datetime.now().isoformat(),
        'results_directory': results_dir,
        'experiments': {},
        'summary_statistics': {
            'total_experiments': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_phases': 0
        }
    }
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return all_experiments
    
    # Find all experiment directories
    experiment_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.startswith('experiment_'):
            experiment_dirs.append(item_path)
    
    all_experiments['summary_statistics']['total_experiments'] = len(experiment_dirs)
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    for experiment_dir in sorted(experiment_dirs):
        experiment_name = os.path.basename(experiment_dir)
        print(f"Analyzing experiment: {experiment_name}")
        
        try:
            composition_data = assess_agent_composition_per_phase(experiment_dir)
            all_experiments['experiments'][experiment_name] = composition_data
            all_experiments['summary_statistics']['successful_analyses'] += 1
            all_experiments['summary_statistics']['total_phases'] += len(composition_data.get('phases', {}))
            
            # Save individual experiment analysis
            log_composition_to_experiment_folder(experiment_dir, composition_data)
            
            # Generate and save comprehensive summary report
            summary_report = generate_composition_summary_report(composition_data)
            report_file = os.path.join(experiment_dir, "agent_composition_summary.txt")
            with open(report_file, 'w') as f:
                f.write(summary_report)
            print(f"Summary report saved to: {report_file}")
            
        except Exception as e:
            print(f"Error analyzing experiment {experiment_name}: {e}")
            all_experiments['experiments'][experiment_name] = {'error': str(e)}
            all_experiments['summary_statistics']['failed_analyses'] += 1
    
    return all_experiments

def main():
    """Main function to run composition analysis on all experiments."""
    print("Starting comprehensive agent composition analysis for all experiments...")
    
    # Analyze all experiments
    all_results = analyze_all_experiments()
    
    # Save consolidated results
    consolidated_file = os.path.join("results", "all_experiments_composition_analysis.json")
    try:
        with open(consolidated_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Consolidated analysis saved to: {consolidated_file}")
    except Exception as e:
        print(f"Error saving consolidated analysis: {e}")
    
    # Print summary statistics
    stats = all_results['summary_statistics']
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total experiments found: {stats['total_experiments']}")
    print(f"Successfully analyzed: {stats['successful_analyses']}")
    print(f"Failed analyses: {stats['failed_analyses']}")
    print(f"Total phases analyzed: {stats['total_phases']}")
    
    if stats['successful_analyses'] > 0:
        avg_phases = stats['total_phases'] / stats['successful_analyses']
        print(f"Average phases per experiment: {avg_phases:.1f}")

if __name__ == "__main__":
    main()