#!/usr/bin/env python3
"""
Script to evaluate LLM experiments by analyzing reasoning content and move validation.
Usage: python evaluate_experiments.py <experiment_number>
Example: python evaluate_experiments.py 20250808_124710
"""

import argparse
import pandas as pd
import os
import glob
import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict, Counter


def find_experiment_directory(experiment_number: str) -> str:
    """Find the experiment directory matching the given number."""
    results_dir = "results"
    pattern = f"*{experiment_number}*"
    matches = glob.glob(os.path.join(results_dir, pattern))
    
    if not matches:
        raise FileNotFoundError(f"No experiment directory found for: {experiment_number}")
    
    if len(matches) > 1:
        print(f"Warning: Multiple matches found for {experiment_number}:")
        for match in matches:
            print(f"  - {match}")
        print(f"Using: {matches[0]}")
    
    return matches[0]


def is_valid_move(move: str) -> bool:
    """Check if a move is valid (C or D)."""
    return move in ['C', 'D']


def has_reasoning(reasoning_text: str) -> bool:
    """Check if reasoning text contains actual reasoning content."""
    if pd.isna(reasoning_text) or reasoning_text == "":
        return False
    
    # Remove whitespace and check if there's actual content
    cleaned = reasoning_text.strip()
    if len(cleaned) == 0:
        return False
    
    # Additional checks for minimal reasoning content
    # Consider it has reasoning if it's longer than a few characters
    return len(cleaned) > 10


def identify_llm_agents(agent_name: str) -> str:
    """Identify if an agent is an LLM and return the LLM type."""
    llm_patterns = {
        'GPT': r'GPT4?o?Mini?|GPT-?4',
        'Claude': r'Claude[\d\.-]+',
        'Gemini': r'Gemini[\d\.]+',
        'Mistral': r'Mistral|Ministral'
    }
    
    for llm_type, pattern in llm_patterns.items():
        if re.search(pattern, agent_name, re.IGNORECASE):
            return llm_type
    
    return None


def setup_logging(experiment_number: str, experiment_dir: str) -> str:
    """Set up logging to file and return the log file path."""
    log_filename = f"experiment_evaluation_{experiment_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(experiment_dir, log_filename)
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
    return log_filepath


def log_and_print(message: str):
    """Log message and print to console."""
    logging.info(message)


def analyze_csv_file(filepath: str) -> Dict:
    """Analyze a single CSV file and return statistics."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        log_and_print(f"Error reading {filepath}: {e}")
        return {}
    
    results = {}
    
    # Check required columns
    required_cols = ['agent1', 'agent2', 'agent1_move', 'agent2_move', 
                     'agent1_reasoning', 'agent2_reasoning']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log_and_print(f"Warning: Missing columns in {filepath}: {missing_cols}")
        return {}
    
    total_moves = 0
    valid_moves = 0
    llm_data = {}
    agent_stats = defaultdict(lambda: {'matches': 0, 'total_moves': 0, 'valid_moves': 0, 'has_reasoning': 0, 'reasoning_opportunities': 0, 'cooperate_moves': 0, 'defect_moves': 0})
    unique_agents = set()
    match_counts = Counter()
    
    for _, row in df.iterrows():
        agent1_name = str(row['agent1'])
        agent2_name = str(row['agent2'])
        
        # Track unique agents
        unique_agents.add(agent1_name)
        unique_agents.add(agent2_name)
        
        # Count matches per agent
        match_counts[agent1_name] += 1
        match_counts[agent2_name] += 1
        
        # Analyze agent1
        agent1_llm = identify_llm_agents(agent1_name)
        agent_stats[agent1_name]['total_moves'] += 1
        
        if is_valid_move(str(row['agent1_move'])):
            agent_stats[agent1_name]['valid_moves'] += 1
            # Track C/D moves
            if str(row['agent1_move']).upper() == 'C':
                agent_stats[agent1_name]['cooperate_moves'] += 1
            elif str(row['agent1_move']).upper() == 'D':
                agent_stats[agent1_name]['defect_moves'] += 1
        
        # Check if agent has reasoning opportunity (is LLM)
        if agent1_llm:
            agent_stats[agent1_name]['reasoning_opportunities'] += 1
            if has_reasoning(str(row['agent1_reasoning'])):
                agent_stats[agent1_name]['has_reasoning'] += 1
        
        if agent1_llm:
            if agent1_llm not in llm_data:
                llm_data[agent1_llm] = {'total_moves': 0, 'valid_moves': 0, 
                                       'has_reasoning': 0, 'total_reasoning_opportunities': 0}
            
            llm_data[agent1_llm]['total_moves'] += 1
            llm_data[agent1_llm]['total_reasoning_opportunities'] += 1
            
            if is_valid_move(str(row['agent1_move'])):
                llm_data[agent1_llm]['valid_moves'] += 1
            
            if has_reasoning(str(row['agent1_reasoning'])):
                llm_data[agent1_llm]['has_reasoning'] += 1
        
        # Analyze agent2
        agent2_llm = identify_llm_agents(agent2_name)
        agent_stats[agent2_name]['total_moves'] += 1
        
        if is_valid_move(str(row['agent2_move'])):
            agent_stats[agent2_name]['valid_moves'] += 1
            # Track C/D moves
            if str(row['agent2_move']).upper() == 'C':
                agent_stats[agent2_name]['cooperate_moves'] += 1
            elif str(row['agent2_move']).upper() == 'D':
                agent_stats[agent2_name]['defect_moves'] += 1
        
        # Check if agent has reasoning opportunity (is LLM)
        if agent2_llm:
            agent_stats[agent2_name]['reasoning_opportunities'] += 1
            if has_reasoning(str(row['agent2_reasoning'])):
                agent_stats[agent2_name]['has_reasoning'] += 1
        
        if agent2_llm:
            if agent2_llm not in llm_data:
                llm_data[agent2_llm] = {'total_moves': 0, 'valid_moves': 0, 
                                       'has_reasoning': 0, 'total_reasoning_opportunities': 0}
            
            llm_data[agent2_llm]['total_moves'] += 1
            llm_data[agent2_llm]['total_reasoning_opportunities'] += 1
            
            if is_valid_move(str(row['agent2_move'])):
                llm_data[agent2_llm]['valid_moves'] += 1
            
            if has_reasoning(str(row['agent2_reasoning'])):
                llm_data[agent2_llm]['has_reasoning'] += 1
        
        # Overall statistics
        total_moves += 2  # Two agents per row
        if is_valid_move(str(row['agent1_move'])):
            valid_moves += 1
        if is_valid_move(str(row['agent2_move'])):
            valid_moves += 1
    
    results['file'] = os.path.basename(filepath)
    results['total_moves'] = total_moves
    results['valid_moves'] = valid_moves
    results['valid_move_percentage'] = (valid_moves / total_moves * 100) if total_moves > 0 else 0
    results['llm_data'] = llm_data
    results['unique_agents'] = unique_agents
    results['agent_stats'] = dict(agent_stats)
    results['match_counts'] = dict(match_counts)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM experiments')
    parser.add_argument('experiment_number', help='Experiment number (e.g., 20250808_124710)')
    args = parser.parse_args()
    
    try:
        experiment_dir = find_experiment_directory(args.experiment_number)
        
        # Set up logging in the experiment directory
        log_file = setup_logging(args.experiment_number, experiment_dir)
        
        log_and_print(f"Analyzing experiment: {experiment_dir}")
        log_and_print(f"Log file: {log_file}")
        log_and_print("=" * 60)
        
        # Find all CSV files in the experiment directory
        csv_files = glob.glob(os.path.join(experiment_dir, "*.csv"))
        
        if not csv_files:
            log_and_print(f"No CSV files found in {experiment_dir}")
            return
        
        overall_stats = {}
        file_results = []
        all_unique_agents = set()
        all_agent_stats = defaultdict(lambda: {'matches': 0, 'total_moves': 0, 'valid_moves': 0, 'has_reasoning': 0, 'reasoning_opportunities': 0, 'cooperate_moves': 0, 'defect_moves': 0})
        all_match_counts = Counter()
        
        for csv_file in sorted(csv_files):
            log_and_print(f"\nAnalyzing: {os.path.basename(csv_file)}")
            result = analyze_csv_file(csv_file)
            
            if result:
                file_results.append(result)
                
                # Add to overall tracking
                all_unique_agents.update(result['unique_agents'])
                for agent, stats in result['agent_stats'].items():
                    all_agent_stats[agent]['total_moves'] += stats['total_moves']
                    all_agent_stats[agent]['valid_moves'] += stats['valid_moves']
                    all_agent_stats[agent]['has_reasoning'] += stats['has_reasoning']
                    all_agent_stats[agent]['reasoning_opportunities'] += stats['reasoning_opportunities']
                    all_agent_stats[agent]['cooperate_moves'] += stats['cooperate_moves']
                    all_agent_stats[agent]['defect_moves'] += stats['defect_moves']
                
                for agent, count in result['match_counts'].items():
                    all_match_counts[agent] += count
                
                # Log file-specific results
                log_and_print(f"  Total moves: {result['total_moves']}")
                log_and_print(f"  Valid moves: {result['valid_moves']} ({result['valid_move_percentage']:.1f}%)")
                log_and_print(f"  Unique agents in file: {len(result['unique_agents'])}")
                
                if result['llm_data']:
                    log_and_print("  LLM Performance:")
                    for llm_type, data in result['llm_data'].items():
                        move_pct = (data['valid_moves'] / data['total_moves'] * 100) if data['total_moves'] > 0 else 0
                        reasoning_pct = (data['has_reasoning'] / data['total_reasoning_opportunities'] * 100) if data['total_reasoning_opportunities'] > 0 else 0
                        log_and_print(f"    {llm_type}:")
                        log_and_print(f"      Valid moves: {data['valid_moves']}/{data['total_moves']} ({move_pct:.1f}%)")
                        log_and_print(f"      Has reasoning: {data['has_reasoning']}/{data['total_reasoning_opportunities']} ({reasoning_pct:.1f}%)")
                
                # Accumulate overall statistics
                for llm_type, data in result['llm_data'].items():
                    if llm_type not in overall_stats:
                        overall_stats[llm_type] = {'total_moves': 0, 'valid_moves': 0, 
                                                  'has_reasoning': 0, 'total_reasoning_opportunities': 0}
                    
                    overall_stats[llm_type]['total_moves'] += data['total_moves']
                    overall_stats[llm_type]['valid_moves'] += data['valid_moves']
                    overall_stats[llm_type]['has_reasoning'] += data['has_reasoning']
                    overall_stats[llm_type]['total_reasoning_opportunities'] += data['total_reasoning_opportunities']
        
        # Log overall summary
        log_and_print("\n" + "=" * 60)
        log_and_print("EXPERIMENT SUMMARY")
        log_and_print("=" * 60)
        
        log_and_print(f"\nTotal unique agents: {len(all_unique_agents)}")
        
        # Log agent specifications and match counts
        log_and_print("\nAGENT SPECIFICATIONS AND PERFORMANCE:")
        log_and_print("-" * 60)
        
        # Sort agents by type (LLM vs non-LLM) and then alphabetically
        sorted_agents = sorted(all_unique_agents, key=lambda x: (identify_llm_agents(x) is None, x))
        
        for agent in sorted_agents:
            stats = all_agent_stats[agent]
            matches = all_match_counts[agent]
            llm_type = identify_llm_agents(agent)
            
            agent_type = f" ({llm_type} LLM)" if llm_type else " (Non-LLM)"
            move_pct = (stats['valid_moves'] / stats['total_moves'] * 100) if stats['total_moves'] > 0 else 0
            
            log_and_print(f"\n{agent}{agent_type}:")
            log_and_print(f"  Matches participated: {matches}")
            log_and_print(f"  Total moves: {stats['total_moves']}")
            log_and_print(f"  Valid moves: {stats['valid_moves']} ({move_pct:.1f}%)")
            
            # Log C/D move breakdown
            if stats['total_moves'] > 0:
                cooperate_pct = (stats['cooperate_moves'] / stats['total_moves'] * 100)
                defect_pct = (stats['defect_moves'] / stats['total_moves'] * 100)
                log_and_print(f"  Cooperate (C): {stats['cooperate_moves']} ({cooperate_pct:.1f}%)")
                log_and_print(f"  Defect (D): {stats['defect_moves']} ({defect_pct:.1f}%)")
            
            if llm_type and stats['reasoning_opportunities'] > 0:
                reasoning_pct = (stats['has_reasoning'] / stats['reasoning_opportunities'] * 100)
                log_and_print(f"  Contains reasoning: {stats['has_reasoning']}/{stats['reasoning_opportunities']} ({reasoning_pct:.1f}%)")
        
        # Log LLM summary
        log_and_print("\n" + "=" * 60)
        log_and_print("LLM TYPE SUMMARY")
        log_and_print("=" * 60)
        
        if overall_stats:
            for llm_type, data in overall_stats.items():
                move_pct = (data['valid_moves'] / data['total_moves'] * 100) if data['total_moves'] > 0 else 0
                reasoning_pct = (data['has_reasoning'] / data['total_reasoning_opportunities'] * 100) if data['total_reasoning_opportunities'] > 0 else 0
                
                log_and_print(f"\n{llm_type}:")
                log_and_print(f"  Total moves: {data['total_moves']}")
                log_and_print(f"  Valid moves: {data['valid_moves']} ({move_pct:.1f}%)")
                log_and_print(f"  Contains reasoning: {data['has_reasoning']}/{data['total_reasoning_opportunities']} ({reasoning_pct:.1f}%)")
        else:
            log_and_print("No LLM agents found in the experiment data.")
        
        log_and_print(f"\n" + "=" * 60)
        log_and_print(f"Analysis complete. Results saved to: {log_file}")
        log_and_print("=" * 60)
    
    except Exception as e:
        log_and_print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())