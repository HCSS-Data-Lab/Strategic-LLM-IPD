#!/usr/bin/env python3
"""
Main experiment runner for IPD research
Implements all experiments from Payne & Alloui-Cros (2025) plus extensions
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ipd_suite import (
    # Classical strategies
    TitForTat, GrimTrigger, WinStayLoseShift, Random,
    GenerousTitForTat, SuspiciousTitForTat, Prober, Gradual, Alternator, Bayesian,
    
    # Behavioral strategies  
    ForgivingGrimTrigger, Detective, SoftGrudger,
    
    # Adaptive strategies
    QLearningAgent, ThompsonSampling, GradientMetaLearner,
    
    # LLM agents
    GPT4Agent, ClaudeAgent, MistralAgent, GeminiAgent,
    
    # Tournament and analysis
    Tournament, LLMShowdown, MatchHistoryManager,
    generate_comprehensive_report
)

from ipd_suite.utils import (
    load_env_vars, validate_api_keys, create_experiment_config,
    save_experiment_metadata, estimate_api_costs, Timer,
    create_progress_file, update_progress
)

# Import evolution functions
from collections import defaultdict
import glob


def save_population_checkpoint(experiment_dir: str, shadow: float, phase: int, 
                              current_population: dict, population_history: list,
                              initial_agents: list, experiment_config: dict):
    """Save population checkpoint after each phase for resume functionality"""
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'shadow_condition': shadow,
        'completed_phases': phase,
        'current_population': current_population,
        'population_history': population_history,
        'initial_agents_config': [
            {
                'name': agent.name,
                'class': agent.__class__.__name__,
                'model': getattr(agent, 'model', None),
                'temperature': getattr(agent, 'temperature', None),
                'is_llm': any(x in agent.name for x in ['GPT4', 'GPT5', 'Claude', 'Mistral', 'Gemini'])
            }
            for agent in initial_agents
        ],
        'experiment_config': experiment_config
    }
    
    checkpoint_file = os.path.join(experiment_dir, 
                                   f"checkpoint_shadow{int(shadow*100)}_phase{phase}.json")
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"📁 Saved checkpoint: {checkpoint_file}")
    return checkpoint_file


def load_population_checkpoint(checkpoint_file: str):
    """Load population checkpoint for resuming experiments"""
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
    
    print(f"📂 Loaded checkpoint: {checkpoint_file}")
    print(f"   Shadow condition: {checkpoint_data['shadow_condition']}")
    print(f"   Completed phases: {checkpoint_data['completed_phases']}")
    print(f"   Current population: {checkpoint_data['current_population']}")
    
    return checkpoint_data


def find_latest_checkpoint(experiment_dir: str, shadow: float):
    """Find the latest checkpoint file for a given shadow condition"""
    pattern = os.path.join(experiment_dir, f"checkpoint_shadow{int(shadow*100)}_phase*.json")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    
    # Extract phase numbers and find the latest
    phase_files = []
    for file in checkpoint_files:
        basename = os.path.basename(file)
        # Extract phase number from filename like "checkpoint_shadow75_phase3.json"
        parts = basename.split('_')
        for part in parts:
            if part.startswith('phase') and part.endswith('.json'):
                try:
                    phase_num = int(part.replace('phase', '').replace('.json', ''))
                    phase_files.append((phase_num, file))
                except ValueError:
                    continue
    
    if not phase_files:
        return None
    
    # Return the file with the highest phase number
    latest_phase, latest_file = max(phase_files, key=lambda x: x[0])
    return latest_file


def find_checkpoint_with_fallback(experiment_dir: str, shadow: float, target_phase: int, 
                                  initial_agents: list):
    """
    Find the best available checkpoint for resuming, with fallback logic.
    
    Fallback chain:
    1. Try to find checkpoint for (target_phase - 1) - the completed phase before target
    2. If missing, try previous phases in descending order
    3. If no checkpoints exist, return initial population
    
    Returns:
        tuple: (start_phase, current_population, population_history, found_checkpoint_path)
    """
    # Create initial population as ultimate fallback
    initial_population = {}
    for agent in initial_agents:
        agent_name = agent.name
        if agent_name in initial_population:
            initial_population[agent_name] += 1
        else:
            initial_population[agent_name] = 1
    
    # If target_phase is 0 (start from beginning), return initial population
    if target_phase == 0:
        return 0, initial_population.copy(), [initial_population.copy()], None
    
    # Try to find the most recent checkpoint before target_phase
    for phase in range(target_phase - 1, -1, -1):  # target_phase-1 down to 0
        checkpoint_file = os.path.join(experiment_dir, 
                                     f"checkpoint_shadow{int(shadow*100)}_phase{phase+1}.json")
        
        if os.path.exists(checkpoint_file):
            try:
                print(f"🔍 Attempting to load checkpoint: {checkpoint_file}")
                checkpoint_data = load_population_checkpoint(checkpoint_file)
                
                # Validate checkpoint data
                if ('current_population' in checkpoint_data and 
                    'population_history' in checkpoint_data and
                    'completed_phases' in checkpoint_data):
                    
                    # The checkpoint represents the state AFTER phase (phase+1) was completed
                    # So we resume from phase (phase+2) = checkpoint_data['completed_phases'] + 1
                    resume_phase = checkpoint_data['completed_phases']
                    
                    print(f"✅ Successfully loaded checkpoint from phase {resume_phase}")
                    print(f"🔄 Resuming from phase {resume_phase + 1}")
                    
                    # Validate and fix population if needed
                    expected_total = len(initial_agents)
                    validated_population = validate_and_fix_population(
                        checkpoint_data['current_population'], 
                        initial_agents, 
                        expected_total
                    )
                    
                    return (resume_phase, 
                            validated_population,
                            checkpoint_data['population_history'],
                            checkpoint_file)
                else:
                    print(f"⚠️  Invalid checkpoint data in {checkpoint_file}, trying previous phase...")
                    continue
                    
            except Exception as e:
                print(f"⚠️  Could not load checkpoint {checkpoint_file}: {e}")
                print(f"🔄 Trying previous phase checkpoint...")
                continue
    
    # No valid checkpoints found, return initial population
    print(f"📋 No valid checkpoints found, starting from initial population")
    return 0, initial_population.copy(), [initial_population.copy()], None


def find_completed_phases_from_csv(experiment_dir: str, shadow: float):
    """
    Find completed phases by looking for CSV result files.
    Returns the highest phase number that has a corresponding CSV file.
    """
    pattern = os.path.join(experiment_dir, f"evolutionary_shadow{int(shadow*100)}_phase*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        return 0  # No phases completed
    
    # Extract phase numbers
    completed_phases = []
    for file in csv_files:
        basename = os.path.basename(file)
        # Extract phase number from filename like "evolutionary_shadow75_phase3.csv"
        parts = basename.split('_')
        for part in parts:
            if part.startswith('phase') and part.endswith('.csv'):
                try:
                    phase_num = int(part.replace('phase', '').replace('.csv', ''))
                    completed_phases.append(phase_num)
                except ValueError:
                    continue
    
    if not completed_phases:
        return 0
    
    return max(completed_phases)


def reconstruct_initial_population_from_config(experiment_dir: str, api_keys: dict, 
                                              temperature_settings: dict, shadow: float):
    """
    Reconstruct the initial population from the experiment configuration file.
    This recreates the population as it was at the start of the experiment.
    """
    config_file = os.path.join(experiment_dir, "config.json")
    
    if not os.path.exists(config_file):
        print(f"⚠️  No config.json found in {experiment_dir}")
        return None, None
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"📋 Reconstructing initial population from config.json")
        print(f"   Original experiment timestamp: {config.get('timestamp', 'unknown')}")
        
        # Use the temperature settings from the config if they exist
        config_temp_settings = config.get('temperature_settings', temperature_settings)
        
        # Create agents exactly as they were in the original experiment
        initial_agents = create_agents(api_keys, config_temp_settings, shadow, include_classical=True)
        
        # Convert to population dictionary
        initial_population = {}
        for agent in initial_agents:
            agent_name = agent.name
            if agent_name in initial_population:
                initial_population[agent_name] += 1
            else:
                initial_population[agent_name] = 1
        
        print(f"📊 Reconstructed initial population with {len(initial_agents)} agents:")
        agent_counts = {}
        for agent in initial_agents:
            agent_type = "LLM" if any(x in agent.name for x in ['GPT', 'Claude', 'Mistral', 'Gemini']) else "Classical"
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        
        for agent_type, count in agent_counts.items():
            print(f"   - {count} {agent_type} agents")
        
        return initial_population, initial_agents
        
    except Exception as e:
        print(f"⚠️  Error reading config file: {e}")
        return None, None


def find_best_resume_point_with_csv_fallback(experiment_dir: str, shadow: float, 
                                            target_phases: int, api_keys: dict,
                                            temperature_settings: dict):
    """
    Enhanced checkpoint finder that falls back to CSV detection when checkpoints are missing.
    
    Fallback order:
    1. Try checkpoint files (newest to oldest)
    2. If no checkpoints, detect completed phases from CSV files
    3. If CSV files found, reconstruct initial population and resume from next phase
    4. If nothing found, start from beginning with initial population
    
    Returns:
        tuple: (start_phase, current_population, population_history, source_info)
    """
    # First try the normal checkpoint fallback
    initial_agents = create_agents(api_keys, temperature_settings, shadow, include_classical=True)
    start_phase, current_population, population_history, checkpoint_file = find_checkpoint_with_fallback(
        experiment_dir, shadow, target_phases, initial_agents
    )
    
    if checkpoint_file:
        return start_phase, current_population, population_history, f"checkpoint:{os.path.basename(checkpoint_file)}"
    
    # If no checkpoints found, try CSV detection
    print(f"🔍 No checkpoint files found, looking for completed CSV files...")
    completed_phases = find_completed_phases_from_csv(experiment_dir, shadow)
    
    if completed_phases > 0:
        print(f"📄 Found CSV files for phases 1-{completed_phases}")
        
        if completed_phases >= target_phases:
            print(f"✅ All {target_phases} phases already completed (CSV files exist)")
            # Reconstruct final population (same as initial for now)
            reconstructed_pop, reconstructed_agents = reconstruct_initial_population_from_config(
                experiment_dir, api_keys, temperature_settings, shadow
            )
            if reconstructed_pop:
                return target_phases, reconstructed_pop, [reconstructed_pop], f"csv:completed_all_{completed_phases}_phases"
        else:
            print(f"🔄 Resuming from phase {completed_phases + 1} (CSV-based detection)")
            print(f"📋 Will use initial population distribution for phase {completed_phases + 1}")
            
            # Reconstruct initial population to continue from
            reconstructed_pop, reconstructed_agents = reconstruct_initial_population_from_config(
                experiment_dir, api_keys, temperature_settings, shadow
            )
            
            if reconstructed_pop:
                # Create a population history with just the initial population
                # We don't have the evolution history, so we start fresh with initial population
                return completed_phases, reconstructed_pop, [reconstructed_pop], f"csv:resume_from_phase_{completed_phases + 1}"
    
    # Final fallback - no CSV files or checkpoints found
    print(f"📋 No CSV files or checkpoints found, starting fresh")
    initial_population = {}
    for agent in initial_agents:
        agent_name = agent.name
        if agent_name in initial_population:
            initial_population[agent_name] += 1
        else:
            initial_population[agent_name] = 1
    
    return 0, initial_population, [initial_population], "fresh_start"


def validate_and_fix_population(current_population: dict, initial_agents: list, 
                               total_expected: int = None):
    """
    Validate population integrity and fix if needed.
    Ensures that:
    1. All strategies in population exist in initial_agents
    2. Total population count matches expected (if provided)
    3. No negative or zero counts
    """
    # Get list of valid agent names from initial agents
    valid_agent_names = {agent.name for agent in initial_agents}
    
    # Remove invalid agents and fix counts
    cleaned_population = {}
    for agent_name, count in current_population.items():
        if agent_name in valid_agent_names and count > 0:
            cleaned_population[agent_name] = count
        elif agent_name not in valid_agent_names:
            print(f"⚠️  Removing invalid agent '{agent_name}' from population")
        elif count <= 0:
            print(f"⚠️  Removing agent '{agent_name}' with invalid count {count}")
    
    # If total count is specified, adjust population to match
    if total_expected:
        current_total = sum(cleaned_population.values())
        
        if current_total != total_expected:
            print(f"⚠️  Population total mismatch: expected {total_expected}, found {current_total}")
            
            if current_total == 0:
                # If population is empty, recreate initial population
                print(f"🔄 Population is empty, recreating initial population")
                for agent in initial_agents:
                    if agent.name in cleaned_population:
                        cleaned_population[agent.name] += 1
                    else:
                        cleaned_population[agent.name] = 1
            elif current_total < total_expected:
                # Add missing agents (distribute evenly among existing strategies)
                deficit = total_expected - current_total
                strategies = list(cleaned_population.keys())
                if strategies:
                    per_strategy = deficit // len(strategies)
                    remainder = deficit % len(strategies)
                    
                    for i, strategy in enumerate(strategies):
                        cleaned_population[strategy] += per_strategy
                        if i < remainder:  # Distribute remainder
                            cleaned_population[strategy] += 1
                    
                    print(f"🔧 Added {deficit} agents to reach target population of {total_expected}")
            elif current_total > total_expected:
                # Remove excess agents (prefer removing from strategies with highest counts)
                excess = current_total - total_expected
                while excess > 0 and cleaned_population:
                    # Find strategy with highest count
                    max_strategy = max(cleaned_population.keys(), 
                                     key=lambda x: cleaned_population[x])
                    if cleaned_population[max_strategy] > 1:
                        cleaned_population[max_strategy] -= 1
                        excess -= 1
                    else:
                        # If all strategies have count 1, remove entire strategy
                        del cleaned_population[max_strategy]
                        excess -= 1
                
                print(f"🔧 Removed {current_total - sum(cleaned_population.values())} agents to reach target population")
    
    return cleaned_population


def detect_resume_opportunity(output_dir: str, shadow_conditions: list, n_phases: int):
    """Detect if there are any incomplete experiments that can be resumed"""
    resume_options = []
    
    # Look for existing experiment directories
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            experiment_path = os.path.join(output_dir, item)
            if os.path.isdir(experiment_path) and item.startswith('experiment_'):
                
                # Check for checkpoints in this experiment
                for shadow in shadow_conditions:
                    latest_checkpoint = find_latest_checkpoint(experiment_path, shadow)
                    if latest_checkpoint:
                        try:
                            checkpoint_data = load_population_checkpoint(latest_checkpoint)
                            completed_phases = checkpoint_data['completed_phases']
                            
                            # Check if experiment is incomplete
                            if completed_phases < n_phases:
                                resume_options.append({
                                    'experiment_dir': experiment_path,
                                    'shadow_condition': shadow,
                                    'completed_phases': completed_phases,
                                    'remaining_phases': n_phases - completed_phases,
                                    'checkpoint_file': latest_checkpoint,
                                    'experiment_name': item
                                })
                        except Exception as e:
                            print(f"Warning: Could not load checkpoint {latest_checkpoint}: {e}")
                            continue
    
    return resume_options


def create_agents(api_keys: Dict[str, str], 
                 temperature_settings: Dict[str, List[float]],
                 termination_prob: float,
                 include_classical: bool = True,
                 history_manager: MatchHistoryManager = None) -> List:
    """Create all agent instances for experiments"""
    agents = []
    
    # Agents 1-16: Classical, behavioral, and adaptive strategies
    if include_classical:
        agents.extend([
            # 1-10: Classical strategies
            TitForTat("TitForTat"),                        # 1
            GrimTrigger("GrimTrigger"),                    # 2  
            WinStayLoseShift("WinStayLoseShift"),          # 3
            GenerousTitForTat("GenerousTitForTat"),        # 4
            SuspiciousTitForTat("SuspiciousTitForTat"),    # 5
            Prober("Prober"),                              # 6
            Random("Random"),                              # 7
            Gradual("Gradual"),                            # 8
            Alternator("Alternator"),                      # 9
            Bayesian("Bayesian"),                          # 10
            
            # 11-13: Behavioral strategies
            ForgivingGrimTrigger("ForgivingGrimTrigger"),  # 11
            Detective("Detective"),                        # 12
            SoftGrudger("SoftGrudger"),                    # 13
            
            # 14-16: Adaptive learning strategies
            QLearningAgent("QLearning"),                   # 14
            ThompsonSampling("ThompsonSampling"),          # 15
            GradientMetaLearner("GradientMetaLearner")     # 16
        ])
    
    # Agents 17-28: LLM agents with 3 temperatures each (4 providers × 3 temperatures = 12 agents)
    # OpenAI agents (Different Models) - Agents 17-19
    if api_keys.get('OPENAI_API_KEY'):
        openai_models = ['gpt-5-mini', 'gpt-5-nano', 'gpt-4.1-mini']
        for i, model in enumerate(openai_models, 17):
            model_name = model.replace('-', '').replace('gpt', 'GPT')
            agent = GPT4Agent(f"{model_name}_T1", 
                            api_keys['OPENAI_API_KEY'],
                            model=model,
                            temperature=1.0,
                            termination_prob=termination_prob,
                            match_history=history_manager.get_history_for_agent(
                                GPT4Agent("temp", api_keys['OPENAI_API_KEY'], model=model, temperature=1.0)
                            ) if history_manager else None)
            agents.append(agent)
    
    # Claude agents (Anthropic) - Agents 20-22
    if api_keys.get('ANTHROPIC_API_KEY') and 'anthropic' in temperature_settings:
        for i, temp in enumerate(temperature_settings['anthropic'][:3], 20):  # Ensure exactly 3 temperatures
            temp_suffix = f"_T{str(temp).replace('.', '')}"
            agent = ClaudeAgent(f"Claude4-Sonnet{temp_suffix}",
                              api_keys['ANTHROPIC_API_KEY'],
                              model="claude-sonnet-4-20250514", #claude-3-5-haiku-latest
                              temperature=temp,
                              termination_prob=termination_prob,
                              match_history=history_manager.get_history_for_agent(
                                  ClaudeAgent("temp", api_keys['ANTHROPIC_API_KEY'], temperature=temp)
                              ) if history_manager else None)
            agents.append(agent)
    
    # Mistral agents - Agents 23-25
    if api_keys.get('MISTRAL_API_KEY') and 'mistral' in temperature_settings:
        for i, temp in enumerate(temperature_settings['mistral'][:3], 23):  # Ensure exactly 3 temperatures
            temp_suffix = f"_T{str(temp).replace('.', '')}"
            agent = MistralAgent(f"Mistral-Medium{temp_suffix}",
                               api_keys['MISTRAL_API_KEY'],
                               model="mistral-medium-2508",
                               temperature=temp,
                               termination_prob=termination_prob,
                               match_history=history_manager.get_history_for_agent(
                                   MistralAgent("temp", api_keys['MISTRAL_API_KEY'], temperature=temp)
                               ) if history_manager else None)
            agents.append(agent)
    
    # Gemini agents - Agents 26-28
    if api_keys.get('GOOGLE_API_KEY') and 'gemini' in temperature_settings:
        for i, temp in enumerate(temperature_settings['gemini'][:3], 26):  # Ensure exactly 3 temperatures
            temp_suffix = f"_T{str(temp).replace('.', '')}"
            agent = GeminiAgent(f"Gemini20Flash{temp_suffix}",
                              api_keys['GOOGLE_API_KEY'],
                              model="gemini-2.0-flash",
                              temperature=temp,
                              termination_prob=termination_prob,
                              match_history=history_manager.get_history_for_agent(
                                  GeminiAgent("temp", api_keys['GOOGLE_API_KEY'], temperature=temp)
                              ) if history_manager else None)
            agents.append(agent)
    
    print(f"Created {len(agents)} agents")
    return agents


def convert_tournament_result_to_phase_result(tournament_result):
    """Convert TournamentResult to phase_result format expected by evolve_population"""
    # Get agent stats from tournament result
    agent_stats = tournament_result.agent_stats
    
    # Convert to strategy stats by grouping agents by strategy name
    strategy_stats = {}
    
    for agent_name, stats in agent_stats.items():
        # Extract strategy name (remove instance numbers like _p1i1)
        if any(x in agent_name for x in ['GPT4', 'Claude', 'Mistral', 'Gemini']):
            # For LLM agents, remove the phase/instance suffix (_p1i1) but keep temperature
            # Examples: 'Gemini25Flash_T07_p1i1' -> 'Gemini25Flash_T07'
            parts = agent_name.split('_')
            # Keep all parts except the last one if it looks like phase/instance (pXiX)
            if len(parts) > 1 and parts[-1].startswith('p') and 'i' in parts[-1]:
                strategy = '_'.join(parts[:-1])
            else:
                strategy = agent_name
        else:
            # For classical agents, extract base strategy name (remove instance IDs like _p1i1)
            parts = agent_name.split('_')
            if len(parts) > 1 and parts[-1].startswith('p') and 'i' in parts[-1]:
                strategy = '_'.join(parts[:-1])
            else:
                strategy = parts[0]
        
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                'total_score': 0,
                'matches_played': 0,
                'total_rounds': 0
            }
        
        # Aggregate stats for this strategy
        strategy_stats[strategy]['total_score'] += stats['total_score']
        strategy_stats[strategy]['matches_played'] += stats['matches_played']
        strategy_stats[strategy]['total_rounds'] += stats['total_moves']
    
    # Calculate average scores for each strategy
    for strategy, stats in strategy_stats.items():
        if stats['matches_played'] > 0:
            stats['avg_score_per_match'] = stats['total_score'] / stats['matches_played']
            
            if stats['total_rounds'] > 0:
                stats['avg_score_per_move'] = stats['total_score'] / stats['total_rounds']
            else:
                stats['avg_score_per_move'] = 0
                
            stats['avg_score_per_round'] = stats['avg_score_per_match']  # For compatibility
        else:
            stats['avg_score_per_match'] = 0
            stats['avg_score_per_move'] = 0
            stats['avg_score_per_round'] = 0
    
    return {
        "strategy_stats": strategy_stats,
        "matches": []  # Not needed for evolution but included for completeness
    }


def evolve_population(current_population, phase_result, min_count=0, verbose=True):
    """
    Update the population based on performance in the last phase.
    """
    # Extract average score per move for each strategy (true per-move performance)
    strategy_stats = phase_result["strategy_stats"]
    strategy_fitness = {
        strategy: stats["avg_score_per_move"] 
        for strategy, stats in strategy_stats.items()
    }
    
    # Print detailed fitness values
    if verbose:
        print("\nDetailed Strategy Fitness Calculation:")
        print("----------------------------------------")
        print("Strategy    | Score/Move | Score/Match | Total Score | Rounds | Matches")
        print("------------|------------|-------------|-------------|--------|--------")
        for strategy, stats in sorted(strategy_stats.items(), 
                                     key=lambda x: x[1]['avg_score_per_move'], 
                                     reverse=True):
            print(f"{strategy:12}|    {stats['avg_score_per_move']:.3f}    |    {stats['avg_score_per_match']:.3f}    | {stats['total_score']:.1f}     | {stats['total_rounds']}   | {stats['matches_played']}")
    
    # Calculate total fitness
    total_fitness = sum(strategy_fitness.values())
    if total_fitness == 0:  # Avoid division by zero
        print("Warning: Total fitness is zero. Using equal distribution.")
        new_population = {strategy: min_count for strategy in current_population}
        return new_population
    
    # Calculate new population sizes with enhanced selection pressure
    total_agents = sum(current_population.values())
    new_population = {}
    
    # Calculate mean fitness
    mean_fitness = total_fitness / len(strategy_fitness)
    
    # Print evolutionary calculations
    if verbose:
        print("\nEvolutionary Calculation:")
        print("-------------------------")
        print("Strategy    | Fitness | Relative | Raw Count | Final Count")
        print("------------|---------|----------|-----------|------------")
    
    for strategy, current_count in current_population.items():
        if strategy not in strategy_fitness:
            print(f"Warning: No fitness data for {strategy}, using minimum count.")
            new_population[strategy] = min_count
            continue
            
        # Enhanced fitness-proportional reproduction
        fitness = strategy_fitness[strategy]
        # Calculate relative fitness compared to mean (amplifies differences)
        relative_fitness = (fitness / mean_fitness) ** 2  # Square to amplify differences
        raw_count = (relative_fitness * current_count)
        # Use more aggressive rounding to increase selective pressure
        new_count = max(min_count, int(round(raw_count)))
        new_population[strategy] = new_count
        
        if verbose:
            print(f"{strategy:12}| {fitness:.3f}  | {relative_fitness:.3f}   | {raw_count:.2f}    | {new_count}")
    
    # Adjust to maintain total population size
    original_total = sum(new_population.values())
    adjustment_attempts = 0
    max_adjustments = 100  # Safety limit to prevent infinite loops
    
    while sum(new_population.values()) > total_agents and adjustment_attempts < max_adjustments:
        adjustment_attempts += 1
        # Find strategy with lowest fitness that has more than min_count
        adjustable = [s for s in new_population if new_population[s] > min_count]
        if not adjustable:
            # If all at minimum, reduce the one with the most counts
            strategy = max(new_population, key=new_population.get)
        else:
            # Otherwise reduce the one with lowest fitness
            strategy = min(adjustable, key=lambda s: strategy_fitness.get(s, 0))
        new_population[strategy] -= 1
        if verbose and adjustment_attempts <= 5:  # Only show first few adjustments
            print(f"Adjusting down: {strategy} (lowest adjustable fitness)")
    
    while sum(new_population.values()) < total_agents and adjustment_attempts < max_adjustments:
        adjustment_attempts += 1
        # Find strategy with highest fitness to increment
        strategy = max(strategy_fitness, key=strategy_fitness.get)
        if strategy in new_population:
            new_population[strategy] += 1
        else:
            # Reintroduce eliminated strategy
            new_population[strategy] = 1
        if verbose and adjustment_attempts <= 5:  # Only show first few adjustments
            print(f"Adjusting up: {strategy} (highest fitness)")
    
    # Verify all strategies have at least min_count
    for strategy in new_population:
        if new_population[strategy] < min_count:
            new_population[strategy] = min_count
            if verbose:
                print(f"Warning: Adjusted {strategy} to minimum count of {min_count}")
    
    if verbose and original_total != sum(new_population.values()):
        print(f"Population adjusted from {original_total} to {sum(new_population.values())} to maintain total of {total_agents} agents")
    
    # Special handling: if a strategy has 0 count, remove it entirely from the population
    # This ensures eliminated strategies really disappear
    new_population = {k: v for k, v in new_population.items() if v > 0}
    
    return new_population


def run_main_experiments(shadow_conditions: List[float] = [0.1, 0.25, 0.75],
                        temperature_settings: Dict[str, List[float]] = None,
                        n_tournaments: int = 5,
                        n_phases: int = 5,
                        output_dir: str = "results",
                        evolutionary: bool = False,
                        auto_confirm: bool = False,
                        resume_experiment: str = None,
                        max_concurrent: int = 50,
                        enable_opponent_tracking: bool = False):
    """Run the main experimental suite
    
    Args:
        shadow_conditions: List of termination probabilities to test
        temperature_settings: Temperature settings for each LLM provider
        n_tournaments: Number of tournaments per condition (standard mode only)
        n_phases: Number of evolutionary phases per condition (evolutionary mode only)
        output_dir: Directory to save results
        evolutionary: If True, use evolutionary mode where population changes
                     based on performance. If False, run repeated identical tournaments.
        auto_confirm: If True, automatically confirm all prompts (skip cost confirmation)
        resume_experiment: Path to experiment directory to resume from checkpoint
        max_concurrent: Maximum number of concurrent matches within each tournament
        enable_opponent_tracking: If True, LLM agents receive cross-references to previous encounters with the same opponent
    """
    print("="*60)
    print("IPD EXPERIMENT RUNNER")
    print("Based on Payne & Alloui-Cros (2025)")
    if evolutionary:
        print("Mode: EVOLUTIONARY (population evolves based on performance)")
    else:
        print("Mode: STANDARD (repeated identical tournaments)")
    
    # Handle resume functionality
    if resume_experiment:
        print(f"Mode: RESUME from {resume_experiment}")
    elif evolutionary and not resume_experiment:
        # Check for resume opportunities
        resume_options = detect_resume_opportunity(output_dir, shadow_conditions, n_phases)
        if resume_options:
            print(f"🔄 Found {len(resume_options)} incomplete experiment(s) that can be resumed:")
            for i, option in enumerate(resume_options, 1):
                print(f"   {i}. {option['experiment_name']}: Shadow {option['shadow_condition']*100}% - {option['completed_phases']}/{n_phases} phases complete")
            
            if not auto_confirm:
                response = input(f"\nResume incomplete experiment? (1-{len(resume_options)}/n): ")
                if response.isdigit() and 1 <= int(response) <= len(resume_options):
                    resume_experiment = resume_options[int(response)-1]['experiment_dir']
                    print(f"🔄 Resuming: {resume_experiment}")
                elif response.lower() != 'n':
                    print("Invalid selection. Starting new experiment.")
    
    print("="*60)
    
    # Default temperature settings - model-specific to respect API constraints
    # Note: OpenAI uses different models instead of temperature variations
    if temperature_settings is None:
        temperature_settings = {
            'anthropic': [0.2, 0.5, 0.8],  # Anthropic supports 0-1 range
            'mistral': [0.2, 0.7, 1.2],    # Mistral supports 0-1 range (capped at 1.0)
            'gemini': [0.2, 0.7, 1.2]      # Gemini supports 0-2 range
        }
    
    # Load API keys
    api_keys = load_env_vars()
    
    # Check which APIs are available
    available_apis = [k.replace('_API_KEY', '') for k, v in api_keys.items() if v]
    if not available_apis:
        print("ERROR: No API keys found! Please set up axelrod.env")
        return
    
    print(f"\nAvailable APIs: {', '.join(available_apis)}")
    
    # Map API key names to temperature setting keys
    api_to_temp_key = {
        'OPENAI': 'openai',
        'ANTHROPIC': 'anthropic', 
        'MISTRAL': 'mistral',
        'GOOGLE': 'gemini'  # Google API key maps to gemini temperature settings
    }
    
    # Calculate number of LLM agents based on available APIs and their settings
    n_llm_agents = 0
    for api in available_apis:
        temp_key = api_to_temp_key.get(api, api.lower())
        if api == 'OPENAI':
            n_llm_agents += 3  # Always 3 OpenAI models
        elif temp_key in temperature_settings:
            n_llm_agents += len(temperature_settings[temp_key])
    n_total_agents = n_llm_agents + 16  # 16 classical/behavioral/adaptive
    n_matches = n_total_agents * (n_total_agents - 1) // 2
    
    print(f"\nExperiment scale:")
    print(f"- Shadow conditions: {shadow_conditions}")
    print(f"- Model configurations:")
    # Show OpenAI models explicitly
    if 'OPENAI' in available_apis:
        print("  - OpenAI: 3 different models (gpt-5-mini, gpt-5-nano, gpt-4.1-mini) all at temperature 1.0")
    # Show other providers with their temperatures
    for api in available_apis:
        temp_key = api_to_temp_key.get(api, api.lower())
        if temp_key in temperature_settings and temp_key != 'openai':
            print(f"  - {temp_key.capitalize()}: {temperature_settings[temp_key]}")
    print(f"- Total agents: {n_total_agents}")
    print(f"- Matches per tournament: {n_matches}")
    print(f"- Tournaments per condition: {n_tournaments}")
    
    # Cost estimation
    avg_rounds = 1 / shadow_conditions[0]  # Expected rounds for first condition
    api_counts = {api: len(temperature_settings.get(api_to_temp_key.get(api, api.lower()), [])) 
                  for api in available_apis 
                  if api_to_temp_key.get(api, api.lower()) in temperature_settings}
    costs = estimate_api_costs(api_counts, int(avg_rounds), n_matches)
    
    total_cost = sum(costs.values()) * len(shadow_conditions) * n_tournaments
    print(f"\nEstimated total cost: ${total_cost:.2f}")
    
    # Auto-confirm for non-interactive mode, if AUTO_CONFIRM is set, or if --yes flag is used
    import sys
    if auto_confirm or not sys.stdin.isatty() or os.environ.get('AUTO_CONFIRM') == 'yes':
        print("\nAuto-confirming experiment start")
    else:
        response = input("\nProceed with experiments? (y/n): ")
        if response.lower() != 'y':
            print("Experiments cancelled.")
            return
    
    # Handle experiment directory creation or resumption
    if resume_experiment:
        experiment_dir = resume_experiment
        print(f"📁 Using existing experiment directory: {experiment_dir}")
        # Load existing config
        config_file = os.path.join(experiment_dir, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            # Create config if missing
            config = create_experiment_config(shadow_conditions, temperature_settings, 
                                            {api: "default" for api in available_apis})
            save_experiment_metadata(config, config_file)
        
        # Load existing progress file
        progress_file = os.path.join(experiment_dir, "progress.json")
        if not os.path.exists(progress_file):
            progress_file = create_progress_file(progress_file)
            update_progress(progress_file, {
                'total_conditions': len(shadow_conditions),
                'total_matches': n_matches * len(shadow_conditions) * n_tournaments
            })
    else:
        # Create new experiment directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save experiment configuration
        config = create_experiment_config(shadow_conditions, temperature_settings, 
                                         {api: "default" for api in available_apis})
        save_experiment_metadata(config, os.path.join(experiment_dir, "config.json"))
        
        # Create progress file
        progress_file = create_progress_file(os.path.join(experiment_dir, "progress.json"))
        update_progress(progress_file, {
            'total_conditions': len(shadow_conditions),
            'total_matches': n_matches * len(shadow_conditions) * n_tournaments
        })
    
    # Run experiments for each shadow condition
    all_results = {}
    all_agents_used = []  # Track all agents used across conditions for LLM showdown
    
    for i, shadow in enumerate(shadow_conditions):
        print(f"\n{'='*60}")
        print(f"SHADOW CONDITION: {shadow*100}% termination probability")
        print(f"{'='*60}")
        
        update_progress(progress_file, {
            'current_condition': f"shadow_{int(shadow*100)}",
            'completed_conditions': i
        })
        
        if evolutionary:
            # Create match history manager for tracking LLM agent histories across phases
            history_manager = MatchHistoryManager(enable_opponent_tracking=enable_opponent_tracking)
            history_file = os.path.join(experiment_dir, f"match_history_shadow{int(shadow*100)}.json")
            
            # Load existing match histories if resuming
            if resume_experiment and os.path.exists(history_file):
                history_manager.load_histories(history_file)
                print(f"📚 Loaded existing match histories for shadow {shadow*100}%")
            
            # Create initial agents first (needed for fallback logic)
            initial_agents = create_agents(api_keys, temperature_settings, shadow, include_classical=True, history_manager=history_manager)
            all_agents_used.extend(initial_agents)  # Track agents for LLM showdown
            
            # Use enhanced fallback system with CSV detection
            if resume_experiment:
                # Look for checkpoints first, then fall back to CSV detection
                start_phase, current_population, population_history, source_info = find_best_resume_point_with_csv_fallback(
                    experiment_dir, shadow, n_phases, api_keys, temperature_settings
                )
                
                if start_phase >= n_phases:
                    print(f"✅ Shadow condition {shadow*100}% already completed ({start_phase}/{n_phases} phases)")
                    continue
                
                # Provide detailed information about the resume source
                if source_info.startswith("checkpoint:"):
                    checkpoint_name = source_info.split(":", 1)[1]
                    print(f"\n🔄 Resuming evolutionary mode from phase {start_phase+1}/{n_phases}...")
                    print(f"📂 Using checkpoint: {checkpoint_name}")
                elif source_info.startswith("csv:resume_from_phase_"):
                    resume_phase = source_info.split("_")[-1]
                    print(f"\n🔄 Resuming evolutionary mode from phase {resume_phase}/{n_phases}...")
                    print(f"📄 CSV-based resume: phases 1-{start_phase} completed, starting with initial population")
                elif source_info.startswith("csv:completed_all_"):
                    completed_count = source_info.split("_")[3]
                    print(f"✅ All phases already completed (found CSV files for {completed_count} phases)")
                    continue
                else:  # fresh_start
                    print(f"\n🆕 Starting evolutionary mode from beginning")
                    print(f"📋 No previous progress found - using initial population configuration")
            else:
                print(f"\nRunning evolutionary mode with {n_phases} phases...")
                
                # Create initial population from agents
                initial_population = {}
                for agent in initial_agents:
                    agent_name = agent.name
                    if agent_name in initial_population:
                        initial_population[agent_name] += 1
                    else:
                        initial_population[agent_name] = 1
                
                start_phase = 0
                current_population = initial_population.copy()
                population_history = [initial_population.copy()]
            
            results = []
            
            with Timer(f"Shadow {shadow*100}% evolutionary tournaments"):
                for phase in range(start_phase, n_phases):
                    print(f"\n{'='*50}")
                    print(f"Phase {phase+1}/{n_phases} - Shadow {shadow*100}%")
                    print(f"{'='*50}")
                    print(f"Population: {current_population}")
                    
                    # Create agents based on current population
                    phase_agents = []
                    for agent_name, count in current_population.items():
                        for instance in range(count):
                            # Find the original agent template
                            original_agent = next(a for a in initial_agents if a.name == agent_name)
                            
                            # Create new instance with proper API key handling
                            if any(x in agent_name for x in ['GPT4', 'GPT5', 'Claude', 'Mistral', 'Gemini']):
                                # For LLM agents, determine which API key to use based on agent type
                                if any(x in agent_name for x in ['GPT4', 'GPT5']):
                                    api_key = api_keys['OPENAI_API_KEY']
                                elif 'Claude' in agent_name:
                                    api_key = api_keys['ANTHROPIC_API_KEY']
                                elif 'Mistral' in agent_name:
                                    api_key = api_keys['MISTRAL_API_KEY']
                                elif 'Gemini' in agent_name:
                                    api_key = api_keys['GOOGLE_API_KEY']
                                else:
                                    continue  # Skip unknown LLM agent types
                                
                                new_agent = original_agent.__class__(
                                    f"{agent_name}_p{phase+1}i{instance+1}",
                                    api_key,
                                    model=original_agent.model,
                                    temperature=original_agent.temperature,
                                    termination_prob=shadow,
                                    match_history=history_manager.get_history_for_agent(original_agent)
                                )
                            else:
                                # For classical agents
                                new_agent = original_agent.__class__(f"{agent_name}_p{phase+1}i{instance+1}")
                            
                            phase_agents.append(new_agent)
                    
                    # Run tournament for this phase
                    tournament = Tournament(phase_agents, termination_prob=shadow, verbose=True, 
                                          max_concurrent=max_concurrent, history_manager=history_manager, 
                                          current_phase=phase+1)
                    result = tournament.run_tournament()
                    
                    # Save phase results
                    result.save_to_csv(
                        os.path.join(experiment_dir, 
                                   f"evolutionary_shadow{int(shadow*100)}_phase{phase+1}.csv")
                    )
                    results.append(result)
                    
                    # Save match histories after each phase
                    history_manager.save_histories(history_file)
                    print(f"📚 Saved match histories for phase {phase+1}")
                    
                    # Update progress
                    completed_matches = (i * n_phases + phase + 1) * len(phase_agents) * (len(phase_agents) - 1) // 2
                    update_progress(progress_file, {
                        'completed_matches': completed_matches
                    })
                    
                    # Print phase results
                    print("\nTop 5 performers this phase:")
                    summary = result.get_summary_stats()
                    for idx, row in summary.head().iterrows():
                        print(f"{idx+1}. {row['agent']}: {row['avg_score_per_move']:.3f}")
                    
                    # Evolve population for next phase (except last phase)
                    if phase < n_phases - 1:
                        phase_result = convert_tournament_result_to_phase_result(result)
                        current_population = evolve_population(current_population, phase_result, verbose=True)
                        population_history.append(current_population.copy())
                        
                        print("\nPopulation changes:")
                        prev_pop = population_history[-2]
                        for strategy in sorted(set(list(prev_pop.keys()) + list(current_population.keys()))):
                            prev_count = prev_pop.get(strategy, 0)
                            curr_count = current_population.get(strategy, 0)
                            if prev_count != curr_count:
                                change = curr_count - prev_count
                                change_str = f"(+{change})" if change > 0 else f"({change})" if change < 0 else ""
                                print(f"  {strategy}: {prev_count} → {curr_count} {change_str}")
                    
                    # Save checkpoint after each phase
                    save_population_checkpoint(
                        experiment_dir, shadow, phase + 1, 
                        current_population, population_history, 
                        initial_agents, config
                    )
        else:
            # Run standard tournaments (original behavior)
            # Create match history manager
            history_manager = MatchHistoryManager(enable_opponent_tracking=enable_opponent_tracking)
            
            # Create agents for this condition
            agents = create_agents(api_keys, temperature_settings, shadow, 
                                 include_classical=True, history_manager=history_manager)
            all_agents_used.extend(agents)  # Track agents for LLM showdown
            
            # Run tournaments
            tournament = Tournament(agents, termination_prob=shadow, verbose=True, 
                                  max_concurrent=max_concurrent, history_manager=history_manager, current_phase=1)
            
            with Timer(f"Shadow {shadow*100}% tournaments"):
                results = []
                for t in range(n_tournaments):
                    print(f"\nTournament {t+1}/{n_tournaments}")
                    result = tournament.run_tournament()
                    
                    # Save individual tournament results
                    result.save_to_csv(
                        os.path.join(experiment_dir, 
                                   f"tournament_shadow{int(shadow*100)}_run{t+1}.csv")
                    )
                    results.append(result)
                    
                    # Update progress
                    completed_matches = (i * n_tournaments + t + 1) * n_matches
                    update_progress(progress_file, {
                        'completed_matches': completed_matches
                    })
                    
                    # Print intermediate results
                    print("\nTop 5 performers:")
                    summary = result.get_summary_stats()
                    for idx, row in summary.head().iterrows():
                        print(f"{idx+1}. {row['agent']}: {row['avg_score_per_move']:.3f}")
        
        all_results[f"shadow_{int(shadow*100)}"] = results
    
    # Run LLM Showdown
    print(f"\n{'='*60}")
    print("LLM SHOWDOWN")
    print(f"{'='*60}")
    
    # Get LLM agents from all agents used across conditions
    llm_agents = [a for a in all_agents_used if any(x in a.name for x in ['GPT', 'Claude', 'Mistral', 'Gemini'])]
    
    if len(llm_agents) >= 2:
        showdown = LLMShowdown(llm_agents, shadow_conditions, 
                              rounds_per_condition=3, verbose=True)
        showdown_results = showdown.run()
        
        # Merge showdown results
        for condition, results in showdown_results.items():
            if condition in all_results:
                all_results[condition].extend(results)
            else:
                all_results[condition] = results
    
    # Generate comprehensive report
    print(f"\n{'='*60}")
    print("GENERATING ANALYSIS REPORT")
    print(f"{'='*60}")
    
    with Timer("Analysis generation"):
        summary_df = generate_comprehensive_report(all_results, experiment_dir)
    
    # Print final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {experiment_dir}")
    print("\nOverall Performance Rankings:")
    
    overall_avg = summary_df.groupby('agent')['avg_score'].mean().sort_values(ascending=False)
    for i, (agent, score) in enumerate(overall_avg.head(10).items(), 1):
        print(f"{i}. {agent}: {score:.3f} points/move")
    
    # Calculate total API usage
    total_api_calls = 0
    total_tokens = 0
    for agent in all_agents_used:
        if hasattr(agent, 'api_calls'):
            total_api_calls += agent.api_calls
            total_tokens += agent.total_tokens
    
    print(f"\nAPI Usage:")
    print(f"- Total API calls: {total_api_calls}")
    print(f"- Total tokens: {total_tokens:,}")
    
    update_progress(progress_file, {
        'completed_conditions': len(shadow_conditions),
        'completed_matches': n_matches * len(shadow_conditions) * n_tournaments,
        'status': 'complete'
    })


def test_mistral_temperature():
    """Test Mistral with temperature 1.2 against simple strategies to verify API support"""
    print("="*60)
    print("MISTRAL TEMPERATURE 1.2 TEST")
    print("Testing if Mistral API accepts temperature=1.2")
    print("="*60)
    
    # Create output directory for test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join("results", f"mistral_temp_test_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)
    print(f"📁 Test results will be saved to: {test_dir}")
    
    api_keys = load_env_vars()
    
    if not api_keys.get('MISTRAL_API_KEY'):
        print("❌ No Mistral API key found. Please add MISTRAL_API_KEY to your .env file.")
        return
    
    print("✓ Mistral API key found")
    
    # Create simple opponent strategies
    simple_agents = [
        AlwaysCooperate("AlwaysCooperate"),
        AlwaysDefect("AlwaysDefect"), 
        TitForTat("TitForTat")
    ]
    
    # Test different Mistral temperatures
    test_temperatures = [0.7, 1.0, 1.2]
    mistral_agents = []
    
    for temp in test_temperatures:
        print(f"\n🧪 Testing Mistral with temperature={temp}")
        try:
            agent = MistralAgent(
                f"Mistral_T{str(temp).replace('.', '')}", 
                api_keys['MISTRAL_API_KEY'],
                model="mistral-large-latest",
                temperature=temp,
                termination_prob=0.7  # High termination for quick test
            )
            mistral_agents.append(agent)
            print(f"✓ Successfully created Mistral agent with temperature={temp}")
        except Exception as e:
            print(f"❌ Failed to create Mistral agent with temperature={temp}: {e}")
            continue
    
    if not mistral_agents:
        print("❌ Could not create any Mistral agents. Check your API key and connection.")
        return
    
    print(f"\n✓ Created {len(mistral_agents)} Mistral agents")
    
    # Test each Mistral agent against simple strategies
    for mistral_agent in mistral_agents:
        print(f"\n{'='*40}")
        print(f"TESTING {mistral_agent.name} (temp={mistral_agent.temperature})")
        print(f"{'='*40}")
        
        test_agents = [mistral_agent] + simple_agents
        
        try:
            # Run very short tournament
            tournament = Tournament(test_agents, termination_prob=0.7, max_rounds=5, max_concurrent=100)
            
            with Timer(f"{mistral_agent.name} test"):
                result = tournament.run_tournament()
            
            # Save tournament results
            result.save_to_csv(os.path.join(test_dir, f"{mistral_agent.name}_results.csv"))
            
            # Show results
            summary = result.get_summary_stats()
            print(f"\nResults for {mistral_agent.name}:")
            for idx, row in summary.iterrows():
                if row['agent'] == mistral_agent.name:
                    print(f"📊 {row['agent']}: {row['avg_score_per_move']:.3f} points/move")
                    print(f"   API calls: {mistral_agent.api_calls}")
                    print(f"   Total tokens: {mistral_agent.total_tokens}")
                    
                    # Show a sample reasoning if available
                    if mistral_agent.last_reasoning:
                        reasoning_preview = mistral_agent.last_reasoning[:200] + "..." if len(mistral_agent.last_reasoning) > 200 else mistral_agent.last_reasoning
                        print(f"   Sample reasoning: {reasoning_preview}")
                    break
            
            print("✅ Test completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during tournament with {mistral_agent.name}: {e}")
            # Try a single API call to diagnose the issue
            try:
                print("🔍 Testing single API call...")
                move = mistral_agent.make_move([], [])
                print(f"✓ Single API call successful. Move: {move}")
                print(f"  Temperature used: {mistral_agent.temperature}")
            except Exception as api_error:
                print(f"❌ Single API call failed: {api_error}")
                if "temperature" in str(api_error).lower():
                    print("🚨 Temperature-related error detected!")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEMPERATURE TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_temps = []
    failed_temps = []
    
    for agent in mistral_agents:
        if agent.api_calls > 0:
            successful_temps.append(agent.temperature)
        else:
            failed_temps.append(agent.temperature)
    
    if successful_temps:
        print(f"✅ Successfully tested temperatures: {successful_temps}")
        max_working_temp = max(successful_temps)
        print(f"📈 Maximum working temperature: {max_working_temp}")
        
        if 1.2 in successful_temps:
            print("🎉 Mistral DOES support temperature=1.2!")
        elif 1.0 in successful_temps:
            print("⚠️  Mistral supports up to temperature=1.0 only")
        
    if failed_temps:
        print(f"❌ Failed temperatures: {failed_temps}")
    
    # Save test summary report
    test_summary = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'mistral_temperature_test',
        'tested_temperatures': test_temperatures,
        'successful_temperatures': successful_temps,
        'failed_temperatures': failed_temps,
        'max_working_temperature': max(successful_temps) if successful_temps else None,
        'supports_1_2': 1.2 in successful_temps,
        'api_usage': {
            agent.name: {
                'temperature': agent.temperature,
                'api_calls': agent.api_calls,
                'total_tokens': agent.total_tokens,
                'success': agent.api_calls > 0
            } for agent in mistral_agents
        }
    }
    
    with open(os.path.join(test_dir, 'test_summary.json'), 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    print(f"📊 Test summary saved to: {os.path.join(test_dir, 'test_summary.json')}")
    print(f"{'='*60}")


def run_test_experiment():
    """Run a comprehensive test experiment with all strategies and LLM temperature settings"""
    print("="*60)
    print("COMPREHENSIVE TEST EXPERIMENT")
    print("Testing all strategies + LLMs with assigned temperatures")
    print("="*60)
    
    # Create output directory for test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join("results", f"comprehensive_test_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)
    print(f"📁 Test results will be saved to: {test_dir}")
    
    api_keys = load_env_vars()
    
    # Test temperature settings (OpenAI uses different models instead)
    temperature_settings = {
        'anthropic': [0.2, 0.5, 0.8],  # Anthropic supports 0-1 range
        'mistral': [0.2, 0.7, 1.2],    # Mistral supports 0-1 range (capped at 1.0)
        'gemini': [0.2, 0.7, 1.2]      # Gemini supports 0-2 range
    }
    
    # Create all classical, behavioral, and adaptive agents (agents 1-16)
    agents = [
        # 1-10: Classical strategies
        TitForTat("TitForTat"),                        # 1
        GrimTrigger("GrimTrigger"),                    # 2  
        WinStayLoseShift("WinStayLoseShift"),          # 3
        GenerousTitForTat("GenerousTitForTat"),        # 4
        SuspiciousTitForTat("SuspiciousTitForTat"),    # 5
        Prober("Prober"),                              # 6
        Random("Random"),                              # 7
        Gradual("Gradual"),                            # 8
        Alternator("Alternator"),                      # 9
        Bayesian("Bayesian"),                          # 10
        
        # 11-13: Behavioral strategies
        ForgivingGrimTrigger("ForgivingGrimTrigger"),  # 11
        Detective("Detective"),                        # 12
        SoftGrudger("SoftGrudger"),                    # 13
        
        # 14-16: Adaptive learning strategies
        QLearningAgent("QLearning"),                   # 14
        ThompsonSampling("ThompsonSampling"),          # 15
        GradientMetaLearner("GradientMetaLearner")     # 16
    ]
    
    # Add LLM agents with their assigned temperature settings
    test_termination_prob = 0.5  # Higher termination for faster testing
    
    # OpenAI agents (Different Models)
    if api_keys.get('OPENAI_API_KEY'):
        openai_models = ['gpt-5-mini', 'gpt-5-nano', 'gpt-4.1-mini']
        for model in openai_models:
            model_name = model.replace('-', '').replace('gpt', 'GPT')
            agents.append(
                GPT4Agent(f"{model_name}_T1", 
                         api_keys['OPENAI_API_KEY'],
                         model=model,
                         temperature=1.0,
                         termination_prob=test_termination_prob)
            )
    
    # Claude agents (Anthropic supports 0-1 range)
    if api_keys.get('ANTHROPIC_API_KEY') and 'anthropic' in temperature_settings:
        for temp in temperature_settings['anthropic']:
            temp_suffix = f"_T{str(temp).replace('.', '')}"
            agents.append(
                ClaudeAgent(f"Claude3Sonnet{temp_suffix}",
                           api_keys['ANTHROPIC_API_KEY'],
                           model="claude-3-5-sonnet-20241022",
                           temperature=temp,
                           termination_prob=test_termination_prob)
            )
    
    # Mistral agents (Mistral supports 0-1 range)
    if api_keys.get('MISTRAL_API_KEY') and 'mistral' in temperature_settings:
        for temp in temperature_settings['mistral']:
            temp_suffix = f"_T{str(temp).replace('.', '')}"
            agents.append(
                MistralAgent(f"MistralLarge{temp_suffix}",
                            api_keys['MISTRAL_API_KEY'],
                            model="mistral-medium-2508",
                            temperature=temp,
                            termination_prob=test_termination_prob)
            )
    
    # Gemini agents (Gemini supports 0-2 range)
    if api_keys.get('GOOGLE_API_KEY') and 'gemini' in temperature_settings:
        for temp in temperature_settings['gemini']:
            temp_suffix = f"_T{str(temp).replace('.', '')}"
            agents.append(
                GeminiAgent(f"Gemini20Flash{temp_suffix}",
                           api_keys['GOOGLE_API_KEY'],
                           model="gemini-2.0-flash",
                           temperature=temp,
                           termination_prob=test_termination_prob)
            )
    
    print(f"\nCreated {len(agents)} total agents:")
    print(f"- 16 Classical/Behavioral/Adaptive agents")
    
    # Count LLM agents by provider
    llm_counts = {'GPT4': 0, 'Claude': 0, 'Mistral': 0, 'Gemini': 0}
    for agent in agents:
        if 'GPT4' in agent.name:
            llm_counts['GPT4'] += 1
        elif 'Claude' in agent.name:
            llm_counts['Claude'] += 1
        elif 'Mistral' in agent.name:
            llm_counts['Mistral'] += 1
        elif 'Gemini' in agent.name:
            llm_counts['Gemini'] += 1
    
    # Map display names to temperature setting keys
    provider_key_mapping = {
        'GPT4': 'openai',
        'Claude': 'anthropic', 
        'Mistral': 'mistral',
        'Gemini': 'gemini'
    }
    
    for provider, count in llm_counts.items():
        if count > 0:
            temp_key = provider_key_mapping.get(provider, provider.lower())
            temps = temperature_settings.get(temp_key, [])
            print(f"- {count} {provider} agents with temperatures {temps}")
    
    # Run short tournament (fewer rounds for quick testing)
    print(f"\nRunning test tournament with {test_termination_prob*100}% termination probability...")
    tournament = Tournament(agents, termination_prob=test_termination_prob, max_rounds=8, max_concurrent=100)
    
    with Timer("Test tournament"):
        result = tournament.run_tournament()
    
    # Save tournament results to CSV
    result.save_to_csv(os.path.join(test_dir, "comprehensive_test_results.csv"))
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    summary = result.get_summary_stats()
    print("\nTop 10 performers:")
    for idx, row in summary.head(10).iterrows():
        print(f"{idx+1:2d}. {row['agent']:25s}: {row['avg_score_per_move']:.3f} points/move")
    
    # Count API usage
    total_api_calls = 0
    total_tokens = 0
    for agent in agents:
        if hasattr(agent, 'api_calls'):
            total_api_calls += agent.api_calls
            total_tokens += agent.total_tokens
    
    print(f"\nAPI Usage Summary:")
    print(f"- Total API calls: {total_api_calls}")
    print(f"- Total tokens used: {total_tokens:,}")
    
    # Temperature performance analysis
    llm_agents = [a for a in agents if any(x in a.name for x in ['GPT', 'Claude', 'Mistral', 'Gemini'])]
    if llm_agents:
        print(f"\nLLM Temperature Performance:")
        for agent in llm_agents:
            agent_stats = summary[summary['agent'] == agent.name]
            if not agent_stats.empty:
                score = agent_stats['avg_score_per_move'].iloc[0]
                temp = agent.temperature
                print(f"- {agent.name}: {score:.3f} (temp={temp})")
    
    # Save comprehensive test summary
    test_summary = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'comprehensive_test',
        'total_agents': len(agents),
        'classical_agents': 16,
        'llm_agent_counts': {k: v for k, v in llm_counts.items() if v > 0},
        'temperature_settings': temperature_settings,
        'termination_probability': test_termination_prob,
        'max_rounds': 8,
        'api_usage_summary': {
            'total_api_calls': total_api_calls,
            'total_tokens': total_tokens
        },
        'top_performers': [
            {
                'rank': idx + 1,
                'agent': row['agent'],
                'avg_score_per_move': float(row['avg_score_per_move']),
                'is_llm': any(x in row['agent'] for x in ['GPT', 'Claude', 'Mistral', 'Gemini'])
            }
            for idx, row in summary.head(10).iterrows()
        ]
    }
    
    # Add individual LLM performance details
    if llm_agents:
        llm_performance = []
        for agent in llm_agents:
            agent_stats = summary[summary['agent'] == agent.name]
            if not agent_stats.empty:
                score = agent_stats['avg_score_per_move'].iloc[0]
                llm_performance.append({
                    'agent': agent.name,
                    'temperature': agent.temperature,
                    'score': float(score),
                    'api_calls': agent.api_calls,
                    'tokens': agent.total_tokens
                })
        test_summary['llm_performance'] = llm_performance
    
    with open(os.path.join(test_dir, 'test_summary.json'), 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    # Save experiment configuration 
    config = create_experiment_config([test_termination_prob], temperature_settings, 
                                     {api: "default" for api in ['OPENAI', 'ANTHROPIC', 'MISTRAL', 'GOOGLE'] 
                                      if api_keys.get(f'{api}_API_KEY')})
    save_experiment_metadata(config, os.path.join(test_dir, "config.json"))
    
    print(f"\n📊 Complete test results saved to: {test_dir}")
    print(f"📄 Files created:")
    print(f"   - comprehensive_test_results.csv (detailed tournament data)")
    print(f"   - test_summary.json (summary statistics)")
    print(f"   - config.json (experiment configuration)")
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE! All strategies and temperature settings verified.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IPD experiments")
    parser.add_argument("--test", action="store_true", 
                       help="Run test experiment to verify setup")
    parser.add_argument("--test-mistral-temp", action="store_true",
                       help="Test Mistral temperature support (1.2 vs 1.0)")
    parser.add_argument("--shadow", type=float, nargs="+", 
                       default=[0.1, 0.25, 0.75],
                       help="Shadow conditions (termination probabilities)")
    parser.add_argument("--temperature", type=str,
                       default='{"anthropic": [0.2, 0.5, 0.8], "mistral": [0.2, 0.7, 1.2], "gemini": [0.2, 0.7, 1.2]}',
                       help="Temperature settings JSON string for each model (e.g., '{\"openai\": [0.2, 0.7, 1.2], \"anthropic\": [0.2, 0.5, 0.8]}')")
    parser.add_argument("--tournaments", type=int, default=5,
                       help="Number of tournaments per condition (standard mode only, default: 5)")
    parser.add_argument("--phases", type=int, default=5,
                       help="Number of evolutionary phases per condition (evolutionary mode only, default: 5)")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--evolutionary", action="store_true",
                       help="Use evolutionary mode: population evolves based on performance across phases")
    parser.add_argument("--test-evolutionary", action="store_true",
                       help="Run test experiment in evolutionary mode")
    parser.add_argument("--yes", "-y", action="store_true",
                       help="Automatically confirm all prompts (skip cost confirmation)")
    parser.add_argument("--resume", type=str,
                       help="Resume experiment from checkpoint (provide experiment directory path)")
    parser.add_argument("--max-concurrent", type=int, default=50,
                       help="Maximum number of concurrent matches within each tournament (default: 50)")
    parser.add_argument("--enable-opponent-tracking", action="store_true",
                       help="Enable opponent tracking: LLM agents receive cross-references to previous encounters")
    
    args = parser.parse_args()
    
    if args.test:
        run_test_experiment()
    elif args.test_mistral_temp:
        test_mistral_temperature()
    elif args.test_evolutionary:
        # Run a quick evolutionary test with fewer agents and phases
        run_main_experiments(
            shadow_conditions=[0.5],  # Single condition for speed
            temperature_settings={},  # No longer using temperature_settings for OpenAI
            n_tournaments=5,  # Not used in evolutionary mode
            n_phases=3,  # 3 phases instead of 5
            output_dir="results",
            evolutionary=True,
            auto_confirm=args.yes,
            enable_opponent_tracking=args.enable_opponent_tracking
        )
    else:
        # Parse temperature settings from JSON string
        try:
            temperature_settings = json.loads(args.temperature)
        except json.JSONDecodeError as e:
            print(f"Error parsing temperature settings JSON: {e}")
            print("Using default temperature settings.")
            temperature_settings = {
                'anthropic': [0.2, 0.5, 0.8],
                'mistral': [0.2, 0.7, 1.2],
                'gemini': [0.2, 0.7, 1.2]
            }
        
        run_main_experiments(
            shadow_conditions=args.shadow,
            temperature_settings=temperature_settings,
            n_tournaments=args.tournaments,
            n_phases=args.phases,
            output_dir=args.output,
            evolutionary=args.evolutionary,
            auto_confirm=args.yes,
            resume_experiment=args.resume,
            max_concurrent=args.max_concurrent,
            enable_opponent_tracking=args.enable_opponent_tracking
        )