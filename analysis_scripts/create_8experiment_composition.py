#!/usr/bin/env python3
"""
Create composition analysis for the 8 main experiments
"""

import os
import json
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import glob

def analyze_experiment_data(experiment_path: str, experiment_name: str):
    """Analyze a single experiment to extract population data across phases"""

    exp_path = Path(experiment_path)

    # Find all CSV files for this experiment
    csv_files = list(exp_path.glob("evolutionary_*.csv"))

    if not csv_files:
        return None

    experiment_data = {
        "experiment_path": str(exp_path),
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "phases": {}
    }

    # Process each CSV file (each represents a phase)
    for csv_file in csv_files:
        try:
            # Extract phase information from filename
            filename = csv_file.stem
            if "_phase" in filename:
                phase_parts = filename.split("_phase")
                shadow_info = phase_parts[0].replace("evolutionary_shadow", "")
                phase_num = int(phase_parts[1])
                shadow_condition = float(shadow_info) / 100.0
            else:
                continue

            # Read CSV file
            df = pd.read_csv(csv_file)

            if df.empty:
                continue

            # Count agents by strategy
            agents_by_strategy = Counter()
            agents_by_type = defaultdict(int)

            # Get unique agents in this phase and clean strategy names
            agents = set()
            for col in ['agent1', 'agent2']:
                if col in df.columns:
                    agents.update(df[col].dropna().unique())

            # Clean strategy names and count each agent type
            for agent in agents:
                # Clean strategy name - remove phase and instance info (_p1i1, _p2i1, etc.)
                clean_agent = agent.split('_p')[0] if '_p' in agent else agent
                agents_by_strategy[clean_agent] += 1

                # Categorize agent by type (use clean_agent for categorization)
                if any(llm in clean_agent for llm in ['Claude4-Sonnet', 'Claude']):
                    agents_by_type['LLM_Anthropic'] += 1
                elif any(llm in clean_agent for llm in ['Mistral-Medium', 'Mistral']):
                    agents_by_type['LLM_Mistral'] += 1
                elif any(llm in clean_agent for llm in ['Gemini20Flash', 'Gemini']):
                    agents_by_type['LLM_Gemini'] += 1
                elif any(llm in clean_agent for llm in ['GPT5mini', 'GPT5nano', 'GPT4.1mini', 'GPT', 'o3_T1', 'GPT4o']):
                    agents_by_type['LLM_OpenAI'] += 1
                elif agent in ['TitForTat', 'GrimTrigger', 'AlwaysCooperate', 'AlwaysDefect',
                             'Random', 'Pavlov', 'Gradual', 'Prober', 'Detective', 'Alternator',
                             'GenerousTitForTat', 'SuspiciousTitForTat', 'WinStayLoseShift',
                             'Bayesian']:
                    agents_by_type['Classical'] += 1
                elif agent in ['ForgivingGrimTrigger', 'SoftGrudger']:
                    agents_by_type['Behavioral'] += 1
                elif agent in ['QLearning', 'ThompsonSampling', 'GradientMetaLearner']:
                    agents_by_type['Adaptive'] += 1
                else:
                    agents_by_type['Other'] += 1

            phase_key = f"shadow_{int(shadow_condition*100)}_phase_{phase_num}"

            experiment_data["phases"][phase_key] = {
                "shadow_condition": shadow_condition,
                "phase_number": phase_num,
                "csv_file": str(csv_file),
                "agents_by_type": dict(agents_by_type),
                "agents_by_strategy": dict(agents_by_strategy)
            }

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    return experiment_data

def main():
    """Create composition analysis for all 8 experiments"""

    # Define experiment mapping
    experiments = {
        # Anonymous Memory Mode
        "experiment_20250908_081108": {"shadow": 0.05, "memory": "anonymous"},
        "experiment_20250905_152754": {"shadow": 0.10, "memory": "anonymous"},
        "experiment_20250905_152805": {"shadow": 0.25, "memory": "anonymous"},
        "experiment_20250905_152321": {"shadow": 0.75, "memory": "anonymous"},
        # Opponent Tracking Mode
        "experiment_20250912_153315": {"shadow": 0.05, "memory": "tracking"},
        "experiment_20250911_112723": {"shadow": 0.10, "memory": "tracking"},
        "experiment_20250910_074608": {"shadow": 0.25, "memory": "tracking"},
        "experiment_20250910_074557": {"shadow": 0.75, "memory": "tracking"}
    }

    results_dir = "/mnt/c/Github/LLM-IPD-ARXIV2/results"

    # Collect data for all experiments
    all_experiments_data = {
        "analysis_timestamp": datetime.now().isoformat(),
        "results_directory": "results",
        "experiments": {}
    }

    for experiment_name, config in experiments.items():
        experiment_path = os.path.join(results_dir, experiment_name)

        if os.path.exists(experiment_path):
            print(f"Analyzing {experiment_name} (Shadow {config['shadow']}, Memory {config['memory']})...")
            experiment_data = analyze_experiment_data(experiment_path, experiment_name)

            if experiment_data:
                all_experiments_data["experiments"][experiment_name] = experiment_data
                print(f"  Found {len(experiment_data['phases'])} phases")
            else:
                print(f"  No data found for {experiment_name}")
        else:
            print(f"  Directory not found: {experiment_path}")

    # Save the analysis
    output_file = "/mnt/c/Github/LLM-IPD-ARXIV2/visualization_results/all_8experiments_composition_analysis.json"

    with open(output_file, 'w') as f:
        json.dump(all_experiments_data, f, indent=2)

    print(f"\nComposition analysis saved to: {output_file}")
    print(f"Total experiments analyzed: {len(all_experiments_data['experiments'])}")

    # Print summary
    for exp_name, exp_data in all_experiments_data["experiments"].items():
        phases_count = len(exp_data.get("phases", {}))
        print(f"  {exp_name}: {phases_count} phases")

if __name__ == "__main__":
    main()