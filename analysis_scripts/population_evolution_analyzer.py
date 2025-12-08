#!/usr/bin/env python3
"""
Population Evolution Analysis for All IPD Experiments
Analyzes population evolution across 8 experiments (4 shadow conditions x 2 memory modes)

Author: Population Evolution Analysis Pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PopulationEvolutionAnalyzer:
    """Comprehensive analyzer for population evolution across all experiments"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

        # Define experiment mapping based on analysis
        self.experiments = {
            # Anonymous Memory Mode (September 5, 8)
            "anonymous": {
                0.05: "experiment_20250908_081108",
                0.10: "experiment_20250905_152754",
                0.25: "experiment_20250905_152805",
                0.75: "experiment_20250905_152321"
            },
            # Opponent Tracking Mode (September 10, 11, 12)
            "tracking": {
                0.05: "experiment_20250912_153315",
                0.10: "experiment_20250911_112723",
                0.25: "experiment_20250910_074608",
                0.75: "experiment_20250910_074557"
            }
        }

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 12)

    def load_experiment_data(self, experiment_name: str):
        """Load population evolution data from experiment"""
        exp_dir = self.results_dir / experiment_name

        if not exp_dir.exists():
            print(f"Experiment directory not found: {exp_dir}")
            return None

        # Find evolutionary CSV files
        csv_files = list(exp_dir.glob("evolutionary_shadow*.csv"))
        if not csv_files:
            print(f"No evolutionary CSV files found in {exp_dir}")
            return None

        # Load all phases
        phases_data = []
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)

                # Extract phase number from filename
                phase_match = csv_file.name
                if 'phase1' in phase_match:
                    phase = 1
                elif 'phase2' in phase_match:
                    phase = 2
                elif 'phase3' in phase_match:
                    phase = 3
                elif 'phase4' in phase_match:
                    phase = 4
                elif 'phase5' in phase_match:
                    phase = 5
                else:
                    continue

                df['phase'] = phase
                phases_data.append(df)

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue

        if not phases_data:
            return None

        combined_data = pd.concat(phases_data, ignore_index=True)

        # Load config for metadata
        config_file = exp_dir / "config.json"
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)

        return {
            'data': combined_data,
            'config': config,
            'experiment_name': experiment_name
        }

    def calculate_population_evolution(self, experiment_data):
        """Calculate population counts across phases"""
        if not experiment_data:
            return None

        df = experiment_data['data']

        # Count agent occurrences per phase
        population_evolution = []

        for phase in sorted(df['phase'].unique()):
            phase_data = df[df['phase'] == phase]

            # Count each agent type
            agent_counts = {}
            for _, row in phase_data.iterrows():
                agent1 = row['agent1']
                agent2 = row['agent2']

                # Extract base agent name (remove phase/instance info)
                agent1_base = self.extract_base_agent_name(agent1)
                agent2_base = self.extract_base_agent_name(agent2)

                agent_counts[agent1_base] = agent_counts.get(agent1_base, 0) + 1
                agent_counts[agent2_base] = agent_counts.get(agent2_base, 0) + 1

            # Normalize to get relative population sizes
            total_agents = sum(agent_counts.values()) / 2  # Divide by 2 since each match counts each agent twice
            for agent, count in agent_counts.items():
                population_evolution.append({
                    'phase': phase,
                    'agent': agent,
                    'count': count / 2,  # Actual count
                    'proportion': (count / 2) / total_agents if total_agents > 0 else 0
                })

        return pd.DataFrame(population_evolution)

    def extract_base_agent_name(self, agent_name: str):
        """Extract base agent name without phase/instance info"""
        # Remove phase and instance info (e.g., _p1i1, _p2i2)
        import re
        base_name = re.sub(r'_p\d+i\d+$', '', agent_name)

        # Categorize agents
        if any(x in base_name for x in ['GPT4', 'GPT5', 'Claude', 'Gemini', 'Mistral']):
            return 'LLM_' + base_name
        elif any(x in base_name for x in ['TitForTat', 'GrimTrigger', 'Random', 'AlwaysCooperate', 'AlwaysDefect', 'Pavlov']):
            return 'Classical_' + base_name
        elif any(x in base_name for x in ['QLearning', 'ThompsonSampling', 'GradientMetaLearner']):
            return 'Adaptive_' + base_name
        else:
            return 'Other_' + base_name

    def create_population_evolution_plot(self, pop_data, memory_mode: str, shadow: float, save_dir: Path):
        """Create population evolution plot for single experiment"""
        if pop_data is None or pop_data.empty:
            return None

        # Group by agent category
        llm_agents = pop_data[pop_data['agent'].str.startswith('LLM_')]
        classical_agents = pop_data[pop_data['agent'].str.startswith('Classical_')]
        adaptive_agents = pop_data[pop_data['agent'].str.startswith('Adaptive_')]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Population Evolution - {memory_mode.title()} Memory, Shadow {shadow}',
                     fontsize=16, fontweight='bold')

        # Plot 1: LLM Agents
        ax1 = axes[0, 0]
        if not llm_agents.empty:
            for agent in llm_agents['agent'].unique():
                agent_data = llm_agents[llm_agents['agent'] == agent]
                ax1.plot(agent_data['phase'], agent_data['proportion'],
                        marker='o', label=agent.replace('LLM_', ''), linewidth=2)
        ax1.set_title('LLM Agents Evolution', fontweight='bold')
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Population Proportion')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Classical Agents
        ax2 = axes[0, 1]
        if not classical_agents.empty:
            for agent in classical_agents['agent'].unique():
                agent_data = classical_agents[classical_agents['agent'] == agent]
                ax2.plot(agent_data['phase'], agent_data['proportion'],
                        marker='s', label=agent.replace('Classical_', ''), linewidth=2)
        ax2.set_title('Classical Agents Evolution', fontweight='bold')
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Population Proportion')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Adaptive Agents
        ax3 = axes[1, 0]
        if not adaptive_agents.empty:
            for agent in adaptive_agents['agent'].unique():
                agent_data = adaptive_agents[adaptive_agents['agent'] == agent]
                ax3.plot(agent_data['phase'], agent_data['proportion'],
                        marker='^', label=agent.replace('Adaptive_', ''), linewidth=2)
        ax3.set_title('Adaptive Learning Agents Evolution', fontweight='bold')
        ax3.set_xlabel('Phase')
        ax3.set_ylabel('Population Proportion')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Overall Category Comparison
        ax4 = axes[1, 1]
        category_evolution = pop_data.groupby(['phase', pop_data['agent'].str.split('_').str[0]])['proportion'].sum().unstack(fill_value=0)

        for category in category_evolution.columns:
            ax4.plot(category_evolution.index, category_evolution[category],
                    marker='o', label=category, linewidth=3)

        ax4.set_title('Agent Category Evolution', fontweight='bold')
        ax4.set_xlabel('Phase')
        ax4.set_ylabel('Population Proportion')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        filename = save_dir / f'population_evolution_{memory_mode}_shadow{int(shadow*100)}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved population evolution plot: {filename}")
        return filename

    def generate_latex_table(self, pop_data, memory_mode: str, shadow: float):
        """Generate LaTeX table for experiment results"""
        if pop_data is None or pop_data.empty:
            return ""

        # Calculate final phase statistics
        final_phase = pop_data['phase'].max()
        final_data = pop_data[pop_data['phase'] == final_phase].copy()

        # Sort by proportion (descending)
        final_data = final_data.sort_values('proportion', ascending=False)

        # Generate LaTeX table
        latex_content = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Population Evolution Results - {memory_mode.title()} Memory, Shadow Condition {shadow}}}
\\label{{tab:population_{memory_mode}_shadow{int(shadow*100)}}}
\\begin{{tabular}}{{lcc}}
\\toprule
Agent Type & Final Population & Proportion \\\\
\\midrule
"""

        for _, row in final_data.head(10).iterrows():  # Top 10 agents
            agent_name = row['agent'].replace('_', '\\_')
            count = int(row['count'])
            proportion = row['proportion']
            latex_content += f"{agent_name} & {count} & {proportion:.3f} \\\\\n"

        latex_content += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: Results show final population distribution after 5 evolutionary phases.
\\item Only top 10 agents by population proportion are shown.
\\end{tablenotes}
\\end{table}
"""

        return latex_content

    def analyze_all_experiments(self):
        """Analyze population evolution across all experiments"""
        print("POPULATION EVOLUTION ANALYZER")
        print("="*60)

        # Create output directories
        latex_dir = Path("latex_artefacts")
        latex_dir.mkdir(exist_ok=True)

        all_results = {}
        latex_tables = []

        for memory_mode in ["anonymous", "tracking"]:
            for shadow in [0.05, 0.10, 0.25, 0.75]:
                experiment_name = self.experiments[memory_mode][shadow]

                print(f"\nAnalyzing {experiment_name} ({memory_mode}, shadow {shadow})")

                # Load experiment data
                exp_data = self.load_experiment_data(experiment_name)
                if not exp_data:
                    print(f"  Failed to load data")
                    continue

                # Calculate population evolution
                pop_evolution = self.calculate_population_evolution(exp_data)
                if pop_evolution is None:
                    print(f"  Failed to calculate population evolution")
                    continue

                print(f"  Processed {len(pop_evolution)} agent-phase combinations")

                # Create plot in experiment directory
                exp_dir = self.results_dir / experiment_name
                plot_file = self.create_population_evolution_plot(
                    pop_evolution, memory_mode, shadow, exp_dir
                )

                # Generate LaTeX table
                latex_table = self.generate_latex_table(pop_evolution, memory_mode, shadow)
                latex_tables.append(latex_table)

                # Store results
                all_results[(memory_mode, shadow)] = {
                    'population_data': pop_evolution,
                    'experiment_data': exp_data,
                    'plot_file': plot_file
                }

        # Save all LaTeX tables
        latex_file = latex_dir / "population_evolution_tables.tex"
        with open(latex_file, 'w') as f:
            f.write("% Population Evolution Tables for All Experiments\n")
            f.write("% Generated by Population Evolution Analyzer\n\n")

            for table in latex_tables:
                f.write(table)
                f.write("\n\\clearpage\n\n")

        print(f"\nSaved LaTeX tables: {latex_file}")

        # Generate summary comparison
        self.generate_comparative_analysis(all_results, latex_dir)

        return all_results

    def generate_comparative_analysis(self, all_results, output_dir):
        """Generate comparative analysis across memory modes and shadow conditions"""
        print("\nGenerating comparative analysis...")

        # Create summary statistics
        summary_data = []

        for (memory_mode, shadow), results in all_results.items():
            pop_data = results['population_data']
            if pop_data is None or pop_data.empty:
                continue

            final_phase = pop_data['phase'].max()
            final_data = pop_data[pop_data['phase'] == final_phase]

            # Calculate diversity metrics
            num_surviving_agents = len(final_data[final_data['proportion'] > 0.01])  # Agents with >1% population
            top_agent_dominance = final_data['proportion'].max()

            # Categorize agents
            llm_proportion = final_data[final_data['agent'].str.startswith('LLM_')]['proportion'].sum()
            classical_proportion = final_data[final_data['agent'].str.startswith('Classical_')]['proportion'].sum()
            adaptive_proportion = final_data[final_data['agent'].str.startswith('Adaptive_')]['proportion'].sum()

            summary_data.append({
                'memory_mode': memory_mode,
                'shadow_condition': shadow,
                'surviving_agents': num_surviving_agents,
                'top_agent_dominance': top_agent_dominance,
                'llm_proportion': llm_proportion,
                'classical_proportion': classical_proportion,
                'adaptive_proportion': adaptive_proportion
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary to CSV
        summary_file = output_dir / "population_evolution_summary.csv"
        summary_df.to_csv(summary_file, index=False)

        # Generate summary LaTeX table
        summary_latex = self.generate_summary_latex_table(summary_df)

        with open(output_dir / "population_evolution_summary.tex", 'w') as f:
            f.write(summary_latex)

        print(f"Saved comparative analysis: {summary_file}")

        return summary_df

    def generate_summary_latex_table(self, summary_df):
        """Generate summary LaTeX table comparing all experiments"""
        latex_content = """
\\begin{table*}[htbp]
\\centering
\\caption{Population Evolution Summary Across All Experiments}
\\label{tab:population_evolution_summary}
\\begin{tabular}{llcccccc}
\\toprule
Memory Mode & Shadow & Surviving & Top Agent & LLM & Classical & Adaptive & Other \\\\
            & Condition & Agents & Dominance & Proportion & Proportion & Proportion & Proportion \\\\
\\midrule
"""

        for _, row in summary_df.iterrows():
            memory = row['memory_mode'].title()
            shadow = row['shadow_condition']
            surviving = int(row['surviving_agents'])
            dominance = row['top_agent_dominance']
            llm_prop = row['llm_proportion']
            classical_prop = row['classical_proportion']
            adaptive_prop = row['adaptive_proportion']
            other_prop = 1 - (llm_prop + classical_prop + adaptive_prop)

            latex_content += f"{memory} & {shadow} & {surviving} & {dominance:.3f} & {llm_prop:.3f} & {classical_prop:.3f} & {adaptive_prop:.3f} & {other_prop:.3f} \\\\\n"

        latex_content += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: Surviving Agents = agents with >1\\% final population share.
\\item Top Agent Dominance = largest single agent population share.
\\item Proportions sum to 1.0 within each experiment.
\\end{tablenotes}
\\end{table*}
"""

        return latex_content


def main():
    """Main execution function"""
    analyzer = PopulationEvolutionAnalyzer()
    results = analyzer.analyze_all_experiments()

    print(f"\nAnalysis complete! Processed {len(results)} experiments")
    print("Check latex_artefacts/ for LaTeX tables and comparative analysis")


if __name__ == "__main__":
    main()