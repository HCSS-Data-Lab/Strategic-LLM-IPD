#!/usr/bin/env python3
"""
Performance Tables Generator for All IPD Experiments
Creates LaTeX performance tables in the exact format specified by the user

Author: Performance Tables Analysis Pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

class PerformanceTablesGenerator:
    """Generates performance tables for all experiments in specified format"""

    def __init__(self, results_dir: str = "../results"):
        self.results_dir = Path(results_dir)

        # Define experiment mapping
        self.experiments = {
            # Anonymous Memory Mode
            "anonymous": {
                0.05: "experiment_20250908_081108",
                0.10: "experiment_20250905_152754",
                0.25: "experiment_20250905_152805",
                0.75: "experiment_20250905_152321"
            },
            # Opponent Tracking Mode
            "tracking": {
                0.05: "experiment_20250912_153315",
                0.10: "experiment_20250911_112723",
                0.25: "experiment_20250910_074608",
                0.75: "experiment_20250910_074557"
            }
        }

    def load_experiment_data(self, experiment_name: str):
        """Load all phase data from experiment"""
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
        phases_data = {}
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)

                # Extract phase number from filename
                if 'phase1' in csv_file.name:
                    phase = 1
                elif 'phase2' in csv_file.name:
                    phase = 2
                elif 'phase3' in csv_file.name:
                    phase = 3
                elif 'phase4' in csv_file.name:
                    phase = 4
                elif 'phase5' in csv_file.name:
                    phase = 5
                else:
                    continue

                phases_data[phase] = df

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue

        return phases_data

    def extract_base_agent_name(self, agent_name: str):
        """Extract base agent name without phase/instance info"""
        import re
        # Remove phase and instance info (e.g., _p1i1, _p2i2)
        base_name = re.sub(r'_p\d+i\d+$', '', agent_name)
        return base_name

    def calculate_agent_performance(self, phases_data):
        """Calculate average score per move and match counts for each agent per phase"""
        agent_stats = defaultdict(lambda: defaultdict(lambda: {'total_score': 0, 'total_moves': 0, 'unique_matches': set()}))

        for phase, df in phases_data.items():
            # Group by match_id to get final scores for each match
            matches = df.groupby('match_id')

            for match_id, match_df in matches:
                # Get the final round (last row) to get total score for the entire match
                final_row = match_df.iloc[-1]

                agent1 = self.extract_base_agent_name(final_row['agent1'])
                agent2 = self.extract_base_agent_name(final_row['agent2'])

                # Final total score for the entire match
                agent1_final_score = final_row['agent1_total_score']
                agent2_final_score = final_row['agent2_total_score']

                # Number of rounds in this match
                rounds_played = final_row['round']

                # Update agent1 stats
                agent_stats[agent1][phase]['total_score'] += agent1_final_score
                agent_stats[agent1][phase]['total_moves'] += rounds_played
                agent_stats[agent1][phase]['unique_matches'].add(match_id)

                # Update agent2 stats
                agent_stats[agent2][phase]['total_score'] += agent2_final_score
                agent_stats[agent2][phase]['total_moves'] += rounds_played
                agent_stats[agent2][phase]['unique_matches'].add(match_id)

        # Calculate average scores per move
        performance_data = {}
        for agent, phase_data in agent_stats.items():
            performance_data[agent] = {}
            for phase, stats in phase_data.items():
                if stats['total_moves'] > 0:
                    avg_score = stats['total_score'] / stats['total_moves']
                    performance_data[agent][phase] = {
                        'score': avg_score,
                        'matches': len(stats['unique_matches'])
                    }

        return performance_data

    def categorize_agents(self, performance_data):
        """Categorize agents into groups for table organization"""
        categories = {
            'Classical': [],
            'Adaptive': [],
            'Anthropic': [],
            'Mistral': [],
            'Gemini': [],
            'OpenAI': []
        }

        for agent in performance_data.keys():
            agent_lower = agent.lower()

            if any(x in agent_lower for x in ['claude', 'sonnet']):
                categories['Anthropic'].append(agent)
            elif any(x in agent_lower for x in ['mistral']):
                categories['Mistral'].append(agent)
            elif any(x in agent_lower for x in ['gemini']):
                categories['Gemini'].append(agent)
            elif any(x in agent_lower for x in ['gpt', 'o3']):
                categories['OpenAI'].append(agent)
            elif any(x in agent_lower for x in ['qlearning', 'thompson', 'gradient']):
                categories['Adaptive'].append(agent)
            else:
                categories['Classical'].append(agent)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def format_agent_name_for_latex(self, agent_name):
        """Format agent name for LaTeX display"""
        # Handle specific agent name patterns
        if 'Claude4-Sonnet' in agent_name:
            temp_match = agent_name.split('_T')
            if len(temp_match) > 1:
                temp = temp_match[1]
                if temp == '02':
                    return 'Claude4-Sonnet (T=0.2)'
                elif temp == '05':
                    return 'Claude4-Sonnet (T=0.5)'
                elif temp == '08':
                    return 'Claude4-Sonnet (T=0.8)'
            return agent_name

        elif 'Mistral-Medium' in agent_name or 'Mistral-Large' in agent_name:
            if '_T02' in agent_name:
                return agent_name.replace('_T02', ' (T=0.2)')
            elif '_T07' in agent_name:
                return agent_name.replace('_T07', ' (T=0.7)')
            elif '_T12' in agent_name:
                return agent_name.replace('_T12', ' (T=1.2)')
            return agent_name

        elif 'Gemini' in agent_name:
            if '_T02' in agent_name:
                return agent_name.replace('_T02', ' (T=0.2)')
            elif '_T07' in agent_name:
                return agent_name.replace('_T07', ' (T=0.7)')
            elif '_T12' in agent_name:
                return agent_name.replace('_T12', ' (T=1.2)')
            return agent_name.replace('Gemini20Flash', 'Gemini25Pro')

        elif 'GPT' in agent_name:
            if '_T1' in agent_name:
                return agent_name.replace('_T1', ' (T=0.1)')
            return agent_name

        # Handle classical and adaptive agents
        name_mapping = {
            'TitForTat': 'TFT',
            'GrimTrigger': 'Grim',
            'SuspiciousTitForTat': 'Suspicious TFT',
            'GenerousTitForTat': 'Generous TFT',
            'ForgivingGrimTrigger': 'Forgiving Grim',
            'WinStayLoseShift': 'WSLS',
            'QLearning': 'Q-Learning',
            'ThompsonSampling': 'Thompson',
            'GradientMetaLearner': 'Gradient Meta'
        }

        return name_mapping.get(agent_name, agent_name)

    def generate_performance_table(self, performance_data, memory_mode, shadow, experiment_name):
        """Generate LaTeX performance table in the exact specified format"""

        # Categorize agents
        categories = self.categorize_agents(performance_data)

        # Start building the table
        shadow_percent = int(shadow * 100)

        latex_content = f"""\\begin{{landscape}}
\\begin{{sidewaystable}}[htbp]
\\centering
\\scriptsize
\\caption{{Performance Metrics ({memory_mode.title()} Memory, Shadow={shadow_percent}\\%). Score represents the average payoff per move per phase.}}
\\label{{tab:{shadow_percent}performance_{memory_mode}}}
\\begin{{tabular}}{{|l|cc|cc|cc|cc|cc|}}
\\hline
\\textbf{{Strategy}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{P1}}}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{P2}}}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{P3}}}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{P4}}}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{P5}}}} \\\\
\\hline
 & \\textbf{{Score}} & \\textbf{{Matches}} & \\textbf{{Score}} & \\textbf{{Matches}} & \\textbf{{Score}} & \\textbf{{Matches}} & \\textbf{{Score}} & \\textbf{{Matches}} & \\textbf{{Score}} & \\textbf{{Matches}} \\\\
\\hline
"""

        # Add each category
        for category, agents in categories.items():
            if category in ['Anthropic', 'Mistral', 'Gemini', 'OpenAI']:
                # Add category header for LLM providers
                latex_content += f"\\multicolumn{{11}}{{|c|}}{{\\textbf{{{category}}}}} \\\\\n\\hline\n"

            # Sort agents within category
            sorted_agents = sorted(agents)

            for agent in sorted_agents:
                agent_data = performance_data[agent]
                formatted_name = self.format_agent_name_for_latex(agent)

                latex_content += formatted_name

                # Add data for each phase (1-5)
                for phase in range(1, 6):
                    if phase in agent_data:
                        score = agent_data[phase]['score']
                        matches = agent_data[phase]['matches']
                        latex_content += f" & {score:.3f} & {matches}"
                    else:
                        latex_content += " & -- & --"

                latex_content += " \\\\\n"

            # Add separator line after each category except the last
            if category != list(categories.keys())[-1]:
                latex_content += "\\hline\n"

        latex_content += """\\hline
\\end{tabular}
\\end{sidewaystable}
\\end{landscape}

"""

        return latex_content

    def generate_all_performance_tables(self):
        """Generate performance tables for all experiments"""
        print("PERFORMANCE TABLES GENERATOR")
        print("="*60)

        # Create output directory
        output_dir = Path("latex_artefacts")
        output_dir.mkdir(exist_ok=True)

        all_tables = []

        for memory_mode in ["anonymous", "tracking"]:
            for shadow in [0.05, 0.10, 0.25, 0.75]:
                experiment_name = self.experiments[memory_mode][shadow]

                print(f"\nProcessing {experiment_name} ({memory_mode}, shadow {shadow})")

                # Load experiment data
                phases_data = self.load_experiment_data(experiment_name)
                if not phases_data:
                    print(f"  Failed to load data")
                    continue

                # Calculate performance metrics
                performance_data = self.calculate_agent_performance(phases_data)
                print(f"  Calculated performance for {len(performance_data)} agents")

                # Generate LaTeX table
                table_latex = self.generate_performance_table(
                    performance_data, memory_mode, shadow, experiment_name
                )
                all_tables.append(table_latex)

        # Save all tables to file
        output_file = output_dir / "all_performance_tables.tex"
        with open(output_file, 'w') as f:
            f.write("% Performance Tables for All Experiments\n")
            f.write("% Generated by Performance Tables Generator\n")
            f.write("% Format matches user specification exactly\n\n")

            for table in all_tables:
                f.write(table)
                f.write("\\clearpage\n\n")

        print(f"\nGenerated {len(all_tables)} performance tables")
        print(f"Saved to: {output_file}")

        return output_file


def main():
    """Main execution function"""
    generator = PerformanceTablesGenerator()
    output_file = generator.generate_all_performance_tables()

    print(f"\nPerformance tables generation complete!")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()