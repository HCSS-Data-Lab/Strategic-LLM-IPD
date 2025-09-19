#!/usr/bin/env python3
"""
Strategic Reasoning Analysis for LLM IPD Experiments
Comprehensive analysis of strategic reasoning patterns across all experiments

This script analyzes:
1. Match history awareness across phases
2. Shadow of future consideration
3. Formal game theory usage with temperature sensitivity breakdown

Author: Strategic Reasoning Analysis Pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
import re
import json
import glob
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class StrategicReasoningAnalyzer:
    """Comprehensive analyzer for strategic reasoning patterns in LLM IPD experiments"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.analysis_results = {}

        # Define search patterns and keywords
        self.match_history_patterns = self._define_match_history_patterns()
        self.shadow_future_patterns = self._define_shadow_future_patterns()
        self.formal_methods_patterns = self._define_formal_methods_patterns()

    def _define_match_history_patterns(self):
        """Define explicit patterns for detecting match history awareness"""
        return {
            'explicit_history_keywords': [
                'history', 'previous round', 'last round', 'last move', 'earlier round',
                'before', 'previously', 'past move', 'prior round', 'history shows'
            ],
            'opponent_behavior_tracking': [
                'opponent cooperated', 'opponent defected', 'they cooperated', 'they defected',
                'my opponent chose', 'the opponent has', 'opponent always', 'opponent tends'
            ],
            'reciprocity_concepts': [
                'reciprocate', 'retaliate', 'tit for tat', 'copy', 'mirror', 'respond to',
                'based on their', 'following their', 'matching their'
            ],
            'trust_betrayal': [
                'trust', 'betrayed', 'betrayal', 'trustworthy', 'reliable',
                'broke trust', 'established trust', 'breach of trust'
            ],
            'pattern_recognition': [
                'pattern', 'consistent', 'always cooperates', 'always defects',
                'alternating', 'strategy seems', 'appears to be', 'strategy is'
            ]
        }

    def _define_shadow_future_patterns(self):
        """Define patterns for detecting shadow of future awareness"""
        return {
            'termination_probability': [
                'termination prob', 'continuation prob', 'shadow of', 'future is',
                'game ends', 'match ends', 'probability.*continue', 'chance.*end',
                'expected.*rounds', 'horizon'
            ],
            'future_payoffs': [
                'future payoff', 'long.term', 'future rounds', 'expected value',
                'future benefit', 'future cooperation', 'ongoing', 'sustained'
            ],
            'discount_factor': [
                'discount', 'present value', 'future value', 'time preference'
            ],
            'repeated_game_awareness': [
                'repeated', 'iteration', 'multiple rounds', 'many rounds',
                'extended game', 'ongoing game', 'series of'
            ]
        }

    def _define_formal_methods_patterns(self):
        """Define patterns for detecting formal game theory and modeling approaches"""
        return {
            'game_theory_concepts': [
                'game theory', 'nash equilibrium', 'dominant strategy', 'pareto',
                'prisoner.*dilemma', 'zero.sum', 'non.zero.sum', 'payoff matrix',
                'equilibrium', 'dominant', 'dominated'
            ],
            'decision_theory': [
                'expected utility', 'utility function', 'decision theory',
                'rational choice', 'optimization', 'maximize.*expected'
            ],
            'probability_modeling': [
                'bayesian', 'prior', 'posterior', 'likelihood', 'probability distribution',
                'belief', 'uncertain', 'stochastic', 'random variable'
            ],
            'opponent_modeling': [
                'opponent model', 'opponent.*strategy', 'model.*opponent',
                'predict.*opponent', 'opponent.*type', 'classify.*opponent',
                'infer.*strategy', 'estimate.*strategy'
            ],
            'algorithmic_approaches': [
                'algorithm', 'heuristic', 'rule', 'policy', 'strategy.*based',
                'systematic', 'computational', 'calculate'
            ]
        }

    def extract_reasoning_data(self, experiment_dir: Path):
        """Extract all reasoning data from experiment directory"""
        reasoning_data = []

        # Find all CSV files in experiment directory
        csv_files = list(experiment_dir.glob("*.csv"))
        evolutionary_files = [f for f in csv_files if 'evolutionary' in f.name]

        if not evolutionary_files:
            print(f"No evolutionary CSV files found in {experiment_dir}")
            return pd.DataFrame()

        for csv_file in evolutionary_files:
            try:
                df = pd.read_csv(csv_file)

                # Extract phase and shadow condition from filename
                filename = csv_file.name
                shadow_match = re.search(r'shadow(\d+)', filename)
                phase_match = re.search(r'phase(\d+)', filename)

                shadow_condition = int(shadow_match.group(1)) / 100 if shadow_match else None
                phase = int(phase_match.group(1)) if phase_match else None

                # Process each row for reasoning data
                for _, row in df.iterrows():
                    # Agent 1 reasoning
                    if pd.notna(row.get('agent1_reasoning')) and str(row['agent1_reasoning']) != 'nan':
                        reasoning_data.append({
                            'experiment': experiment_dir.name,
                            'shadow_condition': shadow_condition,
                            'phase': phase,
                            'agent': row['agent1'],
                            'reasoning': str(row['agent1_reasoning']),
                            'move': row['agent1_move'],
                            'opponent': row['agent2'],
                            'round': row['round'],
                            'match_id': row['match_id'],
                            'agent_position': 1
                        })

                    # Agent 2 reasoning
                    if pd.notna(row.get('agent2_reasoning')) and str(row['agent2_reasoning']) != 'nan':
                        reasoning_data.append({
                            'experiment': experiment_dir.name,
                            'shadow_condition': shadow_condition,
                            'phase': phase,
                            'agent': row['agent2'],
                            'reasoning': str(row['agent2_reasoning']),
                            'move': row['agent2_move'],
                            'opponent': row['agent1'],
                            'round': row['round'],
                            'match_id': row['match_id'],
                            'agent_position': 2
                        })

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

        return pd.DataFrame(reasoning_data)

    def classify_agent_type(self, agent_name: str):
        """Classify agent type and extract metadata"""
        agent_info = {
            'agent_name': agent_name,
            'is_llm': False,
            'provider': None,
            'model': None,
            'temperature': None,
            'base_name': agent_name
        }

        # LLM patterns
        llm_patterns = {
            'GPT4': r'GPT4[.\w]*_T(\d+)',
            'GPT5': r'GPT5[.\w]*_T(\d+)',
            'Claude': r'Claude[\w-]*_T(\d+)',
            'Gemini': r'Gemini[\w-]*_T(\d+)',
            'Mistral': r'Mistral[\w-]*_T(\d+)'
        }

        for provider, pattern in llm_patterns.items():
            match = re.search(pattern, agent_name)
            if match:
                agent_info['is_llm'] = True
                agent_info['provider'] = provider
                agent_info['model'] = provider
                temp_str = match.group(1)
                # Convert temperature (e.g., T02 -> 0.2, T1 -> 1.0, T05 -> 0.5, T12 -> 1.2)
                if len(temp_str) == 1:
                    agent_info['temperature'] = float(temp_str)
                elif len(temp_str) == 2:
                    # Handle two-digit temperatures like T02, T05, T08, T12
                    if temp_str.startswith('0'):
                        agent_info['temperature'] = float(temp_str) / 10  # T02 -> 0.2, T05 -> 0.5
                    else:
                        agent_info['temperature'] = float(temp_str) / 10  # T12 -> 1.2
                else:
                    agent_info['temperature'] = float(temp_str) / 10
                agent_info['base_name'] = f"{provider}_T{temp_str}"
                break

        return agent_info

    def analyze_match_history_awareness(self, reasoning_df: pd.DataFrame):
        """Analyze match history awareness patterns"""
        print("=== ANALYZING MATCH HISTORY AWARENESS ===")

        results = []

        for _, row in reasoning_df.iterrows():
            reasoning_text = row['reasoning'].lower()
            agent_info = self.classify_agent_type(row['agent'])

            # Skip non-LLM agents
            if not agent_info['is_llm']:
                continue

            analysis = {
                'experiment': row['experiment'],
                'shadow_condition': row['shadow_condition'],
                'phase': row['phase'],
                'round': row['round'],
                'agent': row['agent'],
                'provider': agent_info['provider'],
                'temperature': agent_info['temperature'],
                'base_name': agent_info['base_name']
            }

            # Check each pattern category
            for category, keywords in self.match_history_patterns.items():
                found_keywords = []
                for keyword in keywords:
                    if re.search(keyword.lower(), reasoning_text):
                        found_keywords.append(keyword)

                analysis[f'{category}_count'] = len(found_keywords)
                analysis[f'{category}_present'] = len(found_keywords) > 0
                analysis[f'{category}_keywords'] = '; '.join(found_keywords)

            # Overall history awareness score
            total_matches = sum(analysis[f'{cat}_count'] for cat in self.match_history_patterns.keys())
            analysis['history_awareness_score'] = total_matches
            analysis['has_history_awareness'] = total_matches > 0

            # Special case: first round vs later rounds
            analysis['is_first_round'] = row['round'] == 1
            analysis['first_round_no_history'] = (row['round'] == 1 and
                                                 any(phrase in reasoning_text for phrase in
                                                     ['no history', 'first round', 'first move', 'no information']))

            results.append(analysis)

        return pd.DataFrame(results)

    def analyze_shadow_future_awareness(self, reasoning_df: pd.DataFrame):
        """Analyze shadow of future consideration"""
        print("=== ANALYZING SHADOW OF FUTURE AWARENESS ===")

        results = []

        for _, row in reasoning_df.iterrows():
            reasoning_text = row['reasoning'].lower()
            agent_info = self.classify_agent_type(row['agent'])

            # Skip non-LLM agents
            if not agent_info['is_llm']:
                continue

            analysis = {
                'experiment': row['experiment'],
                'shadow_condition': row['shadow_condition'],
                'phase': row['phase'],
                'round': row['round'],
                'agent': row['agent'],
                'provider': agent_info['provider'],
                'temperature': agent_info['temperature'],
                'base_name': agent_info['base_name']
            }

            # Check each pattern category
            for category, keywords in self.shadow_future_patterns.items():
                found_keywords = []
                for keyword in keywords:
                    if re.search(keyword.lower(), reasoning_text):
                        found_keywords.append(keyword)

                analysis[f'{category}_count'] = len(found_keywords)
                analysis[f'{category}_present'] = len(found_keywords) > 0
                analysis[f'{category}_keywords'] = '; '.join(found_keywords)

            # Specific shadow condition awareness
            shadow_val = row['shadow_condition']
            if shadow_val:
                # Look for specific mentions of the shadow condition
                shadow_patterns = [
                    f"{int(shadow_val*100)}%",
                    f"{shadow_val}",
                    f"0.{int(shadow_val*100)}",
                    f"{100-int(shadow_val*100)}% chance"
                ]

                shadow_specific = any(pattern in reasoning_text for pattern in shadow_patterns)
                analysis['mentions_specific_shadow'] = shadow_specific

            # Overall shadow awareness score
            total_matches = sum(analysis[f'{cat}_count'] for cat in self.shadow_future_patterns.keys())
            analysis['shadow_awareness_score'] = total_matches
            analysis['has_shadow_awareness'] = total_matches > 0

            results.append(analysis)

        return pd.DataFrame(results)

    def analyze_formal_methods_usage(self, reasoning_df: pd.DataFrame):
        """Analyze formal game theory and modeling approaches"""
        print("=== ANALYZING FORMAL METHODS USAGE ===")

        results = []

        for _, row in reasoning_df.iterrows():
            reasoning_text = row['reasoning'].lower()
            agent_info = self.classify_agent_type(row['agent'])

            # Skip non-LLM agents
            if not agent_info['is_llm']:
                continue

            analysis = {
                'experiment': row['experiment'],
                'shadow_condition': row['shadow_condition'],
                'phase': row['phase'],
                'round': row['round'],
                'agent': row['agent'],
                'provider': agent_info['provider'],
                'temperature': agent_info['temperature'],
                'base_name': agent_info['base_name']
            }

            # Check each pattern category
            for category, keywords in self.formal_methods_patterns.items():
                found_keywords = []
                for keyword in keywords:
                    if re.search(keyword.lower(), reasoning_text):
                        found_keywords.append(keyword)

                analysis[f'{category}_count'] = len(found_keywords)
                analysis[f'{category}_present'] = len(found_keywords) > 0
                analysis[f'{category}_keywords'] = '; '.join(found_keywords)

            # Overall formal methods score
            total_matches = sum(analysis[f'{cat}_count'] for cat in self.formal_methods_patterns.keys())
            analysis['formal_methods_score'] = total_matches
            analysis['uses_formal_methods'] = total_matches > 0

            # Sophistication level
            if analysis['game_theory_concepts_present'] or analysis['decision_theory_present']:
                analysis['sophistication_level'] = 'high'
            elif analysis['probability_modeling_present'] or analysis['opponent_modeling_present']:
                analysis['sophistication_level'] = 'medium'
            elif analysis['algorithmic_approaches_present']:
                analysis['sophistication_level'] = 'low'
            else:
                analysis['sophistication_level'] = 'none'

            results.append(analysis)

        return pd.DataFrame(results)

    def generate_summary_tables(self, history_df, shadow_df, formal_df):
        """Generate comprehensive summary tables"""
        print("=== GENERATING SUMMARY TABLES ===")

        tables = {}

        # 1. Match History Analysis Summary
        if not history_df.empty:
            history_summary = history_df.groupby(['provider', 'temperature']).agg({
                'has_history_awareness': ['count', 'sum', 'mean'],
                'explicit_history_keywords_present': 'mean',
                'opponent_behavior_tracking_present': 'mean',
                'reciprocity_concepts_present': 'mean',
                'pattern_recognition_present': 'mean',
                'history_awareness_score': 'mean'
            }).round(3)

            history_summary.columns = ['_'.join(col).strip() for col in history_summary.columns]
            tables['match_history_summary'] = history_summary

        # 2. Shadow of Future Analysis Summary
        if not shadow_df.empty:
            shadow_summary = shadow_df.groupby(['provider', 'temperature', 'shadow_condition']).agg({
                'has_shadow_awareness': ['count', 'sum', 'mean'],
                'termination_probability_present': 'mean',
                'future_payoffs_present': 'mean',
                'repeated_game_awareness_present': 'mean',
                'shadow_awareness_score': 'mean'
            }).round(3)

            shadow_summary.columns = ['_'.join(col).strip() for col in shadow_summary.columns]
            tables['shadow_future_summary'] = shadow_summary

        # 3. Formal Methods Analysis Summary
        if not formal_df.empty:
            formal_summary = formal_df.groupby(['provider', 'temperature']).agg({
                'uses_formal_methods': ['count', 'sum', 'mean'],
                'game_theory_concepts_present': 'mean',
                'decision_theory_present': 'mean',
                'probability_modeling_present': 'mean',
                'opponent_modeling_present': 'mean',
                'formal_methods_score': 'mean'
            }).round(3)

            formal_summary.columns = ['_'.join(col).strip() for col in formal_summary.columns]
            tables['formal_methods_summary'] = formal_summary

            # Temperature sensitivity analysis
            if len(formal_df['temperature'].unique()) > 1:
                temp_analysis = formal_df.groupby(['provider', 'temperature']).agg({
                    'sophistication_level': lambda x: (x == 'high').mean(),
                    'uses_formal_methods': 'mean',
                    'formal_methods_score': 'mean'
                }).round(3)

                temp_analysis.columns = ['high_sophistication_rate', 'formal_methods_rate', 'avg_formal_score']
                tables['temperature_sensitivity'] = temp_analysis

        return tables

    def save_results(self, output_dir: Path, tables: dict, history_df, shadow_df, formal_df):
        """Save all analysis results"""
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        if not history_df.empty:
            history_df.to_csv(output_dir / f"match_history_analysis_{timestamp}.csv", index=False)
        if not shadow_df.empty:
            shadow_df.to_csv(output_dir / f"shadow_future_analysis_{timestamp}.csv", index=False)
        if not formal_df.empty:
            formal_df.to_csv(output_dir / f"formal_methods_analysis_{timestamp}.csv", index=False)

        # Save summary tables
        with open(output_dir / f"strategic_reasoning_summary_{timestamp}.txt", 'w') as f:
            f.write("STRATEGIC REASONING ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")

            for table_name, table_data in tables.items():
                f.write(f"{table_name.upper().replace('_', ' ')}\n")
                f.write("-" * 40 + "\n")
                f.write(str(table_data))
                f.write("\n\n")

        # Save as Excel for better readability
        try:
            with pd.ExcelWriter(output_dir / f"strategic_reasoning_analysis_{timestamp}.xlsx") as writer:
                if not history_df.empty:
                    history_df.to_excel(writer, sheet_name='Match_History_Detail', index=False)
                if not shadow_df.empty:
                    shadow_df.to_excel(writer, sheet_name='Shadow_Future_Detail', index=False)
                if not formal_df.empty:
                    formal_df.to_excel(writer, sheet_name='Formal_Methods_Detail', index=False)

                for table_name, table_data in tables.items():
                    table_data.to_excel(writer, sheet_name=table_name[:31])  # Excel sheet name limit
        except ImportError:
            print("Note: openpyxl not available, skipping Excel output")

        print(f"Results saved to {output_dir}")
        return output_dir / f"strategic_reasoning_summary_{timestamp}.txt"


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze strategic reasoning patterns in LLM IPD experiments')
    parser.add_argument('--results_dir', default='results', help='Directory containing experiment results')
    parser.add_argument('--output_dir', default='analysis_scripts/strategic_reasoning_results',
                       help='Directory to save analysis results')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = StrategicReasoningAnalyzer(args.results_dir)
    output_dir = Path(args.output_dir)

    print("STRATEGIC REASONING ANALYZER")
    print("="*50)
    print(f"Analyzing experiments in: {analyzer.results_dir}")
    print(f"Output directory: {output_dir}")

    # Find all experiment directories
    experiment_dirs = [d for d in analyzer.results_dir.iterdir()
                      if d.is_dir() and d.name.startswith('experiment_')]

    if not experiment_dirs:
        print("No experiment directories found!")
        return

    print(f"Found {len(experiment_dirs)} experiment directories")

    # Process all experiments
    all_reasoning_data = []

    for exp_dir in experiment_dirs:
        print(f"\nProcessing: {exp_dir.name}")
        reasoning_data = analyzer.extract_reasoning_data(exp_dir)
        if not reasoning_data.empty:
            all_reasoning_data.append(reasoning_data)
            print(f"  Extracted {len(reasoning_data)} reasoning entries")
        else:
            print(f"  No reasoning data found")

    if not all_reasoning_data:
        print("No reasoning data found across all experiments!")
        return

    # Combine all data
    combined_reasoning = pd.concat(all_reasoning_data, ignore_index=True)
    print(f"\nTotal reasoning entries: {len(combined_reasoning)}")
    print(f"Unique LLM agents: {combined_reasoning[combined_reasoning['agent'].str.contains('GPT|Claude|Gemini|Mistral')]['agent'].nunique()}")

    # Perform analyses
    history_analysis = analyzer.analyze_match_history_awareness(combined_reasoning)
    shadow_analysis = analyzer.analyze_shadow_future_awareness(combined_reasoning)
    formal_analysis = analyzer.analyze_formal_methods_usage(combined_reasoning)

    # Generate summary tables
    tables = analyzer.generate_summary_tables(history_analysis, shadow_analysis, formal_analysis)

    # Save results
    summary_file = analyzer.save_results(output_dir, tables, history_analysis, shadow_analysis, formal_analysis)

    print(f"\nAnalysis complete! Summary saved to: {summary_file}")

    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS PREVIEW")
    print("="*50)

    if 'match_history_summary' in tables:
        print("\nMATCH HISTORY AWARENESS by Provider/Temperature:")
        history_table = tables['match_history_summary']
        if 'has_history_awareness_mean' in history_table.columns:
            print(history_table['has_history_awareness_mean'].sort_values(ascending=False))

    if 'temperature_sensitivity' in tables:
        print("\nTEMPERATURE SENSITIVITY (Formal Methods Usage):")
        print(tables['temperature_sensitivity'])


if __name__ == "__main__":
    main()