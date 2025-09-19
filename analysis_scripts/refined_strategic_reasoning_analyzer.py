#!/usr/bin/env python3
"""
REFINED Strategic Reasoning Analysis for LLM IPD Experiments
Enhanced analysis with precise keyword definitions to avoid misinterpretation

Key Improvements:
1. Distinguishes within-game history vs cross-phase/match history
2. Sharpened keywords to reduce false positives
3. More precise formal methods detection
4. Context-aware pattern matching

Author: Strategic Reasoning Analysis Pipeline (Refined)
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

class RefinedStrategicReasoningAnalyzer:
    """Enhanced analyzer with precise keyword definitions to minimize false positives"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.analysis_results = {}

        # Define refined search patterns
        self.within_game_history_patterns = self._define_within_game_history_patterns()
        self.cross_match_history_patterns = self._define_cross_match_history_patterns()
        self.shadow_future_patterns = self._define_refined_shadow_future_patterns()
        self.formal_methods_patterns = self._define_refined_formal_methods_patterns()

    def _define_within_game_history_patterns(self):
        """Define patterns for within-game history awareness (round-to-round within single match)"""
        return {
            'explicit_round_history': [
                # Specific round references
                r'previous round', r'last round', r'earlier round', r'round \d+',
                r'first round', r'second round', r'third round',
                r'in round \d+', r'from round \d+', r'since round \d+'
            ],
            'move_sequence_tracking': [
                # Tracking specific move sequences
                r'last move', r'previous move', r'their last move', r'my last move',
                r'they played [CD]', r'opponent played [CD]', r'I played [CD]',
                r'[CD] in the last', r'previously [CD]'
            ],
            'immediate_reciprocity': [
                # Direct reciprocity based on recent moves
                r'since they cooperated', r'since they defected',
                r'because they cooperated', r'because they defected',
                r'in response to their', r'reciprocating their',
                r'copying their last', r'mirroring their'
            ],
            'within_game_patterns': [
                # Pattern recognition within current game
                r'they always cooperate', r'they always defect',
                r'opponent is consistent', r'opponent keeps',
                r'their strategy appears', r'they seem to be',
                r'following.*pattern', r'consistent.*behavior'
            ]
        }

    def _define_cross_match_history_patterns(self):
        """Define patterns for cross-match/cross-phase history awareness"""
        return {
            'cross_match_references': [
                # References to previous matches or encounters
                r'previous match', r'last match', r'earlier match',
                r'in our last encounter', r'when we played before',
                r'from previous games', r'past encounters',
                r'history with this opponent', r'played.*before'
            ],
            'population_learning': [
                # Learning from population or multiple opponents
                r'learned from.*opponents', r'experience with.*players',
                r'other.*players.*tend', r'most opponents',
                r'players like this', r'similar opponents',
                r'population.*behavior', r'overall.*strategy'
            ],
            'meta_game_awareness': [
                # Awareness of evolutionary/meta-game dynamics
                r'evolutionary.*pressure', r'survival.*depends',
                r'population.*selection', r'successful.*strategies',
                r'adaptation.*required', r'evolving.*strategy'
            ]
        }

    def _define_refined_shadow_future_patterns(self):
        """Define precise patterns for shadow of future awareness (avoiding generic future mentions)"""
        return {
            'termination_probability_specific': [
                # Specific mentions of termination/continuation probability
                r'termination.*prob', r'continuation.*prob',
                r'prob.*\d+%.*end', r'prob.*\d+%.*continue',
                r'\d+%.*chance.*end', r'\d+%.*chance.*continue',
                r'shadow of.*future', r'discount.*factor'
            ],
            'expected_game_length': [
                # Explicit calculation or mention of expected rounds
                r'expected.*\d+.*rounds', r'average.*\d+.*rounds',
                r'about \d+.*rounds', r'roughly \d+.*rounds',
                r'game.*length.*\d+', r'horizon.*\d+',
                r'short.*game', r'long.*game', r'brief.*interaction'
            ],
            'future_payoff_calculation': [
                # Explicit future payoff considerations
                r'expected.*payoff', r'future.*payoff.*calculation',
                r'discounted.*value', r'present.*value.*future',
                r'sum.*future.*payoffs', r'total.*expected.*score'
            ],
            'strategic_horizon_reasoning': [
                # Strategic reasoning about time horizon effects
                r'short.*horizon.*means', r'long.*horizon.*allows',
                r'given.*short.*game', r'since.*game.*short',
                r'horizon.*too.*short', r'not.*enough.*rounds'
            ]
        }

    def _define_refined_formal_methods_patterns(self):
        """Define precise patterns for formal game theory and methods (avoiding casual usage)"""
        return {
            'game_theory_technical': [
                # Technical game theory terms (not casual usage)
                r'nash.*equilibrium', r'pareto.*optimal', r'pareto.*efficient',
                r'dominant.*strategy', r'dominated.*strategy',
                r'prisoner.*dilemma.*theory', r'payoff.*matrix',
                r'zero.*sum.*game', r'repeated.*game.*theory'
            ],
            'decision_theory_formal': [
                # Formal decision theory concepts
                r'expected.*utility.*theory', r'utility.*function',
                r'decision.*theory', r'rational.*choice.*theory',
                r'maximize.*expected.*utility', r'optimize.*expected'
            ],
            'probability_modeling_technical': [
                # Technical probability/Bayesian modeling
                r'bayesian.*inference', r'bayesian.*updating',
                r'prior.*probability', r'posterior.*probability',
                r'likelihood.*function', r'probability.*distribution',
                r'stochastic.*process', r'markov.*chain'
            ],
            'opponent_modeling_formal': [
                # Formal opponent modeling approaches
                r'opponent.*model.*based', r'model.*opponent.*as',
                r'classify.*opponent.*type', r'opponent.*type.*estimation',
                r'belief.*about.*opponent', r'infer.*opponent.*strategy',
                r'opponent.*strategy.*space', r'strategy.*classification'
            ],
            'algorithmic_approaches_specific': [
                # Specific algorithmic approaches
                r'minimax.*algorithm', r'dynamic.*programming',
                r'backward.*induction', r'forward.*induction',
                r'best.*response.*calculation', r'iterated.*elimination'
            ]
        }

    def extract_reasoning_data(self, experiment_dir: Path):
        """Extract all reasoning data from experiment directory (unchanged from original)"""
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
        """Classify agent type and extract metadata (unchanged from original)"""
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

    def analyze_refined_match_history_awareness(self, reasoning_df: pd.DataFrame):
        """Analyze match history awareness with distinction between within-game and cross-match"""
        print("=== ANALYZING REFINED MATCH HISTORY AWARENESS ===")

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

            # Analyze within-game history patterns
            for category, patterns in self.within_game_history_patterns.items():
                found_matches = []
                for pattern in patterns:
                    matches = re.findall(pattern, reasoning_text)
                    if matches:
                        found_matches.extend(matches)

                analysis[f'within_game_{category}_count'] = len(found_matches)
                analysis[f'within_game_{category}_present'] = len(found_matches) > 0
                analysis[f'within_game_{category}_matches'] = '; '.join(found_matches[:3])  # Limit to first 3

            # Analyze cross-match history patterns
            for category, patterns in self.cross_match_history_patterns.items():
                found_matches = []
                for pattern in patterns:
                    matches = re.findall(pattern, reasoning_text)
                    if matches:
                        found_matches.extend(matches)

                analysis[f'cross_match_{category}_count'] = len(found_matches)
                analysis[f'cross_match_{category}_present'] = len(found_matches) > 0
                analysis[f'cross_match_{category}_matches'] = '; '.join(found_matches[:3])

            # Overall scores
            within_game_score = sum(analysis[f'within_game_{cat}_count']
                                  for cat in self.within_game_history_patterns.keys())
            cross_match_score = sum(analysis[f'cross_match_{cat}_count']
                                  for cat in self.cross_match_history_patterns.keys())

            analysis['within_game_history_score'] = within_game_score
            analysis['cross_match_history_score'] = cross_match_score
            analysis['has_within_game_awareness'] = within_game_score > 0
            analysis['has_cross_match_awareness'] = cross_match_score > 0
            analysis['total_history_awareness'] = within_game_score + cross_match_score

            # Context-specific analysis
            analysis['is_first_round'] = row['round'] == 1
            analysis['is_later_round'] = row['round'] > 1

            # Validate first round claims (should not reference previous rounds in same game)
            if row['round'] == 1:
                invalid_first_round = any(analysis[f'within_game_{cat}_present']
                                        for cat in ['explicit_round_history', 'move_sequence_tracking'])
                analysis['invalid_first_round_history'] = invalid_first_round
            else:
                analysis['invalid_first_round_history'] = False

            results.append(analysis)

        return pd.DataFrame(results)

    def analyze_refined_shadow_future_awareness(self, reasoning_df: pd.DataFrame):
        """Analyze shadow of future awareness with precise pattern matching"""
        print("=== ANALYZING REFINED SHADOW OF FUTURE AWARENESS ===")

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

            # Check each refined pattern category
            for category, patterns in self.shadow_future_patterns.items():
                found_matches = []
                for pattern in patterns:
                    matches = re.findall(pattern, reasoning_text)
                    if matches:
                        found_matches.extend(matches)

                analysis[f'{category}_count'] = len(found_matches)
                analysis[f'{category}_present'] = len(found_matches) > 0
                analysis[f'{category}_matches'] = '; '.join(found_matches[:3])

            # Specific shadow condition awareness (more precise)
            shadow_val = row['shadow_condition']
            if shadow_val:
                # Look for specific numerical mentions
                shadow_patterns = [
                    f"{int(shadow_val*100)}%",
                    f"0.{int(shadow_val*100):02d}",
                    f"{100-int(shadow_val*100)}%.*end",
                    f"{int(shadow_val*100)}%.*continue"
                ]

                specific_shadow_matches = []
                for pattern in shadow_patterns:
                    matches = re.findall(pattern, reasoning_text)
                    if matches:
                        specific_shadow_matches.extend(matches)

                analysis['mentions_specific_shadow_value'] = len(specific_shadow_matches) > 0
                analysis['specific_shadow_matches'] = '; '.join(specific_shadow_matches)

            # Overall shadow awareness score (only counts precise matches)
            total_matches = sum(analysis[f'{cat}_count'] for cat in self.shadow_future_patterns.keys())
            analysis['refined_shadow_awareness_score'] = total_matches
            analysis['has_precise_shadow_awareness'] = total_matches > 0

            # Distinguish between generic future mentions and strategic horizon reasoning
            analysis['strategic_vs_generic_future'] = (
                analysis['strategic_horizon_reasoning_present'] or
                analysis['termination_probability_specific_present']
            )

            results.append(analysis)

        return pd.DataFrame(results)

    def analyze_refined_formal_methods_usage(self, reasoning_df: pd.DataFrame):
        """Analyze formal methods usage with precise technical pattern matching"""
        print("=== ANALYZING REFINED FORMAL METHODS USAGE ===")

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

            # Check each refined formal methods category
            for category, patterns in self.formal_methods_patterns.items():
                found_matches = []
                for pattern in patterns:
                    matches = re.findall(pattern, reasoning_text)
                    if matches:
                        found_matches.extend(matches)

                analysis[f'{category}_count'] = len(found_matches)
                analysis[f'{category}_present'] = len(found_matches) > 0
                analysis[f'{category}_matches'] = '; '.join(found_matches[:3])

            # Overall formal methods score (only technical usage)
            total_matches = sum(analysis[f'{cat}_count'] for cat in self.formal_methods_patterns.keys())
            analysis['refined_formal_methods_score'] = total_matches
            analysis['uses_technical_formal_methods'] = total_matches > 0

            # Sophistication classification (more stringent)
            if (analysis['game_theory_technical_present'] or
                analysis['decision_theory_formal_present'] or
                analysis['algorithmic_approaches_specific_present']):
                analysis['refined_sophistication_level'] = 'high_technical'
            elif (analysis['probability_modeling_technical_present'] or
                  analysis['opponent_modeling_formal_present']):
                analysis['refined_sophistication_level'] = 'medium_technical'
            else:
                analysis['refined_sophistication_level'] = 'low_or_none'

            # Distinguish between technical and casual usage
            technical_indicators = (
                analysis['game_theory_technical_present'] +
                analysis['decision_theory_formal_present'] +
                analysis['probability_modeling_technical_present']
            )
            analysis['technical_vs_casual'] = technical_indicators > 0

            results.append(analysis)

        return pd.DataFrame(results)

    def generate_refined_summary_tables(self, history_df, shadow_df, formal_df):
        """Generate refined summary tables with precise measurements"""
        print("=== GENERATING REFINED SUMMARY TABLES ===")

        tables = {}

        # 1. Within-Game vs Cross-Match History Analysis
        if not history_df.empty:
            history_summary = history_df.groupby(['provider', 'temperature']).agg({
                'has_within_game_awareness': ['count', 'sum', 'mean'],
                'has_cross_match_awareness': ['count', 'sum', 'mean'],
                'within_game_history_score': 'mean',
                'cross_match_history_score': 'mean',
                'invalid_first_round_history': 'sum'  # Count of invalid claims
            }).round(4)

            history_summary.columns = ['_'.join(col).strip() for col in history_summary.columns]
            tables['refined_match_history_summary'] = history_summary

        # 2. Precise Shadow of Future Analysis
        if not shadow_df.empty:
            shadow_summary = shadow_df.groupby(['provider', 'temperature', 'shadow_condition']).agg({
                'has_precise_shadow_awareness': ['count', 'sum', 'mean'],
                'strategic_vs_generic_future': 'mean',
                'mentions_specific_shadow_value': 'mean',
                'termination_probability_specific_present': 'mean',
                'expected_game_length_present': 'mean'
            }).round(4)

            shadow_summary.columns = ['_'.join(col).strip() for col in shadow_summary.columns]
            tables['refined_shadow_future_summary'] = shadow_summary

        # 3. Technical Formal Methods Analysis
        if not formal_df.empty:
            formal_summary = formal_df.groupby(['provider', 'temperature']).agg({
                'uses_technical_formal_methods': ['count', 'sum', 'mean'],
                'technical_vs_casual': 'mean',
                'game_theory_technical_present': 'mean',
                'opponent_modeling_formal_present': 'mean',
                'refined_formal_methods_score': 'mean'
            }).round(4)

            formal_summary.columns = ['_'.join(col).strip() for col in formal_summary.columns]
            tables['refined_formal_methods_summary'] = formal_summary

            # Refined sophistication analysis
            sophistication_summary = formal_df.groupby(['provider', 'temperature']).agg({
                'refined_sophistication_level': lambda x: (x == 'high_technical').mean()
            }).round(4)
            sophistication_summary.columns = ['technical_sophistication_rate']
            tables['technical_sophistication_analysis'] = sophistication_summary

        return tables

    def save_refined_results(self, output_dir: Path, tables: dict, history_df, shadow_df, formal_df):
        """Save refined analysis results with clear methodology documentation"""
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        if not history_df.empty:
            history_df.to_csv(output_dir / f"refined_match_history_analysis_{timestamp}.csv", index=False)
        if not shadow_df.empty:
            shadow_df.to_csv(output_dir / f"refined_shadow_future_analysis_{timestamp}.csv", index=False)
        if not formal_df.empty:
            formal_df.to_csv(output_dir / f"refined_formal_methods_analysis_{timestamp}.csv", index=False)

        # Save methodology documentation
        methodology_file = output_dir / f"refined_analysis_methodology_{timestamp}.md"
        with open(methodology_file, 'w') as f:
            f.write("# REFINED STRATEGIC REASONING ANALYSIS METHODOLOGY\n\n")

            f.write("## 1. MATCH HISTORY ANALYSIS\n\n")
            f.write("### Within-Game History (Round-to-Round in Single Match):\n")
            for category, patterns in self.within_game_history_patterns.items():
                f.write(f"**{category}**: {patterns}\n\n")

            f.write("### Cross-Match History (Previous Matches/Phases):\n")
            for category, patterns in self.cross_match_history_patterns.items():
                f.write(f"**{category}**: {patterns}\n\n")

            f.write("## 2. SHADOW OF FUTURE ANALYSIS\n\n")
            f.write("### Precise Patterns (Avoiding Generic Future Mentions):\n")
            for category, patterns in self.shadow_future_patterns.items():
                f.write(f"**{category}**: {patterns}\n\n")

            f.write("## 3. FORMAL METHODS ANALYSIS\n\n")
            f.write("### Technical Patterns (Excluding Casual Usage):\n")
            for category, patterns in self.formal_methods_patterns.items():
                f.write(f"**{category}**: {patterns}\n\n")

        # Save refined summary tables
        with open(output_dir / f"refined_strategic_reasoning_summary_{timestamp}.txt", 'w') as f:
            f.write("REFINED STRATEGIC REASONING ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write("METHODOLOGY: Enhanced analysis with precise keyword definitions\n")
            f.write("to minimize false positives and distinguish different types of reasoning.\n\n")

            for table_name, table_data in tables.items():
                f.write(f"{table_name.upper().replace('_', ' ')}\n")
                f.write("-" * 50 + "\n")
                f.write(str(table_data))
                f.write("\n\n")

        print(f"Refined results saved to {output_dir}")
        return output_dir / f"refined_strategic_reasoning_summary_{timestamp}.txt"


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Refined strategic reasoning analysis with precise keyword definitions')
    parser.add_argument('--results_dir', default='results', help='Directory containing experiment results')
    parser.add_argument('--output_dir', default='analysis_scripts/refined_strategic_reasoning_results',
                       help='Directory to save refined analysis results')

    args = parser.parse_args()

    # Initialize refined analyzer
    analyzer = RefinedStrategicReasoningAnalyzer(args.results_dir)
    output_dir = Path(args.output_dir)

    print("REFINED STRATEGIC REASONING ANALYZER")
    print("="*60)
    print("Enhanced analysis with precise keyword definitions")
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

    # Perform refined analyses
    history_analysis = analyzer.analyze_refined_match_history_awareness(combined_reasoning)
    shadow_analysis = analyzer.analyze_refined_shadow_future_awareness(combined_reasoning)
    formal_analysis = analyzer.analyze_refined_formal_methods_usage(combined_reasoning)

    # Generate refined summary tables
    tables = analyzer.generate_refined_summary_tables(history_analysis, shadow_analysis, formal_analysis)

    # Save results
    summary_file = analyzer.save_refined_results(output_dir, tables, history_analysis, shadow_analysis, formal_analysis)

    print(f"\nRefined analysis complete! Summary saved to: {summary_file}")

    # Print key refined findings
    print("\n" + "="*60)
    print("KEY REFINED FINDINGS")
    print("="*60)

    if 'refined_match_history_summary' in tables:
        print("\nWITHIN-GAME vs CROSS-MATCH HISTORY AWARENESS:")
        history_table = tables['refined_match_history_summary']
        print("Within-game awareness rates:")
        if 'has_within_game_awareness_mean' in history_table.columns:
            print(history_table['has_within_game_awareness_mean'].sort_values(ascending=False))
        print("\nCross-match awareness rates:")
        if 'has_cross_match_awareness_mean' in history_table.columns:
            print(history_table['has_cross_match_awareness_mean'].sort_values(ascending=False))

    if 'technical_sophistication_analysis' in tables:
        print("\nTECHNICAL SOPHISTICATION (Refined Formal Methods):")
        print(tables['technical_sophistication_analysis'])


if __name__ == "__main__":
    main()