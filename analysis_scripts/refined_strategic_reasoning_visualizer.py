#!/usr/bin/env python3
"""
Refined Strategic Reasoning Visualizer
Creates comprehensive visualizations for strategic reasoning patterns across LLM variants
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class RefinedStrategicReasoningVisualizer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "refined_figures"
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        self.load_data()

        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")

    def load_data(self):
        """Load the refined strategic reasoning data"""
        # Try to load from the complete dataset first
        complete_data_dir = Path("refined_strategic_reasoning_all_models_fixed")
        if complete_data_dir.exists():
            print("Loading complete dataset with all LLM models...")
            history_files = list(complete_data_dir.glob("refined_match_history_analysis_*.csv"))
            shadow_files = list(complete_data_dir.glob("refined_shadow_future_analysis_*.csv"))
            formal_files = list(complete_data_dir.glob("refined_formal_methods_analysis_*.csv"))
        else:
            # Fall back to original directory
            history_files = list(self.data_dir.glob("refined_match_history_analysis_*.csv"))
            shadow_files = list(self.data_dir.glob("refined_shadow_future_analysis_*.csv"))
            formal_files = list(self.data_dir.glob("refined_formal_methods_analysis_*.csv"))

        if not history_files or not shadow_files or not formal_files:
            raise FileNotFoundError("Required data files not found in directory")

        # Load the most recent files
        self.history_df = pd.read_csv(sorted(history_files)[-1])
        self.shadow_df = pd.read_csv(sorted(shadow_files)[-1])
        self.formal_df = pd.read_csv(sorted(formal_files)[-1])

        print(f"Loaded history data: {len(self.history_df)} entries")
        print(f"Loaded shadow data: {len(self.shadow_df)} entries")
        print(f"Loaded formal data: {len(self.formal_df)} entries")

    def map_shadow_condition_to_length(self, shadow_condition):
        """Map shadow condition to expected game length"""
        shadow_map = {
            0.05: "5% (20 rounds)",
            0.10: "10% (10 rounds)",
            0.25: "25% (4 rounds)",
            0.75: "75% (1.3 rounds)"
        }
        return shadow_map.get(shadow_condition, f"{shadow_condition}")

    def create_agent_labels(self, df):
        """Create comprehensive agent labels including all variants"""
        agent_labels = []
        for _, row in df.iterrows():
            provider = row['provider']
            temp = row['temperature']

            if provider == 'GPT4':
                label = f"GPT4.1mini (T={temp:.1f})"
            elif provider == 'GPT5':
                # Check if we can distinguish between nano and mini variants
                # Look in the agent column for 'nano'
                agent_name = str(row.get('agent', ''))
                if 'nano' in agent_name.lower():
                    label = f"GPT5nano (T={temp:.1f})"
                else:
                    label = f"GPT5mini (T={temp:.1f})"
            elif provider == 'Gemini':
                label = f"Gemini-2.0-Flash (T={temp:.1f})"
            elif provider == 'Claude':
                label = f"Claude4-Sonnet (T={temp:.1f})"
            elif provider == 'Mistral':
                label = f"Mistral-Medium (T={temp:.1f})"
            else:
                label = f"{provider} (T={temp:.1f})"

            agent_labels.append(label)
        return agent_labels

    def figure1_match_history_usage(self):
        """Figure 1: Match history usage by LLM variants split by shadow length"""

        # Aggregate match history usage by agent and shadow condition
        df = self.history_df.copy()
        df['shadow_length'] = df['shadow_condition'].apply(self.map_shadow_condition_to_length)

        # Calculate within-game history awareness rate
        agg_data = df.groupby(['provider', 'temperature', 'shadow_condition', 'shadow_length']).agg({
            'has_within_game_awareness': 'mean',
            'has_cross_match_awareness': 'mean'
        }).reset_index()

        agg_data['agent_label'] = self.create_agent_labels(agg_data)

        # Create more compact layout with all shadow conditions in one figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        fig.suptitle('Match History Usage by LLM Variants Across Shadow Conditions', fontsize=14, fontweight='bold')

        # Pivot data for easier plotting
        pivot_within = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='has_within_game_awareness', fill_value=0)
        pivot_cross = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='has_cross_match_awareness', fill_value=0)

        # Get all unique shadow lengths
        shadow_lengths = sorted(agg_data['shadow_length'].unique())
        n_agents = len(pivot_within.index)
        n_shadows = len(shadow_lengths)

        # Create positions for bars
        x_pos = np.arange(n_agents)
        bar_width = 0.8 / (n_shadows * 2)  # 2 metrics per shadow condition

        # Colors for each shadow condition
        colors_within = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        colors_cross = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        for i, shadow in enumerate(shadow_lengths):
            offset = (i * 2 * bar_width) - (n_shadows * bar_width) + bar_width

            # Within-game bars
            within_values = [pivot_within.loc[agent, shadow] if shadow in pivot_within.columns else 0
                           for agent in pivot_within.index]
            bars1 = ax.bar(x_pos + offset, within_values, bar_width,
                          label=f'Within-Game ({shadow})', alpha=0.8,
                          color=colors_within[i % len(colors_within)])

            # Cross-match bars
            cross_values = [pivot_cross.loc[agent, shadow] if shadow in pivot_cross.columns else 0
                          for agent in pivot_cross.index]
            bars2 = ax.bar(x_pos + offset + bar_width, cross_values, bar_width,
                          label=f'Cross-Match ({shadow})', alpha=0.6,
                          color=colors_cross[i % len(colors_cross)])

        ax.set_xlabel('LLM Variants', fontweight='bold')
        ax.set_ylabel('History Awareness Rate', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_within.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / "match_history_usage_by_shadow_length.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 1: {output_path}")

    def figure2_shadow_future_reasoning(self):
        """Figure 2: Shadow of future reasoning by LLM variants split by shadow length"""

        df = self.shadow_df.copy()
        df['shadow_length'] = df['shadow_condition'].apply(self.map_shadow_condition_to_length)

        # Calculate shadow awareness metrics
        agg_data = df.groupby(['provider', 'temperature', 'shadow_condition', 'shadow_length']).agg({
            'has_precise_shadow_awareness': 'mean',
            'termination_probability_specific_present': 'mean',
            'expected_game_length_present': 'mean'
        }).reset_index()

        agg_data['agent_label'] = self.create_agent_labels(agg_data)

        # Create compact layout with all shadow conditions in one figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        fig.suptitle('Shadow of Future Reasoning by LLM Variants Across Shadow Conditions', fontsize=14, fontweight='bold')

        # Pivot data for easier plotting
        pivot_shadow = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='has_precise_shadow_awareness', fill_value=0)
        pivot_term = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='termination_probability_specific_present', fill_value=0)
        pivot_length = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='expected_game_length_present', fill_value=0)

        # Get all unique shadow lengths
        shadow_lengths = sorted(agg_data['shadow_length'].unique())
        n_agents = len(pivot_shadow.index)
        n_shadows = len(shadow_lengths)

        # Create positions for bars
        x_pos = np.arange(n_agents)
        bar_width = 0.8 / (n_shadows * 3)  # 3 metrics per shadow condition

        # Colors for each metric
        colors_shadow = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']
        colors_term = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        colors_length = ['#17becf', '#bcbd22', '#ff9896', '#c5b0d5']

        for i, shadow in enumerate(shadow_lengths):
            offset = (i * 3 * bar_width) - (n_shadows * 1.5 * bar_width) + 1.5 * bar_width

            # Shadow awareness bars
            shadow_values = [pivot_shadow.loc[agent, shadow] if shadow in pivot_shadow.columns else 0
                           for agent in pivot_shadow.index]
            bars1 = ax.bar(x_pos + offset, shadow_values, bar_width,
                          label=f'Shadow Awareness ({shadow})', alpha=0.8,
                          color=colors_shadow[i % len(colors_shadow)])

            # Termination probability bars
            term_values = [pivot_term.loc[agent, shadow] if shadow in pivot_term.columns else 0
                         for agent in pivot_term.index]
            bars2 = ax.bar(x_pos + offset + bar_width, term_values, bar_width,
                          label=f'Termination Prob ({shadow})', alpha=0.7,
                          color=colors_term[i % len(colors_term)])

            # Expected game length bars
            length_values = [pivot_length.loc[agent, shadow] if shadow in pivot_length.columns else 0
                           for agent in pivot_length.index]
            bars3 = ax.bar(x_pos + offset + 2*bar_width, length_values, bar_width,
                          label=f'Game Length ({shadow})', alpha=0.6,
                          color=colors_length[i % len(colors_length)])

        ax.set_xlabel('LLM Variants', fontweight='bold')
        ax.set_ylabel('Shadow Reasoning Rate', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_shadow.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

        plt.tight_layout()
        output_path = self.output_dir / "shadow_future_reasoning_by_shadow_length.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 2: {output_path}")

    def figure3_formal_methods_usage(self):
        """Figure 3: Formal methods and game theory usage by LLM variants split by shadow length"""

        df = self.formal_df.copy()
        df['shadow_length'] = df['shadow_condition'].apply(self.map_shadow_condition_to_length)

        # Calculate formal methods usage
        agg_data = df.groupby(['provider', 'temperature', 'shadow_condition', 'shadow_length']).agg({
            'uses_technical_formal_methods': 'mean',
            'opponent_modeling_formal_present': 'mean',
            'game_theory_technical_present': 'mean'
        }).reset_index()

        agg_data['agent_label'] = self.create_agent_labels(agg_data)

        # Create compact layout with all shadow conditions in one figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Formal Methods & Game Theory Usage by LLM Variants Across Shadow Conditions', fontsize=14, fontweight='bold')

        # Pivot data for easier plotting
        pivot_formal = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='uses_technical_formal_methods', fill_value=0)
        pivot_opponent = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='opponent_modeling_formal_present', fill_value=0)
        pivot_game = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='game_theory_technical_present', fill_value=0)

        # Get all unique shadow lengths
        shadow_lengths = sorted(agg_data['shadow_length'].unique())
        n_agents = len(pivot_formal.index)
        n_shadows = len(shadow_lengths)

        # Create positions for bars
        x_pos = np.arange(n_agents)
        bar_width = 0.8 / (n_shadows * 3)  # 3 metrics per shadow condition

        # Colors for each metric
        colors_formal = ['#8B0000', '#FF4500', '#DC143C', '#B22222']
        colors_opponent = ['#191970', '#4169E1', '#0000FF', '#6495ED']
        colors_game = ['#B8860B', '#DAA520', '#FFD700', '#F0E68C']

        for i, shadow in enumerate(shadow_lengths):
            offset = (i * 3 * bar_width) - (n_shadows * 1.5 * bar_width) + 1.5 * bar_width

            # Formal methods bars
            formal_values = [pivot_formal.loc[agent, shadow] if shadow in pivot_formal.columns else 0
                           for agent in pivot_formal.index]
            bars1 = ax.bar(x_pos + offset, formal_values, bar_width,
                          label=f'Formal Methods ({shadow})', alpha=0.8,
                          color=colors_formal[i % len(colors_formal)])

            # Opponent modeling bars
            opponent_values = [pivot_opponent.loc[agent, shadow] if shadow in pivot_opponent.columns else 0
                             for agent in pivot_opponent.index]
            bars2 = ax.bar(x_pos + offset + bar_width, opponent_values, bar_width,
                          label=f'Opponent Modeling ({shadow})', alpha=0.7,
                          color=colors_opponent[i % len(colors_opponent)])

            # Game theory bars
            game_values = [pivot_game.loc[agent, shadow] if shadow in pivot_game.columns else 0
                         for agent in pivot_game.index]
            bars3 = ax.bar(x_pos + offset + 2*bar_width, game_values, bar_width,
                          label=f'Game Theory ({shadow})', alpha=0.6,
                          color=colors_game[i % len(colors_game)])

        ax.set_xlabel('LLM Variants', fontweight='bold')
        ax.set_ylabel('Usage Rate', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_formal.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

        plt.tight_layout()
        output_path = self.output_dir / "formal_methods_usage_by_shadow_length.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 3: {output_path}")

    def figure4_tracking_mode_usage(self):
        """Figure 4: Tracking mode cross-phase opponent linking by shadow length"""

        # Filter for tracking experiments only
        tracking_experiments = [
            "experiment_20250912_153315",  # 5%
            "experiment_20250911_112723",  # 10%
            "experiment_20250910_074608",  # 25%
            "experiment_20250910_074557"   # 75%
        ]

        df = self.history_df[self.history_df['experiment'].isin(tracking_experiments)].copy()
        df['shadow_length'] = df['shadow_condition'].apply(self.map_shadow_condition_to_length)

        # Calculate cross-match awareness (which indicates tracking mode usage)
        agg_data = df.groupby(['provider', 'temperature', 'shadow_condition', 'shadow_length']).agg({
            'has_cross_match_awareness': 'mean',
            'cross_match_cross_match_references_present': 'mean',
            'cross_match_population_learning_present': 'mean'
        }).reset_index()

        agg_data['agent_label'] = self.create_agent_labels(agg_data)

        # Create compact layout with all shadow conditions in one figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Tracking Mode: Cross-Phase Opponent Linking by LLM Variants', fontsize=14, fontweight='bold')

        # Pivot data for easier plotting
        pivot_cross = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='has_cross_match_awareness', fill_value=0)
        pivot_refs = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='cross_match_cross_match_references_present', fill_value=0)
        pivot_learning = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='cross_match_population_learning_present', fill_value=0)

        # Get all unique shadow lengths
        shadow_lengths = sorted(agg_data['shadow_length'].unique())
        n_agents = len(pivot_cross.index)
        n_shadows = len(shadow_lengths)

        # Create positions for bars
        x_pos = np.arange(n_agents)
        bar_width = 0.8 / (n_shadows * 3)  # 3 metrics per shadow condition

        # Colors for each metric
        colors_cross = ['#483D8B', '#6A5ACD', '#8A2BE2', '#9370DB']
        colors_refs = ['#008B8B', '#20B2AA', '#48D1CC', '#00CED1']
        colors_learning = ['#DC143C', '#B22222', '#CD5C5C', '#F08080']

        for i, shadow in enumerate(shadow_lengths):
            offset = (i * 3 * bar_width) - (n_shadows * 1.5 * bar_width) + 1.5 * bar_width

            # Cross-match awareness bars
            cross_values = [pivot_cross.loc[agent, shadow] if shadow in pivot_cross.columns else 0
                          for agent in pivot_cross.index]
            bars1 = ax.bar(x_pos + offset, cross_values, bar_width,
                          label=f'Cross-Match Awareness ({shadow})', alpha=0.8,
                          color=colors_cross[i % len(colors_cross)])

            # Match references bars
            refs_values = [pivot_refs.loc[agent, shadow] if shadow in pivot_refs.columns else 0
                         for agent in pivot_refs.index]
            bars2 = ax.bar(x_pos + offset + bar_width, refs_values, bar_width,
                          label=f'Match References ({shadow})', alpha=0.7,
                          color=colors_refs[i % len(colors_refs)])

            # Population learning bars
            learning_values = [pivot_learning.loc[agent, shadow] if shadow in pivot_learning.columns else 0
                             for agent in pivot_learning.index]
            bars3 = ax.bar(x_pos + offset + 2*bar_width, learning_values, bar_width,
                          label=f'Population Learning ({shadow})', alpha=0.6,
                          color=colors_learning[i % len(colors_learning)])

        ax.set_xlabel('LLM Variants', fontweight='bold')
        ax.set_ylabel('Cross-Phase Tracking Rate', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_cross.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

        plt.tight_layout()
        output_path = self.output_dir / "tracking_mode_cross_phase_linking.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 4: {output_path}")

    def generate_all_figures(self):
        """Generate all four refined strategic reasoning figures"""
        print("Generating refined strategic reasoning visualizations...")
        print("=" * 60)

        self.figure1_match_history_usage()
        self.figure2_shadow_future_reasoning()
        self.figure3_formal_methods_usage()
        self.figure4_tracking_mode_usage()

        print("=" * 60)
        print(f"All figures saved to: {self.output_dir}")
        print("Generated 4 refined strategic reasoning figures:")
        print("1. match_history_usage_by_shadow_length.png")
        print("2. shadow_future_reasoning_by_shadow_length.png")
        print("3. formal_methods_usage_by_shadow_length.png")
        print("4. tracking_mode_cross_phase_linking.png")

def main():
    # Set up data directory
    data_dir = Path(__file__).parent / "refined_strategic_reasoning_results"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    # Create visualizer and generate figures
    visualizer = RefinedStrategicReasoningVisualizer(data_dir)
    visualizer.generate_all_figures()

if __name__ == "__main__":
    main()