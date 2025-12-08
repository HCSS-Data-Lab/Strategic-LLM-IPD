#!/usr/bin/env python3
"""
Simplified Refined Strategic Reasoning Visualizer
Creates 4 focused visualizations with max 2 insights per plot (8 colors total)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class SimplifiedStrategicReasoningVisualizer:
    def __init__(self, data_dir="refined_strategic_reasoning_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "refined_figures_simplified"
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
            0.05: "5%",
            0.10: "10%",
            0.25: "25%",
            0.75: "75%"
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
                    label = f"GPT5nano"
                else:
                    label = f"GPT5mini"
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

    def figure1_history_awareness(self):
        """Figure 1: Within-Game vs Cross-Match History Awareness (2 insights)"""

        df = self.history_df.copy()
        df['shadow_length'] = df['shadow_condition'].apply(self.map_shadow_condition_to_length)

        # Calculate history awareness metrics
        agg_data = df.groupby(['provider', 'temperature', 'shadow_condition', 'shadow_length', 'agent']).agg({
            'has_within_game_awareness': 'mean',
            'has_cross_match_awareness': 'mean'
        }).reset_index()

        agg_data['agent_label'] = self.create_agent_labels(agg_data)

        # Create compact layout
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('History Awareness: Within-Game vs Cross-Phase by LLM Variants', fontsize=14, fontweight='bold')

        # Order shadow lengths from longest to shortest game (75% → 5%)
        shadow_order = ["75%", "25%", "10%", "5%"]

        # Pivot data for easier plotting
        pivot_within = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='has_within_game_awareness', fill_value=0)
        pivot_cross = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='has_cross_match_awareness', fill_value=0)

        # Reorder columns according to shadow_order
        available_shadows = [s for s in shadow_order if s in pivot_within.columns]
        pivot_within = pivot_within[available_shadows]
        pivot_cross = pivot_cross[available_shadows]

        n_agents = len(pivot_within.index)
        n_shadows = len(available_shadows)

        # Create positions for bars
        x_pos = np.arange(n_agents)
        bar_width = 0.8 / (n_shadows * 2)  # 2 metrics per shadow condition

        # Colors for shadow conditions (ordered from longest to shortest game)
        colors_within = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        colors_cross = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        for i, shadow in enumerate(available_shadows):
            offset = (i * 2 * bar_width) - (n_shadows * bar_width) + bar_width

            # Within-game bars
            within_values = [pivot_within.loc[agent, shadow] if shadow in pivot_within.columns else 0
                           for agent in pivot_within.index]
            ax.bar(x_pos + offset, within_values, bar_width,
                   label=f'Within-Game ({shadow})', alpha=0.8,
                   color=colors_within[i % len(colors_within)])

            # Cross-phase bars
            cross_values = [pivot_cross.loc[agent, shadow] if shadow in pivot_cross.columns else 0
                          for agent in pivot_cross.index]
            ax.bar(x_pos + offset + bar_width, cross_values, bar_width,
                   label=f'Cross-Phase ({shadow})', alpha=0.6,
                   color=colors_cross[i % len(colors_cross)])

        ax.set_xlabel('LLM Variants', fontweight='bold')
        ax.set_ylabel('History Awareness Rate', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_within.index, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        output_path = self.output_dir / "history_awareness_within_vs_cross.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 1: {output_path}")

    def figure2_shadow_reasoning(self):
        """Figure 2: Shadow Awareness vs Game Length Reasoning (2 insights)"""

        df = self.shadow_df.copy()
        df['shadow_length'] = df['shadow_condition'].apply(self.map_shadow_condition_to_length)

        # Calculate shadow awareness metrics
        agg_data = df.groupby(['provider', 'temperature', 'shadow_condition', 'shadow_length', 'agent']).agg({
            'has_precise_shadow_awareness': 'mean',
            'expected_game_length_present': 'mean'
        }).reset_index()

        agg_data['agent_label'] = self.create_agent_labels(agg_data)

        # Create compact layout
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Shadow Reasoning: Awareness vs Game Length Consideration', fontsize=14, fontweight='bold')

        # Order shadow lengths from longest to shortest game (75% → 5%)
        shadow_order = ["75%", "25%", "10%", "5%"]

        # Pivot data for easier plotting
        pivot_shadow = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='has_precise_shadow_awareness', fill_value=0)
        pivot_length = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='expected_game_length_present', fill_value=0)

        # Reorder columns according to shadow_order
        available_shadows = [s for s in shadow_order if s in pivot_shadow.columns]
        pivot_shadow = pivot_shadow[available_shadows]
        pivot_length = pivot_length[available_shadows]

        n_agents = len(pivot_shadow.index)
        n_shadows = len(available_shadows)

        # Create positions for bars
        x_pos = np.arange(n_agents)
        bar_width = 0.8 / (n_shadows * 2)  # 2 metrics per shadow condition

        # Colors for shadow conditions
        colors_shadow = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']
        colors_length = ['#17becf', '#bcbd22', '#ff9896', '#c5b0d5']

        for i, shadow in enumerate(available_shadows):
            offset = (i * 2 * bar_width) - (n_shadows * bar_width) + bar_width

            # Shadow awareness bars
            shadow_values = [pivot_shadow.loc[agent, shadow] if shadow in pivot_shadow.columns else 0
                           for agent in pivot_shadow.index]
            ax.bar(x_pos + offset, shadow_values, bar_width,
                   label=f'Shadow Awareness ({shadow})', alpha=0.8,
                   color=colors_shadow[i % len(colors_shadow)])

            # Game length bars
            length_values = [pivot_length.loc[agent, shadow] if shadow in pivot_length.columns else 0
                           for agent in pivot_length.index]
            ax.bar(x_pos + offset + bar_width, length_values, bar_width,
                   label=f'Game Length ({shadow})', alpha=0.6,
                   color=colors_length[i % len(colors_length)])

        ax.set_xlabel('LLM Variants', fontweight='bold')
        ax.set_ylabel('Shadow Reasoning Rate', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_shadow.index, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        output_path = self.output_dir / "shadow_reasoning_awareness_vs_length.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 2: {output_path}")

    def figure3_game_theory_opponent_modeling(self):
        """Figure 3: Game Theory vs Opponent Modeling (2 insights)"""

        df = self.formal_df.copy()
        df['shadow_length'] = df['shadow_condition'].apply(self.map_shadow_condition_to_length)

        # Calculate formal methods usage
        agg_data = df.groupby(['provider', 'temperature', 'shadow_condition', 'shadow_length', 'agent']).agg({
            'game_theory_technical_present': 'mean',
            'opponent_modeling_formal_present': 'mean'
        }).reset_index()

        agg_data['agent_label'] = self.create_agent_labels(agg_data)

        # Create compact layout
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Strategic Concepts: Game Theory vs Opponent Modeling Usage', fontsize=14, fontweight='bold')

        # Order shadow lengths from longest to shortest game (75% → 5%)
        shadow_order = ["75%", "25%", "10%", "5%"]

        # Pivot data for easier plotting
        pivot_game = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='game_theory_technical_present', fill_value=0)
        pivot_opponent = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='opponent_modeling_formal_present', fill_value=0)

        # Reorder columns according to shadow_order
        available_shadows = [s for s in shadow_order if s in pivot_game.columns]
        pivot_game = pivot_game[available_shadows]
        pivot_opponent = pivot_opponent[available_shadows]

        n_agents = len(pivot_game.index)
        n_shadows = len(available_shadows)

        # Create positions for bars
        x_pos = np.arange(n_agents)
        bar_width = 0.8 / (n_shadows * 2)  # 2 metrics per shadow condition

        # Colors for shadow conditions
        colors_game = ['#B8860B', '#DAA520', '#FFD700', '#F0E68C']
        colors_opponent = ['#191970', '#4169E1', '#0000FF', '#6495ED']

        for i, shadow in enumerate(available_shadows):
            offset = (i * 2 * bar_width) - (n_shadows * bar_width) + bar_width

            # Game theory bars
            game_values = [pivot_game.loc[agent, shadow] if shadow in pivot_game.columns else 0
                         for agent in pivot_game.index]
            ax.bar(x_pos + offset, game_values, bar_width,
                   label=f'Game Theory ({shadow})', alpha=0.8,
                   color=colors_game[i % len(colors_game)])

            # Opponent modeling bars
            opponent_values = [pivot_opponent.loc[agent, shadow] if shadow in pivot_opponent.columns else 0
                             for agent in pivot_opponent.index]
            ax.bar(x_pos + offset + bar_width, opponent_values, bar_width,
                   label=f'Opponent Modeling ({shadow})', alpha=0.7,
                   color=colors_opponent[i % len(colors_opponent)])

        ax.set_xlabel('LLM Variants', fontweight='bold')
        ax.set_ylabel('Usage Rate', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_game.index, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0, max(0.1, max(pivot_game.max().max(), pivot_opponent.max().max()) * 1.1))
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        output_path = self.output_dir / "game_theory_vs_opponent_modeling.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 3: {output_path}")

    def figure4_match_references(self):
        """Figure 4: Cross-Match References vs Population Learning (2 insights for tracking)"""

        # Filter for tracking experiments only (those with cross-match awareness)
        tracking_experiments = [
            "experiment_20250912_153315",  # 5%
            "experiment_20250911_112723",  # 10%
            "experiment_20250910_074608",  # 25%
            "experiment_20250910_074557"   # 75%
        ]

        df = self.history_df[self.history_df['experiment'].isin(tracking_experiments)].copy()
        df['shadow_length'] = df['shadow_condition'].apply(self.map_shadow_condition_to_length)

        # Calculate tracking mode usage
        agg_data = df.groupby(['provider', 'temperature', 'shadow_condition', 'shadow_length', 'agent']).agg({
            'cross_match_cross_match_references_present': 'mean',
            'cross_match_population_learning_present': 'mean'
        }).reset_index()

        agg_data['agent_label'] = self.create_agent_labels(agg_data)

        # Create compact layout
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Cross-Phase Reasoning Types: Specific Match References vs Population Learning', fontsize=14, fontweight='bold')

        # Order shadow lengths from longest to shortest game (75% → 5%)
        shadow_order = ["75%", "25%", "10%", "5%"]

        # Pivot data for easier plotting
        pivot_refs = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='cross_match_cross_match_references_present', fill_value=0)
        pivot_learning = agg_data.pivot_table(index='agent_label', columns='shadow_length', values='cross_match_population_learning_present', fill_value=0)

        # Reorder columns according to shadow_order
        available_shadows = [s for s in shadow_order if s in pivot_refs.columns]
        pivot_refs = pivot_refs[available_shadows]
        pivot_learning = pivot_learning[available_shadows]

        n_agents = len(pivot_refs.index)
        n_shadows = len(available_shadows)

        # Create positions for bars
        x_pos = np.arange(n_agents)
        bar_width = 0.8 / (n_shadows * 2)  # 2 metrics per shadow condition

        # Colors for shadow conditions
        colors_refs = ['#008B8B', '#20B2AA', '#48D1CC', '#00CED1']
        colors_learning = ['#DC143C', '#B22222', '#CD5C5C', '#F08080']

        for i, shadow in enumerate(available_shadows):
            offset = (i * 2 * bar_width) - (n_shadows * bar_width) + bar_width

            # Specific match references bars
            refs_values = [pivot_refs.loc[agent, shadow] if shadow in pivot_refs.columns else 0
                         for agent in pivot_refs.index]
            ax.bar(x_pos + offset, refs_values, bar_width,
                   label=f'Specific Match Refs ({shadow})', alpha=0.8,
                   color=colors_refs[i % len(colors_refs)])

            # Population learning bars
            learning_values = [pivot_learning.loc[agent, shadow] if shadow in pivot_learning.columns else 0
                             for agent in pivot_learning.index]
            ax.bar(x_pos + offset + bar_width, learning_values, bar_width,
                   label=f'Population Learning ({shadow})', alpha=0.6,
                   color=colors_learning[i % len(colors_learning)])

        ax.set_xlabel('LLM Variants', fontweight='bold')
        ax.set_ylabel('Cross-Phase Reasoning Type Rate', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_refs.index, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0, max(0.1, max(pivot_refs.max().max(), pivot_learning.max().max()) * 1.1))
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        output_path = self.output_dir / "match_references_vs_population_learning.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 4: {output_path}")

    def generate_all_figures(self):
        """Generate all four simplified strategic reasoning figures"""
        print("Generating simplified strategic reasoning visualizations...")
        print("=" * 60)

        self.figure1_history_awareness()
        self.figure2_shadow_reasoning()
        self.figure3_game_theory_opponent_modeling()
        self.figure4_match_references()

        print("=" * 60)
        print(f"All figures saved to: {self.output_dir}")
        print("Generated 4 simplified strategic reasoning figures:")
        print("1. history_awareness_within_vs_cross.png")
        print("2. shadow_reasoning_awareness_vs_length.png")
        print("3. game_theory_vs_opponent_modeling.png")
        print("4. match_references_vs_population_learning.png")

def main():
    # Create visualizer and generate figures
    visualizer = SimplifiedStrategicReasoningVisualizer()
    visualizer.generate_all_figures()

if __name__ == "__main__":
    main()