#!/usr/bin/env python3
"""
Strategic Reasoning Visualization Script
Creates comprehensive visualizations and tables for strategic reasoning analysis

This script generates:
1. Temperature sensitivity heatmaps
2. Provider comparison charts
3. Phase evolution plots
4. Shadow condition sensitivity analysis
5. Formal methods usage breakdown

Author: Strategic Reasoning Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StrategicReasoningVisualizer:
    """Creates visualizations for strategic reasoning analysis results"""

    def __init__(self, output_dir: str = 'analysis_scripts/strategic_reasoning_results'):
        self.output_dir = Path(output_dir)
        self.figure_dir = self.output_dir / 'figures'
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 8)

    def load_analysis_results(self):
        """Load the most recent analysis results"""
        # Find the most recent analysis files
        history_files = list(self.output_dir.glob("match_history_analysis_*.csv"))
        shadow_files = list(self.output_dir.glob("shadow_future_analysis_*.csv"))
        formal_files = list(self.output_dir.glob("formal_methods_analysis_*.csv"))

        if not any([history_files, shadow_files, formal_files]):
            raise FileNotFoundError(f"No analysis results found in {self.output_dir}")

        # Load most recent files
        data = {}
        if history_files:
            latest_history = max(history_files, key=lambda x: x.stat().st_mtime)
            data['history'] = pd.read_csv(latest_history)
            print(f"Loaded history analysis: {latest_history.name}")

        if shadow_files:
            latest_shadow = max(shadow_files, key=lambda x: x.stat().st_mtime)
            data['shadow'] = pd.read_csv(latest_shadow)
            print(f"Loaded shadow analysis: {latest_shadow.name}")

        if formal_files:
            latest_formal = max(formal_files, key=lambda x: x.stat().st_mtime)
            data['formal'] = pd.read_csv(latest_formal)
            print(f"Loaded formal methods analysis: {latest_formal.name}")

        return data

    def create_temperature_sensitivity_heatmap(self, formal_df):
        """Create heatmap showing temperature sensitivity for formal methods usage"""
        if formal_df.empty or 'temperature' not in formal_df.columns:
            print("Insufficient data for temperature sensitivity analysis")
            return None

        # Aggregate by provider and temperature
        temp_data = formal_df.groupby(['provider', 'temperature']).agg({
            'uses_formal_methods': 'mean',
            'game_theory_concepts_present': 'mean',
            'opponent_modeling_present': 'mean',
            'formal_methods_score': 'mean'
        }).round(3)

        # Create pivot tables for heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temperature Sensitivity Analysis: Formal Methods Usage', fontsize=16, fontweight='bold')

        metrics = [
            ('uses_formal_methods', 'Overall Formal Methods Usage'),
            ('game_theory_concepts_present', 'Game Theory Concepts'),
            ('opponent_modeling_present', 'Opponent Modeling'),
            ('formal_methods_score', 'Formal Methods Score')
        ]

        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            # Create pivot table
            pivot_data = temp_data[metric].unstack(level=0, fill_value=0)

            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd',
                       ax=ax, cbar_kws={'label': 'Rate'})
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Provider')
            ax.set_ylabel('Temperature')

        plt.tight_layout()
        filename = self.figure_dir / 'temperature_sensitivity_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved temperature sensitivity heatmap: {filename}")
        return filename

    def create_provider_comparison_chart(self, data_dict):
        """Create comprehensive provider comparison chart"""
        if not data_dict:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Provider Comparison: Strategic Reasoning Capabilities', fontsize=16, fontweight='bold')

        # 1. Match History Awareness
        if 'history' in data_dict:
            history_df = data_dict['history']
            history_summary = history_df.groupby('provider').agg({
                'has_history_awareness': 'mean',
                'explicit_history_keywords_present': 'mean',
                'reciprocity_concepts_present': 'mean',
                'pattern_recognition_present': 'mean'
            }).round(3)

            ax1 = axes[0, 0]
            history_summary.plot(kind='bar', ax=ax1)
            ax1.set_title('Match History Awareness', fontweight='bold')
            ax1.set_ylabel('Rate')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.tick_params(axis='x', rotation=45)

        # 2. Shadow of Future Awareness
        if 'shadow' in data_dict:
            shadow_df = data_dict['shadow']
            shadow_summary = shadow_df.groupby('provider').agg({
                'has_shadow_awareness': 'mean',
                'termination_probability_present': 'mean',
                'future_payoffs_present': 'mean',
                'repeated_game_awareness_present': 'mean'
            }).round(3)

            ax2 = axes[0, 1]
            shadow_summary.plot(kind='bar', ax=ax2)
            ax2.set_title('Shadow of Future Awareness', fontweight='bold')
            ax2.set_ylabel('Rate')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.tick_params(axis='x', rotation=45)

        # 3. Formal Methods Usage
        if 'formal' in data_dict:
            formal_df = data_dict['formal']
            formal_summary = formal_df.groupby('provider').agg({
                'uses_formal_methods': 'mean',
                'game_theory_concepts_present': 'mean',
                'opponent_modeling_present': 'mean',
                'probability_modeling_present': 'mean'
            }).round(3)

            ax3 = axes[1, 0]
            formal_summary.plot(kind='bar', ax=ax3)
            ax3.set_title('Formal Methods Usage', fontweight='bold')
            ax3.set_ylabel('Rate')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.tick_params(axis='x', rotation=45)

            # 4. Sophistication Levels
            soph_data = formal_df.groupby(['provider', 'sophistication_level']).size().unstack(fill_value=0)
            soph_pct = soph_data.div(soph_data.sum(axis=1), axis=0)

            ax4 = axes[1, 1]
            soph_pct.plot(kind='bar', stacked=True, ax=ax4,
                         color=['#ff7f7f', '#ffcc99', '#99ccff', '#99ff99'])
            ax4.set_title('Sophistication Level Distribution', fontweight='bold')
            ax4.set_ylabel('Proportion')
            ax4.legend(title='Sophistication', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        filename = self.figure_dir / 'provider_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved provider comparison chart: {filename}")
        return filename

    def create_phase_evolution_plots(self, data_dict):
        """Create plots showing evolution across phases"""
        if not data_dict:
            return None

        # Check if we have phase data
        has_phase_data = any('phase' in df.columns for df in data_dict.values() if not df.empty)
        if not has_phase_data:
            print("No phase data available for evolution analysis")
            return None

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Evolution Across Phases', fontsize=16, fontweight='bold')

        # 1. History Awareness Evolution
        if 'history' in data_dict and 'phase' in data_dict['history'].columns:
            history_evolution = data_dict['history'].groupby(['phase', 'provider'])['has_history_awareness'].mean().unstack()

            ax1 = axes[0]
            for provider in history_evolution.columns:
                ax1.plot(history_evolution.index, history_evolution[provider],
                        marker='o', label=provider, linewidth=2)
            ax1.set_title('Match History Awareness', fontweight='bold')
            ax1.set_xlabel('Phase')
            ax1.set_ylabel('Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Shadow Awareness Evolution
        if 'shadow' in data_dict and 'phase' in data_dict['shadow'].columns:
            shadow_evolution = data_dict['shadow'].groupby(['phase', 'provider'])['has_shadow_awareness'].mean().unstack()

            ax2 = axes[1]
            for provider in shadow_evolution.columns:
                ax2.plot(shadow_evolution.index, shadow_evolution[provider],
                        marker='s', label=provider, linewidth=2)
            ax2.set_title('Shadow of Future Awareness', fontweight='bold')
            ax2.set_xlabel('Phase')
            ax2.set_ylabel('Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Formal Methods Evolution
        if 'formal' in data_dict and 'phase' in data_dict['formal'].columns:
            formal_evolution = data_dict['formal'].groupby(['phase', 'provider'])['uses_formal_methods'].mean().unstack()

            ax3 = axes[2]
            for provider in formal_evolution.columns:
                ax3.plot(formal_evolution.index, formal_evolution[provider],
                        marker='^', label=provider, linewidth=2)
            ax3.set_title('Formal Methods Usage', fontweight='bold')
            ax3.set_xlabel('Phase')
            ax3.set_ylabel('Rate')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = self.figure_dir / 'phase_evolution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved phase evolution plots: {filename}")
        return filename

    def create_shadow_condition_analysis(self, data_dict):
        """Create analysis of how reasoning changes with shadow conditions"""
        if 'shadow' not in data_dict or data_dict['shadow'].empty:
            print("No shadow condition data available")
            return None

        shadow_df = data_dict['shadow']
        if 'shadow_condition' not in shadow_df.columns:
            print("No shadow_condition column found")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Shadow Condition Sensitivity Analysis', fontsize=16, fontweight='bold')

        # 1. Overall awareness by shadow condition
        shadow_summary = shadow_df.groupby('shadow_condition').agg({
            'has_shadow_awareness': 'mean',
            'termination_probability_present': 'mean',
            'future_payoffs_present': 'mean',
            'mentions_specific_shadow': 'mean'
        }).round(3)

        ax1 = axes[0, 0]
        shadow_summary.plot(kind='bar', ax=ax1)
        ax1.set_title('Shadow Awareness by Condition', fontweight='bold')
        ax1.set_xlabel('Shadow Condition')
        ax1.set_ylabel('Rate')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=0)

        # 2. Provider differences by shadow condition
        provider_shadow = shadow_df.groupby(['shadow_condition', 'provider'])['has_shadow_awareness'].mean().unstack()

        ax2 = axes[0, 1]
        provider_shadow.plot(kind='bar', ax=ax2)
        ax2.set_title('Provider Differences by Shadow Condition', fontweight='bold')
        ax2.set_xlabel('Shadow Condition')
        ax2.set_ylabel('Shadow Awareness Rate')
        ax2.legend(title='Provider', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=0)

        # 3. Specific shadow mention rates
        specific_mentions = shadow_df.groupby('shadow_condition')['mentions_specific_shadow'].mean()

        ax3 = axes[1, 0]
        specific_mentions.plot(kind='bar', ax=ax3, color='skyblue')
        ax3.set_title('Specific Shadow Condition Mentions', fontweight='bold')
        ax3.set_xlabel('Shadow Condition')
        ax3.set_ylabel('Mention Rate')
        ax3.tick_params(axis='x', rotation=0)

        # 4. Temperature effects on shadow awareness
        if 'temperature' in shadow_df.columns:
            temp_shadow = shadow_df.groupby(['temperature', 'shadow_condition'])['has_shadow_awareness'].mean().unstack()

            ax4 = axes[1, 1]
            temp_shadow.plot(kind='bar', ax=ax4)
            ax4.set_title('Temperature Effects on Shadow Awareness', fontweight='bold')
            ax4.set_xlabel('Temperature')
            ax4.set_ylabel('Shadow Awareness Rate')
            ax4.legend(title='Shadow Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.tick_params(axis='x', rotation=0)

        plt.tight_layout()
        filename = self.figure_dir / 'shadow_condition_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved shadow condition analysis: {filename}")
        return filename

    def create_comprehensive_summary_table(self, data_dict):
        """Create a comprehensive LaTeX table for publication"""
        if not data_dict:
            return None

        # Collect all summary statistics
        summary_data = {}

        # History analysis
        if 'history' in data_dict:
            history_df = data_dict['history']
            history_summary = history_df.groupby('provider').agg({
                'has_history_awareness': 'mean',
                'explicit_history_keywords_present': 'mean',
                'reciprocity_concepts_present': 'mean'
            }).round(3)

            for provider in history_summary.index:
                if provider not in summary_data:
                    summary_data[provider] = {}
                summary_data[provider]['History_Awareness'] = history_summary.loc[provider, 'has_history_awareness']
                summary_data[provider]['Explicit_History'] = history_summary.loc[provider, 'explicit_history_keywords_present']

        # Shadow analysis
        if 'shadow' in data_dict:
            shadow_df = data_dict['shadow']
            shadow_summary = shadow_df.groupby('provider').agg({
                'has_shadow_awareness': 'mean',
                'termination_probability_present': 'mean'
            }).round(3)

            for provider in shadow_summary.index:
                if provider not in summary_data:
                    summary_data[provider] = {}
                summary_data[provider]['Shadow_Awareness'] = shadow_summary.loc[provider, 'has_shadow_awareness']
                summary_data[provider]['Termination_Prob'] = shadow_summary.loc[provider, 'termination_probability_present']

        # Formal methods analysis
        if 'formal' in data_dict:
            formal_df = data_dict['formal']
            formal_summary = formal_df.groupby('provider').agg({
                'uses_formal_methods': 'mean',
                'game_theory_concepts_present': 'mean',
                'opponent_modeling_present': 'mean'
            }).round(3)

            for provider in formal_summary.index:
                if provider not in summary_data:
                    summary_data[provider] = {}
                summary_data[provider]['Formal_Methods'] = formal_summary.loc[provider, 'uses_formal_methods']
                summary_data[provider]['Game_Theory'] = formal_summary.loc[provider, 'game_theory_concepts_present']
                summary_data[provider]['Opponent_Modeling'] = formal_summary.loc[provider, 'opponent_modeling_present']

        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data).T
        summary_df = summary_df.fillna(0).round(3)

        # Save as CSV
        csv_filename = self.output_dir / 'comprehensive_summary_table.csv'
        summary_df.to_csv(csv_filename)

        # Create LaTeX table
        latex_filename = self.output_dir / 'strategic_reasoning_table.tex'
        with open(latex_filename, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Strategic Reasoning Analysis Summary by LLM Provider}\n")
            f.write("\\label{tab:strategic_reasoning}\n")
            f.write("\\begin{tabular}{l" + "c" * len(summary_df.columns) + "}\n")
            f.write("\\toprule\n")

            # Header
            headers = ["Provider"] + [col.replace('_', ' ') for col in summary_df.columns]
            f.write(" & ".join(headers) + " \\\\\n")
            f.write("\\midrule\n")

            # Data rows
            for provider, row in summary_df.iterrows():
                values = [provider] + [f"{val:.3f}" for val in row.values]
                f.write(" & ".join(values) + " \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")

            # Add notes
            f.write("\\begin{tablenotes}\n")
            f.write("\\small\n")
            f.write("\\item Note: Values represent the proportion of reasoning instances that exhibit each characteristic.\n")
            f.write("\\item History Awareness: References to game history or opponent behavior.\n")
            f.write("\\item Shadow Awareness: Consideration of future rounds and termination probability.\n")
            f.write("\\item Formal Methods: Use of game theory, decision theory, or formal modeling approaches.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{table}\n")

        print(f"Saved comprehensive summary: {csv_filename}")
        print(f"Saved LaTeX table: {latex_filename}")

        return latex_filename, summary_df

    def generate_all_visualizations(self):
        """Generate all visualizations and tables"""
        print("STRATEGIC REASONING VISUALIZER")
        print("="*50)

        # Load data
        try:
            data_dict = self.load_analysis_results()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run strategic_reasoning_analyzer.py first")
            return

        print(f"Generating visualizations in: {self.figure_dir}")

        generated_files = []

        # Generate visualizations
        if 'formal' in data_dict:
            temp_heatmap = self.create_temperature_sensitivity_heatmap(data_dict['formal'])
            if temp_heatmap:
                generated_files.append(temp_heatmap)

        provider_chart = self.create_provider_comparison_chart(data_dict)
        if provider_chart:
            generated_files.append(provider_chart)

        phase_plots = self.create_phase_evolution_plots(data_dict)
        if phase_plots:
            generated_files.append(phase_plots)

        shadow_analysis = self.create_shadow_condition_analysis(data_dict)
        if shadow_analysis:
            generated_files.append(shadow_analysis)

        # Generate summary table
        latex_table, summary_df = self.create_comprehensive_summary_table(data_dict)
        if latex_table:
            generated_files.append(latex_table)

        print(f"\nGenerated {len(generated_files)} visualization files:")
        for file in generated_files:
            print(f"  {file}")

        # Print key insights
        self._print_key_insights(data_dict, summary_df)

        return generated_files

    def _print_key_insights(self, data_dict, summary_df):
        """Print key insights from the analysis"""
        print("\n" + "="*50)
        print("KEY INSIGHTS")
        print("="*50)

        if summary_df is not None and not summary_df.empty:
            print("\nTop performers by category:")

            if 'History_Awareness' in summary_df.columns:
                top_history = summary_df['History_Awareness'].idxmax()
                print(f"  History Awareness: {top_history} ({summary_df.loc[top_history, 'History_Awareness']:.3f})")

            if 'Shadow_Awareness' in summary_df.columns:
                top_shadow = summary_df['Shadow_Awareness'].idxmax()
                print(f"  Shadow Awareness: {top_shadow} ({summary_df.loc[top_shadow, 'Shadow_Awareness']:.3f})")

            if 'Formal_Methods' in summary_df.columns:
                top_formal = summary_df['Formal_Methods'].idxmax()
                print(f"  Formal Methods: {top_formal} ({summary_df.loc[top_formal, 'Formal_Methods']:.3f})")

        # Temperature sensitivity insights
        if 'formal' in data_dict and 'temperature' in data_dict['formal'].columns:
            formal_df = data_dict['formal']
            temp_corr = formal_df.groupby(['provider', 'temperature'])['uses_formal_methods'].mean().unstack()

            print("\nTemperature sensitivity patterns:")
            for provider in temp_corr.index:
                temps = temp_corr.loc[provider].dropna()
                if len(temps) > 1:
                    trend = "increases" if temps.iloc[-1] > temps.iloc[0] else "decreases"
                    print(f"  {provider}: Formal methods usage {trend} with temperature")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate visualizations for strategic reasoning analysis')
    parser.add_argument('--output_dir', default='analysis_scripts/strategic_reasoning_results',
                       help='Directory containing analysis results')

    args = parser.parse_args()

    # Create visualizer and generate all plots
    visualizer = StrategicReasoningVisualizer(args.output_dir)
    generated_files = visualizer.generate_all_visualizations()

    if generated_files:
        print(f"\nVisualization complete! Generated {len(generated_files)} files.")
    else:
        print("\nNo visualizations generated. Check that analysis data is available.")


if __name__ == "__main__":
    main()