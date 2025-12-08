#!/usr/bin/env python3
"""
Strategic Reasoning Analysis Runner
Convenience script to run the complete strategic reasoning analysis pipeline

Usage:
    python analysis_scripts/run_strategic_analysis.py
    python analysis_scripts/run_strategic_analysis.py --results_dir custom_results
    python analysis_scripts/run_strategic_analysis.py --visualize_only  # Skip analysis, just visualize existing results

Author: Strategic Reasoning Analysis Pipeline
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_analysis(results_dir, output_dir):
    """Run the strategic reasoning analyzer"""
    print("="*60)
    print("STEP 1: RUNNING STRATEGIC REASONING ANALYSIS")
    print("="*60)

    analyzer_script = Path(__file__).parent / "strategic_reasoning_analyzer.py"

    cmd = [
        sys.executable, str(analyzer_script),
        "--results_dir", results_dir,
        "--output_dir", output_dir
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Analysis failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def run_visualization(output_dir):
    """Run the strategic reasoning visualizer"""
    print("\n" + "="*60)
    print("STEP 2: GENERATING VISUALIZATIONS")
    print("="*60)

    visualizer_script = Path(__file__).parent / "strategic_reasoning_visualizer.py"

    cmd = [
        sys.executable, str(visualizer_script),
        "--output_dir", output_dir
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Visualization failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install with: pip install " + " ".join(missing))
        return False

    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run complete strategic reasoning analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete analysis on default results directory
    python analysis_scripts/run_strategic_analysis.py

    # Run on custom results directory
    python analysis_scripts/run_strategic_analysis.py --results_dir my_results

    # Only generate visualizations (skip analysis)
    python analysis_scripts/run_strategic_analysis.py --visualize_only

    # Custom output directory
    python analysis_scripts/run_strategic_analysis.py --output_dir my_analysis_output
        """
    )

    parser.add_argument('--results_dir', default='results',
                       help='Directory containing experiment results (default: results)')
    parser.add_argument('--output_dir', default='analysis_scripts/strategic_reasoning_results',
                       help='Directory to save analysis results (default: analysis_scripts/strategic_reasoning_results)')
    parser.add_argument('--visualize_only', action='store_true',
                       help='Only run visualization (skip analysis step)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only run analysis (skip visualization step)')

    args = parser.parse_args()

    print("STRATEGIC REASONING ANALYSIS PIPELINE")
    print("="*60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check if results directory exists
    if not Path(args.results_dir).exists():
        print(f"Error: Results directory '{args.results_dir}' does not exist!")
        return 1

    success = True

    # Run analysis step
    if not args.visualize_only:
        success = run_analysis(args.results_dir, args.output_dir)
        if not success:
            print("Analysis step failed!")
            return 1

    # Run visualization step
    if not args.analyze_only:
        success = run_visualization(args.output_dir)
        if not success:
            print("Visualization step failed!")
            return 1

    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)

    output_path = Path(args.output_dir)
    if output_path.exists():
        print(f"Results saved to: {output_path.absolute()}")

        # List key output files
        key_files = [
            "strategic_reasoning_summary_*.txt",
            "comprehensive_summary_table.csv",
            "strategic_reasoning_table.tex",
            "figures/*.png"
        ]

        print("\nKey output files:")
        for pattern in key_files:
            files = list(output_path.glob(pattern))
            if files:
                if len(files) == 1:
                    print(f"  ✓ {files[0].name}")
                else:
                    print(f"  ✓ {len(files)} files matching {pattern}")
            else:
                print(f"  - No files matching {pattern}")

    print("\nAnalysis methodology:")
    print("="*30)
    print("1. MATCH HISTORY ANALYSIS:")
    print("   - Explicit history keywords: 'history', 'previous round', 'last move', etc.")
    print("   - Opponent behavior tracking: 'opponent cooperated', 'they defected', etc.")
    print("   - Reciprocity concepts: 'reciprocate', 'retaliate', 'tit for tat', etc.")
    print("   - Pattern recognition: 'pattern', 'consistent', 'always cooperates', etc.")

    print("\n2. SHADOW OF FUTURE ANALYSIS:")
    print("   - Termination probability: 'termination prob', 'continuation prob', 'shadow of', etc.")
    print("   - Future payoffs: 'future payoff', 'long-term', 'expected value', etc.")
    print("   - Repeated game awareness: 'repeated', 'iteration', 'multiple rounds', etc.")

    print("\n3. FORMAL METHODS ANALYSIS:")
    print("   - Game theory concepts: 'game theory', 'nash equilibrium', 'dominant strategy', etc.")
    print("   - Decision theory: 'expected utility', 'utility function', 'rational choice', etc.")
    print("   - Opponent modeling: 'opponent model', 'predict opponent', 'infer strategy', etc.")
    print("   - Temperature sensitivity: Breakdown by provider and temperature settings")

    return 0

if __name__ == "__main__":
    exit(main())