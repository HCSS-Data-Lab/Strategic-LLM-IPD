#!/usr/bin/env python3
"""
LaTeX Table Generator for IPD Experiment Results
Generates comprehensive LaTeX tables from agent composition analysis data
"""

import json
import os
from typing import Dict, Any, List
from collections import defaultdict

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text

def get_strategy_order() -> Dict[str, int]:
    """Define the canonical ordering of strategies for tables."""
    # 13 Canonical strategies (1-13)
    canonical_strategies = [
        'TitForTat', 'GrimTrigger', 'SuspiciousTitForTat', 'GenerousTitForTat',
        'ForgivingGrimTrigger', 'Gradual', 'SoftGrudger', 'Prober', 'Detective',
        'Alternator', 'Random', 'WinStayLoseShift', 'Bayesian'
    ]
    
    # Adaptive learning strategies (14-16)
    adaptive_strategies = [
        'QLearning', 'ThompsonSampling', 'GradientMetaLearner'
    ]
    
    # LLM strategies grouped by model with temperature order
    # Claude strategies
    claude_strategies = [
        'Claude4-Sonnet_T02', 'Claude4-Sonnet_T05', 'Claude4-Sonnet_T08'
    ]
    
    # Mistral strategies
    mistral_strategies = [
        'Mistral-Large_T02', 'Mistral-Large_T07', 'Mistral-Large_T12'
    ]
    
    # Gemini strategies
    gemini_strategies = [
        'Gemini25Pro_T02', 'Gemini25Pro_T07', 'Gemini25Pro_T12'
    ]
    
    # OpenAI strategies (GPT models before GPT4mini)
    openai_strategies = [
        'GPT5nanomini_T1', 'GPT5nano_T1', 'GPT4mini_T1'
    ]
    
    # Build ordering dictionary
    order_dict = {}
    order = 1
    
    # Add canonical strategies
    for strategy in canonical_strategies:
        order_dict[strategy] = order
        order += 1
    
    # Add adaptive strategies
    for strategy in adaptive_strategies:
        order_dict[strategy] = order
        order += 1
    
    # Add LLM strategies in groups
    for strategy in claude_strategies:
        order_dict[strategy] = order
        order += 1
    
    for strategy in mistral_strategies:
        order_dict[strategy] = order
        order += 1
    
    for strategy in gemini_strategies:
        order_dict[strategy] = order
        order += 1
    
    for strategy in openai_strategies:
        order_dict[strategy] = order
        order += 1
    
    return order_dict

def get_strategy_category(strategy: str) -> str:
    """Get the category of a strategy for section headers."""
    if any(llm in strategy for llm in ['Claude']):
        return "Anthropic"
    elif any(llm in strategy for llm in ['Mistral']):
        return "Mistral"
    elif any(llm in strategy for llm in ['Gemini']):
        return "Gemini"
    elif any(llm in strategy for llm in ['GPT', 'GPT4mini']):
        return "OpenAI"
    elif strategy in ['QLearning', 'ThompsonSampling', 'GradientMetaLearner']:
        return "Adaptive"
    else:
        return "Classical"

def sort_strategies_by_canonical_order(strategies: List[str]) -> List[str]:
    """Sort strategies according to canonical ordering."""
    order_dict = get_strategy_order()
    
    def get_order(strategy: str) -> int:
        # Try exact match first
        if strategy in order_dict:
            return order_dict[strategy]
        
        # Try without temperature suffix for LLM agents
        if '_T' in strategy:
            base_strategy = strategy
            return order_dict.get(base_strategy, 1000)  # Unknown strategies go to end
        
        # Unknown strategies go to end
        return 1000
    
    return sorted(strategies, key=get_order)

def format_strategy_name(strategy: str) -> str:
    """Format strategy names for better LaTeX display."""
    # Handle temperature variants for LLM agents
    if '_T' in strategy and any(llm in strategy for llm in ['GPT', 'Claude', 'Mistral', 'Gemini', 'GPT4mini']):
        parts = strategy.split('_T')
        if len(parts) == 2:
            base_name = parts[0]
            temp = parts[1]
            # Convert temperature format
            if temp.startswith('0') and len(temp) == 2:
                temp_val = f"0.{temp[1:]}"
            elif temp.isdigit() and len(temp) <= 2:
                temp_val = f"0.{temp}" if len(temp) == 1 else f"1.{temp[1:]}" if temp.startswith('1') else temp
            else:
                temp_val = temp
            return f"{base_name} (T={temp_val})"
    
    # Handle other naming conventions - very compact names
    replacements = {
        'TitForTat': 'TFT',
        'WinStayLoseShift': 'WSLS',
        'GenerousTitForTat': 'Generous TFT',
        'SuspiciousTitForTat': 'Suspicious TFT',
        'ForgivingGrimTrigger': 'Forgiving Grim',
        'GrimTrigger': 'Grim',
        'GradientMetaLearner': 'Gradient Meta',
        'ThompsonSampling': 'Thompson',
        'QLearning': 'Q-Learning',
        'SoftGrudger': 'Soft Grudger'
    }
    
    return replacements.get(strategy, strategy)

def generate_experiment_table(experiment_data: Dict[str, Any], table_type: str = "comprehensive") -> str:
    """
    Generate a LaTeX table for a single experiment.
    
    Args:
        experiment_data: Data for a single experiment
        table_type: Type of table ("comprehensive", "performance", "evolutionary")
    
    Returns:
        LaTeX table string
    """
    experiment_name = experiment_data['experiment_name']
    phases = experiment_data.get('phases', {})
    
    if not phases:
        return f"% No phase data available for experiment {experiment_name}\n"
    
    # Sort phases by phase number
    sorted_phases = sorted(phases.items(), key=lambda x: x[1].get('phase_number', 0))
    
    # Extract experiment metadata
    shadow_conditions = set()
    for phase_key, phase_data in sorted_phases:
        shadow_conditions.add(phase_data.get('shadow_condition', 0))
    
    shadow_condition = next(iter(shadow_conditions)) if shadow_conditions else 0
    
    if table_type == "comprehensive":
        return generate_comprehensive_table(experiment_name, sorted_phases, shadow_condition)
    elif table_type == "performance":
        return generate_performance_table(experiment_name, sorted_phases, shadow_condition)
    elif table_type == "evolutionary":
        return generate_evolutionary_table(experiment_name, sorted_phases, shadow_condition)
    else:
        raise ValueError(f"Unknown table type: {table_type}")

def generate_comprehensive_table(experiment_name: str, sorted_phases: List, shadow_condition: float) -> str:
    """Generate a comprehensive table with all metrics."""
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\scriptsize")  # Use smaller font for comprehensive table
    latex.append(f"\\caption{{Comprehensive Results for Experiment {escape_latex(experiment_name)} (Shadow={shadow_condition*100:.0f}\\%)}}")
    latex.append(f"\\label{{tab:{experiment_name.lower()}_comprehensive}}")
    
    # Define column format - using landscape orientation for comprehensive table
    latex.append("\\begin{adjustbox}{width=\\textwidth,center}")
    latex.append("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
    latex.append("\\hline")
    
    # Table header
    latex.append("\\textbf{Strategy} & \\textbf{Phase} & \\textbf{Score/Move} & \\textbf{Score/Match} & \\textbf{Total Score} & \\textbf{Total Rounds} & \\textbf{Matches} & \\textbf{Rel. Fitness} & \\textbf{Raw Count} \\\\")
    latex.append("\\hline")
    
    # Collect all strategies across phases
    all_strategies = set()
    for phase_key, phase_data in sorted_phases:
        performance = phase_data.get('strategy_performance', {})
        all_strategies.update(performance.keys())
    
    # Sort strategies by canonical order
    sorted_strategies = sort_strategies_by_canonical_order(list(all_strategies))
    
    # Generate table rows
    for strategy in sorted_strategies:
        strategy_display = format_strategy_name(strategy)
        strategy_escaped = escape_latex(strategy_display)
        
        first_row = True
        for phase_key, phase_data in sorted_phases:
            phase_num = phase_data.get('phase_number', 0)
            performance = phase_data.get('strategy_performance', {})
            evolutionary = phase_data.get('evolutionary_metrics', {})
            
            if strategy in performance:
                perf = performance[strategy]
                evo = evolutionary.get(strategy, {})
                
                if first_row:
                    # Multi-row cell for strategy name
                    num_phases_with_data = sum(1 for _, pd in sorted_phases if strategy in pd.get('strategy_performance', {}))
                    latex.append(f"\\multirow{{{num_phases_with_data}}}{{*}}{{{strategy_escaped}}} & ")
                    first_row = False
                else:
                    latex.append(" & ")
                
                # Add data
                latex.append(f"{phase_num} & ")
                latex.append(f"{perf['score_per_move']:.3f} & ")
                latex.append(f"{perf['score_per_match']:.2f} & ")
                latex.append(f"{perf['total_score']:.0f} & ")
                latex.append(f"{perf['total_rounds']} & ")
                latex.append(f"{perf['matches_played']} & ")
                latex.append(f"{evo.get('relative_fitness', 0):.3f} & ")
                latex.append(f"{evo.get('raw_count', 0):.3f} \\\\")
        
        latex.append("\\hline")
    
    latex.append("\\end{tabular}")
    latex.append("\\end{adjustbox}")
    latex.append("\\end{table}")
    latex.append("")
    
    return "\n".join(latex)

def generate_performance_table(experiment_name: str, sorted_phases: List, shadow_condition: float) -> str:
    """Generate a performance-focused table."""
    
    latex = []
    latex.append("\\begin{sidewaystable}[htbp]")
    latex.append("\\centering")
    latex.append("\\scriptsize")
    latex.append(f"\\caption{{Performance Metrics (Shadow={shadow_condition*100:.0f}\\%). Score represents the average payoff per move for each strategy across evolutionary phases.}}")
    latex.append(f"\\label{{tab:{experiment_name.lower()}_performance}}")
    
    # Column headers based on number of phases - very compact
    num_phases = len(sorted_phases)
    col_spec = "|l|" + "cc|" * num_phases  # No separator between Score and Matches
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\hline")
    
    # Multi-column headers - very compact
    header_line = "\\textbf{Strategy}"
    for phase_key, phase_data in sorted_phases:
        phase_num = phase_data.get('phase_number', 0)
        header_line += f" & \\multicolumn{{2}}{{c|}}{{\\textbf{{P{phase_num}}}}}"  # Just "P" for Phase
    header_line += " \\\\"
    latex.append(header_line)
    latex.append("\\hline")
    
    # Sub-headers - keep full column names
    subheader_line = ""
    for phase_key, phase_data in sorted_phases:
        subheader_line += " & \\textbf{Score/Move} & \\textbf{Matches}"  # Full column names
    subheader_line += " \\\\"
    latex.append(subheader_line)
    latex.append("\\hline")
    
    # Collect all strategies
    all_strategies = set()
    for phase_key, phase_data in sorted_phases:
        performance = phase_data.get('strategy_performance', {})
        all_strategies.update(performance.keys())
    
    # Sort strategies by canonical order
    sorted_strategies = sort_strategies_by_canonical_order(list(all_strategies))
    
    # Generate table rows with section headers
    prev_category = None
    for strategy in sorted_strategies:
        # Add section headers for LLM groups
        current_category = get_strategy_category(strategy)
        if current_category != prev_category and current_category in ["Anthropic", "Mistral", "Gemini", "OpenAI"]:
            latex.append("\\hline")
            latex.append(f"\\multicolumn{{{1 + 2*num_phases}}}{{|c|}}{{\\textbf{{{current_category}}}}} \\\\")
            latex.append("\\hline")
        prev_category = current_category
        
        strategy_display = format_strategy_name(strategy)
        strategy_escaped = escape_latex(strategy_display)
        
        row = f"{strategy_escaped}"
        
        for phase_key, phase_data in sorted_phases:
            performance = phase_data.get('strategy_performance', {})
            
            if strategy in performance:
                perf = performance[strategy]
                row += f" & {perf['score_per_move']:.3f}"
                row += f" & {perf['matches_played']}"
            else:
                row += " & -- & --"
        
        row += " \\\\"
        latex.append(row)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{sidewaystable}")
    latex.append("")
    
    return "\n".join(latex)

def generate_evolutionary_table(experiment_name: str, sorted_phases: List, shadow_condition: float) -> str:
    """Generate an evolutionary metrics focused table."""
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\footnotesize")
    latex.append(f"\\caption{{Evolutionary Fitness (Shadow={shadow_condition*100:.0f}\\%)}}")
    latex.append(f"\\label{{tab:{experiment_name.lower()}_evolutionary}}")
    
    # Column headers based on number of phases - only Fitness
    num_phases = len(sorted_phases)
    col_spec = "|l|" + "c|" * num_phases
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\hline")
    
    # Multi-column headers
    header_line = "\\textbf{Strategy}"
    for phase_key, phase_data in sorted_phases:
        phase_num = phase_data.get('phase_number', 0)
        header_line += f" & \\textbf{{Phase {phase_num}}}"
    header_line += " \\\\"
    latex.append(header_line)
    latex.append("\\hline")
    
    # Collect all strategies
    all_strategies = set()
    for phase_key, phase_data in sorted_phases:
        evolutionary = phase_data.get('evolutionary_metrics', {})
        all_strategies.update(evolutionary.keys())
    
    # Sort strategies by canonical order
    sorted_strategies = sort_strategies_by_canonical_order(list(all_strategies))
    
    # Generate table rows with section headers
    prev_category = None
    for strategy in sorted_strategies:
        # Add section headers for LLM groups
        current_category = get_strategy_category(strategy)
        if current_category != prev_category and current_category in ["Claude", "Mistral", "Gemini", "OpenAI"]:
            latex.append("\\hline")
            latex.append(f"\\multicolumn{{{1 + num_phases}}}{{|c|}}{{\\textbf{{{current_category}}}}} \\\\")
            latex.append("\\hline")
        prev_category = current_category
        
        strategy_display = format_strategy_name(strategy)
        strategy_escaped = escape_latex(strategy_display)
        
        row = f"{strategy_escaped}"
        
        for phase_key, phase_data in sorted_phases:
            evolutionary = phase_data.get('evolutionary_metrics', {})
            
            if strategy in evolutionary:
                evo = evolutionary[strategy]
                row += f" & {evo['fitness']:.3f}"
            else:
                row += " & --"
        
        row += " \\\\"
        latex.append(row)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    
    return "\n".join(latex)

def generate_all_experiment_tables(analysis_file: str, output_file: str = None, table_types: List[str] = None) -> str:
    """
    Generate LaTeX tables for all experiments from the analysis file.
    
    Args:
        analysis_file: Path to the JSON analysis file
        output_file: Optional output file path
        table_types: List of table types to generate ["comprehensive", "performance", "evolutionary"]
    
    Returns:
        LaTeX content as string
    """
    if table_types is None:
        table_types = ["performance", "evolutionary"]  # Default to most useful tables
    
    # Load analysis data
    try:
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading analysis file {analysis_file}: {e}")
    
    experiments = analysis_data.get('experiments', {})
    
    if not experiments:
        raise ValueError("No experiments found in analysis file")
    
    latex_content = []
    
    # Add LaTeX preamble
    latex_content.append("% LaTeX tables generated from IPD experiment analysis")
    latex_content.append("% Requires packages: multirow, array, rotating")
    latex_content.append("% \\usepackage{multirow}")
    latex_content.append("% \\usepackage{array}")
    latex_content.append("% \\usepackage{rotating}")
    latex_content.append("")
    
    # Generate tables for each experiment
    for experiment_name, experiment_data in sorted(experiments.items()):
        latex_content.append(f"% Tables for experiment: {experiment_name}")
        latex_content.append("")
        
        for table_type in table_types:
            try:
                table_latex = generate_experiment_table(experiment_data, table_type)
                latex_content.append(table_latex)
            except Exception as e:
                latex_content.append(f"% Error generating {table_type} table for {experiment_name}: {e}")
                latex_content.append("")
        
        latex_content.append("\\clearpage")
        latex_content.append("")
    
    # Join all content
    full_latex = "\n".join(latex_content)
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(full_latex)
            print(f"LaTeX tables saved to: {output_file}")
        except Exception as e:
            print(f"Error saving to {output_file}: {e}")
    
    return full_latex

def generate_summary_table(analysis_file: str) -> str:
    """Generate a summary table across all experiments."""
    
    # Load analysis data
    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)
    
    experiments = analysis_data.get('experiments', {})
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\footnotesize")
    latex.append("\\caption{Experiment Summary Statistics}")
    latex.append("\\label{tab:experiment_summary}")
    
    latex.append("\\begin{tabular}{|l|c|c|c|c|c|}")
    latex.append("\\hline")
    latex.append("\\textbf{Condition} & \\textbf{Shadow \\%} & \\textbf{Phases} & \\textbf{Agents/Phase} & \\textbf{Matches/Phase} & \\textbf{Avg Rounds/Match} \\\\")
    latex.append("\\hline")
    
    for experiment_name, experiment_data in sorted(experiments.items()):
        phases = experiment_data.get('phases', {})
        
        if phases:
            # Get data from first phase for summary stats
            first_phase = next(iter(phases.values()))
            shadow_condition = first_phase.get('shadow_condition', 0)
            num_phases = len(phases)
            agents_per_phase = first_phase.get('total_unique_agents', 0)
            
            # Calculate average matches and rounds per match
            total_matches = 0
            total_avg_rounds = 0
            valid_phases = 0
            
            for phase_data in phases.values():
                match_stats = phase_data.get('match_statistics', {})
                if match_stats.get('total_matches', 0) > 0:
                    total_matches += match_stats['total_matches']
                    total_avg_rounds += match_stats.get('avg_rounds_per_match', 0)
                    valid_phases += 1
            
            avg_matches_per_phase = total_matches / valid_phases if valid_phases > 0 else 0
            overall_avg_rounds = total_avg_rounds / valid_phases if valid_phases > 0 else 0
            
            latex.append(f"Shadow {shadow_condition*100:.0f}\\% & {shadow_condition*100:.0f} & {num_phases} & {agents_per_phase} & {avg_matches_per_phase:.0f} & {overall_avg_rounds:.1f} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    
    return "\n".join(latex)

def main():
    """Main function to generate LaTeX tables."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from IPD experiment analysis")
    parser.add_argument("--analysis-file", default="../results/all_experiments_composition_analysis.json",
                       help="Path to analysis JSON file")
    parser.add_argument("--output-file", default="experiment_tables.tex",
                       help="Output LaTeX file")
    parser.add_argument("--table-types", nargs="+", 
                       choices=["comprehensive", "performance", "evolutionary", "summary"],
                       default=["performance", "summary"],
                       help="Types of tables to generate")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.analysis_file):
        print(f"Error: Analysis file {args.analysis_file} not found")
        print("Run agent_composition_tracker.py first to generate the analysis")
        return
    
    try:
        latex_content = []
        
        # Add summary table if requested
        if "summary" in args.table_types:
            print("Generating summary table...")
            summary_latex = generate_summary_table(args.analysis_file)
            latex_content.append(summary_latex)
            args.table_types.remove("summary")  # Remove from list for main generation
        
        # Generate main tables
        if args.table_types:
            print(f"Generating {', '.join(args.table_types)} tables...")
            main_latex = generate_all_experiment_tables(args.analysis_file, table_types=args.table_types)
            latex_content.append(main_latex)
        
        # Combine and save
        full_latex = "\n".join(latex_content)
        
        with open(args.output_file, 'w') as f:
            f.write(full_latex)
        
        print(f"LaTeX tables generated successfully!")
        print(f"Output saved to: {args.output_file}")
        print(f"Include in your LaTeX document with: \\input{{{args.output_file}}}")
        
    except Exception as e:
        print(f"Error generating tables: {e}")

if __name__ == "__main__":
    main()