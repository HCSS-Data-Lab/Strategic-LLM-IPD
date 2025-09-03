"""
IPD Suite: Comprehensive framework for analyzing LLM strategic behavior
in the Iterated Prisoner's Dilemma based on Payne & Alloui-Cros (2025)
"""

from .agents import (
    # Classical strategies
    TitForTat, AlwaysCooperate, AlwaysDefect, Random,
    GrimTrigger, Pavlov, WinStayLoseShift,
    GenerousTitForTat, SuspiciousTitForTat, Prober, Gradual, Alternator, Bayesian,
    
    # Behavioral strategies
    ForgivingGrimTrigger, Detective, SoftGrudger,
    
    # Adaptive strategies
    QLearningAgent, ThompsonSampling, GradientMetaLearner,
    
    # LLM agents
    GPT4Agent, ClaudeAgent, MistralAgent, GeminiAgent
)

from .tournament import Tournament, TournamentResult, LLMShowdown, MatchHistoryManager
from .analysis import (
    calculate_strategic_fingerprint,
    analyze_rationales,
    create_fingerprint_visualization,
    analyze_horizon_awareness,
    generate_comprehensive_report
)
from .utils import load_env_vars, format_history, calculate_payoff

__version__ = "1.0.0"
__all__ = [
    # Agents
    "TitForTat", "AlwaysCooperate", "AlwaysDefect", "Random",
    "GrimTrigger", "Pavlov", "WinStayLoseShift",
    "GenerousTitForTat", "SuspiciousTitForTat", "Prober", "Gradual", "Alternator", "Bayesian",
    "ForgivingGrimTrigger", "Detective", "SoftGrudger",
    "QLearningAgent", "ThompsonSampling", "GradientMetaLearner",
    "GPT4Agent", "ClaudeAgent", "MistralAgent", "GeminiAgent",
    
    # Tournament
    "Tournament", "TournamentResult", "LLMShowdown", "MatchHistoryManager",
    
    # Analysis
    "calculate_strategic_fingerprint", "analyze_rationales",
    "create_fingerprint_visualization", "analyze_horizon_awareness",
    "generate_comprehensive_report",
    
    # Utils
    "load_env_vars", "format_history", "calculate_payoff"
]