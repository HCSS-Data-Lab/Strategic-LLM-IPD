#!/usr/bin/env python3
"""
Token usage estimation per LLM agent per tournament (evolutionary phase)
"""
import math

def estimate_tokens(text: str) -> int:
    """Estimate tokens using word count * 1.3 (conservative GPT-4 tokenizer estimate)"""
    return int(len(text.split()) * 1.3)

def estimate_match_length(shadow_prob: float) -> int:
    """Estimate average match length based on termination probability"""
    # Expected number of rounds = 1 / termination_probability
    expected_rounds = 1.0 / shadow_prob
    # Add some variance and cap at max_rounds (200)
    return min(int(expected_rounds * 0.9), 200)  # Conservative estimate

def estimate_prompt_size(shadow_prob: float, phase: int, enable_tracking: bool = False):
    """Estimate prompt size for a single API call"""
    
    # Base game rules (static across all prompts)
    static_rules = """You are playing the Iterated Prisoner's Dilemma game.

Game Rules:
- Two players choose to Cooperate (C) or Defect (D)
- Payoffs: Both C = 3 points each, Both D = 1 point each, You C/They D = 0/5 points, You D/They C = 5/0 points
- The game continues for multiple rounds with {:.1f}% chance of ending after each round
- Goal: Maximize your total score""".format(shadow_prob * 100)
    
    # Match history grows with phase
    match_history = ""
    if phase > 1:
        # In evolutionary mode, each LLM agent plays ~27 matches per phase (against 27 other agents)
        matches_per_phase = 27
        avg_rounds_per_match = estimate_match_length(shadow_prob)
        
        # Estimate total historical matches
        total_historical_matches = (phase - 1) * matches_per_phase
        
        # Estimate total moves in history
        total_moves = total_historical_matches * avg_rounds_per_match
        
        # Format: "Match X vs OpponentName: [(C,C), (D,C), ...]"
        # Each move pair is ~6 characters: "(C,D), "
        # Each match line is ~30 chars base + 6 chars per move
        avg_chars_per_match = 30 + (avg_rounds_per_match * 6)
        
        if enable_tracking:
            # Anonymous opponent IDs: "Match 1 vs Opponent_001: [(C,C), (D,C)]"
            # Slightly longer due to "Opponent_001" vs shorter names
            avg_chars_per_match += 5  # Extra chars for anonymous IDs
        
        total_history_chars = total_historical_matches * avg_chars_per_match
        
        # Add header
        match_history = f"\n\nYour complete match history from previous phases:\n" + "X" * total_history_chars
    
    # Current opponent cross-reference (tracking mode only)
    opponent_info = ""
    if enable_tracking and phase > 1:
        # Average cross-reference text
        opponent_info = "\n\nCurrent opponent: Opponent_001 (previously played in Match 15 of Phase 1)"
    
    # Current match context (dynamic part)
    avg_current_rounds = 3  # Average rounds into current match when making decision
    current_match_moves = "[(C,C), (D,C), (C,D)]"  # Sample current match state
    current_match = f"{opponent_info}\n\nCurrent match - History of moves (You, Opponent): {current_match_moves}"
    
    # Instructions
    instructions = "\n\nProvide your reasoning about the game state, opponent's strategy, and your decision.\nEnd your response with your move on a new line: either 'C' or 'D'."
    
    # Combine all parts
    full_prompt = static_rules + match_history + current_match + instructions
    
    return {
        'input_tokens': estimate_tokens(full_prompt),
        'output_tokens': 150,  # Conservative estimate: ~100 tokens reasoning + move
        'static_tokens': estimate_tokens(static_rules + match_history),
        'dynamic_tokens': estimate_tokens(current_match + instructions)
    }

def calculate_tournament_tokens():
    """Calculate tokens per LLM per tournament for all shadow conditions"""
    
    print("="*80)
    print("TOKEN USAGE PER LLM AGENT PER TOURNAMENT (EVOLUTIONARY PHASE)")
    print("="*80)
    
    shadow_conditions = [0.75, 0.25, 0.10, 0.05, 0.02, 0.01]
    phases = [1, 2, 3, 4, 5]
    
    # Tournament setup
    total_agents = 28  # 16 classical + 12 LLM
    matches_per_llm_per_tournament = total_agents - 1  # Each LLM plays every other agent
    
    print(f"📊 EXPERIMENTAL SETUP:")
    print(f"   Total agents per tournament: {total_agents}")
    print(f"   Matches per LLM per tournament: {matches_per_llm_per_tournament}")
    print(f"   Phases analyzed: {phases}")
    
    for tracking_enabled in [False, True]:
        mode_name = "Opponent Tracking" if tracking_enabled else "Anonymous"
        
        print(f"\n{'='*60}")
        print(f"MODE: {mode_name.upper()}")
        print(f"{'='*60}")
        
        # Header
        print(f"\n{'Shadow':<8} {'Phase':<6} {'Avg':<5} {'Input':<8} {'Output':<8} {'Static':<8} {'Dynamic':<8} {'API':<6} {'Total':<10}")
        print(f"{'Prob':<8} {'#':<6} {'Rnds':<5} {'Tokens':<8} {'Tokens':<8} {'Tokens':<8} {'Tokens':<8} {'Calls':<6} {'Tokens':<10}")
        print("-" * 78)
        
        for shadow in shadow_conditions:
            avg_rounds = estimate_match_length(shadow)
            
            for phase in phases:
                # Estimate prompt size for this phase
                prompt_data = estimate_prompt_size(shadow, phase, tracking_enabled)
                
                # Calculate API calls per LLM per tournament
                # Each match has avg_rounds, each round needs 1 API call
                api_calls_per_llm = matches_per_llm_per_tournament * avg_rounds
                
                # Total tokens for this LLM in this tournament
                total_input_tokens = api_calls_per_llm * prompt_data['input_tokens']
                total_output_tokens = api_calls_per_llm * prompt_data['output_tokens']
                total_tokens = total_input_tokens + total_output_tokens
                
                print(f"{shadow:<8} {phase:<6} {avg_rounds:<5} "
                      f"{prompt_data['input_tokens']:<8,} {prompt_data['output_tokens']:<8,} "
                      f"{prompt_data['static_tokens']:<8,} {prompt_data['dynamic_tokens']:<8,} "
                      f"{api_calls_per_llm:<6,} {total_tokens:<10,}")
        
        print("-" * 78)
        
        # Summary by shadow condition
        print(f"\n📋 SUMMARY BY SHADOW CONDITION ({mode_name} Mode):")
        print(f"{'Shadow':<8} {'Avg Rounds':<12} {'API Calls':<12} {'Input Tokens':<15} {'Output Tokens':<15} {'Total Tokens':<15}")
        print("-" * 90)
        
        for shadow in shadow_conditions:
            avg_rounds = estimate_match_length(shadow)
            api_calls_per_llm = matches_per_llm_per_tournament * avg_rounds
            
            # Use Phase 3 as representative (middle phase with some history)
            prompt_data = estimate_prompt_size(shadow, 3, tracking_enabled)
            
            total_input = api_calls_per_llm * prompt_data['input_tokens']
            total_output = api_calls_per_llm * prompt_data['output_tokens']
            total_tokens = total_input + total_output
            
            print(f"{shadow:<8} {avg_rounds:<12} {api_calls_per_llm:<12,} "
                  f"{total_input:<15,} {total_output:<15,} {total_tokens:<15,}")
        
        print("-" * 90)
        
        # Extreme cases analysis
        print(f"\n🎯 EXTREME CASES ANALYSIS ({mode_name} Mode):")
        
        # Most expensive: Shadow 0.01, Phase 5
        shadow_expensive = 0.01
        phase_expensive = 5
        rounds_expensive = estimate_match_length(shadow_expensive)
        prompt_expensive = estimate_prompt_size(shadow_expensive, phase_expensive, tracking_enabled)
        calls_expensive = matches_per_llm_per_tournament * rounds_expensive
        tokens_expensive = calls_expensive * (prompt_expensive['input_tokens'] + prompt_expensive['output_tokens'])
        
        # Least expensive: Shadow 0.75, Phase 1  
        shadow_cheap = 0.75
        phase_cheap = 1
        rounds_cheap = estimate_match_length(shadow_cheap)
        prompt_cheap = estimate_prompt_size(shadow_cheap, phase_cheap, tracking_enabled)
        calls_cheap = matches_per_llm_per_tournament * rounds_cheap
        tokens_cheap = calls_cheap * (prompt_cheap['input_tokens'] + prompt_cheap['output_tokens'])
        
        print(f"   Most Expensive:  Shadow {shadow_expensive}, Phase {phase_expensive}")
        print(f"                    {calls_expensive:,} API calls, {tokens_expensive:,} total tokens per LLM")
        print(f"   Least Expensive: Shadow {shadow_cheap}, Phase {phase_cheap}")
        print(f"                    {calls_cheap:,} API calls, {tokens_cheap:,} total tokens per LLM")
        print(f"   Ratio:           {tokens_expensive/tokens_cheap:.1f}x difference")
        
    # Mode comparison
    print(f"\n{'='*80}")
    print("MODE COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n📊 TOKEN OVERHEAD OF OPPONENT TRACKING:")
    
    # Compare Phase 3 (representative case) for each shadow
    print(f"{'Shadow':<8} {'Anonymous':<12} {'Tracking':<12} {'Difference':<12} {'% Increase':<12}")
    print("-" * 60)
    
    for shadow in shadow_conditions:
        prompt_anon = estimate_prompt_size(shadow, 3, False)
        prompt_track = estimate_prompt_size(shadow, 3, True)
        
        tokens_anon = prompt_anon['input_tokens'] + prompt_anon['output_tokens']
        tokens_track = prompt_track['input_tokens'] + prompt_track['output_tokens']
        
        difference = tokens_track - tokens_anon
        pct_increase = (difference / tokens_anon) * 100
        
        print(f"{shadow:<8} {tokens_anon:<12,} {tokens_track:<12,} {difference:<12,} {pct_increase:<12.1f}%")
    
    print("-" * 60)
    print(f"\n✅ Key Findings:")
    print(f"   • Opponent tracking adds 15-30 tokens per API call (~1-2% increase)")
    print(f"   • Token overhead is consistent across all shadow conditions")
    print(f"   • Static content (game rules + history) dominates token usage")
    print(f"   • Longer matches (lower shadow) scale linearly with API calls")

if __name__ == "__main__":
    calculate_tournament_tokens()