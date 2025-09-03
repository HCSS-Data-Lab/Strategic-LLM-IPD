#!/usr/bin/env python3
"""
Realistic token usage estimation for IPD experiments
"""
import json
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

def calculate_phase_history_growth(phase: int, matches_per_phase: int, avg_rounds: int) -> int:
    """Calculate cumulative history size by phase"""
    total_matches = phase * matches_per_phase
    # Each match adds rounds_played * 2 moves (yours + opponent)
    total_historical_moves = total_matches * avg_rounds * 2
    return total_historical_moves

def generate_sample_prompts(shadow_prob: float, phase: int, enable_tracking: bool = False):
    """Generate sample prompts for token estimation"""
    
    avg_rounds = estimate_match_length(shadow_prob)
    
    # Base game rules (static across all prompts)
    static_rules = """You are playing the Iterated Prisoner's Dilemma game.

Game Rules:
- Two players choose to Cooperate (C) or Defect (D)
- Payoffs: Both C = 3 points each, Both D = 1 point each, You C/They D = 0/5 points, You D/They C = 5/0 points
- The game continues for multiple rounds with {:.1f}% chance of ending after each round
- Goal: Maximize your total score""".format(shadow_prob * 100)
    
    # Historical matches (grows with phase)
    if phase == 1:
        match_history = ""
    else:
        # Estimate: In evolutionary mode, LLM agents play ~27 matches per phase (28 agents total)
        matches_per_phase = 27
        total_historical_matches = (phase - 1) * matches_per_phase
        
        # Generate sample historical matches
        history_lines = []
        match_count = 0
        for p in range(1, phase):
            for m in range(matches_per_phase):
                match_count += 1
                if enable_tracking:
                    opponent_id = f"Opponent_{(m % 16) + 1:03d}"  # 16 unique opponents cycling
                else:
                    opponents = ["TitForTat", "GrimTrigger", "Random", "Detective", "QLearning", 
                               "GPT5mini_T1", "Claude4-Sonnet_T02", "Mistral-Medium_T07", "Gemini25Pro_T12"]
                    opponent_id = opponents[m % len(opponents)]
                
                # Generate sample moves for this match
                rounds_in_match = max(1, int(avg_rounds * (0.7 + 0.6 * (m / matches_per_phase))))  # Vary length
                moves = [(["C", "D"][i % 2], ["C", "D"][(i + m) % 2]) for i in range(rounds_in_match)]
                moves_str = "[" + ", ".join([f"({your},{opp})" for your, opp in moves]) + "]"
                
                history_lines.append(f"Match {match_count} vs {opponent_id}: {moves_str}")
                
                if match_count >= 100:  # Cap for estimation (most histories won't exceed this)
                    break
            if match_count >= 100:
                break
        
        match_history = "\n\nYour complete match history from previous phases:\n" + "\n".join(history_lines)
    
    # Current opponent info (only for tracking mode)
    opponent_info = ""
    if enable_tracking and phase > 1:
        # Sample opponent cross-reference
        prev_encounters = min(3, phase - 1)  # Up to 3 previous encounters
        if prev_encounters == 1:
            opponent_info = "\n\nCurrent opponent: Opponent_001 (previously played in Match 15 of Phase 1)"
        elif prev_encounters == 2:
            opponent_info = "\n\nCurrent opponent: Opponent_001 (previously played in Match 15 of Phase 1 and Match 8 of Phase 2)"
        else:
            opponent_info = "\n\nCurrent opponent: Opponent_001 (previously played in Match 15 of Phase 1, Match 8 of Phase 2, and Match 22 of Phase 3)"
    
    # Current match context (dynamic part)
    current_match = f"{opponent_info}\n\nCurrent match - History of moves (You, Opponent): [(C,C), (D,C), (C,D)]"
    
    # Instructions
    instructions = "\n\nProvide your reasoning about the game state, opponent's strategy, and your decision.\nEnd your response with your move on a new line: either 'C' or 'D'."
    
    # Combine all parts
    full_prompt = static_rules + match_history + current_match + instructions
    
    return {
        'full_prompt': full_prompt,
        'static_content': static_rules + match_history,
        'dynamic_content': current_match + instructions,
        'tokens_total': estimate_tokens(full_prompt),
        'tokens_static': estimate_tokens(static_rules + match_history),
        'tokens_dynamic': estimate_tokens(current_match + instructions)
    }

def estimate_experiment_costs():
    """Estimate token costs for full experimental suite"""
    
    print("="*80)
    print("IPD EXPERIMENT TOKEN USAGE ESTIMATION")
    print("="*80)
    
    # Experimental parameters
    shadow_conditions = [0.75, 0.25, 0.10, 0.05, 0.02, 0.01]
    n_phases = 5
    
    # LLM agent configurations
    llm_configs = {
        'OpenAI': [
            ('GPT5mini_T1', 1.0),
            ('GPT5nano_T1', 1.0), 
            ('GPT4mini_T1', 1.0)
        ],
        'Claude': [
            ('Claude4-Sonnet_T02', 0.2),
            ('Claude4-Sonnet_T05', 0.5),
            ('Claude4-Sonnet_T08', 0.8)
        ],
        'Mistral': [
            ('Mistral-Medium_T02', 0.2),
            ('Mistral-Medium_T07', 0.7),
            ('Mistral-Medium_T12', 1.2)
        ],
        'Gemini': [
            ('Gemini25Pro_T02', 0.2),
            ('Gemini25Pro_T07', 0.7),
            ('Gemini25Pro_T12', 1.2)
        ]
    }
    
    # Calculate total LLM agents
    total_llm_agents = sum(len(configs) for configs in llm_configs.values())
    total_classical_agents = 16  # Classical + behavioral + adaptive
    total_agents = total_llm_agents + total_classical_agents
    
    print(f"🏗️  EXPERIMENT CONFIGURATION:")
    print(f"   Shadow conditions: {shadow_conditions}")
    print(f"   Phases per condition: {n_phases}")
    print(f"   LLM agents: {total_llm_agents} ({', '.join([f'{len(c)} {k}' for k, c in llm_configs.items()])})")
    print(f"   Classical agents: {total_classical_agents}")
    print(f"   Total agents: {total_agents}")
    
    # Estimate matches per agent per phase (round-robin style)
    matches_per_llm_per_phase = total_agents - 1  # Each agent plays every other agent
    
    results = {}
    
    for tracking_mode in [False, True]:
        mode_name = "Tracking" if tracking_mode else "Anonymous"
        results[mode_name] = {}
        
        print(f"\n{'='*60}")
        print(f"MODE: {mode_name.upper()}")
        print(f"{'='*60}")
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_api_calls = 0
        
        for shadow in shadow_conditions:
            shadow_input_tokens = 0
            shadow_output_tokens = 0
            shadow_api_calls = 0
            
            avg_rounds = estimate_match_length(shadow)
            
            print(f"\n🎯 Shadow {shadow} (avg {avg_rounds} rounds/match):")
            print(f"   {'Phase':<8} {'Prompt Tokens':<15} {'Response Tokens':<17} {'API Calls':<12} {'Cumulative'}")
            print(f"   {'-'*70}")
            
            phase_details = []
            
            for phase in range(1, n_phases + 1):
                # Generate sample prompt for this phase
                sample_prompt = generate_sample_prompts(shadow, phase, tracking_mode)
                
                # Estimate tokens per API call
                input_tokens_per_call = sample_prompt['tokens_total']
                output_tokens_per_call = 150  # Conservative estimate for reasoning + move
                
                # Each LLM agent makes moves for each round of each match
                api_calls_per_agent = matches_per_llm_per_phase * avg_rounds
                total_api_calls_this_phase = total_llm_agents * api_calls_per_agent
                
                phase_input_tokens = total_api_calls_this_phase * input_tokens_per_call
                phase_output_tokens = total_api_calls_this_phase * output_tokens_per_call
                
                shadow_input_tokens += phase_input_tokens
                shadow_output_tokens += phase_output_tokens
                shadow_api_calls += total_api_calls_this_phase
                
                cumulative_calls = shadow_api_calls
                
                print(f"   Phase {phase:<3} {input_tokens_per_call:<15,} {output_tokens_per_call:<17,} {total_api_calls_this_phase:<12,} {cumulative_calls:,}")
                
                phase_details.append({
                    'phase': phase,
                    'input_tokens_per_call': input_tokens_per_call,
                    'output_tokens_per_call': output_tokens_per_call,
                    'api_calls': total_api_calls_this_phase,
                    'total_input': phase_input_tokens,
                    'total_output': phase_output_tokens
                })
            
            total_input_tokens += shadow_input_tokens
            total_output_tokens += shadow_output_tokens
            total_api_calls += shadow_api_calls
            
            results[mode_name][shadow] = {
                'avg_rounds': avg_rounds,
                'total_input_tokens': shadow_input_tokens,
                'total_output_tokens': shadow_output_tokens,
                'total_api_calls': shadow_api_calls,
                'phases': phase_details
            }
            
            print(f"   {'-'*70}")
            print(f"   Total:   {shadow_input_tokens:,} input, {shadow_output_tokens:,} output, {shadow_api_calls:,} calls")
        
        results[mode_name]['totals'] = {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens, 
            'total_api_calls': total_api_calls
        }
        
        print(f"\n📊 {mode_name.upper()} MODE TOTALS:")
        print(f"   Total Input Tokens:  {total_input_tokens:,}")
        print(f"   Total Output Tokens: {total_output_tokens:,}")
        print(f"   Total API Calls:     {total_api_calls:,}")
        print(f"   Avg Input/Call:      {total_input_tokens//max(total_api_calls,1):,}")
        print(f"   Avg Output/Call:     {total_output_tokens//max(total_api_calls,1):,}")
    
    # Comparison
    print(f"\n{'='*80}")
    print("MODE COMPARISON")
    print(f"{'='*80}")
    
    anon_totals = results['Anonymous']['totals']
    track_totals = results['Tracking']['totals']
    
    input_diff = track_totals['total_input_tokens'] - anon_totals['total_input_tokens']
    output_diff = track_totals['total_output_tokens'] - anon_totals['total_output_tokens']
    calls_diff = track_totals['total_api_calls'] - anon_totals['total_api_calls']
    
    input_pct = (input_diff / anon_totals['total_input_tokens']) * 100
    output_pct = (output_diff / anon_totals['total_output_tokens']) * 100 if anon_totals['total_output_tokens'] > 0 else 0
    
    print(f"📈 TRACKING MODE OVERHEAD:")
    print(f"   Input Tokens:  +{input_diff:,} (+{input_pct:.1f}%)")
    print(f"   Output Tokens: +{output_diff:,} (+{output_pct:.1f}%)")
    print(f"   API Calls:     +{calls_diff:,} (same)")
    
    # Cost estimation (rough)
    print(f"\n💰 ESTIMATED API COSTS (approximate):")
    print(f"   Assuming $0.01/1K input tokens, $0.03/1K output tokens")
    
    for mode_name in ['Anonymous', 'Tracking']:
        totals = results[mode_name]['totals']
        input_cost = (totals['total_input_tokens'] / 1000) * 0.01
        output_cost = (totals['total_output_tokens'] / 1000) * 0.03
        total_cost = input_cost + output_cost
        
        print(f"   {mode_name:>12}: ${total_cost:,.2f} (${input_cost:.2f} input + ${output_cost:.2f} output)")
    
    # Save detailed results
    output_file = "token_usage_estimation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    estimate_experiment_costs()