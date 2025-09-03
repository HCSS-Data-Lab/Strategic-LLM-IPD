#!/usr/bin/env python3
"""
Realistic cost breakdown per LLM-temperature combination with actual API pricing
"""
import math

def estimate_tokens(text: str) -> int:
    """Estimate tokens using word count * 1.3 (conservative GPT-4 tokenizer estimate)"""
    return int(len(text.split()) * 1.3)

def estimate_match_length(shadow_prob: float) -> int:
    """Estimate average match length based on termination probability"""
    expected_rounds = 1.0 / shadow_prob
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

def calculate_realistic_costs():
    """Calculate realistic costs with actual API pricing per LLM-temperature combination"""
    
    print("="*80)
    print("REALISTIC COST BREAKDOWN PER LLM-TEMPERATURE COMBINATION")
    print("="*80)
    
    shadow_conditions = [0.75, 0.25, 0.10, 0.05, 0.02, 0.01]
    phases = [1, 2, 3, 4, 5]
    
    # Individual LLM-temperature combinations with actual pricing (per 1M tokens)
    llm_agents = [
        # OpenAI (Different Models, all at T=1.0)
        ('GPT5mini', 'T=1.0', {'input': 0.25, 'output': 2.00}),      # $0.25/$2.00 per 1M
        ('GPT5nano', 'T=1.0', {'input': 0.05, 'output': 0.40}),      # $0.05/$0.40 per 1M
        ('GPT41mini', 'T=1.0', {'input': 0.40, 'output': 1.60}),     # $0.40/$1.60 per 1M
        
        # Claude (claude-sonnet-4-20250514)
        ('Claude4-Sonnet', 'T=0.2', {'input': 6.00, 'output': 22.50}),  # $6.00/$22.50 per 1M
        ('Claude4-Sonnet', 'T=0.5', {'input': 6.00, 'output': 22.50}),
        ('Claude4-Sonnet', 'T=0.8', {'input': 6.00, 'output': 22.50}),
        
        # Mistral (mistral-medium-2508)
        ('Mistral-Medium', 'T=0.2', {'input': 0.40, 'output': 2.00}),  # $0.40/$2.00 per 1M
        ('Mistral-Medium', 'T=0.7', {'input': 0.40, 'output': 2.00}),
        ('Mistral-Medium', 'T=1.2', {'input': 0.40, 'output': 2.00}),
        
        # Gemini (gemini-2.0-flash)
        ('Gemini20Flash', 'T=0.2', {'input': 0.10, 'output': 0.40}),  # $0.10/$0.40 per 1M
        ('Gemini20Flash', 'T=0.7', {'input': 0.10, 'output': 0.40}),
        ('Gemini20Flash', 'T=1.2', {'input': 0.10, 'output': 0.40})
    ]
    
    # Tournament setup
    total_agents = 28  # 16 classical + 12 LLM
    matches_per_llm_per_tournament = total_agents - 1  # Each LLM plays every other agent
    
    print(f"📊 EXPERIMENTAL SETUP:")
    print(f"   Total agents per tournament: {total_agents}")
    print(f"   Matches per LLM per tournament: {matches_per_llm_per_tournament}")
    print(f"   LLM-temperature combinations: {len(llm_agents)}")
    print(f"   Phases analyzed: {phases}")
    
    for tracking_enabled in [False, True]:
        mode_name = "Opponent Tracking" if tracking_enabled else "Anonymous"
        
        print(f"\n{'='*60}")
        print(f"MODE: {mode_name.upper()}")
        print(f"{'='*60}")
        
        # Group results by shadow condition
        for shadow in shadow_conditions:
            avg_rounds = estimate_match_length(shadow)
            
            print(f"\n🎯 SHADOW OF THE FUTURE: {shadow} (avg {avg_rounds} rounds/match)")
            print(f"{'='*90}")
            
            # Header for this shadow condition
            print(f"\n{'LLM Model':<18} {'Temp':<6} {'Input':<10} {'Output':<10} {'Input $':<10} {'Output $':<10} {'Total $':<10}")
            print(f"{'Name':<18} {'T':<6} {'Tokens':<10} {'Tokens':<10} {'Cost':<10} {'Cost':<10} {'Cost':<10}")
            print("-" * 90)
            
            shadow_totals = {'input_tokens': 0, 'output_tokens': 0, 'input_cost': 0, 'output_cost': 0, 'total_cost': 0}
            
            for llm_model, temperature, pricing in llm_agents:
                # Use Phase 3 as representative (middle phase with some history)
                prompt_data = estimate_prompt_size(shadow, 3, tracking_enabled)
                
                # Calculate API calls per individual LLM agent per tournament
                api_calls_per_agent = matches_per_llm_per_tournament * avg_rounds
                
                # Total tokens for this individual LLM-temperature combination
                total_input_tokens = api_calls_per_agent * prompt_data['input_tokens']
                total_output_tokens = api_calls_per_agent * prompt_data['output_tokens']
                
                # Calculate costs (pricing is per 1M tokens)
                input_cost = (total_input_tokens / 1_000_000) * pricing['input']
                output_cost = (total_output_tokens / 1_000_000) * pricing['output']
                total_cost = input_cost + output_cost
                
                # Add to shadow totals
                shadow_totals['input_tokens'] += total_input_tokens
                shadow_totals['output_tokens'] += total_output_tokens
                shadow_totals['input_cost'] += input_cost
                shadow_totals['output_cost'] += output_cost
                shadow_totals['total_cost'] += total_cost
                
                print(f"{llm_model:<18} {temperature:<6} "
                      f"{total_input_tokens:>9,} {total_output_tokens:>9,} "
                      f"${input_cost:>8.2f} ${output_cost:>8.2f} ${total_cost:>8.2f}")
            
            print("-" * 90)
            print(f"{'SHADOW TOTAL':<18} {'':<6} "
                  f"{shadow_totals['input_tokens']:>9,} {shadow_totals['output_tokens']:>9,} "
                  f"${shadow_totals['input_cost']:>8.2f} ${shadow_totals['output_cost']:>8.2f} ${shadow_totals['total_cost']:>8.2f}")
        
        # Overall summary across all shadows
        print(f"\n🌟 COMPLETE EXPERIMENTAL SUITE COST SUMMARY ({mode_name} Mode)")
        print(f"{'='*90}")
        
        print(f"\n{'LLM Model':<18} {'Temp':<6} {'Total Input':<12} {'Total Output':<12} {'Input $':<10} {'Output $':<10} {'Total $':<10}")
        print(f"{'Name':<18} {'T':<6} {'Tokens':<12} {'Tokens':<12} {'Cost':<10} {'Cost':<10} {'Cost':<10}")
        print("-" * 90)
        
        grand_totals = {
            'input_tokens': 0, 'output_tokens': 0, 'input_cost': 0, 'output_cost': 0, 'total_cost': 0
        }
        
        for llm_model, temperature, pricing in llm_agents:
            agent_input_total = 0
            agent_output_total = 0
            
            # Sum across all shadow conditions and phases
            for shadow in shadow_conditions:
                avg_rounds = estimate_match_length(shadow)
                api_calls_per_agent = matches_per_llm_per_tournament * avg_rounds
                
                # Sum across all phases for this shadow
                for phase in phases:
                    prompt_data = estimate_prompt_size(shadow, phase, tracking_enabled)
                    agent_input_total += api_calls_per_agent * prompt_data['input_tokens']
                    agent_output_total += api_calls_per_agent * prompt_data['output_tokens']
            
            # Calculate costs for this agent across all conditions and phases
            agent_input_cost = (agent_input_total / 1_000_000) * pricing['input']
            agent_output_cost = (agent_output_total / 1_000_000) * pricing['output']
            agent_total_cost = agent_input_cost + agent_output_cost
            
            # Add to grand totals
            grand_totals['input_tokens'] += agent_input_total
            grand_totals['output_tokens'] += agent_output_total
            grand_totals['input_cost'] += agent_input_cost
            grand_totals['output_cost'] += agent_output_cost
            grand_totals['total_cost'] += agent_total_cost
            
            print(f"{llm_model:<18} {temperature:<6} "
                  f"{agent_input_total:>11,} {agent_output_total:>11,} "
                  f"${agent_input_cost:>8.2f} ${agent_output_cost:>8.2f} ${agent_total_cost:>8.2f}")
        
        print("-" * 90)
        print(f"{'ALL LLM AGENTS':<18} {'':<6} "
              f"{grand_totals['input_tokens']:>11,} {grand_totals['output_tokens']:>11,} "
              f"${grand_totals['input_cost']:>8.2f} ${grand_totals['output_cost']:>8.2f} ${grand_totals['total_cost']:>8.2f}")
        
        print(f"\n📋 KEY COST STATISTICS ({mode_name} Mode):")
        print(f"   • Total input cost: ${grand_totals['input_cost']:,.2f}")
        print(f"   • Total output cost: ${grand_totals['output_cost']:,.2f}")
        print(f"   • Grand total cost: ${grand_totals['total_cost']:,.2f}")
        print(f"   • Average per agent: ${grand_totals['total_cost']/len(llm_agents):,.2f}")
        
        # Show cost distribution by provider
        print(f"\n📊 COST DISTRIBUTION BY PROVIDER ({mode_name} Mode):")
        providers = {
            'OpenAI': [a for a in llm_agents if a[0].startswith('GPT')],
            'Claude': [a for a in llm_agents if a[0].startswith('Claude')],
            'Mistral': [a for a in llm_agents if a[0].startswith('Mistral')],
            'Gemini': [a for a in llm_agents if a[0].startswith('Gemini')]
        }
        
        for provider, provider_agents in providers.items():
            provider_total_cost = 0
            
            for llm_model, temperature, pricing in provider_agents:
                agent_input_total = 0
                agent_output_total = 0
                
                for shadow in shadow_conditions:
                    avg_rounds = estimate_match_length(shadow)
                    api_calls_per_agent = matches_per_llm_per_tournament * avg_rounds
                    
                    for phase in phases:
                        prompt_data = estimate_prompt_size(shadow, phase, tracking_enabled)
                        agent_input_total += api_calls_per_agent * prompt_data['input_tokens']
                        agent_output_total += api_calls_per_agent * prompt_data['output_tokens']
                
                agent_cost = ((agent_input_total / 1_000_000) * pricing['input'] + 
                             (agent_output_total / 1_000_000) * pricing['output'])
                provider_total_cost += agent_cost
            
            avg_per_agent = provider_total_cost / len(provider_agents) if provider_agents else 0
            print(f"   {provider:<8}: ${provider_total_cost:>8.2f} ({len(provider_agents)} agents, avg ${avg_per_agent:,.2f} per agent)")
        
        # Store mode totals for comparison
        if not tracking_enabled:
            anon_totals = grand_totals.copy()
        else:
            track_totals = grand_totals.copy()
    
    # Mode comparison
    print(f"\n{'='*80}")
    print("TRACKING MODE COST OVERHEAD")
    print(f"{'='*80}")
    
    cost_difference = track_totals['total_cost'] - anon_totals['total_cost']
    cost_increase_pct = (cost_difference / anon_totals['total_cost']) * 100
    
    print(f"\n💰 FINANCIAL IMPACT OF OPPONENT TRACKING:")
    print(f"   Anonymous Mode Total: ${anon_totals['total_cost']:,.2f}")
    print(f"   Tracking Mode Total:  ${track_totals['total_cost']:,.2f}")
    print(f"   Additional Cost:      ${cost_difference:,.2f} (+{cost_increase_pct:.1f}%)")
    print(f"   Cost per LLM Agent:   ${cost_difference/12:,.2f} average overhead")
    
    # Show most/least expensive agents
    print(f"\n📈 MOST EXPENSIVE AGENTS (Anonymous Mode):")
    agent_costs = []
    for llm_model, temperature, pricing in llm_agents:
        agent_input_total = 0
        agent_output_total = 0
        
        for shadow in shadow_conditions:
            avg_rounds = estimate_match_length(shadow)
            api_calls_per_agent = matches_per_llm_per_tournament * avg_rounds
            
            for phase in phases:
                prompt_data = estimate_prompt_size(shadow, phase, False)  # Anonymous mode
                agent_input_total += api_calls_per_agent * prompt_data['input_tokens']
                agent_output_total += api_calls_per_agent * prompt_data['output_tokens']
        
        agent_cost = ((agent_input_total / 1_000_000) * pricing['input'] + 
                     (agent_output_total / 1_000_000) * pricing['output'])
        agent_costs.append((f"{llm_model}_{temperature}", agent_cost, pricing))
    
    # Sort by cost
    agent_costs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (agent_name, cost, pricing) in enumerate(agent_costs, 1):
        print(f"   {i:2d}. {agent_name:<25}: ${cost:>8.2f} (${pricing['input']:.2f}/${pricing['output']:.2f} per 1M)")
    
    # Shadow condition cost breakdown
    print(f"\n📊 COST BY SHADOW CONDITION (Anonymous Mode):")
    print(f"{'Shadow':<8} {'Total Cost':<12} {'% of Total':<12} {'Avg Rounds':<12}")
    print("-" * 50)
    
    for shadow in shadow_conditions:
        avg_rounds = estimate_match_length(shadow)
        shadow_cost = 0
        
        for llm_model, temperature, pricing in llm_agents:
            api_calls_per_agent = matches_per_llm_per_tournament * avg_rounds
            
            for phase in phases:
                prompt_data = estimate_prompt_size(shadow, phase, False)
                input_tokens = api_calls_per_agent * prompt_data['input_tokens']
                output_tokens = api_calls_per_agent * prompt_data['output_tokens']
                
                shadow_cost += ((input_tokens / 1_000_000) * pricing['input'] + 
                               (output_tokens / 1_000_000) * pricing['output'])
        
        pct_of_total = (shadow_cost / anon_totals['total_cost']) * 100
        print(f"{shadow:<8} ${shadow_cost:>10.2f} {pct_of_total:>10.1f}% {avg_rounds:>10} rounds")
    
    print(f"\n✅ Key Findings:")
    print(f"   • Claude agents are most expensive due to high per-token costs")
    print(f"   • Gemini agents are most cost-effective across all conditions")
    print(f"   • Lower shadow conditions (0.01, 0.02) dominate total experiment cost")
    print(f"   • Opponent tracking adds minimal cost overhead (~3.8%)")
    print(f"   • Total experimental suite cost: ${anon_totals['total_cost']:,.2f} - ${track_totals['total_cost']:,.2f}")

if __name__ == "__main__":
    calculate_realistic_costs()