#!/usr/bin/env python3
"""
Show exact LLM prompt example with match history
"""
import sys
import os

# Add the parent directory to the path so we can import ipd_suite
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_suite.agents import LLMAgent, TitForTat, Random, GrimTrigger
from ipd_suite.tournament import Tournament, MatchHistoryManager

class ExampleLLMAgent(LLMAgent):
    """Example LLM agent to demonstrate prompt generation"""
    
    def __init__(self, name: str, temperature: float = 0.7, match_history=None):
        super().__init__(name, "claude-sonnet-4-20250514", temperature, 0.1, match_history=match_history)
        self.captured_prompts = []
        
    def _call_api(self, prompt: str) -> str:
        """Capture the full prompt for demonstration"""
        self.captured_prompts.append(prompt)
        return "I analyze the current situation carefully.\n\nGiven the history and strategic considerations, I choose: C"
    
    def _call_api_with_caching(self, static_content: str, dynamic_content: str) -> str:
        """Capture prompt components for demonstration"""
        self.captured_static = static_content
        self.captured_dynamic = dynamic_content
        full_prompt = static_content + dynamic_content
        self.captured_prompts.append(full_prompt)
        return "I analyze the current situation carefully.\n\nGiven the history and strategic considerations, I choose: C"


def simulate_phase_progression():
    """Simulate realistic match history across phases"""
    print("="*80)
    print("LLM AGENT PROMPT EXAMPLE - PHASE 2 FORWARD")
    print("="*80)
    
    # Create history manager
    history_manager = MatchHistoryManager()
    
    # Create agents
    claude_agent = ExampleLLMAgent("Claude4-Sonnet_T02", temperature=0.2)
    tit_for_tat = TitForTat("TitForTat")
    random_agent = Random("Random") 
    grim_trigger = GrimTrigger("GrimTrigger")
    
    print("\n📋 PHASE 1 SIMULATION")
    print("-" * 50)
    
    # Simulate Phase 1 - Multiple matches
    tournament = Tournament([claude_agent, tit_for_tat, random_agent, grim_trigger], 
                           termination_prob=0.3, max_rounds=8, history_manager=history_manager)
    
    # Match 1: Claude vs TitForTat
    print("🎮 Match 1: Claude4-Sonnet_T02 vs TitForTat")
    match1 = tournament.run_match(claude_agent, tit_for_tat)
    print(f"   Result: {match1.rounds_played} rounds, moves: {list(zip(match1.agent1_moves, match1.agent2_moves))}")
    
    # Match 2: Claude vs Random  
    print("🎮 Match 2: Claude4-Sonnet_T02 vs Random")
    match2 = tournament.run_match(claude_agent, random_agent)
    print(f"   Result: {match2.rounds_played} rounds, moves: {list(zip(match2.agent1_moves, match2.agent2_moves))}")
    
    # Match 3: Claude vs GrimTrigger
    print("🎮 Match 3: Claude4-Sonnet_T02 vs GrimTrigger") 
    match3 = tournament.run_match(claude_agent, grim_trigger)
    print(f"   Result: {match3.rounds_played} rounds, moves: {list(zip(match3.agent1_moves, match3.agent2_moves))}")
    
    # Get Claude's accumulated history
    claude_history = history_manager.get_history_for_agent(claude_agent)
    print(f"\n✅ Phase 1 Complete - Claude has {len(claude_history)} matches in history")
    
    print("\n📋 PHASE 2 SIMULATION")  
    print("-" * 50)
    
    # Create Phase 2 Claude agent with history from Phase 1
    claude_phase2 = ExampleLLMAgent("Claude4-Sonnet_T02", temperature=0.2, match_history=claude_history)
    
    print("🎮 Phase 2 Match: Claude4-Sonnet_T02 vs TitForTat (with Phase 1 history)")
    print("📝 Generating prompt for this match...")
    
    # Trigger prompt generation by making a move in current match
    move = claude_phase2.make_move(['C'], ['C'])  # Current match: both cooperated in round 1
    
    print("\n" + "="*80)
    print("📄 EXACT PROMPT RECEIVED BY LLM AGENT:")
    print("="*80)
    
    # Show the captured prompt
    if claude_phase2.captured_prompts:
        full_prompt = claude_phase2.captured_prompts[0]
        print(full_prompt)
        
        print("\n" + "="*80)
        print("📊 PROMPT BREAKDOWN:")
        print("="*80)
        
        if hasattr(claude_phase2, 'captured_static'):
            print("\n🔧 STATIC CONTENT (cached between calls):")
            print("-" * 50)
            print(claude_phase2.captured_static)
            
            print("\n🔄 DYNAMIC CONTENT (changes with current match state):")
            print("-" * 50)
            print(claude_phase2.captured_dynamic)
        
        # Count sections
        prompt_lines = full_prompt.split('\n')
        game_rules_lines = sum(1 for line in prompt_lines if any(word in line.lower() for word in ['prisoner', 'dilemma', 'cooperate', 'defect', 'payoff']))
        history_lines = sum(1 for line in prompt_lines if 'vs ' in line and '(' in line and ')' in line)
        current_lines = sum(1 for line in prompt_lines if 'current match' in line.lower())
        
        print(f"\n📈 PROMPT STATISTICS:")
        print(f"   Total lines: {len(prompt_lines)}")
        print(f"   Game rules section: ~{game_rules_lines} lines")
        print(f"   Match history section: ~{history_lines} lines")
        print(f"   Current match section: ~{current_lines} lines") 
        print(f"   Total characters: {len(full_prompt)}")
        print(f"   Estimated tokens: ~{len(full_prompt.split()) * 1.3:.0f}")
        
        # Show specific history formatting
        print(f"\n🎯 MATCH HISTORY DETAILS:")
        for i, match in enumerate(claude_history, 1):
            opponent = match.get('opponent', 'Unknown')
            rounds = match.get('rounds', [])
            round_pairs = [(r.get('your_move', '?'), r.get('opponent_move', '?')) for r in rounds]
            print(f"   Match {i}: vs {opponent} → {round_pairs}")
    else:
        print("❌ No prompt was captured")
    
    return True


if __name__ == "__main__":
    simulate_phase_progression()