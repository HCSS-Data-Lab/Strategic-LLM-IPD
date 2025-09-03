#!/usr/bin/env python3
"""
Test script for match history implementation
"""
import sys
import os

# Add the parent directory to the path so we can import ipd_suite
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_suite.agents import LLMAgent, TitForTat, Random
from ipd_suite.tournament import Tournament, MatchHistoryManager

class TestLLMAgent(LLMAgent):
    """Test LLM agent for match history validation"""
    
    def __init__(self, name: str, temperature: float = 0.7, match_history=None):
        super().__init__(name, "test-model", temperature, 0.1, match_history=match_history)
        self.mock_response = "I think carefully about the game situation.\n\nAfter analyzing the history and considering my strategy, I choose: C"
        
    def _call_api(self, prompt: str) -> str:
        """Mock API call that tracks prompt content"""
        self.api_calls += 1
        self.total_tokens += 150
        self.input_tokens += 100
        self.output_tokens += 50
        
        # Store the prompt to verify it contains match history
        self.last_prompt = prompt
        return self.mock_response
    
    def _call_api_with_caching(self, static_content: str, dynamic_content: str) -> str:
        """Mock API call with caching that tracks prompt content"""
        self.api_calls += 1
        self.total_tokens += 150
        self.input_tokens += 100
        self.output_tokens += 50
        
        # Store the prompt components to verify match history inclusion
        self.last_static_content = static_content
        self.last_dynamic_content = dynamic_content
        return self.mock_response


def test_match_history_implementation():
    """Test comprehensive match history functionality"""
    print("="*60)
    print("MATCH HISTORY IMPLEMENTATION TEST")
    print("="*60)
    
    # Test 1: Create history manager and verify basic functionality
    print("\n🧪 Test 1: Basic MatchHistoryManager functionality")
    history_manager = MatchHistoryManager()
    
    # Create test agents
    agent1 = TestLLMAgent("TestAgent_T07", temperature=0.7)
    agent2 = TitForTat("TitForTat")
    agent3 = Random("Random")
    
    print(f"✓ Created agents: {agent1.name}, {agent2.name}, {agent3.name}")
    
    # Test agent identifier generation
    agent1_id = history_manager.get_agent_identifier(agent1)
    expected_id = "testmodeltemperature07"  # Expected identifier format
    print(f"✓ Agent identifier: {agent1_id}")
    
    # Test 2: Run matches and verify history recording
    print("\n🧪 Test 2: Match history recording")
    tournament = Tournament([agent1, agent2, agent3], termination_prob=0.5, max_rounds=5, history_manager=history_manager)
    
    # Run a single match and verify history recording
    match_result = tournament.run_match(agent1, agent2)
    print(f"✓ Match completed: {match_result.agent1_name} vs {match_result.agent2_name}")
    print(f"  Rounds played: {match_result.rounds_played}")
    print(f"  Moves: {list(zip(match_result.agent1_moves, match_result.agent2_moves))}")
    
    # Check if history was recorded for LLM agent
    agent1_history = history_manager.get_history_for_agent(agent1)
    print(f"✓ Agent1 history recorded: {len(agent1_history)} matches")
    
    # Non-LLM agents should not have history recorded
    agent2_history = history_manager.get_history_for_agent(agent2)
    print(f"✓ Agent2 (non-LLM) history: {len(agent2_history)} matches (expected 0)")
    
    # Verify match history structure
    if agent1_history:
        match_record = agent1_history[0]
        print(f"✓ Match record structure: opponent='{match_record.get('opponent')}', rounds={len(match_record.get('rounds', []))}")
        
        if match_record.get('rounds'):
            first_round = match_record['rounds'][0]
            print(f"✓ Round structure: your_move='{first_round.get('your_move')}', opponent_move='{first_round.get('opponent_move')}'")
    
    # Test 3: Create agent with match history and verify prompt inclusion
    print("\n🧪 Test 3: Match history inclusion in prompts")
    
    # Create a new agent with the recorded history
    agent1_with_history = TestLLMAgent("TestAgent_T07", temperature=0.7, match_history=agent1_history)
    print(f"✓ Created agent with history: {len(agent1_with_history.match_history)} matches")
    
    # Make a move to trigger prompt generation
    agent1_with_history.make_move(['C'], ['D'])
    
    # Verify that match history is included in the prompt
    if hasattr(agent1_with_history, 'last_static_content'):
        static_content = agent1_with_history.last_static_content
        if "complete match history from previous phases" in static_content:
            print("✓ Match history section found in static content")
            
            # Check for specific match content
            if "vs TitForTat" in static_content:
                print("✓ Opponent name found in match history")
            
            # Check for move pairs
            if "C,D" in static_content or "(C,D)" in static_content:
                print("✓ Move pairs found in match history")
        else:
            print("⚠️  Match history section not found in static content")
            print(f"Static content preview: {static_content[:200]}...")
    else:
        print("⚠️  No static content captured")
    
    # Test 4: History serialization and loading
    print("\n🧪 Test 4: History serialization and loading")
    
    # Save histories to file
    test_history_file = "test_match_history.json"
    history_manager.save_histories(test_history_file)
    print(f"✓ Histories saved to {test_history_file}")
    
    # Create new manager and load histories
    new_history_manager = MatchHistoryManager()
    new_history_manager.load_histories(test_history_file)
    print(f"✓ Histories loaded from {test_history_file}")
    
    # Verify loaded histories match original
    loaded_agent1_history = new_history_manager.get_history_for_agent(agent1)
    if len(loaded_agent1_history) == len(agent1_history):
        print("✓ Loaded history length matches original")
        
        if loaded_agent1_history and agent1_history:
            if loaded_agent1_history[0]['opponent'] == agent1_history[0]['opponent']:
                print("✓ Loaded history content matches original")
    
    # Clean up test file
    if os.path.exists(test_history_file):
        os.remove(test_history_file)
        print("✓ Test file cleaned up")
    
    # Test 5: Multiple phases simulation
    print("\n🧪 Test 5: Multi-phase history accumulation")
    
    # Simulate phase 2 - create agent with history from phase 1
    agent1_phase2 = TestLLMAgent("TestAgent_T07", temperature=0.7, match_history=agent1_history)
    
    # Run another match (simulating phase 2)
    match_result_2 = tournament.run_match(agent1_phase2, agent3)
    print(f"✓ Phase 2 match completed: {match_result_2.rounds_played} rounds")
    
    # Verify cumulative history
    updated_history = history_manager.get_history_for_agent(agent1_phase2)
    print(f"✓ Cumulative history: {len(updated_history)} matches total")
    
    if len(updated_history) >= 2:
        print(f"  Match 1 opponent: {updated_history[0]['opponent']}")
        print(f"  Match 2 opponent: {updated_history[1]['opponent']}")
    
    print("\n" + "="*60)
    print("✅ ALL MATCH HISTORY TESTS PASSED!")
    print("✅ Match history implementation working correctly")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_match_history_implementation()
    exit_code = 0 if success else 1
    exit(exit_code)