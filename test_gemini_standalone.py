#!/usr/bin/env python3
"""
Standalone test for GeminiAgent real API call
"""
import sys
import os
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading
    pass

# Add the parent directory to the path so we can import ipd_suite
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_suite.agents import GeminiAgent


def test_gemini_agent_real_api_call():
    """Test GeminiAgent with real API call"""
    print("="*60)
    print("GEMINI AGENT REAL API CALL TEST")
    print("="*60)
    
    # Skip if no API key provided
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key in the .env file or environment variables")
        return False
    
    print("✓ Google API key found")
    
    try:
        # Create agent with real API key (using working model)
        agent = GeminiAgent(
            name="TestGeminiReal",
            api_key=api_key,
            model="gemini-2.0-flash",
            temperature=1.0,
            termination_prob=0.1
        )
        
        
        print(f"✓ Created GeminiAgent: {agent.name}")
        print(f"  Model: {agent.model}")
        print(f"  Temperature: {agent.temperature}")
        print(f"  Termination probability: {agent.termination_prob}")
        
        # Test initial state
        assert agent.api_calls == 0
        assert agent.total_tokens == 0
        print("✓ Initial state verified (0 API calls, 0 tokens)")
        
        # Test 1: Make a real API call with some history
        print("\n🧪 Test 1: API call with history ['C'], ['D']")
        start_time = time.time()
        move = agent.make_move(['C'], ['D'])
        end_time = time.time()
        
        # Debug: Show what was captured
        print(f"DEBUG: last_reasoning type: {type(agent.last_reasoning)}")
        print(f"DEBUG: last_reasoning is None: {agent.last_reasoning is None}")
        print(f"DEBUG: last_reasoning bool: {bool(agent.last_reasoning)}")
        print(f"DEBUG: last_reasoning repr: {repr(agent.last_reasoning)}")
        if agent.last_reasoning:
            print(f"DEBUG: last_reasoning length: {len(agent.last_reasoning)}")
            print(f"DEBUG: first 100 chars: {repr(agent.last_reasoning[:100])}")
        
        # Verify the call was made
        assert agent.api_calls == 1
        assert agent.total_tokens > 0
        assert move in ['C', 'D']
        
        time_taken = end_time - start_time
        print(f"✓ Move: {move}")
        print(f"✓ API calls: {agent.api_calls}")
        print(f"✓ Total tokens: {agent.total_tokens}")
        print(f"✓ Time taken: {time_taken:.2f} seconds")
        print(f"✓ Efficiency: {agent.total_tokens/time_taken:.0f} tokens/second")
        
        # Check and display reasoning
        if agent.last_reasoning and agent.last_reasoning.strip():
            reasoning_preview = agent.last_reasoning[:200] + "..." if len(agent.last_reasoning) > 200 else agent.last_reasoning
            print(f"✓ Reasoning: {reasoning_preview}")
            print(f"✓ Reasoning length: {len(agent.last_reasoning)} characters")
        else:
            print(f"⚠️  No reasoning captured from Gemini agent (last_reasoning: {repr(agent.last_reasoning)})")
        
        # Test 2: Test with no history
        print("\n🧪 Test 2: API call with empty history")
        start_time = time.time()
        move2 = agent.make_move([], [])
        end_time = time.time()
        assert agent.api_calls == 2
        assert move2 in ['C', 'D']
        
        time_taken = end_time - start_time
        print(f"✓ Move: {move2}")
        print(f"✓ API calls: {agent.api_calls}")
        print(f"✓ Total tokens: {agent.total_tokens}")
        print(f"✓ Time taken: {time_taken:.2f} seconds")
        
        # Check reasoning for test 2
        if agent.last_reasoning and agent.last_reasoning.strip():
            reasoning_preview = agent.last_reasoning[:150] + "..." if len(agent.last_reasoning) > 150 else agent.last_reasoning
            print(f"✓ Reasoning: {reasoning_preview}")
        else:
            print(f"⚠️  No reasoning captured from Gemini agent (last_reasoning: {repr(agent.last_reasoning)})")
        
        # Test 3: Test with longer history
        print("\n🧪 Test 3: API call with longer history")
        start_time = time.time()
        move3 = agent.make_move(['C', 'D', 'C'], ['D', 'C', 'D'])
        end_time = time.time()
        assert agent.api_calls == 3
        assert move3 in ['C', 'D']
        
        time_taken = end_time - start_time
        print(f"✓ Move: {move3}")
        print(f"✓ API calls: {agent.api_calls}")
        print(f"✓ Total tokens: {agent.total_tokens}")
        print(f"✓ Time taken: {time_taken:.2f} seconds")
        
        # Check reasoning for test 3
        if agent.last_reasoning and agent.last_reasoning.strip():
            reasoning_preview = agent.last_reasoning[:150] + "..." if len(agent.last_reasoning) > 150 else agent.last_reasoning
            print(f"✓ Reasoning: {reasoning_preview}")
        else:
            print(f"⚠️  No reasoning captured from Gemini agent (last_reasoning: {repr(agent.last_reasoning)})")
        
        # Test 4: Test different temperature settings (Gemini supports 0-2 range)
        print("\n🧪 Test 4: Different temperature settings")
        test_temperatures = [0.2, 0.7, 1.2]
        
        for temp in test_temperatures:
            print(f"\n  Testing temperature: {temp}")
            temp_agent = GeminiAgent(
                name=f"TestGeminiTemp{temp}",
                api_key=api_key,
                model="gemini-2.0-flash",
                temperature=temp,
                termination_prob=0.1
            )
            
            # Verify temperature is set correctly
            assert temp_agent.temperature == temp
            print(f"  ✓ Temperature set: {temp_agent.temperature}")
            
            # Make API call with this temperature
            start_time = time.time()
            temp_move = temp_agent.make_move(['C'], ['D'])
            end_time = time.time()
            assert temp_move in ['C', 'D']
            assert temp_agent.api_calls == 1
            
            time_taken = end_time - start_time
            print(f"  ✓ Move with temp {temp}: {temp_move}")
            print(f"  ✓ API calls: {temp_agent.api_calls}")
            print(f"  ✓ Tokens used: {temp_agent.total_tokens}")
            print(f"  ✓ Time taken: {time_taken:.2f} seconds")
            
            # Check reasoning for temperature test
            if temp_agent.last_reasoning and temp_agent.last_reasoning.strip():
                reasoning_preview = temp_agent.last_reasoning[:100] + "..." if len(temp_agent.last_reasoning) > 100 else temp_agent.last_reasoning
                print(f"  ✓ Reasoning: {reasoning_preview}")
            else:
                print(f"  ⚠️  No reasoning captured (last_reasoning: {repr(temp_agent.last_reasoning)})")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("✅ GeminiAgent is working correctly with real API calls")
        print("="*60)
        
        # Reasoning validation summary
        print(f"\n🧠 REASONING VALIDATION:")
        reasoning_checks = []
        
        # Check main agent reasoning from last call
        if agent.last_reasoning and agent.last_reasoning.strip():
            reasoning_checks.append(f"✓ Main agent: {len(agent.last_reasoning)} chars")
            print(f"✓ Main agent reasoning captured: {len(agent.last_reasoning)} characters")
            
            # Validate reasoning contains expected elements
            reasoning_lower = agent.last_reasoning.lower()
            reasoning_elements = []
            if 'prisoner' in reasoning_lower or 'dilemma' in reasoning_lower:
                reasoning_elements.append("game context")
            if 'cooperat' in reasoning_lower or 'defect' in reasoning_lower:
                reasoning_elements.append("move options")
            if 'history' in reasoning_lower or 'previous' in reasoning_lower or 'opponent' in reasoning_lower:
                reasoning_elements.append("history analysis")
            if 'strategy' in reasoning_lower or 'decision' in reasoning_lower:
                reasoning_elements.append("strategic thinking")
            
            if reasoning_elements:
                print(f"✓ Reasoning includes: {', '.join(reasoning_elements)}")
            else:
                print("⚠️  Reasoning lacks expected game theory elements")
                
            # Show full reasoning from last call
            print(f"\n📋 FULL REASONING FROM LAST CALL:")
            print("-" * 50)
            print(agent.last_reasoning)
            print("-" * 50)
        else:
            reasoning_checks.append("❌ Main agent: No reasoning")
            print("❌ Main agent reasoning not captured")
        
        # Summary
        total_api_calls = agent.api_calls + len(test_temperatures)  # main agent + temp test agents
        print(f"\n📊 SUMMARY:")
        print(f"- Total API calls made: {total_api_calls}")
        print(f"- Main agent total tokens: {agent.total_tokens}")
        print(f"- All moves were valid: C or D")
        print(f"- All temperature settings worked: {test_temperatures}")
        print(f"- Model used: {agent.model}")
        print(f"- Reasoning capture rate: {len([r for r in reasoning_checks if '✓' in r])}/{len(reasoning_checks)} successful")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gemini_agent_real_api_call()
    exit_code = 0 if success else 1
    exit(exit_code)