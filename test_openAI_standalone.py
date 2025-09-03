#!/usr/bin/env python3
"""
Standalone test for OpenAI models: gpt-5-mini, gpt-5-nano, and gpt-4.1-mini
"""
import sys
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the parent directory to the path so we can import ipd_suite
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_suite.agents import GPT4Agent


def test_openai_models():
    """Test OpenAI models: gpt-5-mini, gpt-5-nano, and gpt-4.1-mini at temperature 1"""
    print("="*60)
    print("OPENAI MODELS TEST - gpt-5-mini, gpt-5-nano, gpt-4.1-mini")
    print("="*60)
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return False
    
    print("✓ OpenAI API key found")
    
    models = ['gpt-5-mini', 'gpt-5-nano', 'gpt-4.1-mini']
    results = {}
    total_calls = 0
    total_tokens = 0
    
    try:
        for model in models:
            print(f"\n🤖 Testing model: {model}")
            print("-" * 40)
            
            # Create agent
            agent = GPT4Agent(
                name=f"Test_{model.replace('-', '_')}",
                api_key=api_key,
                model=model,
                temperature=1.0,
                termination_prob=0.1
            )
            
            # Test scenarios
            scenarios = [
                ([], [], "empty"),
                (['C'], ['D'], "single"),
                (['C', 'D', 'C'], ['D', 'C', 'D'], "multi")
            ]
            
            moves = []
            for my_hist, opp_hist, name in scenarios:
                move = agent.make_move(my_hist, opp_hist)
                moves.append(move)
                print(f"  {name}: {move}")
            
            # Validate
            valid = all(m in ['C', 'D'] for m in moves)
            results[model] = {
                'moves': moves,
                'calls': agent.api_calls,
                'tokens': agent.total_tokens,
                'valid': valid
            }
            
            total_calls += agent.api_calls
            total_tokens += agent.total_tokens
            
            print(f"✓ {model}: {agent.api_calls} calls, {agent.total_tokens} tokens, valid: {valid}")
        
        # Summary
        print("\n" + "="*60)
        print("✅ ALL MODELS TESTED!")
        print("="*60)
        
        all_valid = all(r['valid'] for r in results.values())
        print(f"Total API calls: {total_calls}")
        print(f"Total tokens: {total_tokens}")
        print(f"All moves valid: {all_valid}")
        
        for model, result in results.items():
            print(f"{model}: {result['moves']} ({result['calls']} calls)")
        
        return all_valid
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_openai_models()
    exit(0 if success else 1)