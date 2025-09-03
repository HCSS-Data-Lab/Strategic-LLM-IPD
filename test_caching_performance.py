#!/usr/bin/env python3
"""
Test script for LLM agent caching performance
"""
import sys
import os

# Add the parent directory to the path so we can import ipd_suite
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_suite.agents import LLMAgent

class TestLLMAgent(LLMAgent):
    """Test LLM agent for caching validation"""
    
    def __init__(self, name: str = "TestAgent"):
        super().__init__(name, "test-model", 0.7, 0.1)
        self.mock_response = "I think carefully about the game situation.\n\nAfter analyzing the history and considering my strategy, I choose: C"
        
    def _call_api(self, prompt: str) -> str:
        """Mock API call that counts invocations"""
        print(f"🔄 API Call made with prompt length: {len(prompt)} chars")
        self.api_calls += 1
        self.total_tokens += 150  # Mock token usage
        self.input_tokens += 100
        self.output_tokens += 50
        return self.mock_response
    
    def _call_api_with_caching(self, static_content: str, dynamic_content: str) -> str:
        """Mock API call with caching that counts invocations"""
        print(f"🚀 Caching-optimized API Call: static={len(static_content)}, dynamic={len(dynamic_content)} chars")
        self.api_calls += 1
        self.total_tokens += 150  # Mock token usage
        self.input_tokens += 100
        self.output_tokens += 50
        return self.mock_response


def test_caching_performance():
    """Test caching functionality and performance tracking"""
    print("="*60)
    print("CACHING PERFORMANCE TEST")
    print("="*60)
    
    agent = TestLLMAgent("CacheTestAgent")
    
    print("✓ Created test agent with caching support")
    print(f"  Cache enabled: {agent.enable_caching}")
    print(f"  Cache size limit: {agent.cache_size}")
    
    # Test initial state
    assert agent.api_calls == 0
    assert agent.cache_hits == 0
    assert agent.cache_misses == 0
    print("✓ Initial state verified")
    
    # Test 1: First call should be a cache miss
    print("\n🧪 Test 1: First API call (should be cache miss)")
    move1 = agent.make_move(['C'], ['D'])
    print(f"  Move: {move1}")
    print(f"  API calls: {agent.api_calls}")
    print(f"  Cache hits: {agent.cache_hits}")
    print(f"  Cache misses: {agent.cache_misses}")
    
    assert agent.api_calls == 1
    assert agent.cache_hits == 0
    assert agent.cache_misses == 1
    assert move1 == 'C'
    print("✓ First call correctly tracked as cache miss")
    
    # Test 2: Same call should be a cache hit
    print("\n🧪 Test 2: Identical API call (should be cache hit)")
    move2 = agent.make_move(['C'], ['D'])
    print(f"  Move: {move2}")
    print(f"  API calls: {agent.api_calls}")  # Should still be 1
    print(f"  Cache hits: {agent.cache_hits}")  # Should be 1
    print(f"  Cache misses: {agent.cache_misses}")  # Should still be 1
    
    assert agent.api_calls == 1  # No new API call
    assert agent.cache_hits == 1
    assert agent.cache_misses == 1
    assert move2 == 'C'
    print("✓ Second identical call correctly served from cache")
    
    # Test 3: Different call should be another cache miss
    print("\n🧪 Test 3: Different API call (should be cache miss)")
    move3 = agent.make_move(['C', 'D'], ['D', 'C'])
    print(f"  Move: {move3}")
    print(f"  API calls: {agent.api_calls}")
    print(f"  Cache hits: {agent.cache_hits}")
    print(f"  Cache misses: {agent.cache_misses}")
    
    assert agent.api_calls == 2  # New API call
    assert agent.cache_hits == 1  # Still 1
    assert agent.cache_misses == 2  # Now 2
    assert move3 == 'C'
    print("✓ Different call correctly tracked as cache miss")
    
    # Test 4: Cache statistics
    print("\n🧪 Test 4: Cache performance statistics")
    stats = agent.get_cache_stats()
    print(f"  Cache statistics: {stats}")
    
    expected_hit_rate = 1/3  # 1 hit out of 3 total calls
    assert abs(stats['hit_rate'] - expected_hit_rate) < 0.01
    total_calls = stats['cache_hits'] + stats['cache_misses'] 
    assert total_calls == 3
    assert stats['cache_hits'] == 1
    assert stats['cache_misses'] == 2
    print("✓ Cache statistics correctly calculated")
    
    # Test 5: Multiple identical calls to verify cache efficiency
    print("\n🧪 Test 5: Multiple identical calls for cache efficiency")
    initial_api_calls = agent.api_calls
    
    for i in range(5):
        move = agent.make_move(['C'], ['D'])  # Same as first call
        assert move == 'C'
    
    # Should have no additional API calls
    assert agent.api_calls == initial_api_calls
    assert agent.cache_hits == 6  # 1 + 5 more
    print(f"✓ 5 additional identical calls served from cache")
    print(f"  Final API calls: {agent.api_calls} (no increase)")
    print(f"  Final cache hits: {agent.cache_hits}")
    
    # Final statistics
    print("\n📊 FINAL PERFORMANCE METRICS:")
    final_stats = agent.get_cache_stats()
    total_calls = final_stats['cache_hits'] + final_stats['cache_misses']
    print(f"  Total calls made: {total_calls}")
    print(f"  Cache hits: {final_stats['cache_hits']}")
    print(f"  Cache misses: {final_stats['cache_misses']}")
    print(f"  Cache hit rate: {final_stats['hit_rate']:.1%}")
    print(f"  Tokens saved: {agent.tokens_saved_from_cache}")
    print(f"  API efficiency: {final_stats['cache_hits']}/{total_calls} calls avoided")
    
    print("\n" + "="*60)
    print("✅ ALL CACHING TESTS PASSED!")
    print("✅ Caching infrastructure working correctly")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_caching_performance()
    exit_code = 0 if success else 1
    exit(exit_code)