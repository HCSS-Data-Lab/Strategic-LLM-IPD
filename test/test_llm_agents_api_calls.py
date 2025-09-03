import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import openai
import anthropic
import google.generativeai as genai
from mistralai import Mistral

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading
    pass

# Add the parent directory to the path so we can import ipd_suite
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_suite.agents import GPT4Agent, ClaudeAgent, MistralAgent, GeminiAgent


class TestRealLLMAgentAPICalls:
    """Test suite for real LLM agent API calls"""
    
    def test_gpt4_agent_real_api_call(self):
        """Test GPT4Agent with real API call"""
        # Skip if no API key provided
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        # Create agent with real API key
        agent = GPT4Agent(
            name="TestGPT4Real",
            api_key=api_key,
            model="gpt-5-mini",
            temperature=1,
            termination_prob=0.1
        )
        
        # Test initial state
        assert agent.api_calls == 0
        assert agent.total_tokens == 0
        
        # Make a real API call
        move = agent.make_move(['C'], ['D'])
        
        # Verify the call was made
        assert agent.api_calls == 1
        assert agent.total_tokens > 0
        assert move in ['C', 'D']
        
        # Test with no history
        move2 = agent.make_move([], [])
        assert agent.api_calls == 2
        assert move2 in ['C', 'D']
        
        # Test with longer history
        move3 = agent.make_move(['C', 'D', 'C'], ['D', 'C', 'D'])
        assert agent.api_calls == 3
        assert move3 in ['C', 'D']
        
    def test_claude_agent_real_api_call(self):
        """Test ClaudeAgent with real API call"""
        # Skip if no API key provided
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY environment variable not set")
        
        # Create agent with real API key
        agent = ClaudeAgent(
            name="TestClaudeReal",
            api_key=api_key,
            model="claude-opus-4-20250514",  # claude-3-haiku-20240307
            temperature=0.7,
            termination_prob=0.1
        )
        
        # Test initial state
        assert agent.api_calls == 0
        assert agent.total_tokens == 0
        
        # Make a real API call
        move = agent.make_move(['C'], ['D'])
        
        # Verify the call was made
        assert agent.api_calls == 1
        assert agent.total_tokens > 0
        assert move in ['C', 'D']
        
        # Test with no history
        move2 = agent.make_move([], [])
        assert agent.api_calls == 2
        assert move2 in ['C', 'D']
        
        # Test with longer history
        move3 = agent.make_move(['C', 'D', 'C'], ['D', 'C', 'D'])
        assert agent.api_calls == 3
        assert move3 in ['C', 'D']
        
        # Test different temperature settings (Claude supports 0-1 range)
        test_temperatures = [0.2, 0.7, 1.0]
        for temp in test_temperatures:
            temp_agent = ClaudeAgent(
                name=f"TestClaudeTemp{temp}",
                api_key=api_key,
                model="claude-opus-4-20250514",
                temperature=temp,
                termination_prob=0.1
            )
            
            # Verify temperature is set correctly
            assert temp_agent.temperature == temp
            
            # Make API call with this temperature
            temp_move = temp_agent.make_move(['C'], ['D'])
            assert temp_move in ['C', 'D']
            assert temp_agent.api_calls == 1
    
    def test_mistral_agent_real_api_call(self):
        """Test MistralAgent with real API call"""
        # Skip if no API key provided
        api_key = os.environ.get('MISTRAL_API_KEY')
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
        
        # Create agent with real API key
        agent = MistralAgent(
            name="TestMistralReal",
            api_key=api_key,
            model="mistral-medium-2508",
            temperature=1,
            termination_prob=0.1
        )
        
        # Test initial state
        assert agent.api_calls == 0
        assert agent.total_tokens == 0
        
        # Make a real API call
        move = agent.make_move(['C'], ['D'])
        
        # Verify the call was made
        assert agent.api_calls == 1
        assert agent.total_tokens > 0
        assert move in ['C', 'D']
        
        # Test with no history
        move2 = agent.make_move([], [])
        assert agent.api_calls == 2
        assert move2 in ['C', 'D']
        
        # Test with longer history
        move3 = agent.make_move(['C', 'D', 'C'], ['D', 'C', 'D'])
        assert agent.api_calls == 3
        assert move3 in ['C', 'D']
        
        # Test different temperature settings (Mistral supports 0-1 range)
        test_temperatures = [0.2, 0.7, 1.0]
        for temp in test_temperatures:
            temp_agent = MistralAgent(
                name=f"TestMistralTemp{temp}",
                api_key=api_key,
                model="mistral-medium-2508",
                temperature=temp,
                termination_prob=0.1
            )
            
            # Verify temperature is set correctly
            assert temp_agent.temperature == temp
            
            # Make API call with this temperature
            temp_move = temp_agent.make_move(['C'], ['D'])
            assert temp_move in ['C', 'D']
            assert temp_agent.api_calls == 1
    
    def test_gemini_agent_real_api_call(self):
        """Test GeminiAgent with real API call"""
        # Skip if no API key provided
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            pytest.skip("GOOGLE_API_KEY environment variable not set")
        
        # Create agent with real API key
        agent = GeminiAgent(
            name="TestGeminiReal",
            api_key=api_key,
            model="gemini-2.0-flash",
            temperature=0.7,
            termination_prob=0.1
        )
        
        # Test initial state
        assert agent.api_calls == 0
        assert agent.total_tokens == 0
        
        # Make a real API call
        move = agent.make_move(['C'], ['D'])
        
        # Verify the call was made
        assert agent.api_calls == 1
        assert agent.total_tokens > 0
        assert move in ['C', 'D']
        
        # Test with no history
        move2 = agent.make_move([], [])
        assert agent.api_calls == 2
        assert move2 in ['C', 'D']
        
        # Test with longer history
        move3 = agent.make_move(['C', 'D', 'C'], ['D', 'C', 'D'])
        assert agent.api_calls == 3
        assert move3 in ['C', 'D']
        
        # Test different temperature settings (Gemini supports 0-2 range)
        test_temperatures = [0.2, 0.7, 1.2]
        for temp in test_temperatures:
            temp_agent = GeminiAgent(
                name=f"TestGeminiTemp{temp}",
                api_key=api_key,
                model="gemini-2.0-flash",
                temperature=temp,
                termination_prob=0.1
            )
            
            # Verify temperature is set correctly
            assert temp_agent.temperature == temp
            
            # Make API call with this temperature
            temp_move = temp_agent.make_move(['C'], ['D'])
            assert temp_move in ['C', 'D']
            assert temp_agent.api_calls == 1
    
    def test_real_api_call_consistency(self):
        """Test that multiple real API calls with same input are reasonably consistent"""
        # Skip if no API key provided
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        # Use low temperature for more consistent results
        agent = GPT4Agent(
            name="TestConsistency",
            api_key=api_key,
            model="gpt-4o-mini",
            temperature=0.1,  # Low temperature for consistency
            termination_prob=0.1
        )
        
        # Make multiple calls with same input
        moves = []
        for i in range(3):
            move = agent.make_move(['C', 'D'], ['D', 'C'])
            moves.append(move)
            assert move in ['C', 'D']
        
        # Verify all calls were made
        assert agent.api_calls == 3
        assert agent.total_tokens > 0
        
        # With low temperature, we should get some consistency
        # (though not perfect due to the nature of LLMs)
        assert len(set(moves)) <= 2  # Should not be completely random
    
    def test_real_api_call_error_handling(self):
        """Test error handling with invalid API key"""
        # Create agent with invalid API key
        agent = GPT4Agent(
            name="TestErrorHandling",
            api_key="invalid-key-123",
            model="gpt-4o-mini",
            temperature=0.7,
            termination_prob=0.1
        )
        
        # Should handle error gracefully and default to 'C'
        move = agent.make_move(['C'], ['D'])
        assert move == 'C'
        
        # API call counter should not increment on error
        assert agent.api_calls == 0
        assert agent.total_tokens == 0
    
    def test_real_api_call_with_different_termination_probs(self):
        """Test that different termination probabilities affect the prompt"""
        # Skip if no API key provided
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        # Test with different termination probabilities
        termination_probs = [0.05, 0.2, 0.5]
        
        for term_prob in termination_probs:
            agent = GPT4Agent(
                name=f"TestTermProb{term_prob}",
                api_key=api_key,
                model="gpt-4o-mini",
                temperature=0.7,
                termination_prob=term_prob
            )
            
            # Test that the prompt includes the termination probability
            prompt = agent._create_prompt(['C'], ['D'])
            assert f"{term_prob*100}%" in prompt
            
            # Make a real API call
            move = agent.make_move(['C'], ['D'])
            assert move in ['C', 'D']
            assert agent.api_calls == 1


class TestLLMAgentAPICalls:
    """Test suite for LLM agent API calls to verify proper parameter passing"""
    
    def test_gpt4_agent_api_call_parameters(self):
        """Test that GPT4Agent passes temperature and other parameters correctly to OpenAI API"""
        # Test parameters
        test_temperature = 0.8
        test_termination_prob = 0.15
        test_model = "gpt-4o-mini"
        test_api_key = "test-key-123"
        
        # Create agent
        agent = GPT4Agent(
            name="TestGPT4", 
            api_key=test_api_key,
            model=test_model,
            temperature=test_temperature,
            termination_prob=test_termination_prob
        )
        
        # Verify initialization parameters
        assert agent.temperature == test_temperature
        assert agent.termination_prob == test_termination_prob
        assert agent.model == test_model
        
        # Mock the OpenAI API call
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            # Mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "C"
            mock_response.usage.total_tokens = 100
            mock_create.return_value = mock_response
            
            # Make a move to trigger API call
            move = agent.make_move([], [])
            
            # Verify API call was made with correct parameters
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            
            # Check that temperature was passed correctly
            assert call_args.kwargs['temperature'] == test_temperature
            assert call_args.kwargs['model'] == test_model
            assert call_args.kwargs['max_tokens'] == 500
            
            # Check message structure
            messages = call_args.kwargs['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert messages[1]['role'] == 'user'
            
            # Check that termination_prob is used in prompt
            prompt_content = messages[1]['content']
            assert f"{test_termination_prob*100}%" in prompt_content
            
            # Verify return value
            assert move == "C"
    
    def test_claude_agent_api_call_parameters(self):
        """Test that ClaudeAgent passes temperature and other parameters correctly to Anthropic API"""
        # Test parameters
        test_temperature = 0.9
        test_termination_prob = 0.25
        test_model = "claude-3-sonnet-20240229"
        test_api_key = "test-claude-key"
        
        # Create agent
        agent = ClaudeAgent(
            name="TestClaude",
            api_key=test_api_key,
            model=test_model,
            temperature=test_temperature,
            termination_prob=test_termination_prob
        )
        
        # Verify initialization parameters
        assert agent.temperature == test_temperature
        assert agent.termination_prob == test_termination_prob
        assert agent.model == test_model
        
        # Mock the Anthropic API call
        with patch.object(agent.client.messages, 'create') as mock_create:
            # Mock response
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = "D"
            mock_create.return_value = mock_response
            
            # Make a move to trigger API call
            move = agent.make_move(['C'], ['D'])
            
            # Verify API call was made with correct parameters
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            
            # Check that temperature was passed correctly
            assert call_args.kwargs['temperature'] == test_temperature
            assert call_args.kwargs['model'] == test_model
            assert call_args.kwargs['max_tokens'] == 500
            
            # Check message structure
            messages = call_args.kwargs['messages']
            assert len(messages) == 1
            assert messages[0]['role'] == 'user'
            
            # Check that termination_prob is used in prompt
            prompt_content = messages[0]['content']
            assert f"{test_termination_prob*100}%" in prompt_content
            
            # Verify return value
            assert move == "D"
    
    def test_mistral_agent_api_call_parameters(self):
        """Test that MistralAgent passes temperature and other parameters correctly to Mistral API"""
        # Test parameters
        test_temperature = 0.6
        test_termination_prob = 0.35
        test_model = "mistral-medium-2508"
        test_api_key = "test-mistral-key"
        
        # Create agent
        agent = MistralAgent(
            name="TestMistral",
            api_key=test_api_key,
            model=test_model,
            temperature=test_temperature,
            termination_prob=test_termination_prob
        )
        
        # Verify initialization parameters
        assert agent.temperature == test_temperature
        assert agent.termination_prob == test_termination_prob
        assert agent.model == test_model
        
        # Mock the Mistral API call
        with patch.object(agent.client.chat, 'complete') as mock_complete:
            # Mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "C"
            mock_response.usage.total_tokens = 150
            mock_complete.return_value = mock_response
            
            # Make a move to trigger API call
            move = agent.make_move(['D', 'C'], ['C', 'D'])
            
            # Verify API call was made with correct parameters
            mock_complete.assert_called_once()
            call_args = mock_complete.call_args
            
            # Check that temperature was passed correctly
            assert call_args.kwargs['temperature'] == test_temperature
            assert call_args.kwargs['model'] == test_model
            assert call_args.kwargs['max_tokens'] == 500
            
            # Check message structure
            messages = call_args.kwargs['messages']
            assert len(messages) == 1
            assert messages[0]['role'] == 'user'
            
            # Check that termination_prob is used in prompt
            prompt_content = messages[0]['content']
            assert f"{test_termination_prob*100}%" in prompt_content
            
            # Verify return value
            assert move == "C"
    
    def test_gemini_agent_api_call_parameters(self):
        """Test that GeminiAgent passes temperature and other parameters correctly to Google API"""
        # Test parameters
        test_temperature = 1.0
        test_termination_prob = 0.10
        test_model = "gemini-2.0-flash"
        test_api_key = "test-gemini-key"
        
        # Mock the genai.configure call
        with patch('google.generativeai.configure') as mock_configure:
            # Mock the GenerativeModel
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model_instance = Mock()
                mock_model_class.return_value = mock_model_instance
                
                # Create agent
                agent = GeminiAgent(
                    name="TestGemini",
                    api_key=test_api_key,
                    model=test_model,
                    temperature=test_temperature,
                    termination_prob=test_termination_prob
                )
                
                # Verify initialization parameters
                assert agent.temperature == test_temperature
                assert agent.termination_prob == test_termination_prob
                assert agent.model == test_model
                
                # Verify genai.configure was called with API key
                mock_configure.assert_called_once_with(api_key=test_api_key)
                
                # Verify GenerativeModel was instantiated with correct model
                mock_model_class.assert_called_once_with(test_model)
                
                # Mock the generate_content method
                mock_response = Mock()
                mock_response.text = "D"
                mock_model_instance.generate_content.return_value = mock_response
                
                # Make a move to trigger API call
                move = agent.make_move(['C', 'C'], ['C', 'D'])
                
                # Verify API call was made with correct parameters
                mock_model_instance.generate_content.assert_called_once()
                call_args = mock_model_instance.generate_content.call_args
                
                # Check that generation config includes temperature
                generation_config = call_args.kwargs['generation_config']
                assert generation_config.temperature == test_temperature
                assert generation_config.max_output_tokens == 500
                
                # Check that termination_prob is used in prompt
                prompt_content = call_args.args[0]
                assert f"{test_termination_prob*100}%" in prompt_content
                
                # Verify return value
                assert move == "D"
    
    def test_llm_agent_prompt_generation(self):
        """Test that LLM agents generate prompts with correct termination probability"""
        # Test parameters
        test_termination_prob = 0.33
        
        # Create a GPT4Agent to test prompt generation
        agent = GPT4Agent(
            name="TestPrompt",
            api_key="test-key",
            temperature=0.7,
            termination_prob=test_termination_prob
        )
        
        # Test prompt generation
        prompt = agent._create_prompt(['C', 'D'], ['D', 'C'])
        
        # Verify termination probability is correctly included
        assert f"{test_termination_prob*100}%" in prompt
        assert "33.0%" in prompt  # Specific value check
        
        # Verify other prompt elements
        assert "Iterated Prisoner's Dilemma" in prompt
        assert "Cooperate (C) or Defect (D)" in prompt
        assert "Payoffs:" in prompt
        assert "Both C = 3" in prompt
        assert "Both D = 1" in prompt
        assert "You C/They D = 0/5" in prompt
        assert "You D/They C = 5/0" in prompt
        assert "History of moves (You, Opponent): [(C,D), (D,C)]" in prompt
        assert "either 'C' or 'D'" in prompt
    
    def test_llm_agent_error_handling(self):
        """Test that LLM agents handle API errors gracefully"""
        # Create agent
        agent = GPT4Agent(
            name="TestError",
            api_key="test-key",
            temperature=0.7,
            termination_prob=0.1
        )
        
        # Mock the OpenAI API call to raise an exception
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            # Make a move - should handle error gracefully
            move = agent.make_move(['C'], ['D'])
            
            # Should default to cooperation on error
            assert move == 'C'
    
    def test_llm_agent_move_extraction(self):
        """Test that LLM agents correctly extract moves from API responses"""
        # Test various response formats
        # Note: The extraction looks at the last line first, then checks for 'C' or 'D' presence
        test_cases = [
            ("C", "C"),                                    # Exact match
            ("D", "D"),                                    # Exact match
            ("My move is C", "C"),                         # 'C' present, 'D' not present
            ("I will defect", "C"),                        # Both 'C' and 'D' present, defaults to 'C'
            ("Only D here", "D"),                          # 'D' present, 'C' not present
            ("Cooperation is the best strategy.\nC", "C"), # Last line is 'C'
            ("After analysis, I decide to defect.\nD", "D"),  # Last line is 'D'
            ("Unclear response with both C and D", "C"),   # Defaults to C when both present
            ("No clear move here", "C"),                   # Defaults to C when no clear move
        ]
        
        # Create agent
        agent = GPT4Agent(
            name="TestExtraction",
            api_key="test-key",
            temperature=0.7,
            termination_prob=0.1
        )
        
        for response_text, expected_move in test_cases:
            # Mock the API response
            with patch.object(agent.client.chat.completions, 'create') as mock_create:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = response_text
                mock_response.usage.total_tokens = 50
                mock_create.return_value = mock_response
                
                # Make a move
                move = agent.make_move(['C'], ['D'])
                
                # Verify correct extraction
                assert move == expected_move, f"Failed for response: '{response_text}'"
    
    def test_llm_agent_token_tracking(self):
        """Test that LLM agents track API calls and tokens correctly"""
        # Create agent
        agent = GPT4Agent(
            name="TestTracking",
            api_key="test-key",
            temperature=0.7,
            termination_prob=0.1
        )
        
        # Initial state
        assert agent.api_calls == 0
        assert agent.total_tokens == 0
        
        # Mock API responses
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "C"
            mock_response.usage.total_tokens = 75
            mock_create.return_value = mock_response
            
            # Make first move
            agent.make_move(['C'], ['D'])
            assert agent.api_calls == 1
            assert agent.total_tokens == 75
            
            # Make second move
            mock_response.usage.total_tokens = 80
            agent.make_move(['C', 'C'], ['D', 'C'])
            assert agent.api_calls == 2
            assert agent.total_tokens == 155  # 75 + 80


if __name__ == "__main__":
    pytest.main([__file__])