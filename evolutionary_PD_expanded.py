"""
Evolutionary Prisoner's Dilemma Simulation with Expanded Agents
This version includes an expanded set of classic agents and a Bayesian agent.
"""

import os
import sys
import time
import random
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Callable, Literal, Any, Optional
from datetime import datetime
import json
import math
from dotenv import load_dotenv
import openai
import requests

# Try to load environment variables from .env file
try:
    load_dotenv("Axelrod.env")
    print("Environment variables loaded from Axelrod.env")
    
    # Set up API keys
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Print masked API keys for debugging
    if openai.api_key:
        masked_key = openai.api_key[:4] + "..." + openai.api_key[-4:]
        print(f"Found OpenAI API key: {masked_key}")
    if GEMINI_API_KEY:
        masked_key = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:]
        print(f"Found Gemini API key: {masked_key}")
        
    HAS_API_AGENTS = bool(openai.api_key and GEMINI_API_KEY)
    if HAS_API_AGENTS:
        print("API-based agents will be available")
    else:
        print("API-based agents will not be available")
except Exception as e:
    print(f"Warning: Could not load environment variables: {e}")
    HAS_API_AGENTS = False

# Base Agent class
class Agent:
    def __init__(self, name):
        self.name = name
        self.history = []
        self.last_reasoning = None
        self.opponent_history = []
        self.score = 0
        
    def make_move(self, own_history, opponent_history):
        """Base method to be overridden by subclasses"""
        raise NotImplementedError
        
    def reset(self):
        """Reset agent state between matches"""
        self.history = []
        self.opponent_history = []
        self.last_reasoning = None
        self.score = 0
        
    def update(self, my_move, opponent_move):
        """Update the agent's history with both moves."""
        self.history.append(my_move)
        self.opponent_history.append(opponent_move)
        
    def add_score(self, points):
        """Add points to the agent's score."""
        self.score += points

# Constants for payoffs (standard Prisoner's Dilemma values)
PAYOFF_BOTH_COOPERATE = 3
PAYOFF_BOTH_DEFECT = 1
PAYOFF_DEFECTOR = 5
PAYOFF_COOPERATOR = 0

# ---- MODIFIED Endpoint and Gemini Agent ----
GEMINI_MODEL_NAME_FOR_API = "gemini-2.5-flash-preview-05-20"
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME_FOR_API}:generateContent"

# AI Agent implementations
class OpenAIAgent(Agent):
    def __init__(self, name="OpenAI Agent"):
        super().__init__(name)
        self.model = "gpt-4o-mini-2024-07-18"
        
    def make_move(self, own_history, opponent_history, termination_prob=0.1):
        if not HAS_API_AGENTS:
            self.last_reasoning = "API key not available, defaulting to cooperation"
            return "C"
            
        # Only consider last 20 move pairs to stay within context limits
        recent_own_history = own_history[-20:] if own_history else []
        recent_opponent_history = opponent_history[-20:] if opponent_history else []
        total_moves = len(opponent_history)
        
        # Create paired history representation
        paired_history = []
        for i in range(len(recent_own_history)):
            paired_history.append((recent_own_history[i], recent_opponent_history[i]))
            
        history_str = f"PAIRED MOVES: [] (Total moves played: 0)" if not paired_history else \
                     f"PAIRED MOVES (You, Opponent): {paired_history} (Total moves played: {total_moves})"
        
        system_message = f"""You are playing the Prisoner's Dilemma game. You will be given a list of paired moves that both you and your opponent have made.
        Each pair is in the format (Your move, Opponent's move).
        The match will end randomly after any given round with a {termination_prob*100:.0f}% probability. Your goal is to maximize your score over the course of the match, however long it lasts.
        
        Your response MUST be structured as follows:
        - Line 1 (and potentially subsequent lines): Your reasoning for your move.
        - The VERY LAST line: Your move (a single letter: C or D).

        Example:
        Reasoning line 1 explaining my thoughts.
        Potentially more reasoning on line 2.
        C

        Remember that in the Prisoner's Dilemma:
        - If both cooperate (C,C), both get 3 points
        - If both defect (D,D), both get 1 point
        - If one cooperates and one defects (C,D), the defector gets 5 points and the cooperator gets 0 points
        """
        
        user_message = f"""Your and your opponent's moves:
{history_str}

What is your next move? Provide your reasoning first, and then your move (C or D) on the last line."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content.strip()
            lines = [line.strip() for line in full_response.split('\n') if line.strip()]
            
            # Validate response format
            if not lines or len(lines) < 2: # Need at least one line for reasoning and one for move
                print(f"DEBUG: OpenAI - Insufficient lines in response. Raw response: '{full_response}'")
                self.last_reasoning = "Insufficient lines in response (expected reasoning and move), defaulting to cooperation"
                return "C"
            
            move_candidate = lines[-1].strip().upper()
            if move_candidate not in ["C", "D"]:
                print(f"DEBUG: OpenAI - Invalid move in last line. Raw response: '{full_response}'")
                self.last_reasoning = f"Invalid move ('{move_candidate}') in last line, defaulting to cooperation"
                return "C"
                
            # Combine all lines except the last one for reasoning
            self.last_reasoning = ' '.join(lines[:-1]).strip()
            move = move_candidate
            
            return move
            
        except Exception as e:
            self.last_reasoning = f"API error: {str(e)}, defaulting to cooperation"
            return "C"

class GeminiAgent(Agent):
    def __init__(self, name="Gemini Agent"):
        super().__init__(name)
        self.model_name = GEMINI_MODEL_NAME_FOR_API # Use the centrally defined model name
        
    def make_move(self, own_history, opponent_history, termination_prob=0.1):
        if not HAS_API_AGENTS:
            self.last_reasoning = "API key not available, defaulting to cooperation"
            return "C"
            
        # Only consider last 20 move pairs to stay within context limits
        recent_own_history = own_history[-20:] if own_history else []
        recent_opponent_history = opponent_history[-20:] if opponent_history else []
        total_moves = len(opponent_history)
        
        # Create paired history representation
        paired_history = []
        for i in range(len(recent_own_history)):
            paired_history.append((recent_own_history[i], recent_opponent_history[i]))
            
        history_str = f"PAIRED MOVES: [] (Total moves played: 0)" if not paired_history else \
                     f"PAIRED MOVES (You, Opponent): {paired_history} (Total moves played: {total_moves})"
        
        system_message = f"""You are playing the Prisoner's Dilemma game. You will be given a list of paired moves that both you and your opponent have made.
        Each pair is in the format (Your move, Opponent's move).
        The match will end randomly after any given round with a {termination_prob*100:.0f}% probability. Your goal is to maximize your score over the course of the match, however long it lasts.
        
        Your response MUST be structured as follows:
        - Line 1 (and potentially subsequent lines): Your reasoning for your move.
        - The VERY LAST line: Your move (a single letter: C or D).

        Example:
        Reasoning line 1 explaining my thoughts.
        Potentially more reasoning on line 2.
        C

        Remember that in the Prisoner's Dilemma:
        - If both cooperate (C,C), both get 3 points
        - If both defect (D,D), both get 1 point
        - If one cooperates and one defects (C,D), the defector gets 5 points and the cooperator gets 0 points
        """
        
        user_message = f"""Your and your opponent's moves:
{history_str}

What is your next move? Provide your reasoning first, and then your move (C or D) on the last line."""

        try:
            response = requests.post(
                GEMINI_API_ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY
                },
                json={
                    "contents": [{"parts": [{"text": system_message + "\n\n" + user_message}]}]
                }
            )
            response.raise_for_status()
            
            full_response = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            lines = [line.strip() for line in full_response.split('\n') if line.strip()]
            
            # Validate response format
            if not lines or len(lines) < 2: # Need at least one line for reasoning and one for move
                print(f"DEBUG: Gemini - Insufficient lines in response. Raw response: '{full_response}'")
                self.last_reasoning = "Insufficient lines in response (expected reasoning and move), defaulting to cooperation"
                return "C"

            move_candidate = lines[-1].strip().upper()
            if move_candidate not in ["C", "D"]:
                print(f"DEBUG: Gemini - Invalid move in last line. Raw response: '{full_response}'")
                self.last_reasoning = f"Invalid move ('{move_candidate}') in last line, defaulting to cooperation"
                return "C"
                
            # Combine all lines except the last one for reasoning
            self.last_reasoning = ' '.join(lines[:-1]).strip()
            move = move_candidate
            
            return move
            
        except Exception as e:
            self.last_reasoning = f"API error: {str(e)}, defaulting to cooperation"
            return "C"

# Basic strategy implementations
class TitForTatAgent(Agent):
    def make_move(self, own_history, opponent_history):
        self.last_reasoning = "Copying opponent's last move."
        if not opponent_history:
            return "C"  # Cooperate on first move
        return opponent_history[-1]  # Copy opponent's last move

class GrimTriggerAgent(Agent):
    def make_move(self, own_history, opponent_history):
        if not opponent_history:
            self.last_reasoning = "Starting with cooperation."
            return "C"  # Cooperate on first move
        
        if "D" in opponent_history:
            self.last_reasoning = "Opponent has defected before, so I will defect forever."
            return "D"  # Defect forever if opponent has ever defected
        else:
            self.last_reasoning = "Opponent has always cooperated, so I will cooperate."
            return "C"

class WinStayLoseShiftAgent(Agent):
    def make_move(self, own_history, opponent_history):
        if not own_history or not opponent_history:
            self.last_reasoning = "Starting with cooperation."
            return "C"  # Cooperate on first move
        
        last_move = own_history[-1]
        last_opponent_move = opponent_history[-1]
        
        # Win-Stay, Lose-Shift logic
        if (last_move == "C" and last_opponent_move == "C") or (last_move == "D" and last_opponent_move == "D"):
            # Win - repeat last move
            self.last_reasoning = f"Last round was a 'win', staying with {last_move}."
            return last_move
        else:
            # Lose - switch strategy
            next_move = "D" if last_move == "C" else "C"
            self.last_reasoning = f"Last round was a 'loss', shifting from {last_move} to {next_move}."
            return next_move

class RandomAgent(Agent):
    def make_move(self, own_history, opponent_history):
        move = random.choice(["C", "D"])
        self.last_reasoning = f"Randomly chose {move}."
        return move

# New expanded agents

class GenerousTFTAgent(Agent):
    def __init__(self, name, forgiveness_prob=0.1):
        super().__init__(name)
        self.forgiveness_prob = forgiveness_prob
        
    def make_move(self, own_history, opponent_history):
        if not opponent_history:
            self.last_reasoning = "Starting with cooperation."
            return "C"  # Cooperate on first move
        
        if opponent_history[-1] == "D":
            # 10% chance to forgive a defection
            if random.random() < self.forgiveness_prob:
                self.last_reasoning = "Opponent defected, but I'm being generous and forgiving."
                return "C"
            else:
                self.last_reasoning = "Opponent defected, so I'm reciprocating with defection."
                return "D"
        else:
            self.last_reasoning = "Opponent cooperated, so I'm cooperating too."
            return "C"

class SuspiciousTFTAgent(Agent):
    def make_move(self, own_history, opponent_history):
        if not opponent_history:
            self.last_reasoning = "Starting with defection (suspicious)."
            return "D"  # Defect on first move
        
        # Then copy opponent's last move
        self.last_reasoning = f"Copying opponent's last move ({opponent_history[-1]})."
        return opponent_history[-1]

class GradualAgent(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.defect_count = 0
        self.retaliation_count = 0
        self.retaliating = False
        
    def reset(self):
        super().reset()
        self.defect_count = 0
        self.retaliation_count = 0
        self.retaliating = False
        
    def make_move(self, own_history, opponent_history):
        if not opponent_history:
            self.last_reasoning = "Starting with cooperation."
            return "C"  # Cooperate on first move
        
        # If currently retaliating
        if self.retaliating:
            self.retaliation_count -= 1
            if self.retaliation_count > 0:
                self.last_reasoning = f"Continuing retaliation ({self.retaliation_count} more defections)."
                return "D"
            else:
                self.retaliating = False
                self.last_reasoning = "Retaliation complete, offering cooperation as peace."
                return "C"
        
        # Check if opponent defected
        if opponent_history[-1] == "D":
            self.defect_count += 1
            self.retaliation_count = self.defect_count
            self.retaliating = True
            self.last_reasoning = f"Opponent defected, initiating retaliation of {self.retaliation_count} defections."
            return "D"
        
        self.last_reasoning = "Opponent cooperated, so I'm cooperating too."
        return "C"

class ProberAgent(Agent):
    def make_move(self, own_history, opponent_history):
        if len(own_history) == 0:
            self.last_reasoning = "Starting with cooperation."
            return "C"  # First move: cooperate
        elif len(own_history) == 1:
            self.last_reasoning = "Testing opponent with defection on second move."
            return "D"  # Second move: defect to test
        elif len(own_history) == 2:
            self.last_reasoning = "Back to cooperation for the third move."
            return "C"  # Third move: cooperate
        else:
            # If opponent didn't retaliate after our defection, exploit them
            if opponent_history[1] == "C" and opponent_history[2] == "C":
                self.last_reasoning = "Opponent didn't retaliate to my test, so I'll exploit with defection."
                return "D"  # Exploit
            else:
                # Otherwise play TFT
                self.last_reasoning = f"Playing Tit-for-Tat, copying opponent's last move ({opponent_history[-1]})."
                return opponent_history[-1]

class AlternatorAgent(Agent):
    def make_move(self, own_history, opponent_history):
        if len(own_history) % 2 == 0:
            self.last_reasoning = "Alternating pattern: cooperate on even-numbered rounds."
            return "C"
        else:
            self.last_reasoning = "Alternating pattern: defect on odd-numbered rounds."
            return "D"

class BayesianAgent(Agent):
    def __init__(self, name):
        super().__init__(name)
        # Initialize equal probabilities for all strategies
        self.strategy_probs = {
            'tit_for_tat': 0.1,
            'grim_trigger': 0.1,
            'win_stay_lose_shift': 0.1,
            'random': 0.1,
            'generous_tft': 0.1,
            'suspicious_tft': 0.1,
            'gradual': 0.1,
            'prober': 0.1,
            'alternator': 0.1,
            'always_cooperate': 0.1  # Added as a baseline strategy to detect
        }
        self.evidence_count = 0
        
    def reset(self):
        super().reset()
        self.strategy_probs = {
            'tit_for_tat': 0.1,
            'grim_trigger': 0.1,
            'win_stay_lose_shift': 0.1,
            'random': 0.1,
            'generous_tft': 0.1,
            'suspicious_tft': 0.1,
            'gradual': 0.1,
            'prober': 0.1,
            'alternator': 0.1,
            'always_cooperate': 0.1
        }
        self.evidence_count = 0
        
    def predict_strategy_move(self, strategy, opponent_move, prev_opponent_move, prev_my_move, move_history_length):
        """Predict what move a strategy would make in the current situation."""
        if strategy == 'always_cooperate':
            return "C"
        elif strategy == 'tit_for_tat':
            return prev_my_move if prev_my_move else "C"
        elif strategy == 'grim_trigger':
            return "D" if "D" in self.opponent_history else "C"
        elif strategy == 'win_stay_lose_shift':
            if not prev_my_move or not prev_opponent_move:
                return "C"
            # Win conditions: (C,C) or (D,D)
            won_last = (prev_my_move == prev_opponent_move)
            return prev_my_move if won_last else ("D" if prev_my_move == "C" else "C")
        elif strategy == 'random':
            return None  # Special case: handled in likelihood calculation
        elif strategy == 'generous_tft':
            if not prev_my_move:
                return "C"
            return "C" if random.random() < 0.1 else prev_my_move
        elif strategy == 'suspicious_tft':
            return prev_my_move if prev_my_move else "D"
        elif strategy == 'gradual':
            if not prev_opponent_move:
                return "C"
            defection_count = self.opponent_history.count("D")
            if defection_count > 0:
                # In retaliation mode
                return "D"
            return "C"
        elif strategy == 'prober':
            if move_history_length == 0:
                return "C"
            elif move_history_length == 1:
                return "D"
            elif move_history_length == 2:
                return "C"
            else:
                # If opponent didn't retaliate to defection, exploit
                if self.opponent_history[1] == "C" and self.opponent_history[2] == "C":
                    return "D"
                return prev_opponent_move
        elif strategy == 'alternator':
            return "D" if move_history_length % 2 == 1 else "C"
        return "C"  # Default to cooperation for unknown strategies
        
    def update_probs(self, opponent_move, prev_opponent_move=None, prev_my_move=None):
        # Skip update on first move since we have no evidence
        if prev_opponent_move is None:
            return
            
        # Calculate likelihood of this move given each strategy
        likelihoods = {}
        move_history_length = len(self.opponent_history)
        
        for strategy in self.strategy_probs:
            if strategy == 'random':
                # Random strategy: 50% chance of either move
                likelihoods[strategy] = 0.5
            else:
                # Predict what this strategy would have done
                predicted = self.predict_strategy_move(
                    strategy, opponent_move, prev_opponent_move, 
                    prev_my_move, move_history_length - 1
                )
                # High likelihood if prediction matches actual move
                likelihoods[strategy] = 0.95 if predicted == opponent_move else 0.05
        
        # Bayesian update
        total = 0
        for strategy in self.strategy_probs:
            self.strategy_probs[strategy] *= likelihoods[strategy]
            total += self.strategy_probs[strategy]
            
        # Normalize
        if total > 0:
            for strategy in self.strategy_probs:
                self.strategy_probs[strategy] /= total
                
        self.evidence_count += 1
        
    def choose_counter_strategy(self, most_likely, highest_prob):
        """Choose the best counter-strategy based on opponent identification."""
        if most_likely in ['always_cooperate', 'generous_tft']:
            self.last_reasoning = f"Determined opponent is likely {most_likely} ({highest_prob:.2f}), exploiting with defection."
            return "D"  # Exploit cooperative strategies
            
        elif most_likely in ['grim_trigger', 'suspicious_tft']:
            if "D" not in self.opponent_history:
                self.last_reasoning = f"Opponent likely {most_likely} ({highest_prob:.2f}) but hasn't defected, cooperating."
                return "C"
            else:
                self.last_reasoning = f"Opponent likely {most_likely} ({highest_prob:.2f}) and has defected, matching defection."
                return "D"
                
        elif most_likely == 'tit_for_tat':
            self.last_reasoning = f"Determined opponent is likely TFT ({highest_prob:.2f}), cooperating to establish mutual cooperation."
            return "C"
            
        elif most_likely == 'win_stay_lose_shift':
            if not self.opponent_history:
                return "C"
            # Try to manipulate WSLS by breaking its pattern
            self.last_reasoning = f"Opponent likely WSLS ({highest_prob:.2f}), attempting to manipulate its pattern."
            return "D" if len(self.history) % 2 == 0 else "C"
            
        elif most_likely == 'gradual':
            if "D" not in self.opponent_history:
                self.last_reasoning = f"Opponent likely Gradual ({highest_prob:.2f}) but hasn't defected, cooperating."
                return "C"
            else:
                self.last_reasoning = f"Opponent likely Gradual ({highest_prob:.2f}) and has defected, matching defection."
                return "D"
                
        elif most_likely == 'prober':
            if len(self.opponent_history) < 3:
                self.last_reasoning = f"Opponent likely Prober ({highest_prob:.2f}), cooperating during testing phase."
                return "C"
            else:
                self.last_reasoning = f"Opponent likely Prober ({highest_prob:.2f}), matching its moves."
                return self.opponent_history[-1]
                
        elif most_likely == 'alternator':
            self.last_reasoning = f"Opponent likely Alternator ({highest_prob:.2f}), exploiting with defection."
            return "D"
            
        elif most_likely == 'random':
            self.last_reasoning = f"Opponent likely Random ({highest_prob:.2f}), defaulting to defection."
            return "D"
            
        return self.opponent_history[-1]  # Default to TFT if unsure
        
    def make_move(self, own_history, opponent_history):
        if not opponent_history:
            self.last_reasoning = "Starting with cooperation to gather data."
            return "C"  # Cooperate on first move to gather data
            
        # Update probabilities based on opponent's last move
        prev_opponent_move = opponent_history[-2] if len(opponent_history) > 1 else None
        prev_my_move = own_history[-2] if len(own_history) > 1 else None
        self.update_probs(opponent_history[-1], prev_opponent_move, prev_my_move)
        
        # Determine most likely strategy
        most_likely = max(self.strategy_probs, key=self.strategy_probs.get)
        highest_prob = self.strategy_probs[most_likely]
        
        # If we're confident enough, use best counter-strategy
        if highest_prob > 0.7 and self.evidence_count >= 3:
            return self.choose_counter_strategy(most_likely, highest_prob)
        else:
            # Not confident enough, use TFT as safe fallback
            self.last_reasoning = f"Not confident in opponent strategy yet. Probs: {', '.join([f'{k}:{v:.2f}' for k, v in self.strategy_probs.items()])}. Using TFT."
            return opponent_history[-1]

def create_agent(strategy_name, agent_id=None):
    """Create an agent instance based on strategy name."""
    if agent_id is None:
        name = strategy_name
    else:
        name = f"{strategy_name}_{agent_id}"
        
    strategy_map = {
        "TitForTat": TitForTatAgent,
        "GrimTrigger": GrimTriggerAgent,
        "WinStayLoseShift": WinStayLoseShiftAgent,
        "Random": RandomAgent,
        "GenerousTFT": GenerousTFTAgent,
        "SuspiciousTFT": SuspiciousTFTAgent,
        "Gradual": GradualAgent,
        "Prober": ProberAgent,
        "Alternator": AlternatorAgent,
        "Bayesian": BayesianAgent,
        "OpenAI": OpenAIAgent if HAS_API_AGENTS else None,
        "Gemini": GeminiAgent if HAS_API_AGENTS else None
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    if strategy_map[strategy_name] is None:
        raise ValueError(f"Strategy {strategy_name} is not available (API agents not loaded)")
        
    return strategy_map[strategy_name](name)

def play_round(agent1, agent2, termination_prob=0.1):
    """
    Play a single round of the Prisoner's Dilemma.
    
    Args:
        agent1: First agent
        agent2: Second agent
        
    Returns:
        Dictionary with results of the round
    """
    # Get agent moves
    try:
        if isinstance(agent1, (OpenAIAgent, GeminiAgent)):
            move1 = agent1.make_move(agent1.history, agent2.history, termination_prob=termination_prob)
        else:
        move1 = agent1.make_move(agent1.history, agent2.history)

        if not move1 or move1 not in ["C", "D"]:
            print(f"Warning: {agent1.name} returned invalid move '{move1}', defaulting to 'C'")
            move1 = "C"
    except Exception as e:
        print(f"Error: {agent1.name} failed to make a move: {e}")
        move1 = "C"  # Default to cooperate on error
    
    try:
        if isinstance(agent2, (OpenAIAgent, GeminiAgent)):
            move2 = agent2.make_move(agent2.history, agent1.history, termination_prob=termination_prob)
        else:
        move2 = agent2.make_move(agent2.history, agent1.history)

        if not move2 or move2 not in ["C", "D"]:
            print(f"Warning: {agent2.name} returned invalid move '{move2}', defaulting to 'C'")
            move2 = "C"
    except Exception as e:
        print(f"Error: {agent2.name} failed to make a move: {e}")
        move2 = "C"  # Default to cooperate on error
    
    # Calculate payoffs
    if move1 == "C" and move2 == "C":
        payoff1 = PAYOFF_BOTH_COOPERATE
        payoff2 = PAYOFF_BOTH_COOPERATE
    elif move1 == "C" and move2 == "D":
        payoff1 = PAYOFF_COOPERATOR
        payoff2 = PAYOFF_DEFECTOR
    elif move1 == "D" and move2 == "C":
        payoff1 = PAYOFF_DEFECTOR
        payoff2 = PAYOFF_COOPERATOR
    else:  # Both defect
        payoff1 = PAYOFF_BOTH_DEFECT
        payoff2 = PAYOFF_BOTH_DEFECT
    
    # Update histories and scores
    agent1.update(move1, move2)
    agent2.update(move2, move1)
    agent1.add_score(payoff1)
    agent2.add_score(payoff2)
    
    # Capture reasoning if present
    reasoning1 = getattr(agent1, 'last_reasoning', None)
    reasoning2 = getattr(agent2, 'last_reasoning', None)
    
    # Return round results
    return {
        "agent1": agent1.name,
        "agent2": agent2.name,
        "move1": move1,
        "move2": move2,
        "payoff1": payoff1,
        "payoff2": payoff2,
        "reasoning1": reasoning1,
        "reasoning2": reasoning2
    }

def run_match_with_termination(agent1, agent2, termination_prob=0.1, max_rounds=30, 
                              phase_num=None, match_num=None, total_matches=None):
    """
    Run a match between two agents with probabilistic termination.
    
    Args:
        agent1: First agent
        agent2: Second agent
        termination_prob: Probability of termination after each round
        max_rounds: Maximum number of rounds to play
        phase_num: Current phase number (for display)
        match_num: Current match number (for display)
        total_matches: Total number of matches (for display)
        
    Returns:
        Dictionary with match results
    """
    # Reset agent states and histories
    agent1.reset()
    agent2.reset()
    agent1.history = []
    agent2.history = []
    
    # Print match header
    if phase_num is not None and match_num is not None:
        match_header = f"Phase {phase_num}, Match {match_num}/{total_matches}: {agent1.name} vs {agent2.name}"
        print("\n" + "=" * len(match_header))
        print(match_header)
        print("=" * len(match_header))
    
    rounds = []
    total_score1 = 0
    total_score2 = 0
    
    # Play rounds until termination or max_rounds
    round_num = 1
    while round_num <= max_rounds:
        round_result = play_round(agent1, agent2, termination_prob=termination_prob)
        rounds.append(round_result)
        
        # Update scores
        total_score1 += round_result["payoff1"]
        total_score2 += round_result["payoff2"]
        
        # Print round result
        print(f"Round {round_num} > {agent1.name}: {round_result['move1']}, {agent2.name}: {round_result['move2']}")
        
        # Print AI reasoning (truncated if very long)
        if round_result["reasoning1"]:
            print(f"Reasoning: {round_result['reasoning1'][:500]}...")
            
        if round_result["reasoning2"]:
            print(f"Reasoning: {round_result['reasoning2'][:500]}...")
        
        # Check for termination
        if random.random() < termination_prob:
            print(f"Match terminated after {round_num} rounds")
            break
            
        round_num += 1
        
    # Calculate average scores per round
    num_rounds = len(rounds)
    avg_score1 = total_score1 / num_rounds if num_rounds > 0 else 0
    avg_score2 = total_score2 / num_rounds if num_rounds > 0 else 0
    
    return {
        "agent1": agent1.name,
        "agent2": agent2.name,
        "rounds": rounds,
        "num_rounds": num_rounds,
        "total_score1": total_score1,
        "total_score2": total_score2,
        "avg_score1": avg_score1,
        "avg_score2": avg_score2
    }

def run_tournament_phase(agents, termination_prob=0.1, max_rounds=30, phase_num=None):
    """
    Run a tournament phase with all agents playing against each other.
    
    Args:
        agents: List of agent instances
        termination_prob: Probability of match termination after each round
        max_rounds: Maximum number of rounds per match
        phase_num: Current phase number (for display)
        
    Returns:
        Dictionary with phase results
    """
    if not agents:
        return {"matches": [], "strategy_stats": {}}
    
    # Count total matches for display
    total_matches = len(agents) * (len(agents) - 1) // 2
    current_match = 0
    
    # Track results
    matches = []
    strategy_stats = defaultdict(lambda: {"total_score": 0, "matches_played": 0, "total_rounds": 0})
    
    # Play all pairs
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1 = agents[i]
            agent2 = agents[j]
            
            current_match += 1
            
            # Run the match
            match_result = run_match_with_termination(
                agent1, agent2, 
                termination_prob=termination_prob,
                max_rounds=max_rounds,
                phase_num=phase_num,
                match_num=current_match,
                total_matches=total_matches
            )
            
            matches.append(match_result)
            
            # Extract strategy names from agent names (removing the _ID part)
            strategy1 = agent1.name.split('_')[0]
            strategy2 = agent2.name.split('_')[0]
            
            # Update strategy statistics
            actual_rounds_played = match_result["num_rounds"]
            
            strategy_stats[strategy1]["total_score"] += match_result["total_score1"]
            strategy_stats[strategy1]["matches_played"] += 1
            strategy_stats[strategy1]["total_rounds"] += actual_rounds_played
            
            strategy_stats[strategy2]["total_score"] += match_result["total_score2"]
            strategy_stats[strategy2]["matches_played"] += 1
            strategy_stats[strategy2]["total_rounds"] += actual_rounds_played
    
    # Calculate average scores
    for strategy, stats in strategy_stats.items():
        if stats["matches_played"] > 0:
            # Original metric (per match average)
            stats["avg_score_per_match"] = stats["total_score"] / stats["matches_played"]
            
            # Better metric (true per move average)
            if stats["total_rounds"] > 0:
                stats["avg_score_per_move"] = stats["total_score"] / stats["total_rounds"]
            else:
                stats["avg_score_per_move"] = 0
                
            # Keep the old metric for backward compatibility
            stats["avg_score_per_round"] = stats["avg_score_per_match"]
        else:
            stats["avg_score_per_match"] = 0
            stats["avg_score_per_move"] = 0
            stats["avg_score_per_round"] = 0
    
    return {
        "matches": matches,
        "strategy_stats": strategy_stats
    }

def evolve_population(current_population, phase_result, min_count=0, verbose=True):
    """
    Update the population based on performance in the last phase.
    """
    # Extract average score per move for each strategy (true per-move performance)
    strategy_stats = phase_result["strategy_stats"]
    strategy_fitness = {
        strategy: stats["avg_score_per_move"] 
        for strategy, stats in strategy_stats.items()
    }
    
    # Print detailed fitness values
    if verbose:
        print("\nDetailed Strategy Fitness Calculation:")
        print("----------------------------------------")
        print("Strategy    | Score/Move | Score/Match | Total Score | Rounds | Matches")
        print("------------|------------|-------------|-------------|--------|--------")
        for strategy, stats in sorted(strategy_stats.items(), 
                                     key=lambda x: x[1]['avg_score_per_move'], 
                                     reverse=True):
            print(f"{strategy:12}|    {stats['avg_score_per_move']:.3f}    |    {stats['avg_score_per_match']:.3f}    | {stats['total_score']:.1f}     | {stats['total_rounds']}   | {stats['matches_played']}")
    
    # Calculate total fitness
    total_fitness = sum(strategy_fitness.values())
    if total_fitness == 0:  # Avoid division by zero
        print("Warning: Total fitness is zero. Using equal distribution.")
        new_population = {strategy: min_count for strategy in current_population}
        return new_population
    
    # Calculate new population sizes with enhanced selection pressure
    total_agents = sum(current_population.values())
    new_population = {}
    
    # Calculate mean fitness
    mean_fitness = total_fitness / len(strategy_fitness)
    
    # Print evolutionary calculations
    if verbose:
        print("\nEvolutionary Calculation:")
        print("-------------------------")
        print("Strategy    | Fitness | Relative | Raw Count | Final Count")
        print("------------|---------|----------|-----------|------------")
    
    for strategy, current_count in current_population.items():
        if strategy not in strategy_fitness:
            print(f"Warning: No fitness data for {strategy}, using minimum count.")
            new_population[strategy] = min_count
            continue
            
        # Enhanced fitness-proportional reproduction
        fitness = strategy_fitness[strategy]
        # Calculate relative fitness compared to mean (amplifies differences)
        relative_fitness = (fitness / mean_fitness) ** 2  # Square to amplify differences
        raw_count = (relative_fitness * current_count)
        # Use more aggressive rounding to increase selective pressure
        new_count = max(min_count, int(round(raw_count)))
        new_population[strategy] = new_count
        
        if verbose:
            print(f"{strategy:12}| {fitness:.3f}  | {relative_fitness:.3f}   | {raw_count:.2f}    | {new_count}")
    
    # Adjust to maintain total population size
    original_total = sum(new_population.values())
    adjustment_attempts = 0
    max_adjustments = 100  # Safety limit to prevent infinite loops
    
    while sum(new_population.values()) > total_agents and adjustment_attempts < max_adjustments:
        adjustment_attempts += 1
        # Find strategy with lowest fitness that has more than min_count
        adjustable = [s for s in new_population if new_population[s] > min_count]
        if not adjustable:
            # If all at minimum, reduce the one with the most counts
            strategy = max(new_population, key=new_population.get)
        else:
            # Otherwise reduce the one with lowest fitness
            strategy = min(adjustable, key=lambda s: strategy_fitness.get(s, 0))
        new_population[strategy] -= 1
        if verbose and adjustment_attempts <= 5:  # Only show first few adjustments
            print(f"Adjusting down: {strategy} (lowest adjustable fitness)")
    
    while sum(new_population.values()) < total_agents and adjustment_attempts < max_adjustments:
        adjustment_attempts += 1
        # Find strategy with highest fitness to increment
        strategy = max(strategy_fitness, key=strategy_fitness.get)
        new_population[strategy] += 1
        if verbose and adjustment_attempts <= 5:  # Only show first few adjustments
            print(f"Adjusting up: {strategy} (highest fitness)")
    
    # Verify all strategies have at least min_count
    for strategy in new_population:
        if new_population[strategy] < min_count:
            new_population[strategy] = min_count
            if verbose:
                print(f"Warning: Adjusted {strategy} to minimum count of {min_count}")
    
    if verbose and original_total != sum(new_population.values()):
        print(f"Population adjusted from {original_total} to {sum(new_population.values())} to maintain total of {total_agents} agents")
    
    # Special handling: if a strategy has 0 count, remove it entirely from the population
    # This ensures eliminated strategies really disappear
    new_population = {k: v for k, v in new_population.items() if v > 0}
    
    return new_population

def save_results_to_csv(population_history, phase_results, filename=None, termination_prob=0.1, openai_model_name=None, gemini_model_name=None, is_final=True):
    """
    Save tournament results to a CSV file.
    
    Args:
        population_history: History of populations across phases
        phase_results: Results of each tournament phase
        filename: Optional filename for the CSV file
        is_final: If False, saves only the last phase's data to a phase-specific file.
    """
    if not is_final:
        # Save only the last phase result
        current_phase_num = len(phase_results)
        timestamp_base = filename.split('_consolidated.csv')[0]
        filename = f"{timestamp_base}_phase{current_phase_num}.csv"
        # We only want the data from the most recent phase
        data_to_save = [phase_results[-1]]
    else:
        # Save all results for the final consolidated file
        data_to_save = phase_results
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(["Evolutionary Prisoner's Dilemma Tournament Results"])
        if not is_final:
            writer.writerow(["Run Type", "Standard"])
        else:
            writer.writerow(["Run Type", "Mutation (Random agent reinjected each phase)"])
        writer.writerow(["Generated on", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["Termination Probability", f"{termination_prob*100:.0f}%"])
        if openai_model_name:
            writer.writerow(["OpenAI Model Used", openai_model_name])
        if gemini_model_name:
            writer.writerow(["Gemini Model Used", gemini_model_name])
        writer.writerow([]) # Add an empty line for spacing before the next section
        
        # SECTION 1: Population History
        writer.writerow(["SECTION 1: POPULATION HISTORY"])
        strategies = sorted(list(population_history[0].keys()))
        header_row = ["Phase"] + strategies
        writer.writerow(header_row)
        
        for i, phase_pop in enumerate(population_history):
            row = [f"Phase {i+1}"] + [phase_pop.get(strategy, 0) for strategy in strategies]
            writer.writerow(row)
        
        writer.writerow([])
        
        # SECTION 2: Strategy Performance by Phase
        writer.writerow(["SECTION 2: STRATEGY PERFORMANCE BY PHASE"])
        writer.writerow(["Phase", "Strategy", "ScorePerMove", "ScorePerMatch", "TotalScore", "TotalRounds", "MatchesPlayed"])
        
        for phase_num, phase_result in enumerate(data_to_save, start=1 if is_final else len(phase_results)):
            for strategy, stats in sorted(phase_result["strategy_stats"].items(), 
                                         key=lambda x: x[1]['avg_score_per_move'], 
                                         reverse=True):
                writer.writerow([
                    f"Phase {phase_num}",
                    strategy,
                    round(stats["avg_score_per_move"], 3),
                    round(stats["avg_score_per_match"], 3),
                    round(stats["total_score"], 1),
                    stats["total_rounds"],
                    stats["matches_played"]
                ])
        
        writer.writerow([])
        
        # SECTION 3: Complete Round-by-Round Data with Reasoning
        writer.writerow(["SECTION 3: COMPLETE MATCH AND ROUND DATA"])
        writer.writerow(["Phase", "Match", "Round", "Agent1", "Agent2", "Move1", "Move2", "Payoff1", "Payoff2", "Reasoning1", "Reasoning2"])
        
        for phase_num, phase_result in enumerate(data_to_save, start=1 if is_final else len(phase_results)):
            for match_num, match in enumerate(phase_result["matches"]):
                for round_num, round_data in enumerate(match["rounds"]):
                    # Clean reasoning text for CSV by replacing any newlines and quotes
                    reasoning1 = str(round_data["reasoning1"]).replace('\n', ' ').replace('\r', ' ').replace('"', '""') if round_data["reasoning1"] else ""
                    reasoning2 = str(round_data["reasoning2"]).replace('\n', ' ').replace('\r', ' ').replace('"', '""') if round_data["reasoning2"] else ""
                    
                    writer.writerow([
                        f"Phase {phase_num}",
                        match_num+1,
                        round_num+1,
                        round_data["agent1"],
                        round_data["agent2"],
                        round_data["move1"],
                        round_data["move2"],
                        round_data["payoff1"],
                        round_data["payoff2"],
                        reasoning1,
                        reasoning2
                    ])
        
        writer.writerow([])
        
        # SECTION 4: Match Summary Statistics
        writer.writerow(["SECTION 4: MATCH SUMMARY STATISTICS"])
        writer.writerow(["Phase", "Match", "Agent1", "Agent2", "TotalRounds", "TotalScore1", "TotalScore2", "AvgScore1", "AvgScore2"])
        
        for phase_num, phase_result in enumerate(data_to_save, start=1 if is_final else len(phase_results)):
            for match_num, match in enumerate(phase_result["matches"]):
                writer.writerow([
                    f"Phase {phase_num}",
                    match_num+1,
                    match["agent1"],
                    match["agent2"],
                    match["num_rounds"],
                    round(match["total_score1"], 1),
                    round(match["total_score2"], 1),
                    round(match["avg_score1"], 3),
                    round(match["avg_score2"], 3)
                ])
    
    print(f"Results saved to {filename}")
    return filename

def create_agents_from_population(population, strategy_counters=None):
    """
    Create agent instances from a population dictionary.
    
    Args:
        population: Dictionary mapping strategy names to counts
        strategy_counters: Dictionary to track IDs for each strategy
        
    Returns:
        List of agent instances
    """
    if strategy_counters is None:
        strategy_counters = {strategy: 0 for strategy in population}
    
    agents = []
    for strategy, count in population.items():
        for _ in range(count):
            strategy_counters[strategy] += 1
            agent_id = strategy_counters[strategy]
            agents.append(create_agent(strategy, agent_id))
    
    return agents, strategy_counters

def run_evolutionary_tournament(num_phases=5, initial_population=None, 
                             termination_prob=0.1, max_rounds=30, 
                             verbose_matches=True):
    """
    Run an evolutionary tournament with multiple phases.
    
    Args:
        num_phases: Number of phases to run
        initial_population: Initial population dictionary
        termination_prob: Probability of match termination after each round
        max_rounds: Maximum number of rounds per match
        verbose_matches: Whether to print detailed match information
        
    Returns:
        Tuple of (population_history, final_population, phase_results)
    """
    # Default initial population with all strategies
    if initial_population is None:
        # Include all implemented strategies
        all_strategies = [
            "TitForTat", "GrimTrigger", "WinStayLoseShift", "Random", 
            "GenerousTFT", "SuspiciousTFT", "Gradual", "Prober", "Alternator", "Bayesian"
        ]
        
        # Add AI strategies if available
        if HAS_API_AGENTS:
            all_strategies.extend(["OpenAI", "Gemini"])
            
        # Default to 2 of each strategy
        initial_population = {strategy: 2 for strategy in all_strategies}
    
    # Track populations across phases
    population_history = [initial_population.copy()]
    current_population = initial_population.copy()
    
    # Track tournament results
    phase_results = []
    
    # Determine LLM models used for logging, if any
    openai_model_name = None
    gemini_model_name = None
    if initial_population and "OpenAI" in initial_population and initial_population["OpenAI"] > 0:
        temp_openai_agent = create_agent("OpenAI")
        openai_model_name = getattr(temp_openai_agent, 'model', 'Unknown') # GPT-3.5-turbo uses 'model'
    if initial_population and "Gemini" in initial_population and initial_population["Gemini"] > 0:
        temp_gemini_agent = create_agent("Gemini")
        gemini_model_name = getattr(temp_gemini_agent, 'model_name', 'Unknown')

    # Track timestamp for consistent file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run tournament phases
    for phase in range(num_phases):
        print(f"\n{'='*60}")
        print(f"Phase {phase + 1} of {num_phases}")
        print(f"{'='*60}")
        print(f"Population: {current_population}")
        
        # Reset strategy counters at the beginning of each phase
        # This ensures agent IDs start from 1 in each phase
        strategy_counters = {strategy: 0 for strategy in current_population}
        
        # Create actual agent instances based on population counts
        agents, _ = create_agents_from_population(
            current_population, strategy_counters
        )
        
        # Run the tournament phase
        print(f"\nRunning tournament with {len(agents)} agents...")
        phase_result = run_tournament_phase(
            agents, 
            termination_prob=termination_prob,
            max_rounds=max_rounds,
            phase_num=phase+1
        )
        phase_results.append(phase_result)
        
        # Save results for the current phase (non-final)
        save_results_to_csv(
            population_history, 
            phase_results, 
            filename=f"expanded_evolutionary_pd_{timestamp}_consolidated.csv",
            termination_prob=termination_prob,
            openai_model_name=openai_model_name,
            gemini_model_name=gemini_model_name,
            is_final=False
        )
        
        # Calculate fitness and update population for next phase
        if phase < num_phases - 1:  # Don't evolve after the last phase
            current_population = evolve_population(current_population, phase_result)
            population_history.append(current_population.copy())
            
            # Print population changes
            print("\nPopulation Evolution:")
            for strategy, count in sorted(current_population.items()):
                prev_count = population_history[-2][strategy] if strategy in population_history[-2] else 0
                change = count - prev_count
                change_str = f"(+{change})" if change > 0 else f"({change})" if change < 0 else "(no change)"
                print(f"{strategy}: {prev_count} → {count} {change_str}")
    
    # Save final consolidated results
    print("\nSaving final results...")
    save_results_to_csv(
        population_history, 
        phase_results, 
        f"expanded_evolutionary_pd_{timestamp}_consolidated.csv",
        termination_prob,
        openai_model_name,
        gemini_model_name
    )
    
    return population_history, current_population, phase_results

def print_ai_reasoning_examples(phase_results):
    """Print examples of AI agent reasoning for analysis."""
    print("\nAI Agent Reasoning Examples:")
    print("----------------------------")
    
    # Look for interesting examples (e.g., defections, changes in strategy)
    for phase_idx, phase_result in enumerate(phase_results):
        for match in phase_result["matches"]:
            if "OpenAI" in match["agent1"] or "OpenAI" in match["agent2"] or "Gemini" in match["agent1"] or "Gemini" in match["agent2"]:
                for round_idx, round_data in enumerate(match["rounds"]):
                    # Find rounds where AI defected
                    if ("OpenAI" in round_data["agent1"] and round_data["move1"] == "D") or \
                       ("OpenAI" in round_data["agent2"] and round_data["move2"] == "D") or \
                       ("Gemini" in round_data["agent1"] and round_data["move1"] == "D") or \
                       ("Gemini" in round_data["agent2"] and round_data["move2"] == "D"):
                        
                        print(f"\nPhase {phase_idx+1}, {round_data['agent1']} vs {round_data['agent2']}, Round {round_idx+1}:")
                        print(f"{round_data['agent1']}: {round_data['move1']}, {round_data['agent2']}: {round_data['move2']}")
                        
                        if round_data["reasoning1"] and ("OpenAI" in round_data["agent1"] or "Gemini" in round_data["agent1"]):
                            print(f"{round_data['agent1']} reasoning: {round_data['reasoning1'][:200]}...")
                            
                        if round_data["reasoning2"] and ("OpenAI" in round_data["agent2"] or "Gemini" in round_data["agent2"]):
                            print(f"{round_data['agent2']} reasoning: {round_data['reasoning2'][:200]}...")
                        
                        print("-" * 40)
                        
                        # Limit examples to keep output manageable
                        if round_idx >= 10:
                            break

def main():
    """
    Main function to run the evolutionary tournament with expanded agents.
    """
    # Set up initial population with 2 of each strategy
    initial_population = {
        "TitForTat": 2,         # Classic, robust strategy
        "GrimTrigger": 2,       # Unforgiving strategy
        "WinStayLoseShift": 2,  # Adaptive strategy
        "Random": 2,            # Noise strategy
        "GenerousTFT": 2,       # Forgiving TFT
        "SuspiciousTFT": 2,     # Starts with defection
        "Gradual": 2,           # Escalating retaliation
        "Prober": 2,            # Tests opponent
        "Alternator": 2,        # Simple alternating pattern
        "Bayesian": 2           # Adaptive Bayesian agent
    }
    
    # Add AI agents if available
    if HAS_API_AGENTS:
        initial_population["OpenAI"] = 2
        initial_population["Gemini"] = 2
    
    print(f"Initial population: {initial_population}")
    
    # Run the evolutionary tournament with expanded agents
    print("Starting Expanded Evolutionary Prisoner's Dilemma Tournament")
    print(f"Initial population: {initial_population}")
    # Use the actual termination_prob for the print statement
    current_termination_prob = 0.75 # This is the value passed to run_evolutionary_tournament
    print(f"Termination probability: {current_termination_prob*100:.0f}% (avg ~{1/current_termination_prob:.1f} rounds per match)")
    print(f"Maximum rounds per match: 30")
    print(f"Number of phases: 5")
    
    # Track the total number of agents for population control
    total_agents = sum(initial_population.values())
    
    # Run the tournament
    population_history, final_population, all_phase_results = run_evolutionary_tournament(
        num_phases=5,  
        initial_population=initial_population,
        termination_prob=0.75,
        max_rounds=30,  
        verbose_matches=True
    )
    
    # Print final results
    print("\nFinal Population Distribution:")
    initial_counts = population_history[0]
    for strategy, final_count in sorted(final_population.items(), key=lambda x: x[1], reverse=True):
        initial_count = initial_counts.get(strategy, 0)
        change = final_count - initial_count
        change_str = f"(+{change})" if change > 0 else f"({change})" if change < 0 else "(no change)"
        print(f"{strategy}: {final_count} {change_str}")
    
    # Print examples of AI reasoning for selected rounds
    if HAS_API_AGENTS:
        print_ai_reasoning_examples(all_phase_results)
    
    # Print overall strategy performance
    print("\nOverall Strategy Performance (average across all phases):")
    print("-------------------------------------------------------")
    
    # Calculate average performance across all phases
    strategy_total_scores = defaultdict(float)
    strategy_total_match_scores = defaultdict(float)
    strategy_count = defaultdict(int)
    
    for phase_result in all_phase_results:
        for strategy, stats in phase_result['strategy_stats'].items():
            strategy_total_scores[strategy] += stats['avg_score_per_move']
            strategy_total_match_scores[strategy] += stats['avg_score_per_match']
            strategy_count[strategy] += 1
    
    # Print average scores
    print("Strategy    | Score/Move | Score/Match | Phases")
    print("------------|------------|-------------|-------")
    for strategy in sorted(strategy_total_scores.keys()):
        avg_score_per_move = strategy_total_scores[strategy] / strategy_count[strategy]
        avg_score_per_match = strategy_total_match_scores[strategy] / strategy_count[strategy]
        print(f"{strategy:12}|   {avg_score_per_move:.3f}    |   {avg_score_per_match:.3f}    | {strategy_count[strategy]}")

if __name__ == "__main__":
    main() 