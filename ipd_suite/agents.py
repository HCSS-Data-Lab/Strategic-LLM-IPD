"""
Agent implementations for IPD experiments
Including classical strategies, behavioral variants, adaptive learning, and LLM agents
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import openai
import anthropic
import google.generativeai as genai
from mistralai import Mistral
import time
import json
import datetime


class Agent(ABC):
    """Base class for all IPD agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.history = []
        self.opponent_history = []
        self.last_reasoning = None
        
    @abstractmethod
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        """Return 'C' for cooperate or 'D' for defect"""
        pass
    
    def reset(self):
        """Reset agent state for new match"""
        self.history = []
        self.opponent_history = []
        self.last_reasoning = None


# Classical Strategies
class AlwaysCooperate(Agent):
    """Always cooperates"""
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        return 'C'


class AlwaysDefect(Agent):
    """Always defects"""
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        return 'D'


class Random(Agent):
    """Randomly cooperates or defects"""
    def __init__(self, name: str, p_cooperate: float = 0.5):
        super().__init__(name)
        self.p_cooperate = p_cooperate
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        return 'C' if random.random() < self.p_cooperate else 'D'


class TitForTat(Agent):
    """Cooperates first, then copies opponent's last move"""
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if not opponent_history:
            return 'C'
        return opponent_history[-1]


class GrimTrigger(Agent):
    """Cooperates until opponent defects, then always defects"""
    def __init__(self, name: str):
        super().__init__(name)
        self.triggered = False
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if 'D' in opponent_history:
            self.triggered = True
        return 'D' if self.triggered else 'C'
    
    def reset(self):
        super().reset()
        self.triggered = False


class WinStayLoseShift(Agent):
    """If last outcome was good (CC or DC), repeat; otherwise switch"""
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if not own_history:
            return 'C'
        
        last_own = own_history[-1]
        last_opp = opponent_history[-1]
        
        # Good outcomes: mutual cooperation or exploiting opponent
        if (last_own == 'C' and last_opp == 'C') or (last_own == 'D' and last_opp == 'C'):
            return last_own
        else:
            return 'D' if last_own == 'C' else 'C'


class Pavlov(Agent):
    """Win-Stay-Lose-Shift variant"""
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        return WinStayLoseShift.make_move(self, own_history, opponent_history)


class GenerousTitForTat(Agent):
    """Tit-for-Tat with forgiveness probability"""
    def __init__(self, name: str, forgiveness_prob: float = 0.1):
        super().__init__(name)
        self.forgiveness_prob = forgiveness_prob
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if not opponent_history:
            self.last_reasoning = "Starting with cooperation."
            return "C"
        
        if opponent_history[-1] == "D":
            if random.random() < self.forgiveness_prob:
                self.last_reasoning = "Opponent defected, but I'm being generous and forgiving."
                return "C"
            else:
                self.last_reasoning = "Opponent defected, so I'm reciprocating with defection."
                return "D"
        else:
            self.last_reasoning = "Opponent cooperated, so I'm cooperating too."
            return "C"


class SuspiciousTitForTat(Agent):
    """Starts with defection, then plays Tit-for-Tat"""
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if not opponent_history:
            self.last_reasoning = "Starting with defection (suspicious)."
            return "D"
        
        self.last_reasoning = f"Copying opponent's last move ({opponent_history[-1]})."
        return opponent_history[-1]


class Gradual(Agent):
    """Escalating retaliation strategy"""
    def __init__(self, name: str):
        super().__init__(name)
        self.defect_count = 0
        self.retaliation_count = 0
        self.retaliating = False
        
    def reset(self):
        super().reset()
        self.defect_count = 0
        self.retaliation_count = 0
        self.retaliating = False
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if not opponent_history:
            self.last_reasoning = "Starting with cooperation."
            return "C"
        
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


class Prober(Agent):
    """Tests opponent with C-D-C sequence, then adapts"""
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if len(own_history) == 0:
            self.last_reasoning = "Starting with cooperation."
            return "C"
        elif len(own_history) == 1:
            self.last_reasoning = "Testing opponent with defection on second move."
            return "D"
        elif len(own_history) == 2:
            self.last_reasoning = "Back to cooperation for the third move."
            return "C"
        else:
            # If opponent didn't retaliate after our defection, exploit them
            if opponent_history[1] == "C" and opponent_history[2] == "C":
                self.last_reasoning = "Opponent didn't retaliate to my test, so I'll exploit with defection."
                return "D"
            else:
                # Otherwise play TFT
                self.last_reasoning = f"Playing Tit-for-Tat, copying opponent's last move ({opponent_history[-1]})."
                return opponent_history[-1]


class Alternator(Agent):
    """Simple alternating pattern (C-D-C-D...)"""
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if len(own_history) % 2 == 0:
            self.last_reasoning = "Alternating pattern: cooperate on even-numbered rounds."
            return "C"
        else:
            self.last_reasoning = "Alternating pattern: defect on odd-numbered rounds."
            return "D"


class Bayesian(Agent):
    """Bayesian inference to identify opponent strategy"""
    def __init__(self, name: str):
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
            'always_cooperate': 0.1
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
                if len(self.opponent_history) >= 3 and self.opponent_history[1] == "C" and self.opponent_history[2] == "C":
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
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if not opponent_history:
            self.last_reasoning = "Starting with cooperation to gather data."
            return "C"
            
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


# Behavioral Strategies
class ForgivingGrimTrigger(Agent):
    """Grim Trigger that forgives after N mutual defections (optimized for shorter games)"""
    def __init__(self, name: str, forgiveness_threshold: int = 2):
        super().__init__(name)
        self.triggered = False
        self.forgiveness_threshold = forgiveness_threshold
        self.mutual_defection_count = 0
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        if not self.triggered and 'D' in opponent_history:
            self.triggered = True
            
        if self.triggered:
            # Count mutual defections from previous round
            if (own_history and opponent_history and 
                own_history[-1] == 'D' and opponent_history[-1] == 'D'):
                self.mutual_defection_count += 1
                
            # Check if we should forgive (before reaching threshold)
            if self.mutual_defection_count >= self.forgiveness_threshold:
                self.triggered = False
                self.mutual_defection_count = 0
                return 'C'
                    
            return 'D'
        return 'C'
    
    def reset(self):
        super().reset()
        self.triggered = False
        self.mutual_defection_count = 0


class Detective(Agent):
    """Tests opponent with C-D-C, then adapts based on response (optimized for shorter games)"""
    def __init__(self, name: str):
        super().__init__(name)
        self.test_sequence = ['C', 'D', 'C']
        self.opponent_type = None  # 'sucker', 'retaliator', 'random'
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        round_num = len(own_history)
        
        # Testing phase
        if round_num < len(self.test_sequence):
            return self.test_sequence[round_num]
        
        # Analyze opponent after testing
        if round_num == len(self.test_sequence):
            self._analyze_opponent(opponent_history)
            
        # Adapt strategy based on opponent type
        if self.opponent_type == 'sucker':
            return 'D'  # Exploit cooperators
        elif self.opponent_type == 'retaliator':
            # TFT against retaliators
            return opponent_history[-1] if opponent_history else 'C'
        else:
            # Mixed strategy against unpredictable opponents
            return 'C' if random.random() < 0.6 else 'D'
    
    def _analyze_opponent(self, opponent_history: List[str]):
        """Classify opponent based on test response (optimized for shorter games)"""
        if len(opponent_history) < 3:
            self.opponent_type = 'random'
            return
            
        # Check if opponent retaliated to our defection (position 1 -> response at position 2)
        retaliated = opponent_history[2] == 'D'
        
        # Check cooperation rate
        coop_rate = opponent_history.count('C') / len(opponent_history)
        
        if not retaliated and coop_rate >= 0.67:  # 2/3 cooperation rate
            self.opponent_type = 'sucker'
        elif retaliated:
            self.opponent_type = 'retaliator'
        else:
            self.opponent_type = 'random'
    
    def reset(self):
        super().reset()
        self.opponent_type = None


class SoftGrudger(Agent):
    """Retaliates with graduated punishment: D-D-C (optimized for shorter games)"""
    def __init__(self, name: str):
        super().__init__(name)
        self.punishment_sequence = ['D', 'D', 'C']
        self.punishment_index = 0
        self.punishing = False
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        # Start punishment if opponent defected and we're not already punishing
        if opponent_history and opponent_history[-1] == 'D' and not self.punishing:
            self.punishing = True
            self.punishment_index = 0
            
        # Execute punishment sequence
        if self.punishing:
            if self.punishment_index < len(self.punishment_sequence):
                move = self.punishment_sequence[self.punishment_index]
                self.punishment_index += 1
                return move
            else:
                # Punishment complete, return to cooperation
                self.punishing = False
                self.punishment_index = 0
                
        return 'C'
    
    def reset(self):
        super().reset()
        self.punishment_index = 0
        self.punishing = False


# Adaptive Learning Strategies
class QLearningAgent(Agent):
    """Q-learning agent with state-action values (default settings optimized for length 4 games)"""
    def __init__(self, name: str, alpha: float = 0.7, gamma: float = 0.5, epsilon: float = 0.3):
        super().__init__(name)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        
        # Use optimistic initialization to encourage exploration
        # This helps the agent try defection early to learn it's better against cooperators
        self._optimistic_init_value = 10.0
    
    @classmethod
    def for_game_length(cls, name: str, expected_game_length: int):
        """Factory method to create QLearningAgent with optimal parameters for game length"""
        if expected_game_length <= 1:
            # Single-shot games: immediate learning, no future, no exploration
            return cls(name, alpha=1.0, gamma=0.0, epsilon=0.0)
        elif expected_game_length <= 4:
            # Short games: fast learning, short horizon, moderate exploration
            return cls(name, alpha=0.7, gamma=0.5, epsilon=0.3)
        elif expected_game_length <= 10:
            # Moderate games: standard learning, moderate horizon, conservative exploration
            return cls(name, alpha=0.3, gamma=0.7, epsilon=0.2)
        else:
            # Long games: gradual learning, long horizon, minimal exploration
            return cls(name, alpha=0.1, gamma=0.9, epsilon=0.1)
        
    def _get_state(self, own_history: List[str], opponent_history: List[str]) -> Tuple:
        """Define state based on last moves"""
        if not own_history:
            return ('START', 'START')
        return (own_history[-1], opponent_history[-1])
    
    def _get_q_value(self, state: Tuple, action: str) -> float:
        """Get Q-value for state-action pair with optimistic initialization"""
        return self.q_table.get((state, action), self._optimistic_init_value)
    
    def _update_q_value(self, state: Tuple, action: str, reward: float, next_state: Tuple):
        """Update Q-value using Q-learning formula"""
        current_q = self._get_q_value(state, action)
        max_next_q = max(self._get_q_value(next_state, 'C'), 
                        self._get_q_value(next_state, 'D'))
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        state = self._get_state(own_history, opponent_history)
        
        # Update Q-value from last round
        if self.last_state is not None and own_history:
            reward = self._calculate_reward(own_history[-1], opponent_history[-1])
            self._update_q_value(self.last_state, self.last_action, reward, state)
        
        # Epsilon-greedy action selection with adaptive exploration
        # Reduce exploration over time to converge to optimal policy
        current_epsilon = self.epsilon * (0.99 ** len(own_history))
        
        if random.random() < current_epsilon:
            action = random.choice(['C', 'D'])
        else:
            q_c = self._get_q_value(state, 'C')
            q_d = self._get_q_value(state, 'D')
            action = 'C' if q_c > q_d else 'D'  # Changed >= to > for slight bias toward defection
            
        self.last_state = state
        self.last_action = action
        return action
    
    def _calculate_reward(self, own_move: str, opp_move: str) -> float:
        """Calculate reward based on payoff matrix"""
        payoffs = {
            ('C', 'C'): 3,
            ('C', 'D'): 0,
            ('D', 'C'): 5,
            ('D', 'D'): 1
        }
        return payoffs[(own_move, opp_move)]
    
    def reset(self):
        super().reset()
        self.last_state = None
        self.last_action = None


class ThompsonSampling(Agent):
    """Thompson sampling for exploration/exploitation (default settings optimized for length 5 games)"""
    def __init__(self, name: str, base_learning_rate: float = 0.2, learning_rate_scale: float = 1.2):
        super().__init__(name)
        # Beta distribution parameters for each action
        self.alpha_c = 1  # Successes for cooperation
        self.beta_c = 1   # Failures for cooperation
        self.alpha_d = 1  # Successes for defection
        self.beta_d = 1   # Failures for defection
        
        # Learning rate parameters for game length adaptation
        self.base_learning_rate = base_learning_rate
        self.learning_rate_scale = learning_rate_scale
    
    @classmethod
    def for_game_length(cls, name: str, expected_game_length: float):
        """Factory method to create ThompsonSampling agent with optimal parameters for game length"""
        if expected_game_length <= 1.5:
            # Very short games (mean ~1.3): fast learning, high sensitivity
            return cls(name, base_learning_rate=0.3, learning_rate_scale=1.5)
        elif expected_game_length <= 5:
            # Short games (mean ~4): moderate learning, balanced sensitivity
            return cls(name, base_learning_rate=0.2, learning_rate_scale=1.2)
        elif expected_game_length <= 12:
            # Moderate games (mean ~10): standard learning, normal sensitivity
            return cls(name, base_learning_rate=0.1, learning_rate_scale=0.9)
        else:
            # Long games: gradual learning, conservative sensitivity
            return cls(name, base_learning_rate=0.05, learning_rate_scale=0.7)
        
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        # Sample from beta distributions
        theta_c = np.random.beta(self.alpha_c, self.beta_c)
        theta_d = np.random.beta(self.alpha_d, self.beta_d)
        
        # Choose action with higher sampled value
        return 'C' if theta_c > theta_d else 'D'
    
    def update(self, own_move: str, opponent_move: str, payoff: float):
        """Update beta parameters based on outcome with adaptive learning rate"""
        # Use adaptive learning rate based on payoff magnitude and game length
        # Scale payoff to learning rate: higher payoffs give stronger updates
        # Payoff range: 0-5, so we normalize and scale appropriately
        
        # Calculate adaptive learning rate based on payoff and game length parameters
        # Base learning rate adjusted by payoff magnitude
        learning_rate = self.base_learning_rate + (payoff / 5.0) * self.learning_rate_scale
        
        # All payoffs are treated as "successes" but with different magnitudes
        # This allows the agent to learn that higher payoffs are better
        if own_move == 'C':
            self.alpha_c += learning_rate
            # Small penalty for low payoffs to maintain exploration
            if payoff < 2.5:  # Below average
                penalty = (2.5 - payoff) * self.base_learning_rate
                self.beta_c += penalty
        else:
            self.alpha_d += learning_rate
            # Small penalty for low payoffs
            if payoff < 2.5:  # Below average
                penalty = (2.5 - payoff) * self.base_learning_rate
                self.beta_d += penalty


class GradientMetaLearner(Agent):
    """Policy gradient approach with feature extraction (default settings optimized for length 5 games)"""
    def __init__(self, name: str, learning_rate: float = 0.05):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.weights = np.zeros(5)  # Feature weights
        self.feature_history = []
        self.action_history = []
        self.reward_history = []
        
    def _extract_features(self, own_history: List[str], opponent_history: List[str]) -> np.ndarray:
        """Extract features from game state"""
        features = np.zeros(5)
        
        if not opponent_history:
            features[0] = 1  # First round
            return features
            
        # Feature 1: Opponent cooperation rate
        features[1] = opponent_history.count('C') / len(opponent_history)
        
        # Feature 2: Recent opponent cooperation (last 3 moves)
        recent = opponent_history[-3:]
        features[2] = recent.count('C') / len(recent) if recent else 0
        
        # Feature 3: Mutual cooperation rate
        mutual_coop = sum(1 for i in range(len(own_history)) 
                         if own_history[i] == 'C' and opponent_history[i] == 'C')
        features[3] = mutual_coop / len(own_history) if own_history else 0
        
        # Feature 4: Rounds played (normalized for ~5 round games)
        features[4] = min(len(own_history) / 5, 1.0)
        
        return features
    
    def _policy(self, features: np.ndarray) -> float:
        """Compute probability of cooperation"""
        logit = np.dot(self.weights, features)
        return 1 / (1 + np.exp(-logit))  # Sigmoid
    
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        features = self._extract_features(own_history, opponent_history)
        p_cooperate = self._policy(features)
        
        # Sample action from policy
        action = 'C' if random.random() < p_cooperate else 'D'
        
        # Store for learning
        self.feature_history.append(features)
        self.action_history.append(1 if action == 'C' else 0)
        
        return action
    
    def update_policy(self):
        """Update weights using policy gradient"""
        if len(self.reward_history) < 2:
            return
            
        # Compute advantages (rewards - baseline)
        rewards = np.array(self.reward_history)
        baseline = np.mean(rewards)
        advantages = rewards - baseline
        
        # Policy gradient update
        for i in range(len(self.feature_history)):
            features = self.feature_history[i]
            action = self.action_history[i]
            advantage = advantages[i] if i < len(advantages) else 0
            
            # Gradient of log probability
            p = self._policy(features)
            grad = features * (action - p)
            
            # Update weights
            self.weights += self.learning_rate * advantage * grad
    
    def reset(self):
        super().reset()
        self.feature_history = []
        self.action_history = []
        self.reward_history = []


# LLM Agents with Caching
import hashlib
from typing import Dict, Tuple, Optional
from functools import lru_cache

class LLMAgent(Agent):
    """Base class for LLM agents with comprehensive caching support"""
    
    def __init__(self, name: str, model: str, temperature: float = 0.7, termination_prob: float = 0.1, 
                 enable_caching: bool = True, cache_size: int = 1000, match_history: List[Dict] = None):
        super().__init__(name)
        self.model = model
        self.temperature = temperature
        self.termination_prob = termination_prob
        self.enable_caching = enable_caching
        
        # Token tracking
        self.api_calls = 0
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        
        # Caching metrics
        self.cached_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.tokens_saved_from_cache = 0
        
        # Response cache for identical game states
        self.response_cache: Dict[str, str] = {}
        self.cache_size = cache_size
        
        # Match history from previous phases
        self.match_history = match_history if match_history is not None else []
        
        # Current opponent cross-reference information
        self.current_opponent_info = ""
        
        # Create static template components
        self._create_static_templates()
    
    def set_current_opponent_info(self, opponent_info: str):
        """Set cross-reference information about the current opponent"""
        self.current_opponent_info = opponent_info
    
    def _create_static_templates(self):
        """Create reusable static template components"""
        self.static_game_rules = """You are playing the Iterated Prisoner's Dilemma game.

Game Rules:
- Two players choose to Cooperate (C) or Defect (D)
- Payoffs: Both C = 3 points each, Both D = 1 point each, You C/They D = 0/5 points, You D/They C = 5/0 points
- The game continues for multiple rounds with {termination_prob}% chance of ending after each round
- Goal: Maximize your total score"""

        self.static_instructions = """Provide your reasoning about the game state, opponent's strategy, and your decision.
End your response with your move on a new line: either 'C' or 'D'."""
        
        # Pre-format the static content with termination probability
        self.formatted_static_rules = self.static_game_rules.format(
            termination_prob=self.termination_prob*100
        )
    
    def _format_history(self, own_history: List[str], opponent_history: List[str]) -> str:
        """Format move history efficiently"""
        if not own_history:
            return "[]"
        history_pairs = list(zip(own_history, opponent_history))
        return '[' + ', '.join([f"({own},{opp})" for own, opp in history_pairs]) + ']'
    
    def _create_cache_key(self, own_history: List[str], opponent_history: List[str]) -> str:
        """Create cache key for identical game states including match history"""
        history_str = f"{','.join(own_history)}|{','.join(opponent_history)}"
        termination_str = f"{self.termination_prob}"
        
        # Include match history in cache key
        match_history_str = ""
        if self.match_history:
            # Create a compact representation of match history for cache key
            match_summaries = []
            for match in self.match_history:
                opponent = match.get('opponent', 'Unknown')
                rounds = match.get('rounds', [])
                round_moves = ''.join([f"{r.get('your_move','?')}{r.get('opponent_move','?')}" for r in rounds])
                match_summaries.append(f"{opponent}:{round_moves}")
            match_history_str = '|'.join(match_summaries)
        
        cache_input = f"{history_str}|{termination_str}|{self.temperature}|{match_history_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()[:16]
    
    def _format_match_history(self) -> str:
        """Format complete match history from previous phases"""
        if not self.match_history:
            return ""
        
        formatted_matches = []
        for i, match in enumerate(self.match_history, 1):
            opponent = match.get('opponent', 'Unknown')
            rounds = match.get('rounds', [])
            
            if not rounds:
                continue
                
            # Format rounds as (Your_Move, Opponent_Move)
            round_pairs = []
            for round_data in rounds:
                your_move = round_data.get('your_move', '?')
                opp_move = round_data.get('opponent_move', '?')
                round_pairs.append(f"({your_move},{opp_move})")
            
            rounds_str = '[' + ', '.join(round_pairs) + ']'
            formatted_matches.append(f"Match {i} vs {opponent}: {rounds_str}")
        
        return "\n".join(formatted_matches)
    
    def _create_prompt_components(self, own_history: List[str], opponent_history: List[str]) -> Tuple[str, str]:
        """Create static and dynamic components for caching optimization"""
        # Static component (cacheable) - includes match history from previous phases
        match_history_text = self._format_match_history()
        if match_history_text:
            static_content = self.formatted_static_rules + f"\n\nYour complete match history from previous phases:\n{match_history_text}"
        else:
            static_content = self.formatted_static_rules
        
        # Dynamic component (changes with current history)
        history_str = self._format_history(own_history, opponent_history)
        
        # Add current opponent info if available
        opponent_info_text = ""
        if self.current_opponent_info:
            opponent_info_text = f"\n\n{self.current_opponent_info}"
        
        dynamic_content = f"{opponent_info_text}\n\nCurrent match - History of moves (You, Opponent): {history_str}\n\n{self.static_instructions}"
        
        return static_content, dynamic_content
    
    def _create_prompt(self, own_history: List[str], opponent_history: List[str]) -> str:
        """Create complete prompt by combining static and dynamic components"""
        static_content, dynamic_content = self._create_prompt_components(own_history, opponent_history)
        return static_content + dynamic_content
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Retrieve cached response if available"""
        if not self.enable_caching:
            return None
        return self.response_cache.get(cache_key)
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache response with LRU eviction"""
        if not self.enable_caching:
            return
        
        # Simple LRU: remove oldest if cache is full
        if len(self.response_cache) >= self.cache_size:
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = response
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (rough approximation)"""
        return len(text.split()) * 1.3
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive caching statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        # Estimate cost savings (assuming $0.01 per 1K input tokens)
        estimated_cost_saved = (self.tokens_saved_from_cache / 1000) * 0.01
        
        return {
            'cache_enabled': self.enable_caching,
            'cache_size': len(self.response_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'tokens_saved': self.tokens_saved_from_cache,
            'estimated_cost_saved': estimated_cost_saved,
            'api_calls': self.api_calls,
            'cached_calls': self.cached_calls
        }
    
    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Call the specific LLM API"""
        pass
    
    @abstractmethod
    def _call_api_with_caching(self, static_content: str, dynamic_content: str) -> str:
        """Call API with provider-specific caching optimizations"""
        pass
    
    def make_move(self, own_history: List[str], opponent_history: List[str]) -> str:
        # Create cache key for response caching
        cache_key = self._create_cache_key(own_history, opponent_history)
        
        # Check for cached response first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            self.cache_hits += 1
            self.cached_calls += 1
            self.last_reasoning = cached_response
            
            # Estimate tokens saved
            prompt = self._create_prompt(own_history, opponent_history)
            estimated_input_tokens = self.estimate_tokens(prompt)
            self.tokens_saved_from_cache += estimated_input_tokens
            
            return self._extract_move_from_response(cached_response)
        
        # Cache miss - make API call
        self.cache_misses += 1
        
        try:
            # Get static and dynamic components for optimized caching
            static_content, dynamic_content = self._create_prompt_components(own_history, opponent_history)
            
            # Use provider-specific caching if available, fallback to standard
            try:
                response = self._call_api_with_caching(static_content, dynamic_content)
            except (NotImplementedError, AttributeError):
                # Fallback to standard API call
                prompt = static_content + dynamic_content
                response = self._call_api(prompt)
            
            self.last_reasoning = response
            
            # Cache the response
            self._cache_response(cache_key, response)
            
            return self._extract_move_from_response(response)
                
        except Exception as e:
            print(f"Error in {self.name}: {e}")
            return 'C'  # Default to cooperation on error
    
    def _extract_move_from_response(self, response: str) -> str:
        """Extract move from LLM response"""
        lines = response.strip().split('\n')
        last_line = lines[-1].strip().upper()
        
        if last_line in ['C', 'D']:
            return last_line
        elif 'C' in last_line and 'D' not in last_line:
            return 'C'
        elif 'D' in last_line and 'C' not in last_line:
            return 'D'
        else:
            # Default to cooperation if unclear
            return 'C'


class GPT4Agent(LLMAgent):
    """OpenAI GPT-4 agent with prompt caching"""
    
    def __init__(self, name: str, api_key: str, model: str = "gpt-5-mini", 
                 temperature: float = 1, termination_prob: float = 0.1, match_history: List[Dict] = None):
        super().__init__(name, model, temperature, termination_prob, match_history=match_history)
        self.client = openai.OpenAI(api_key=api_key)
        
    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert game theory player."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_completion_tokens=2000
        )
        
        self.api_calls += 1
        self.total_tokens += response.usage.total_tokens
        self.input_tokens += response.usage.prompt_tokens
        self.output_tokens += response.usage.completion_tokens
        
        return response.choices[0].message.content
    
    def _call_api_with_caching(self, static_content: str, dynamic_content: str) -> str:
        """OpenAI-specific API call with prompt caching"""
        # Create cached system message for static content
        system_message = {
            "role": "system", 
            "content": f"You are an expert game theory player.\n\n{static_content}"
        }
        
        user_message = {
            "role": "user",
            "content": dynamic_content
        }
        
        # For OpenAI prompt caching, we structure messages to maximize cache hits
        # The system message containing static content is more likely to be cached
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[system_message, user_message],
            temperature=self.temperature,
            max_completion_tokens=2000,
            # Add cache control headers if supported by the model
            extra_headers={
                "OpenAI-Cache-Control": "ephemeral"
            } if "gpt-4" in self.model.lower() or "gpt-5" in self.model.lower() else {}
        )
        
        self.api_calls += 1
        self.total_tokens += response.usage.total_tokens
        self.input_tokens += response.usage.prompt_tokens
        self.output_tokens += response.usage.completion_tokens
        
        return response.choices[0].message.content


class ClaudeAgent(LLMAgent):
    """Anthropic Claude agent with prompt caching beta"""
    
    def __init__(self, name: str, api_key: str, model: str = "claude-3-5-sonnet-20241022",
                 temperature: float = 0.7, termination_prob: float = 0.1, match_history: List[Dict] = None):
        super().__init__(name, model, temperature, termination_prob, match_history=match_history)
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def _call_api(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=500
        )
        
        self.api_calls += 1
        
        # Claude API returns usage information
        if hasattr(response, 'usage'):
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += input_tokens + output_tokens
        else:
            # Fallback estimation if usage not available
            estimated_input = len(prompt.split()) * 1.3
            estimated_output = 200
            self.input_tokens += estimated_input
            self.output_tokens += estimated_output
            self.total_tokens += estimated_input + estimated_output
        
        return response.content[0].text
    
    def _call_api_with_caching(self, static_content: str, dynamic_content: str) -> str:
        """Claude-specific API call with prompt caching beta"""
        # Claude prompt caching beta - mark static content for caching
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": static_content,
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text", 
                        "text": f"\n\n{dynamic_content}"
                    }
                ]
            }
        ]
        
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500,
            extra_headers={
                "anthropic-beta": "prompt-caching-2024-07-31"
            }
        )
        
        self.api_calls += 1
        
        # Claude API returns usage information including cache metrics
        if hasattr(response, 'usage'):
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += input_tokens + output_tokens
            
            # Track cached input tokens if available
            if hasattr(response.usage, 'cache_creation_input_tokens'):
                # This indicates new cache creation
                pass
            if hasattr(response.usage, 'cache_read_input_tokens'):
                # This indicates cache hit
                cache_read_tokens = response.usage.cache_read_input_tokens
                self.tokens_saved_from_cache += cache_read_tokens
        else:
            # Fallback estimation
            estimated_input = len((static_content + dynamic_content).split()) * 1.3
            estimated_output = 200
            self.input_tokens += estimated_input
            self.output_tokens += estimated_output
            self.total_tokens += estimated_input + estimated_output
        
        return response.content[0].text


class MistralAgent(LLMAgent):
    """Mistral AI agent with basic caching optimization"""
    
    def __init__(self, name: str, api_key: str, model: str = "mistral-medium-2508",
                 temperature: float = 0.7, termination_prob: float = 0.1, match_history: List[Dict] = None):
        super().__init__(name, model, temperature, termination_prob, match_history=match_history)
        self.client = Mistral(api_key=api_key)
        
    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=500
        )
        
        self.api_calls += 1
        if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens'):
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
            self.total_tokens += response.usage.total_tokens
        else:
            # Fallback estimation
            estimated_input = len(prompt.split()) * 1.3
            estimated_output = 200
            self.input_tokens += estimated_input
            self.output_tokens += estimated_output
            self.total_tokens += estimated_input + estimated_output
        
        return response.choices[0].message.content
    
    def _call_api_with_caching(self, static_content: str, dynamic_content: str) -> str:
        """Mistral-specific API call with optimized prompt structure for caching"""
        # Mistral doesn't have explicit prompt caching yet, but we can optimize the structure
        # by putting static content in system role and dynamic content in user role
        messages = [
            {"role": "system", "content": f"You are an expert game theory player.\n\n{static_content}"},
            {"role": "user", "content": dynamic_content}
        ]
        
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500
        )
        
        self.api_calls += 1
        if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens'):
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
            self.total_tokens += response.usage.total_tokens
        else:
            # Fallback estimation
            estimated_input = len((static_content + dynamic_content).split()) * 1.3
            estimated_output = 200
            self.input_tokens += estimated_input
            self.output_tokens += estimated_output
            self.total_tokens += estimated_input + estimated_output
        
        return response.choices[0].message.content


class GeminiAgent(LLMAgent):
    """Google Gemini agent with context caching"""
    
    def __init__(self, name: str, api_key: str, model: str = "gemini-2.0-flash",
                 temperature: float = 0.7, termination_prob: float = 0.1, match_history: List[Dict] = None):
        super().__init__(name, model, temperature, termination_prob, match_history=match_history)
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
        # Create cached model instance for static content
        self._cached_model_instance = None
        
    def _call_api(self, prompt: str) -> str:
        response = self.model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=5000,
            )
        )
        
        self.api_calls += 1
        
        # Gemini API returns usage information
        if hasattr(response, 'usage_metadata'):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += input_tokens + output_tokens
        else:
            # Fallback estimation
            estimated_input = len(prompt.split()) * 1.3
            estimated_output = 200
            self.input_tokens += estimated_input
            self.output_tokens += estimated_output
            self.total_tokens += estimated_input + estimated_output
        
        return response.text
    
    def _call_api_with_caching(self, static_content: str, dynamic_content: str) -> str:
        """Gemini-specific API call with context caching"""
        try:
            # Create a cached context with the static content if not already cached
            if self._cached_model_instance is None:
                # Create cached content for reuse
                static_text = f"You are an expert game theory player.\n\n{static_content}"
                
                # Use Gemini's context caching (if available)
                cached_content = genai.caching.CachedContent.create(
                    model=self.model,
                    display_name="ipd_static_context",
                    contents=[static_text],
                    ttl=datetime.timedelta(minutes=15),  # Cache for 15 minutes
                )
                
                self._cached_model_instance = genai.GenerativeModel.from_cached_content(cached_content)
            
            # Generate response using cached context
            response = self._cached_model_instance.generate_content(
                dynamic_content,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=5000,
                )
            )
            
        except (AttributeError, Exception):
            # Fallback to regular API call if context caching not available
            full_prompt = f"You are an expert game theory player.\n\n{static_content}\n\n{dynamic_content}"
            response = self.model_instance.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=5000,
                )
            )
        
        self.api_calls += 1
        
        # Gemini API returns usage information
        if hasattr(response, 'usage_metadata'):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += input_tokens + output_tokens
            
            # Track tokens saved from context caching
            if hasattr(response.usage_metadata, 'cached_content_token_count'):
                cached_tokens = response.usage_metadata.cached_content_token_count
                self.tokens_saved_from_cache += cached_tokens
        else:
            # Fallback estimation
            estimated_input = len((static_content + dynamic_content).split()) * 1.3
            estimated_output = 200
            self.input_tokens += estimated_input
            self.output_tokens += estimated_output
            self.total_tokens += estimated_input + estimated_output
        
        return response.text