import random
import pickle
import os


class QLearningAgent:

    def __init__(
        self,
        actions=6,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    ):

        self.actions = actions

        self.alpha = alpha

        self.gamma = gamma

        self.epsilon = epsilon

        self.epsilon_decay = epsilon_decay

        self.epsilon_min = epsilon_min

        self.q_table = {}

    ####################################################
    # Get Q values for a state
    ####################################################

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = [0.0] * self.actions

        return self.q_table[state]

    ####################################################
    # Choose Action
    ####################################################

    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(0, self.actions - 1)

        q_values = self.get_q_values(state)

        return q_values.index(max(q_values))

    ####################################################
    # Update Q Table
    ####################################################

    def update(self,
               state,
               action,
               reward,
               next_state):

        current_q = self.get_q_values(state)[action]

        max_next_q = max(
            self.get_q_values(next_state)
        )

        new_q = current_q + self.alpha * (

            reward +

            self.gamma * max_next_q -

            current_q

        )

        self.q_table[state][action] = new_q

    ####################################################
    # Reduce Exploration
    ####################################################

    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

    ####################################################
    # Save
    ####################################################

    def save(self,
             filename="q_table.pkl"):

        with open(filename, "wb") as f:

            pickle.dump(
                self.q_table,
                f
            )

    ####################################################
    # Load
    ####################################################

    def load(
            self,
            filename="q_table.pkl"
    ):

        if os.path.exists(filename):

            with open(filename, "rb") as f:

                self.q_table = pickle.load(f)

# --- ADAPTER WRAPPER FOR MEMBER 3 INTEGRATION ---

from traffic_state import TrafficState
from reward_function import RewardFunction

class RLAgent:
    """
    Adapter class to bridge Member 2's rogue implementation with the
    expected Week 3 integration contract.
    """
    def __init__(self):
        self.agent = QLearningAgent(actions=6, alpha=0.1, gamma=0.95)
        self.state_converter = TrafficState()
        self.reward_calculator = RewardFunction()
        self.last_state = None
        self.last_action = None

    def step(self, state_tuple):
        """
        API Contract:
        Input: state_tuple = (congestion_level, queue_length)
        Output: green_time_delta in seconds (int)
        """
        congestion_level, queue_length = state_tuple
        
        # Fabricate the traffic_stats dictionary Member 2's code expects
        traffic_stats = {
            "vehicle_count": queue_length,
            "average_speed": 10 if congestion_level == "Light" else (5 if congestion_level == "Moderate" else 1),
            "queue_length": queue_length,
            "average_queue_time": 5 * queue_length,
            "congestion": congestion_level
        }

        # Get Member 2's discrete state index
        current_state = self.state_converter.get_state(traffic_stats)

        # Update Q-table based on previous action's result
        if self.last_state is not None:
            reward = self.reward_calculator.calculate_reward(traffic_stats)
            self.agent.update(self.last_state, self.last_action, reward, current_state)

        # Choose new action
        action = self.agent.choose_action(current_state)

        # Save for next tick's update
        self.last_state = current_state
        self.last_action = action

        # Map Member 2's action space (0-5) to green_time_delta
        # Action 4 = Increase by 5s, Action 5 = Decrease by 5s, others = Hold
        if action == 4:
            return 5
        elif action == 5:
            return -5
        else:
            return 0

