import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# Define the Q-learning Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  # Reduced learning rate
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table
        self.target_q_table = np.copy(self.q_table)  # Initialize target Q-table
        self.update_target_frequency = 10  # Frequency of target network updates
        self.time_steps = 0  # Time steps counter

    # Update the Q-table
    def update_q_table(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.target_q_table[next_state])  # Greedy next action from target Q-network
        target = reward + self.gamma * self.target_q_table[next_state][best_next_action] * (1 - done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    # Choose action using epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration: random action
        return np.argmax(self.q_table[state])  # Exploitation: choose best action based on Q-table

    # Store the state-action-reward-transition for future learning
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Replay the agent's memory (Not needed for Q-learning, but can be used for experience replay)
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.update_q_table(state, action, reward, next_state, done)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Update the target network periodically
    def update_target_network(self):
        self.target_q_table = np.copy(self.q_table)  # Copy Q-table to target Q-table

# Function to calculate the carbon footprint based on power consumption
def calculate_carbon_footprint(power_kwh, co2_factor=475):
    return power_kwh * co2_factor

# Train the agent for a number of episodes
def train_agent(episodes=100):
    state_size = 2  # Example state size (e.g., energy consumption, temperature)
    action_size = 3  # Example action size (e.g., low, medium, high power usage)
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    losses = []

    for e in range(episodes):
        state = np.random.randint(0, state_size)  # Start state (example)
        total_power_consumed = 0  # For tracking power consumption over episodes
        for time in range(500):  # Arbitrary time steps
            action = agent.act(state)
            next_state = np.random.randint(0, state_size)  # Example next state

            # Simulate power consumption (in kWh) for the action taken
            power_consumed = np.random.uniform(0.01, 0.1)  # Random power consumption in kWh
            total_power_consumed += power_consumed

            # Calculate carbon footprint
            carbon_footprint = calculate_carbon_footprint(power_consumed)

            # **Modified Reward Function**
            # Encourage low power usage and low carbon footprint
            reward = 1.0 / (1 + power_consumed + 0.01 * carbon_footprint)  # Inverse penalty
            reward = max(0, reward)  # Ensure reward is non-negative

            done = time == 499  # Example condition to end the episode
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # End of episode
            if done:
                loss = np.mean(agent.q_table[state] - reward)
                losses.append(loss)

                # Log the results
                print(f"Episode {e}: Loss: {loss:.4f}")
                print(f"Total Power Consumption: {total_power_consumed:.4f} kWh")
                print(f"Estimated Carbon Footprint: {carbon_footprint:.2f} gCO2")
                print(f"Final Reward: {reward:.4f}")
                break

        # Replay the agent's memory
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # Update the target network periodically
        agent.time_steps += 1
        if agent.time_steps % agent.update_target_frequency == 0:
            agent.update_target_network()

    # Plotting loss over episodes
    plt.plot(range(episodes), losses)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Q-Learning Loss over Episodes')
    plt.show()

if __name__ == "__main__":
    train_agent(episodes=100)
