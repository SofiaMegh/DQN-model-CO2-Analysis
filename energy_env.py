import gym
from gym import spaces
import numpy as np
import psutil

class EnergyEnv(gym.Env):
    def __init__(self):
        super(EnergyEnv, self).__init__()
        
        # Define action and observation space
        # Actions represent different possible decisions related to energy management
        self.action_space = spaces.Discrete(3)  # Actions: 0 - No Action, 1 - Optimize CPU, 2 - Optimize Memory
        
        # Observations are CPU usage, memory usage, and battery status
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)
    def _get_system_state(self):
        # Fetch real-time data from system using psutil
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery else 100
        
        # Return state as a NumPy array
        return np.array([cpu_usage, memory_info, battery_percent])

    def step(self, action):
        # Get current system state
        state = self._get_system_state()
        
        # Define reward based on action taken (simplified for demo)
        reward = -state[0]  # Negative of CPU usage, goal is to minimize CPU usage
        
        # Action logic (adjust the reward based on the action)
        if action == 1:
            reward += 10  # Optimization on CPU
        elif action == 2:
            reward += 5   # Optimization on memory
        
        # Check if the episode is done (you can define custom termination conditions)
        done = False
        
        # Return new state, reward, done flag, and extra info
        return state, reward, done, {}

    def reset(self):
        # Reset the environment (not much to reset in this case)
        return self._get_system_state()

    def render(self, mode="human"):
        # Optional: Visualization logic (can be enhanced later)
        pass

if __name__ == "__main__":
    env = EnergyEnv()
    state = env.reset()
    print("Initial State:", state)
