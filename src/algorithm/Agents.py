import logging

import numpy as np

from src.LoadLoggingConfiguration import load_logging_config


class UCB1Agent:
    """
    Initializes a UCB1 agent with a given number of arms.
    :param no_arm: Number of arms in the bandit.
    """

    def __init__(self, no_arm):
        load_logging_config()

        # Get the logger for the 'staging' logger
        self.logger = logging.getLogger('staging')

        # Number of arms
        self.num_arms = no_arm

        # Array to store total rewards for each arm
        self.total_rewards = np.zeros(no_arm)

        # Array to store the number of pulls for each arm
        self.num_pulls = np.zeros(no_arm)

        # Timestep to keep track of the number of iterations
        self.timestep = 0

    def select_arm(self):
        """
        Selects an arm based on the UCB1 strategy.
        :return: The selected arm.
        """
        # Increment timestep
        self.timestep += 1

        # Calculate exploration bonus using UCB1 formula
        exploration_bonus = np.sqrt(2 * np.log(self.timestep) / (self.num_pulls + 1e-6))

        # Calculate UCB values for each arm
        ucb_values = self.total_rewards / (self.num_pulls + 1e-6) + exploration_bonus

        # Select arm with the highest UCB value
        selected_arm = np.argmax(ucb_values)

        # Log the selected arm
        self.logger.info(f"UCB1 Agent selected arm: {selected_arm}")

        return selected_arm

    def update(self, arm, reward):
        """
        Updates the agent's knowledge after pulling an arm and receiving a reward.
        :param arm: The arm that was pulled.
        :param reward: The received reward.
        """
        # Update total rewards and number of pulls for the selected arm
        self.total_rewards[arm] += reward
        self.num_pulls[arm] += 1

        # Log the update information
        self.logger.info(f"UCB1 Agent updated for arm {arm}: Total Rewards={self.total_rewards[arm]}, "
                         f"Num Pulls={self.num_pulls[arm]}")
class EpsilonGreedyAgent:
    def __init__(self, no_arm, eps):

        load_logging_config()

        # Get the logger for the 'staging' logger
        self.logger = logging.getLogger('staging')

        """
        Initializes an Epsilon-Greedy agent with a given number of arms and exploration rate.
        :param no_arm: Number of arms in the bandit.
        :param eps: Exploration rate (probability of exploration).
        """
        # Number of arms
        self.num_arms = no_arm

        # Exploration rate
        self.epsilon = eps

        # Array to store total rewards for each arm
        self.total_rewards = np.zeros(no_arm)

        # Array to store the number of pulls for each arm
        self.num_pulls = np.zeros(no_arm)

    def select_arm(self):
        """
        Selects an arm based on the Epsilon-Greedy strategy.
        :return: The selected arm.
        """
        # Explore with probability epsilon
        if np.random.rand() < self.epsilon:
            selected_arm = np.random.choice(self.num_arms)
            self.logger.info(f"Epsilon-Greedy Agent exploring: Selected arm randomly: {selected_arm}")
        else:
            # Exploit the arm with the highest average reward
            avg_rewards = self.total_rewards / (self.num_pulls + 1e-6)
            selected_arm = np.argmax(avg_rewards)
            self.logger.info(f"Epsilon-Greedy Agent exploiting: Selected arm based on highest average reward: {selected_arm}")
        return selected_arm

        self.logger.info(f"Epsilon-Greedy Agent - Average rewards: {avg_rewards}")

    def update(self, arm, reward):
        """
        Updates the agent's knowledge after pulling an arm and receiving a reward.
        :param arm: The arm that was pulled.
        :param reward: The received reward.
        """
        # Update total rewards and number of pulls for the selected arm
        self.total_rewards[arm] += reward
        self.num_pulls[arm] += 1

        # Log the update information
        self.logger.info(f"Epsilon-Greedy Agent updated for arm {arm}: Total Rewards={self.total_rewards[arm]}, "
                         f"Num Pulls={self.num_pulls[arm]}")
