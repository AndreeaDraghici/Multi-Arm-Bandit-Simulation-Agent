import numpy as np
import logging

class MultiArmedBandit:
    def __init__(self, no_arm):
        """
            Initializes a Multi-Armed Bandit with a given number of arms.

            :param no_arm: Number of arms in the bandit.
        """
        try:
            self.num_arms = no_arm
            # Mean reward for each arm sampled from a normal distribution
            self.true_means = np.random.normal(0, 1, no_arm)
        except Exception as e:
            logging.error(f"Error during MultiArmedBandit initialization due to: {str(e)}")
            raise e

    def pull_arm(self, arm):
        """
            Simulates pulling a specific arm and returns a reward sampled from a normal distribution.

            :param arm: The arm to be pulled.
            :return: The sampled reward.
        """
        try:
            # Check if the arm index is valid
            if arm < 0 or arm >= self.num_arms:
                raise ValueError(f"Invalid arm index: {arm}")

            # Sample a reward from a normal distribution with mean true_means[arm] and standard deviation 1
            reward = np.random.normal(self.true_means[arm], 1)

            return reward
        except Exception as e:
            logging.error(f"Error during arm pulling due to: {str(e)}")
            raise e
