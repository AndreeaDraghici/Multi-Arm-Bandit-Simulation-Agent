import numpy as np


class MultiArmedBandit :
    def __init__(self, no_arm) :
        """
                Initializes a Multi-Armed Bandit with a given number of arms.
                :param no_arm: Number of arms in the bandit.
        """
        self.num_arms = no_arm
        self.true_means = np.random.normal(0, 1, no_arm)  # Mean reward for each arm

    def pull_arm(self, arm) :
        """
                Simulates pulling a specific arm and returns a reward sampled from a normal distribution.
                :param arm: The arm to be pulled.
                :return: The sampled reward.
        """
        return np.random.normal(self.true_means[arm], 1)
