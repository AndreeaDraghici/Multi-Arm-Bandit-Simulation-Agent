import numpy as np
import logging


# This class simulates a multi-armed bandit problem, where each arm of the bandit provides a reward drawn from a normal distribution.
class MultiArmedBandit :
    def __init__(self, no_arm) :
        """
            The constructor initializes the bandit with a given number of arms and sets up the mean rewards for each arm, sampled from a normal distribution.

            :param no_arm: Number of arms in the bandit.
        """
        try :
            # This line sets the number of arms (num_arms) for the bandit instance to the value provided in no_arm.
            self.num_arms = no_arm

            # Generates a list of true_means for each arm. Each mean is drawn from a normal distribution with a mean of 0 and a standard deviation of 1.
            # The length of this list is equal to the number of arms (no_arm).
            # These means represent the expected reward for each arm.
            self.true_means = np.random.normal(0, 1, no_arm)
        except Exception as e :
            logging.error(f"Error during MultiArmedBandit initialization due to: {str(e)}")
            raise e

    def pull_arm(self, arm) :
        """
            Method simulates the action of pulling an arm and receiving a reward based on the underlying distribution of that arm.

            :param arm: The arm to be pulled.
            :return: The sampled reward.
        """
        try :
            # Check if the arm index is valid
            if arm < 0 or arm >= self.num_arms :
                raise ValueError(f"Invalid arm index: {arm}")

            # Samples a reward for the specified arm.
            # The reward is drawn from a normal distribution with the mean equal to the true mean of that arm (self.true_means[arm])
            # and a standard deviation of 1.
            reward = np.random.normal(self.true_means[arm], 1)

            return reward
        except Exception as e :
            logging.error(f"Error during arm pulling due to: {str(e)}")
            raise e
