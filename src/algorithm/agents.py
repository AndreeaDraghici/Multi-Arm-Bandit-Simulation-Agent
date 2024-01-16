import logging
import numpy as np

from src.load_logging_configuration import load_logging_config


class UCB1Agent :
    def __init__(self, no_arm) :
        """
        Initializes a UCB1 agent with a given number of arms.
        :param no_arm: Number of arms in the bandit.

        Initializes two arrays: total_rewards and num_pulls, both of size no_arm, to zero.
        These arrays will track the total rewards and number of times each arm has been pulled, respectively.
        Initializes a timestep variable to 0, which will keep track of the number of times an arm has been selected.
        """
        try :
            # Check if the number of arms is at least 2
            if no_arm < 2 :
                raise ValueError("UCB1Agent requires at least 2 arms.")

            # Initialize logging configuration
            load_logging_config()
            self.logger = logging.getLogger('staging')

            # Number of arms
            self.num_arms = no_arm

            # Array to store total rewards for each arm
            self.total_rewards = np.zeros(no_arm)

            # Array to store the number of pulls for each arm
            self.num_pulls = np.zeros(no_arm)

            # Timestep to keep track of the number of iterations
            self.timestep = 0
        except Exception as e :
            self.logger.error(f"Error during UCB1Agent initialization due to: {str(e)}")
            raise e

    def select_arm(self) :
        """
        Selects an arm based on the UCB1 strategy.
        :return: The selected arm.

        Increments the timestep.
        Calculates an "exploration bonus" for each arm using the UCB1 formula. This bonus encourages exploration of less-pulled arms.
        Calculates the UCB values for each arm, which is the sum of the average reward of the arm and its exploration bonus.
        Selects the arm with the highest UCB value.
        """
        try :
            # Increment timestep
            self.timestep += 1

            # Calculate exploration bonus using UCB1 formula
            # the formula involves dividing by the number of times an arm has been pulled (self.num_pulls).
            # If an arm has never been pulled, this value would be zero, leading to a division by zero error.
            # Adding a small number like 1e-6 ensures that the denominator is never zero, thus avoiding this error.
            exploration_bonus = np.sqrt(2 * np.log(self.timestep) / (self.num_pulls + 1e-6))

            # Calculate UCB values for each arm
            ucb_values = self.total_rewards / (self.num_pulls + 1e-6) + exploration_bonus

            # Select arm with the highest UCB value
            selected_arm = np.argmax(ucb_values)

            self.logger.info(f"UCB1 Agent selected arm: {selected_arm}")

            return selected_arm
        except Exception as e :
            self.logger.error(f"Error during arm selection: {str(e)}")
            raise e

    def update(self, arm, reward) :
        """
        Updates the agent's knowledge after pulling an arm and receiving a reward.
        :param arm: The arm that was pulled.
        :param reward: The received reward.

        Takes the index of an arm and the reward received from pulling that arm.
        Checks if the provided arm index is valid. If not, it raises a ValueError.
        Updates the total_rewards and num_pulls arrays for the pulled arm.
        """
        try :
            # Check if the arm index is valid
            if arm < 0 or arm >= self.num_arms :
                raise ValueError(f"Invalid arm index: {arm}")

            # Update total rewards and number of pulls for the selected arm
            self.total_rewards[arm] += reward
            self.num_pulls[arm] += 1

            self.logger.info(f"UCB1 Agent updated for arm {arm}: Total Rewards={self.total_rewards[arm]}, "
                             f"Num Pulls={self.num_pulls[arm]}")
        except Exception as e :
            self.logger.error(f"Error during UCB1 agent update due to: {str(e)}")
            raise e


class EpsilonGreedyAgent :
    def __init__(self, no_arm, eps) :
        """
        Initializes an Epsilon-Greedy agent with a given number of arms and exploration rate.
        :param no_arm: Number of arms in the bandit.
        :param eps: Exploration rate (probability of exploration).
        """
        try :
            load_logging_config()

            # Get the logger for the 'staging' logger
            self.logger = logging.getLogger('staging')

            # Number of arms
            self.num_arms = no_arm

            # Exploration rate
            self.epsilon = eps

            # Array to store total rewards for each arm
            self.total_rewards = np.zeros(no_arm)

            # Array to store the number of pulls for each arm
            self.num_pulls = np.zeros(no_arm)
        except Exception as e :
            self.logger.error(f"Error during EpsilonGreedyAgent initialization due to: {str(e)}")
            raise e

    def select_arm(self) :
        """
        Selects an arm based on the Epsilon-Greedy strategy.
        :return: The selected arm.
        """
        try :
            # Explore with probability epsilon
            if np.random.rand() < self.epsilon :
                selected_arm = np.random.choice(self.num_arms)
                self.logger.info(f"Epsilon-Greedy Agent exploring: Selected arm randomly: {selected_arm}")
            else :
                # Exploit the arm with the highest average reward.
                # Calculates the average reward for each arm.
                # To avoid division by zero, `1e-6` (a small number) is added to `self.num_pulls`. This ensures numerical stability.
                avg_rewards = self.total_rewards / (self.num_pulls + 1e-6)
                selected_arm = np.argmax(avg_rewards)

                self.logger.info(
                    f"Epsilon-Greedy Agent exploiting: Selected arm based on highest average reward: {selected_arm}")
                self.logger.info(f"Epsilon-Greedy Agent - Average rewards: {avg_rewards}")

            return selected_arm
        except Exception as e :
            self.logger.error(f"Error during arm selection due to: {str(e)}")
            raise e

    def update(self, arm, reward) :
        """
        Updates the agent's knowledge after pulling an arm and receiving a reward.
        :param arm: The arm that was pulled.
        :param reward: The received reward.
        """
        try :
            # Update total rewards and number of pulls for the selected arm
            # Updates the total rewards for the pulled arm by adding the received reward.
            self.total_rewards[arm] += reward
            # Increments the count of how many times the arm has been pulled.
            self.num_pulls[arm] += 1

            self.logger.info(f"Epsilon-Greedy Agent updated for arm {arm}: Total Rewards={self.total_rewards[arm]}, "
                             f"Num Pulls={self.num_pulls[arm]}\n")
        except Exception as e :
            self.logger.error(f"Error during Epsilon Greedy agent update due to: {str(e)}")
            raise e
