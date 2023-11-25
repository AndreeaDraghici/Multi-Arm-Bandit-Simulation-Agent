import numpy as np


class UCB1Agent :
    """
            Initializes a UCB1 agent with a given number of arms.
            :param no_arm: Number of arms in the bandit.
    """

    def __init__(self, no_arm) :
        self.num_arms = no_arm
        self.total_rewards = np.zeros(no_arm)
        self.num_pulls = np.zeros(no_arm)
        self.timestep = 0

    def select_arm(self) :
        """
                Selects an arm based on the UCB1 strategy.
                :return: The selected arm.
        """
        self.timestep += 1
        exploration_bonus = np.sqrt(2 * np.log(self.timestep) / (self.num_pulls + 1e-6))
        ucb_values = self.total_rewards / (self.num_pulls + 1e-6) + exploration_bonus
        selected_arm = np.argmax(ucb_values)
        return selected_arm

    def update(self, arm, reward) :
        """
                Updates the agent's knowledge after pulling an arm and receiving a reward.
                :param arm: The arm that was pulled.
                :param reward: The received reward.
        """
        self.total_rewards[arm] += reward
        self.num_pulls[arm] += 1


class EpsilonGreedyAgent :
    def __init__(self, no_arm, eps) :
        """
               Initializes an Epsilon-Greedy agent with a given number of arms and exploration rate.
               :param no_arm: Number of arms in the bandit.
               :param eps: Exploration rate (probability of exploration).
        """
        self.num_arms = no_arm
        self.epsilon = eps
        self.total_rewards = np.zeros(no_arm)
        self.num_pulls = np.zeros(no_arm)

    def select_arm(self) :
        """
               Selects an arm based on the Epsilon-Greedy strategy.
               :return: The selected arm.
        """
        if np.random.rand() < self.epsilon :
            # Explore with probability epsilon
            selected_arm = np.random.choice(self.num_arms)
        else :
            # Exploit the arm with the highest average reward
            avg_rewards = self.total_rewards / (self.num_pulls + 1e-6)
            selected_arm = np.argmax(avg_rewards)
        return selected_arm

    def update(self, arm, reward) :
        """
               Updates the agent's knowledge after pulling an arm and receiving a reward.
               :param arm: The arm that was pulled.
               :param reward: The received reward.
        """
        self.total_rewards[arm] += reward
        self.num_pulls[arm] += 1
