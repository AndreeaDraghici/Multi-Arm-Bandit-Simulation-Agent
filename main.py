import numpy as np
import matplotlib.pyplot as plt


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


class UCB1Agent :
    """
            Initializes a UCB1 agent with a given number of arms.
            :param no_arms: Number of arms in the bandit.
    """

    def __init__(self, no_arms) :
        self.num_arms = no_arms
        self.total_rewards = np.zeros(no_arms)
        self.num_pulls = np.zeros(no_arms)
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


if __name__ == '__main__' :
    # Reading the number of arms and total iterations from the input file
    with open('input.txt', 'r') as file :
        no_arms = int(file.readline().strip())
        num_iterations = int(file.readline().strip())
        epsilon = float(file.readline().strip())

    # Creating the multi-armed bandit
    bandit = MultiArmedBandit(no_arms)

    # Creating the agents
    ucb1_agent = UCB1Agent(no_arms)
    epsilon_greedy_agent = EpsilonGreedyAgent(no_arms, eps=epsilon)

    # Lists to store average rewards
    avg_rewards_ucb1 = []
    avg_rewards_epsilon_greedy = []

    # Simulating iterations
    for i in range(num_iterations) :
        # Selecting arms for each agent
        arm_ucb1 = ucb1_agent.select_arm()
        arm_epsilon_greedy = epsilon_greedy_agent.select_arm()

        # Simulating pulling arms and obtaining rewards
        reward_ucb1 = bandit.pull_arm(arm_ucb1)
        reward_epsilon_greedy = bandit.pull_arm(arm_epsilon_greedy)

        # Updating agents with obtained rewards
        ucb1_agent.update(arm_ucb1, reward_ucb1)
        epsilon_greedy_agent.update(arm_epsilon_greedy, reward_epsilon_greedy)

        # Calculating average rewards and adding them to the list
        avg_rewards_ucb1.append(np.mean(ucb1_agent.total_rewards / (ucb1_agent.num_pulls + 1e-6)))
        avg_rewards_epsilon_greedy.append(
            np.mean(epsilon_greedy_agent.total_rewards / (epsilon_greedy_agent.num_pulls + 1e-6)))

    # Visualizing the results
    plt.plot(avg_rewards_ucb1, label='UCB1')
    plt.plot(avg_rewards_epsilon_greedy, label='Epsilon-Greedy')
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()
