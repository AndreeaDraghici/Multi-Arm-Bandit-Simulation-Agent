import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit :
    def __init__(self, num_arms) :
        self.num_arms = num_arms
        self.true_means = np.random.normal(0, 1, num_arms)  # Mean reward for each arm

    def pull_arm(self, arm) :
        # Simulate pulling a specific arm and return a reward sampled from a normal distribution
        return np.random.normal(self.true_means[arm], 1)


class UCB1Agent :
    def __init__(self, num_arms) :
        self.num_arms = num_arms
        self.total_rewards = np.zeros(num_arms)
        self.num_pulls = np.zeros(num_arms)
        self.timestep = 0

    def select_arm(self) :
        self.timestep += 1
        exploration_bonus = np.sqrt(2 * np.log(self.timestep) / (self.num_pulls + 1e-6))
        ucb_values = self.total_rewards / (self.num_pulls + 1e-6) + exploration_bonus
        selected_arm = np.argmax(ucb_values)
        return selected_arm

    def update(self, arm, reward) :
        self.total_rewards[arm] += reward
        self.num_pulls[arm] += 1


class EpsilonGreedyAgent :
    def __init__(self, num_arms, epsilon=0.1) :
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.total_rewards = np.zeros(num_arms)
        self.num_pulls = np.zeros(num_arms)

    def select_arm(self) :
        if np.random.rand() < self.epsilon :
            # Explore with probability epsilon
            selected_arm = np.random.choice(self.num_arms)
        else :
            # Exploit the arm with the highest average reward
            avg_rewards = self.total_rewards / (self.num_pulls + 1e-6)
            selected_arm = np.argmax(avg_rewards)
        return selected_arm

    def update(self, arm, reward) :
        self.total_rewards[arm] += reward
        self.num_pulls[arm] += 1


if __name__ == '__main__' :
    # Numărul de brațe ale mașinii cu sloturi
    num_arms = 5

    # Numărul total de iterații
    num_iterations = 1000

    # Crearea mașinii cu sloturi
    bandit = MultiArmedBandit(num_arms)

    # Crearea agenților
    ucb1_agent = UCB1Agent(num_arms)
    epsilon_greedy_agent = EpsilonGreedyAgent(num_arms, epsilon=0.1)

    # Listele pentru a stoca recompensele medii
    avg_rewards_ucb1 = []
    avg_rewards_epsilon_greedy = []

    # Simularea iterațiilor
    for i in range(num_iterations) :
        # Alegerea brațelor pentru fiecare agent
        arm_ucb1 = ucb1_agent.select_arm()
        arm_epsilon_greedy = epsilon_greedy_agent.select_arm()

        # Simularea extragerii brațelor și obținerea recompenselor
        reward_ucb1 = bandit.pull_arm(arm_ucb1)
        reward_epsilon_greedy = bandit.pull_arm(arm_epsilon_greedy)

        # Actualizarea agenților cu recompensele obținute
        ucb1_agent.update(arm_ucb1, reward_ucb1)
        epsilon_greedy_agent.update(arm_epsilon_greedy, reward_epsilon_greedy)

        # Calcularea recompenselor medii și adăugarea acestora la listă
        avg_rewards_ucb1.append(np.mean(ucb1_agent.total_rewards / (ucb1_agent.num_pulls + 1e-6)))
        avg_rewards_epsilon_greedy.append(
            np.mean(epsilon_greedy_agent.total_rewards / (epsilon_greedy_agent.num_pulls + 1e-6)))

    # Vizualizarea rezultatelor
    plt.plot(avg_rewards_ucb1, label='UCB1')
    plt.plot(avg_rewards_epsilon_greedy, label='Epsilon-Greedy')
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()
