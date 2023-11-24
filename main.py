import numpy as np
import matplotlib.pyplot as plt

from agents import UCB1Agent, EpsilonGreedyAgent
from multiarmedbandit import MultiArmedBandit

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
