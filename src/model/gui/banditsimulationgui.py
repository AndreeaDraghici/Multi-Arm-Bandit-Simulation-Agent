import numpy as np
import matplotlib.pyplot as plt
from tkinter import Label, Button, Entry, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.algorithm.agents import UCB1Agent, EpsilonGreedyAgent
from src.model.multiarmedbandit import MultiArmedBandit


class BanditSimulationGUI :
    def __init__(self, root) :
        self.root = root
        self.root.title("Multi Arm Bandit Simulation")

        self.label_path = Label(root, text="Enter path to input file:")
        self.label_path.pack(pady=5)

        self.entry_path = Entry(root)
        self.entry_path.pack(pady=5)

        self.button_browse = Button(root, text="Browse", command=self.browse_file)
        self.button_browse.pack(pady=5)

        self.button_run_simulation = Button(root, text="Run Simulation", command=self.run_simulation)
        self.button_run_simulation.pack(pady=10)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(expand=True, fill="both", padx=10, pady=10)

    def browse_file(self) :
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        self.entry_path.delete(0, 'end')
        self.entry_path.insert(0, file_path)

    def run_simulation(self) :
        file_path = self.entry_path.get()

        # Reading the number of arms, total iterations, and epsilon from the input file
        with open(file_path, 'r') as file :
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

        # Plotting the results
        self.ax.clear()
        self.ax.plot(avg_rewards_ucb1, label='UCB1')
        self.ax.plot(avg_rewards_epsilon_greedy, label='Epsilon-Greedy')
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Average Reward')
        self.ax.legend()

        # Refreshing the canvas
        self.canvas.draw()
