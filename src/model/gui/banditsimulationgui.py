import time

import numpy as np
import matplotlib.pyplot as plt
from tkinter import Label, Button, Entry, filedialog
from tkinter.messagebox import askokcancel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from src.algorithm.agents import UCB1Agent, EpsilonGreedyAgent
from src.model.multiarmedbandit import MultiArmedBandit
from tkinter import Menu


class BanditSimulationGUI :
    def __init__(self, root) :
        """
        Initializes the BanditSimulationGUI.
        :param root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Multi Arm Bandit Simulation")

        # Configure column and row weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)

        # Label, Entry, and Button for the file path
        self.label_path = Label(root, text="Enter path to data file")
        self.label_path.grid(row=0, column=0, pady=5, sticky='e')

        self.entry_path = Entry(root, width=40)
        self.entry_path.grid(row=0, column=1, pady=5, sticky='we')

        self.button_browse = Button(root, text="Browse", command=self.browse_file)
        self.button_browse.grid(row=0, column=2, pady=5, sticky='w')

        # Button to run the simulation
        self.button_run_simulation = Button(root, text="Run Simulation", command=self.run_simulation)
        self.button_run_simulation.grid(row=1, column=1, pady=10)

        # Figure and canvas for the plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

    def browse_file(self) :
        """
        Opens a file dialog to select the data file and updates the entry field with the selected file path.
        """
        # Open a file dialog to select the data file
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])

        # Update the entry field with the selected file path
        self.entry_path.delete(0, 'end')
        self.entry_path.insert(0, file_path)

    def run_simulation(self) :
        """
        Runs the bandit simulation based on the provided data file.

        Reads the number of arms, total iterations, and epsilon from the data file,
        creates a multi-armed bandit and agents, simulates iterations, and plots the results.
        """

        # Get the file path from the entry field
        file_path = self.entry_path.get()

        # Create a menu bar
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # Add a "File" menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        # Read the number of arms, total iterations, and epsilon from the data file
        with open(file_path, 'r') as file :
            no_arms = int(file.readline().strip())
            num_iterations = int(file.readline().strip())
            epsilon = float(file.readline().strip())

        # Create the multi-armed bandit
        bandit = MultiArmedBandit(no_arms)

        # Create the agents
        ucb1_agent = UCB1Agent(no_arms)
        epsilon_greedy_agent = EpsilonGreedyAgent(no_arms, eps=epsilon)

        # Lists to store average rewards
        avg_rewards_ucb1 = []
        avg_rewards_epsilon_greedy = []

        # Simulate iterations
        for i in range(num_iterations) :
            # Select arms for each agent
            arm_ucb1 = ucb1_agent.select_arm()
            arm_epsilon_greedy = epsilon_greedy_agent.select_arm()

            # Simulate pulling arms and obtaining rewards
            reward_ucb1 = bandit.pull_arm(arm_ucb1)
            reward_epsilon_greedy = bandit.pull_arm(arm_epsilon_greedy)

            # Update agents with obtained rewards
            ucb1_agent.update(arm_ucb1, reward_ucb1)
            epsilon_greedy_agent.update(arm_epsilon_greedy, reward_epsilon_greedy)

            # Calculate average rewards and add them to the list
            avg_rewards_ucb1.append(np.mean(ucb1_agent.total_rewards / (ucb1_agent.num_pulls + 1e-6)))
            avg_rewards_epsilon_greedy.append(
                np.mean(epsilon_greedy_agent.total_rewards / (epsilon_greedy_agent.num_pulls + 1e-6)))

        # Plot the results
        self.ax.clear()
        self.ax.plot(avg_rewards_ucb1, label='UCB1')
        self.ax.plot(avg_rewards_epsilon_greedy, label='Epsilon-Greedy')
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Average Reward')
        self.ax.legend()

        # Refresh the canvas
        self.canvas.draw()

        # Add the "Save Plot" option to the menu
        file_menu.add_command(label="Save Plot", command=self.save_plot)

    def save_plot(self) :
        # Save the plot to an output directory
        output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../output')
        if not os.path.exists(output_dir) :
            os.makedirs(output_dir)

        # Generate a unique filename based on timestamp for hour and second
        timestamp = time.strftime("%Y.%m.%d_%H.%M.%S")
        output_filename = os.path.join(output_dir, f'Output_{timestamp}.png')

        # Save the figure to the output file
        self.figure.savefig(output_filename)

        # Open a dialog to confirm the save
        askokcancel("Save Plot", f"Plot saved to:\n{output_filename}")
