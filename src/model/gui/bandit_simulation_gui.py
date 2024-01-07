import logging

import mplcursors
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Label, Button, Entry, filedialog, messagebox

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


from tkinter import Menu
import time

from src.algorithm.agents import UCB1Agent, EpsilonGreedyAgent
from src.load_logging_configuration import load_logging_config
from src.model.multi_armed_bandit import MultiArmedBandit


class BanditSimulationGUI :
    def __init__(self, root) :

        load_logging_config()

        # Get the logger for the 'staging' logger
        self.logger = logging.getLogger('staging')

        self.no_arms = None
        self.epsilon = None
        self.num_iterations = None
        self.iteration_time = None
        self.execution_times = []

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
        try :
            # Open a file dialog to select the data file
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])

            # Update the entry field with the selected file path
            self.entry_path.delete(0, 'end')
            self.entry_path.insert(0, file_path)

            self.logger.info(f"Selected data input file: {file_path}")

        except Exception as e :
            error_message = f"An error occurred while browsing for a input file due to: {str(e)}"
            self.logger.error(error_message)
            # Display a modal error information
            messagebox.showerror("Error", error_message)

    def run_simulation(self) :
        """
        Runs the bandit simulation based on the provided data file.

        Reads the number of arms, total iterations, and epsilon from the data file,
        creates a multi-armed bandit and agents, simulates iterations, and plots the results.
        """
        try :
            # Get the file path from the entry field
            file_path = self.entry_path.get()

            # Check if the file has a '.txt' extension
            if not file_path.lower().endswith('.txt') :
                raise ValueError("Invalid file format. Please select a '.txt' file.")
            self.logger.info("Running bandit simulation based on the provided data file.")

            # Create a menu bar
            menubar = Menu(self.root)
            self.root.config(menu=menubar)

            # Add a "File" menu
            file_menu = Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Save plot", menu=file_menu)

            # Read the number of arms, total iterations, and epsilon from the data file
            with open(file_path, 'r') as file :
                # Verify and parse the structure of the file
                try :
                    self.no_arms = int(file.readline().strip())
                    self.num_iterations = int(file.readline().strip())
                    self.epsilon = float(file.readline().strip())
                except ValueError as ve :
                    raise ValueError("Invalid file structure. Please make sure the file contains valid data.") from ve

            # Create the multi-armed bandit
            bandit = MultiArmedBandit(self.no_arms)

            # Create the agents
            ucb1_agent = UCB1Agent(self.no_arms)
            epsilon_greedy_agent = EpsilonGreedyAgent(self.no_arms, eps=self.epsilon)

            # Lists to store average rewards
            avg_rewards_ucb1 = []
            avg_rewards_epsilon_greedy = []

            self.logger.info(f"Number of arms: {self.no_arms}")
            self.logger.info(f"Number of iterations: {self.num_iterations}")
            self.logger.info(f"Epsilon value: {self.epsilon}\n")

            start_time_iteration = time.time()
            # Simulate iterations
            for i in range(self.num_iterations) :
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

                end_time = time.time()

                # Calculate and log the iteration time
                iteration_time = end_time - start_time_iteration
                self.execution_times.append(iteration_time)
                self.logger.info(f"Iteration {i + 1} took {iteration_time:.6f} seconds")

            # Adjust the length of the lists to reflect the total number of iterations
            total_iterations = self.num_iterations * self.no_arms
            avg_rewards_ucb1 = avg_rewards_ucb1[:total_iterations]
            avg_rewards_epsilon_greedy = avg_rewards_epsilon_greedy[:total_iterations]

            # Clear the axis before adding new lines
            self.ax.clear()

            line_ucb1, = self.ax.plot(avg_rewards_ucb1, label='UCB1')
            line_epsilon_greedy, = self.ax.plot(avg_rewards_epsilon_greedy, label='Epsilon-Greedy')

            # Set the labels for X and Y axes
            self.ax.set_xlabel('Iterations')
            self.ax.set_ylabel('Average Reward')

            # Set the legend to show labels for each line
            self.ax.legend()

            # Add interactive data cursors to the plot
            mplcursors.cursor(hover=True).connect("add", lambda selection : self.show_cursor_data(selection, line_ucb1,
                                                                                                  line_epsilon_greedy))

            # Refresh the canvas
            self.canvas.draw()

            # Add the "Save Plot" option to the menu
            file_menu.add_command(label="Save Plot", command=self.save_plot)


        except Exception as e :
            error_message = f"An error occurred during the simulation due to: {str(e)}"
            self.logger.error(error_message)
            # Display an error box
            messagebox.showerror("Error", error_message)

    def show_cursor_data(self, sel, line_ucb1, line_epsilon_greedy) :
        """
        Display additional information on the plot when hovering over data points.
        """
        try :
            # Clear previous hover labels
            self.clear_hover_labels()

            index = sel.target.index

            # Calculate the arm, iteration number, and epsilon based on the index
            arm = index % self.no_arms
            total_iterations = self.num_iterations * self.no_arms
            iteration_number = index % total_iterations
            eps = self.epsilon

            if not (0 <= arm <= self.no_arms) :
                raise ValueError(f"Invalid value for arm: {arm}")

            if not (0 <= iteration_number < self.num_iterations) :
                raise ValueError(f"Invalid value for iteration_number: {iteration_number}")

            if not (0 <= eps <= 1) :
                raise ValueError(f"Invalid value for epsilon: {eps}")

            # Set the label text and adjust alpha for visibility
            sel.annotation.set_text(f"Arm: {arm}, Iteration: {iteration_number}, Epsilon: {eps:.2f}")
            sel.annotation.get_bbox_patch().set_alpha(0.8)

            # Update the legend to reflect the new labels
            self.ax.legend()

        except ValueError as ve :
            self.logger.warning(f"Invalid value detected during hovering due to: {ve}")

        except Exception as e :
            self.logger.error(f"An error occurred during hovering due to: {str(e)}")

    def clear_hover_labels(self) :
        """
        Clears the hover labels and legend.
        """
        try :
            # Clear the legend
            self.ax.get_legend().remove()

            # Clear existing annotations
            for annotation in self.ax.texts :
                annotation.remove()

        except Exception as e :
            self.logger.error(f"An error occurred while clearing hover labels: {str(e)}")

    def save_plot(self) :
        try :
            # Save the plot to an output directory
            output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../output')
            if not os.path.exists(output_dir) :
                os.makedirs(output_dir)

            # Generate a unique filename based on timestamp for hour and second
            timestamp = time.strftime("%Y.%m.%d_%H.%M.%S")
            output_filename = os.path.join(output_dir, f'Output_{timestamp}.png')

            # Save the figure to the output file
            self.figure.savefig(output_filename)
            self.logger.info(f"Plot saved to: {output_filename}")

            # Open a dialog to confirm the save
            user_response = messagebox.askokcancel("Save Plot", f"Plot saved to:\n{output_filename}")

            # Check user response
            if user_response :
                # Perform any additional actions or close the dialog
                pass
            else :
                # Optionally, undo the save or perform other actions
                os.remove(output_filename)
                self.logger.info(f"Save operation undone. Plot not saved.")

        except Exception as e :
            error_message = f"An error occurred while saving the plot: {str(e)}"
            self.logger.error(error_message)
            # Display an error box
            messagebox.showerror("Error", error_message)
