import os
import random


class RandomInputGenerator :
    def __init__(self, num_files=10) :
        """
        Initializes the RandomInputGenerator.

        :param num_files: Number of random input files to generate.
        """
        self.num_files = num_files
        self.data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                     '../../input')  # Define the path to the 'input' directory

    def generate_inputs(self) :
        """
        Generates random input files.

        For each file, generates a random number of arms, total iterations, and epsilon.
        Saves the files in the 'input' directory.
        """
        # Check if the 'input' directory exists, and if not, create it
        if not os.path.exists(self.data_dir) :
            os.makedirs(self.data_dir)

        for i in range(1, self.num_files + 1) :
            num_arms = random.randint(2, 10)  # Generate a random number of arms
            num_iterations = random.randint(500, 1500)  # Generate a random number of iterations
            epsilon = round(random.uniform(0.05, 0.3), 2)  # Generate a random epsilon

            with open(os.path.join(self.data_dir, f'input{i}.txt'), 'w') as file :
                file.write(f"{num_arms}\n{num_iterations}\n{epsilon}\n")  # Write the values to the input file


if __name__ == "__main__" :
    generator = RandomInputGenerator(num_files=10)
    generator.generate_inputs()
