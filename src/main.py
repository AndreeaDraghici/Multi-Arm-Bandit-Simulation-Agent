from tkinter import Tk

from src.model.gui.bandit_simulation_gui import BanditSimulationGUI

# Instantiate and run the GUI
if __name__ == '__main__' :
    root = Tk()
    app = BanditSimulationGUI(root)
    root.geometry("670x600")  # Set the initial size of the window
    root.mainloop()
