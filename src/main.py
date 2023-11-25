from tkinter import Tk

from src.model.gui.banditsimulationgui import BanditSimulationGUI

if __name__ == '__main__':
    root = Tk()
    app = BanditSimulationGUI(root)
    root.geometry("600x500")  # Set the initial size of the window
    root.mainloop()