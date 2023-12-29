from tkinter import Tk
import logging

from src.model.gui.bandit_simulation_gui import BanditSimulationGUI


def main_driver() :
    root = Tk()
    app = BanditSimulationGUI(root)
    root.geometry("670x600")
    root.mainloop()
    logging.info("GUI execution completed.")


if __name__ == '__main__' :
    try :
        main_driver()
    except Exception as e :
        logging.error(f"An error occurred to run the application due to: {str(e)}")
