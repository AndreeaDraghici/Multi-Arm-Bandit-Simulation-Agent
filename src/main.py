from tkinter import Tk
import logging
from src.model.gui.BanditSimulationGUI import BanditSimulationGUI

try :
    if __name__ == '__main__' :
        root = Tk()
        app = BanditSimulationGUI(root)
        root.geometry("670x600")
        root.mainloop()
        logging.info("GUI execution completed.")
except Exception as e :
    logging.error(f"An error occurred to run the application due to: {str(e)}")
