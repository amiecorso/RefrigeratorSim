import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd

class Visualizer:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data

    def plot(self):
        self.data = pd.read_csv(self.path_to_data)

        fig, axs = plt.subplots(2, 1)
        # plot fridge temp

        def plot_func(df):
            global axs
            color = 'r' if (df['fridge_on'] is False).all() else 'g'
            lw = 2.0
            axs[0].plot(self.data["time"], self.data["fridge_temp"], c=color, linewidth=lw)

        axs[0].set_ylabel('Refrigerator Temperature (F)')
        axs[0].plot(self.data["time"], self.data["fridge_temp"])

        # plot moer
        axs[1].set_ylabel('MOER (lbs CO2 / Mwh)')
        axs[1].plot(self.data["time"], self.data["moer"])

        axs[1].set_xlabel('Elapsed Time (min)')

        # plot whether fridge is on or not (color code fridge line?)
        # plot cumulative CO2 usage
        plt.show()

