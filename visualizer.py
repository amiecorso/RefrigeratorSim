import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


class Visualizer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.path_to_data = simulator.path_to_output

    def plot(self):
        self.data = pd.read_csv(self.path_to_data)

        fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0})

        # plot fridge temp color-coded by on/off
        for t1, t2, status, y1, y2 in zip(self.data['time'],
                                          self.data['time'][1:],
                                          self.data['fridge_on'],
                                          self.data['fridge_temp'],
                                          self.data['fridge_temp'][1:]):
            if status:
                axs[0].plot([t1, t2], [y1, y2], 'b')
            else:
                axs[0].plot([t1, t2], [y1, y2], 'r')
        axs[0].set_ylabel('Refrigerator \nTemperature \n(F)', rotation=0, labelpad=42)
        axs[0].set_ylim([30, 45])
        axs[0].set_yticks(range(33, 44))

        legend_elements = [Line2D([0], [0], color='b', lw=4, label='On'),
                           Line2D([0], [0], color='r', lw = 4, label='Off')]
        axs[0].legend(handles=legend_elements)

        # plot moer
        axs[1].set_ylabel('MOER \n(lbs CO2 / Mwh)', rotation=0, labelpad=42)
        axs[1].plot(self.data["time"], self.data["moer"])

        # plot cumulative CO2 usage
        cumulative_total = 0
        for t1, co2, t2 in zip(self.data['time'], self.data['lbs_co2'], self.data['time'][1:]):
            new_total = cumulative_total + co2
            axs[2].plot([t1, t2], [cumulative_total, new_total], 'b')
            cumulative_total = new_total
        axs[2].set_ylabel('Cumulative \nlbs CO2', rotation=0, labelpad=42)

        axs[2].set_xlabel('Elapsed Time (min)')
        for ax in axs:
            ax.label_outer()

        fig.set_size_inches(12, 7)
        fig.suptitle("Simple AER Simulation")
        fig.tight_layout()

        fridge_on_subset = self.data.loc[self.data['fridge_on'] == True]
        print(fridge_on_subset.head())
        total_run_time = fridge_on_subset.shape[0] * self.simulator.timestep
        print("Total refrigerator run time: ", total_run_time, " mins")
        print("Total lbs CO2 emitted: ", cumulative_total)
        plt.show()
