import matplotlib.pyplot as plt
import pandas as pd

class Visualizer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.path_to_data = simulator.path_to_output

    def plot(self):
        self.data = pd.read_csv(self.path_to_data)

        fig, axs = plt.subplots(3, 1, sharex=True)

        # plot fridge temp color-coded by on/off
        for t1, t2, status, y1, y2 in zip(self.data['time'],
                                  self.data['time'][1:],
                                  self.data['fridge_on'],
                                  self.data['fridge_temp'],
                                  self.data['fridge_temp'][1:]):
            if status:
                axs[0].plot([t1, t2], [y1, y2], 'g')
            else:
                axs[0].plot([t1, t2], [y1, y2], 'r')
        axs[0].set_ylabel('Refrigerator Temperature (F)')

        # plot moer
        axs[1].set_ylabel('MOER (lbs CO2 / Mwh)')
        axs[1].plot(self.data["time"], self.data["moer"])
        axs[1].set_xlabel('Elapsed Time (min)')

        # plot cumulative CO2 usage
        cumulative_total = 0
        for t1, co2, t2 in zip(self.data['time'], self.data['lbs_co2'], self.data['time'][1:]):
            new_total = cumulative_total + co2
            axs[2].plot([t1, t2], [cumulative_total, new_total], 'b')
            cumulative_total = new_total
        axs[2].set_ylabel('Cumulative LBS CO2')

        fridge_on_subset = self.data.loc[self.data['fridge_on'] == True]
        print(fridge_on_subset.head())
        total_run_time = fridge_on_subset.shape[0] * self.simulator.timestep
        print("Total refrigerator run time: ", total_run_time, " mins")
        print("Total lbs CO2 emitted: ", cumulative_total)
        plt.show()
