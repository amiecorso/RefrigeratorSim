import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


class Visualizer:
    def __init__(self, simulator):
        self.simulator = simulator

    def plot(self, path_to_data, show_plot):
        data = pd.read_csv(path_to_data)
        sim_id = path_to_data.lstrip(self.simulator.output_dir).rstrip('.csv').lstrip('/').lstrip('sim_output_')

        fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0})

        # plot fridge temp color-coded by on/off
        for t1, t2, status, y1, y2 in zip(data['time'],
                                          data['time'][1:],
                                          data['fridge_on'],
                                          data['fridge_temp'],
                                          data['fridge_temp'][1:]):
            if status:
                axs[0].plot([t1, t2], [y1, y2], 'b')
            else:
                axs[0].plot([t1, t2], [y1, y2], 'r')
        axs[0].set_ylabel('Refrigerator \nTemperature \n(F)', rotation=0, labelpad=42)
        axs[0].set_ylim([30, 45])
        axs[0].set_yticks(range(33, 44))

        legend_elements = [Line2D([0], [0], color='b', lw=4, label='On'),
                           Line2D([0], [0], color='r', lw=4, label='Off')]
        axs[0].legend(handles=legend_elements)

        # plot moer
        axs[1].set_ylabel('MOER \n(lbs CO2 / Mwh)', rotation=0, labelpad=42)
        axs[1].plot(data["time"], data["moer"])

        # plot cumulative CO2 usage
        cumulative_total = 0
        for t1, co2, t2 in zip(data['time'], data['lbs_co2'], data['time'][1:]):
            new_total = cumulative_total + co2
            axs[2].plot([t1, t2], [cumulative_total, new_total], 'b')
            cumulative_total = new_total
        axs[2].set_ylabel('Cumulative \nlbs CO2', rotation=0, labelpad=42)

        axs[2].set_xlabel('Elapsed Time (min)')
        for ax in axs:
            ax.label_outer()

        fig.set_size_inches(12, 7)
        title_suffix = " ".join([string.capitalize() for string in sim_id.split("_")])
        fig.suptitle("Simple AER Simulation: " + title_suffix)
        fig.tight_layout()

        fridge_on_subset = data.loc[data['fridge_on'] == True]
        #print(fridge_on_subset.head())
        total_run_time = fridge_on_subset.shape[0] * self.simulator.timestep
        print("=" * 50)
        print("Total refrigerator run time: ", total_run_time, " mins")
        print("Total lbs CO2 emitted: ", cumulative_total)

        axs[2].text(0.95, 0.01, 'Total lbs CO2 emitted: ' + str(round(cumulative_total, 4)),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='green', fontsize=15)

        fig.savefig(self.simulator.output_dir.rstrip('/') + '/' + 'plots_' + sim_id + ".pdf")
        if show_plot:
            plt.show()
        return

    def plot_avg_moers(self, path_to_data, show_plot):
        data = pd.read_csv(path_to_data)

        fig, ax = plt.subplots()
        # plot avg moers
        ax.set_ylabel('AVG MOER \n(lbs CO2 / Mwh)', rotation=0, labelpad=42)
        ax.plot(data["time"], data["avg_moer_at_time"])
        fig.set_size_inches(8, 5)
        fig.suptitle("Average Historical MOER: Granularity = " + self.simulator.avgs_granularity.capitalize())
        fig.tight_layout()
        fig.savefig(self.simulator.output_dir.rstrip('/') + '/' + 'avgMOERs_' +
                    self.simulator.avgs_granularity + "_granularity.pdf")
        if show_plot:
            plt.show()
        return