import argparse
import os
import pandas as pd
import time
from simulator import Simulator

path_to_moer_data = "MOER_data/MOERS.csv"
if not os.path.exists(path_to_moer_data):
    raise FileNotFoundError("{} doesn't seem to exist...".format(path_to_moer_data))

output_dir = "./output_data/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Read initial MOER data
all_moer_data = pd.read_csv(path_to_moer_data)
initial_historical_data = all_moer_data[:288]
sim_moer_data = all_moer_data[288:-1].reset_index(drop=True)

# number of timesteps (data rows) to process (smaller for testing speed)
num_timesteps = sim_moer_data.shape[0]  # all rows
#num_timesteps = 300

simulator = Simulator(sim_moer_data, output_dir, num_timesteps)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Run all three models.')
    parser.add_argument('--no_data', action='store_true', help='Run model using no data.')
    parser.add_argument('--forecast_only', action='store_true', help='Run model using 1-hr forecast window.')
    parser.add_argument('--forecast_and_history', action='store_true', help='Run model using 1-hr forecast and historical avgs.')
    return parser.parse_args()

def end_timer(start_time, sim_id):
    end_time = time.time()
    total_seconds = round(end_time - start_time, 2)
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    print("Simulation '{}' duration: {} min {} sec".format(sim_id, minutes, seconds))

if __name__ == '__main__':
    args = parse_args()

    # run simulations
    if args.no_data or args.all:
        start_time = time.time()
        simulator.run_simulation_without_data(show_plot=False)
        end_timer(start_time, 'no_data')

    if args.forecast_only or args.all:
        start_time = time.time()
        simulator.run_simulation_with_forecast(show_plot=False)
        end_timer(start_time, 'forecast_only')

    if args.forecast_and_history or args.all or len(args) == 0:
        start_time = time.time()
        simulator.run_simulation_with_forecast_and_historicals(show_plot=False)
        end_timer(start_time, 'forecast_and_history')

