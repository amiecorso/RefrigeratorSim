import pandas as pd
import os
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
num_timesteps = 400

simulator = Simulator(sim_moer_data, output_dir, num_timesteps)

simulator.run_simulation_without_data(show_plot=False)
simulator.run_simulation_with_forecast()

