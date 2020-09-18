import pandas as pd
from simulator import Simulator
from visualizer import Visualizer

path_to_moer_data = "MOER_Data/MOERS.csv"
path_to_sim_output = "simulation_output.csv"

all_moer_data = pd.read_csv(path_to_moer_data)
initial_historical_data = all_moer_data[:288]
sim_moer_data = all_moer_data[288:-1].reset_index(drop=True)

simulator = Simulator(sim_moer_data, path_to_sim_output)
visualizer = Visualizer(simulator)

simulator.run_simulation_with_data()
visualizer.plot()
