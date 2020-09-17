import pandas as pd
from refrigerator import Refrigerator

path_to_moer_data = "MOER_Data/MOERS.csv"
path_to_output = "simulation_output.csv"

timestep = 5  # start my mimicking granularity of dataset

all_moer_data = pd.read_csv(path_to_moer_data)
initial_historical_data = all_moer_data[:288]
sim_moer_data = all_moer_data[288:-1].reset_index(drop=True)

fridge = Refrigerator()
current_time = 0

with open(path_to_output, 'w') as sim_output:
    # write CSV headers
    for row in sim_moer_data.head().index:
        print(sim_moer_data.iloc[row])
        expected_temp = fridge.expected_temp(current_time + timestep)

        if expected_temp <= 33:
            fridge.turn_off()
        elif expected_temp >= 43:
            fridge.turn_on()

        # should the simulation be its own class???

        # write simulation row

        # update fridge/time variables


def generate_output_row(current_time, fridge, moer):
    lbs_co2 = lbs_co2_produced(fridge, moer, timestep)
    row_data = [current_time, fridge.current_temp, fridge.on, moer, lbs_co2]
    return ",".join(row_data)

def lbs_co2_produced(fridge, moer, mins):
    megawatts_per_watt = 1/1000000
    hours_per_minute = 1/60
    return moer * (fridge.wattage * megawatts_per_watt) * (mins * hours_per_minute)


