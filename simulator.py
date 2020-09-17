from pulp import *
import pandas as pd
import numpy as np
from refrigerator import Refrigerator

class Simulator:
    def __init__(self, moer_data, path_to_output):
        self.fridge = Refrigerator()
        self.timestep = 5
        self.current_time = 0
        self.moer_data = moer_data
        self.path_to_output = path_to_output
        self.outfile = open(path_to_output, 'w')
        self.outfile.write("time,fridge_temp,fridge_on,moer,lbs_co2\n")  # write CSV headers

    def pulptime(self):


    def run_simulation(self):
        for row in self.moer_data.head(200).index:
            #print(self.moer_data.iloc[row])
            expected_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            if expected_temp <= 33:
                self.fridge.turn_off()
            elif expected_temp >= 43:
                self.fridge.turn_on()

            moer = self.moer_data.iloc[row]["MOER"]
            #print("moer: ", moer)

            # write row for timestep
            self.outfile.write(self.generate_output_row(moer))

            self.fridge.current_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            self.current_time += self.timestep
            self.fridge.current_timestamp = self.current_time

        self.outfile.close()


    def generate_output_row(self, moer):
        lbs_co2 = self.lbs_co2_produced_this_timestep(moer)
        row_data = [self.current_time, round(self.fridge.current_temp, 2), self.fridge.on, moer, lbs_co2]
        str_row_data = [str(field) for field in row_data]
        return ",".join(str_row_data) + "\n"

    def lbs_co2_produced_this_timestep(self, moer):
        if not self.fridge.on:
            return 0
        megawatts_per_watt = 1 / 1000000
        hours_per_minute = 1 / 60
        return round(moer * (self.fridge.wattage * megawatts_per_watt) * (self.timestep * hours_per_minute), 8)
