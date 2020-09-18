from pulp import *
import numpy as np
from refrigerator import Refrigerator

class Simulator:
    def __init__(self, moer_data, path_to_output):
        self.fridge = Refrigerator()
        self.timestep = 5
        self.lookahead_window = int(60 / self.timestep)
        self.current_time = 0
        self.path_to_output = path_to_output
        self.outfile = open(path_to_output, 'w')
        self.data = moer_data
        #self.number_timesteps_to_process = self.data.shape[0]  # ALL data
        self.number_timesteps_to_process = 300
        self.add_fridge_co2_to_dataframe()
        self.add_timeslotIDs_to_dataframe()
        print(self.data.head(10))

    def add_fridge_co2_to_dataframe(self):
        megawatts_per_watt = 1 / 1000000
        hours_per_minute = 1 / 60
        self.data['lbs_co2'] = self.data['MOER'].apply(
            lambda moer: moer * (self.fridge.wattage * megawatts_per_watt) * (self.timestep * hours_per_minute)
        )

    def add_timeslotIDs_to_dataframe(self):
        def process_timestamp(timestamp):
            time_part = timestamp.split(" ")[1]
            return "".join(time_part.split(":")[:2])
        self.data['timeslotID'] = self.data['timestamp'].apply(process_timestamp)

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

    def run_simulation_without_data(self):
        self.fridge = Refrigerator()
        self.outfile.write("time,fridge_temp,fridge_on,moer,lbs_co2\n")  # write CSV headers

        for timestep in self.data.head(self.number_timesteps_to_process).index:
            expected_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            if expected_temp <= 33:
                self.fridge.turn_off()
            elif expected_temp >= 43:
                self.fridge.turn_on()

            moer = self.data.iloc[timestep]["MOER"]
            #print("moer: ", moer)

            # write row for timestep
            self.outfile.write(self.generate_output_row(moer))

            self.fridge.current_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            self.current_time += self.timestep
            self.fridge.current_timestamp = self.current_time

        self.outfile.close()

    def run_simulation_with_forecast(self):
        self.fridge = Refrigerator()
        self.outfile.write("time,fridge_temp,fridge_on,moer,lbs_co2\n")  # write CSV headers
        for timestep in self.data.head(self.number_timesteps_to_process).index:
            decision = self.get_next_decision(timestep, self.lookahead_window)
            if decision == 0:
                self.fridge.turn_off()
            else:
                self.fridge.turn_on()

            # write row for timestep
            moer = self.data.iloc[timestep]["MOER"]
            self.outfile.write(self.generate_output_row(moer))

            # update variables
            self.fridge.current_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            self.current_time += self.timestep
            self.fridge.current_timestamp = self.current_time
        self.outfile.close()

    def get_next_decision(self, start_timestep, num_timesteps):
        model = LpProblem("CO2_Minimization_Problem", LpMinimize)

        co2_vector = self.data['lbs_co2'][start_timestep : start_timestep + num_timesteps]

        variable_suffixes = [str(i) for i in range(co2_vector.size)]
        #print("Variable Indices:", variable_suffixes)
        decision_variables = LpVariable.matrix("status", variable_suffixes, cat="Binary")
        status_vector = np.array(decision_variables)
        #print("Decision Variables: ")
        #print(status_vector)

        obj_func = lpSum(status_vector * co2_vector)
        #print(obj_func)
        model += obj_func

        temp_variables = LpVariable.matrix("temp", variable_suffixes, cat="Continuous")
        cooling_rate_per_min = -(10 / 60) * 5
        warming_rate_per_min = (5 / 60) * 5

        for i in range(co2_vector.size - 1):
            model += temp_variables[i + 1] == temp_variables[i] + cooling_rate_per_min * status_vector[i] + \
                     warming_rate_per_min * (1 - status_vector[i])
            model += temp_variables[i + 1] >= 33
            model += temp_variables[i + 1] <= 43
        model += temp_variables[0] == self.fridge.current_temp  # starting temp of fridge
        #print(model)

        model.solve()
        #model.solve(PULP_CBC_CMD())

        #status = LpStatus[model.status]
        #print(status)

        # Decision Variables
        #print("status_0 value: ", model.variablesDict()['status_0'].value())
        '''
        for v in model.variables():
            try:
                print(v.name, "=", v.value())
            except:
                print("error couldnt find value")

        print("Total Cost:", model.objective.value())
        '''
        return model.variablesDict()['status_0'].value()

    def run_simulation_with_historicals(self):
        self.fridge = Refrigerator()
        self.outfile.write("time,fridge_temp,fridge_on,moer,lbs_co2\n")  # write CSV headers
        for timestep in self.data.head(self.number_timesteps_to_process).index:
            decision = self.get_next_decision(timestep, self.lookahead_window)
            if decision == 0:
                self.fridge.turn_off()
            else:
                self.fridge.turn_on()

            # write row for timestep
            moer = self.data.iloc[timestep]["MOER"]
            self.outfile.write(self.generate_output_row(moer))

            # update variables
            self.fridge.current_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            self.current_time += self.timestep
            self.fridge.current_timestamp = self.current_time
        self.outfile.close()