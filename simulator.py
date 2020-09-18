from pulp import *
import numpy as np
from refrigerator import Refrigerator
from visualizer import Visualizer

class Simulator:
    def __init__(self, moer_data, output_dir, num_timesteps):
        self.visualizer = Visualizer(self)
        self.fridge = Refrigerator()
        self.historicals = {}
        self.timestep = 5
        self.lookahead_window = int(60 / self.timestep)
        self.avgs_granularity = "timestep"
        #self.avgs_granularity = "hour"
        self.current_time = 0
        self.output_dir = output_dir
        self.data = moer_data
        #self.number_timesteps_to_process = self.data.shape[0]  # ALL data
        self.number_timesteps_to_process = num_timesteps
        self.add_synthetic_fields_to_dataframe()
        self.data = self.data[:self.number_timesteps_to_process]

    def add_synthetic_fields_to_dataframe(self):
        # add CO2 consumed by fridge as function of MOER
        self.data['lbs_co2'] = self.data['MOER'].apply(self.lbs_co2_from_moer)

        # add "timeslotID" - to tie a timestamp to its relative time of day
        if self.avgs_granularity == "timestep":
            process_timestamp = lambda timestamp: "".join(timestamp.split(" ")[1].split(":")[:2])
            self.data['timeslotID'] = self.data['timestamp'].apply(process_timestamp)
        elif self.avgs_granularity == "hour":
            process_timestamp = lambda timestamp: "".join(timestamp.split(" ")[1].split(":")[:1])
            self.data['timeslotID'] = self.data['timestamp'].apply(process_timestamp)

        # create an empty historical avg column, and empty associated count
        self.data['hist_avg_moer_at_time'] = 0
        self.data['num_datapoints_in_avg'] = 0
        return

    def prepare_new_simulation(self, sim_id):
        self.fridge = Refrigerator()
        self.historicals = {}
        self.current_time = 0
        self.data['hist_avg_moer_at_time'] = 0
        self.data['num_datapoints_in_avg'] = 0
        output_filename = self.output_dir.rstrip("/") + "/" + "sim_output_" + sim_id + ".csv"
        self.outfile = open(output_filename, 'w')
        # write CSV headers
        self.outfile.write("time,fridge_temp,fridge_on,moer,lbs_co2,avg_moer_at_time,datapoints_in_avg\n")
        return output_filename

    def update_historical_avgs(self, timestep):
        timeslotID = self.data.iloc[timestep]['timeslotID']
        moer = self.data.iloc[timestep]['MOER']
        avg = moer
        count = 1
        if timeslotID in self.historicals:
            old_avg = self.historicals[timeslotID][0]
            count = self.historicals[timeslotID][1]
            avg = (count * old_avg + moer) / (count + 1)
            count += 1
        # Update running average for this time of day
        self.historicals[timeslotID] = (avg, count)
        # Update this timestep with new historical information
        self.data.loc[self.data.timeslotID == timeslotID, 'hist_avg_moer_at_time'] = avg
        self.data.loc[self.data.timeslotID == timeslotID, 'num_datapoints_in_avg'] = count
        return

    def generate_output_row(self, data_row):
        moer = data_row['MOER']
        lbs_co2 = self.lbs_co2_produced_this_timestep(moer)
        hist_avg_moer = data_row['hist_avg_moer_at_time']
        num_datapoints = data_row['num_datapoints_in_avg']

        row_data = [self.current_time, round(self.fridge.current_temp, 2), self.fridge.on, moer, lbs_co2,
                    hist_avg_moer, num_datapoints]
        str_row_data = [str(field) for field in row_data]
        return ",".join(str_row_data) + "\n"

    def lbs_co2_from_moer(self, moer):
        megawatts_per_watt = 1 / 1000000
        hours_per_minute = 1 / 60
        return round(moer * (self.fridge.wattage * megawatts_per_watt) * (self.timestep * hours_per_minute), 8)

    def lbs_co2_produced_this_timestep(self, moer):
        if not self.fridge.on:
            return 0
        return self.lbs_co2_from_moer(moer)

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

        for i in range(co2_vector.size - 1):
            model += temp_variables[i + 1] == temp_variables[i] + \
                     self.fridge.cooling_rate * self.timestep * status_vector[i] + \
                     self.fridge.warming_rate * self.timestep * (1 - status_vector[i])
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

    def get_next_decision_using_historicals(self, start_timestep, num_timesteps):
        model = LpProblem("CO2_Minimization_Problem", LpMinimize)

        # how much historical average data should be used?
        # - no more than an hour?  two hours?
        # - the refrigerator can last for a maximum of two hours in the off position if it starts at 33 degrees...
        # - so insight beyond two hours could be useful, but probably not a whole lot more than that.
        # One hour corresponds to 12 timestamps
        # by the end of the simulation, we'll have ~ (1440 mins per day / 5 mins per timesteps) = 288 timesteps per day
        # 9000 timesteps / 288 timesteps per day = ~31 datapoints per historical average by end of sim
        # so maybe the hist-avg lookahead length can be proportional in timesteps to the number of values used in the average...
        # shorter as we have less info, longer as we get more info, eventually extending to 31 timesteps or 2.6 hours past the
        # official lookahead window
        if self.avgs_granularity == "timestep":
            hist_lookahead_window = self.data['num_datapoints_in_avg'][start_timestep]
        elif self.avgs_granularity == "hour":
            hist_lookahead_window = int(self.data['num_datapoints_in_avg'][start_timestep] / 12) # timesteps/hour

        co2_vector_forecast = self.data['lbs_co2'][start_timestep : start_timestep + num_timesteps]
        moer_vector_hist_avg = self.data['hist_avg_moer_at_time'][start_timestep + num_timesteps :
                                                                 start_timestep + num_timesteps + hist_lookahead_window]
        co2_vector_hist_avg = moer_vector_hist_avg.apply(self.lbs_co2_from_moer)
        co2_vector = co2_vector_forecast.append(co2_vector_hist_avg)

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

        for i in range(co2_vector.size - 1):
            model += temp_variables[i + 1] == temp_variables[i] + \
                     self.fridge.cooling_rate * self.timestep * status_vector[i] + \
                     self.fridge.warming_rate * self.timestep * (1 - status_vector[i])
            model += temp_variables[i + 1] >= 33
            model += temp_variables[i + 1] <= 43
        model += temp_variables[0] == self.fridge.current_temp  # starting temp of fridge
        print(model)

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

    def run_simulation_without_data(self, show_plot=True):
        output_filename = self.prepare_new_simulation("no_forecasting")

        for timestep in self.data.head(self.number_timesteps_to_process).index:
            expected_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            if expected_temp <= 33:
                self.fridge.turn_off()
            elif expected_temp >= 43:
                self.fridge.turn_on()

            # write csv row for this timestep
            data_row = self.data.iloc[timestep]
            self.outfile.write(self.generate_output_row(data_row))

            self.fridge.current_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            self.current_time += self.timestep
            self.fridge.current_timestamp = self.current_time

            # update historicals dictionary and dataframe row
            self.update_historical_avgs(timestep)

        print("HISTORICALS: ", self.historicals)
        print(self.data.head()['hist_avg_moer_at_time'])
        print(self.data.head()['num_datapoints_in_avg'])

        self.outfile.close()
        self.visualizer.plot(output_filename, show_plot)
        self.visualizer.plot_avg_moers(output_filename, show_plot)
        return

    def run_simulation_with_forecast(self, show_plot=True):
        output_filename = self.prepare_new_simulation("with_forecasting")

        for timestep in self.data.head(self.number_timesteps_to_process).index:
            decision = self.get_next_decision(timestep, self.lookahead_window)
            if decision == 0:
                self.fridge.turn_off()
            else:
                self.fridge.turn_on()

            # write csv row for this timestep
            data_row = self.data.iloc[timestep]
            self.outfile.write(self.generate_output_row(data_row))

            self.fridge.current_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            self.current_time += self.timestep
            self.fridge.current_timestamp = self.current_time

            # update historicals dictionary and dataframe row
            self.update_historical_avgs(timestep)

        self.outfile.close()
        self.visualizer.plot(output_filename, show_plot)
        return

    def run_simulation_with_forecast_and_historicals(self, show_plot=True):
        output_filename = self.prepare_new_simulation("with_forecasting_and_historicals")

        for timestep in self.data.head(self.number_timesteps_to_process).index:
            decision = self.get_next_decision_using_historicals(timestep, self.lookahead_window)
            if decision == 0:
                self.fridge.turn_off()
            else:
                self.fridge.turn_on()

            # write csv row for this timestep
            data_row = self.data.iloc[timestep]
            self.outfile.write(self.generate_output_row(data_row))

            self.fridge.current_temp = self.fridge.expected_temp(self.current_time + self.timestep)
            self.current_time += self.timestep
            self.fridge.current_timestamp = self.current_time

            # update historicals dictionary and dataframe row
            self.update_historical_avgs(timestep)

        self.outfile.close()
        self.visualizer.plot(output_filename, show_plot)
        return