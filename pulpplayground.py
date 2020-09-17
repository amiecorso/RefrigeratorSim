from pulp import *
import pandas as pd
import numpy as np


path_to_moer_data = "MOER_Data/MOERS.csv"

data = pd.read_csv(path_to_moer_data).head(12)

def lbs_co2_if_fridge_on(moer):
    wattage = 200
    timestep = 5  # minutes
    megawatts_per_watt = 1 / 1000000
    hours_per_minute = 1 / 60
    return round(moer * (wattage * megawatts_per_watt) * (timestep * hours_per_minute), 8)

model = LpProblem("CO2_Minimization_Problem", LpMinimize)

data['lbs_co2'] = data['MOER'].apply(lbs_co2_if_fridge_on)

co2_vector = data['lbs_co2']

variable_suffixes = [str(i) for i in range(co2_vector.size)]
print("Variable Indices:", variable_suffixes)
DV_variables = LpVariable.matrix("status", variable_suffixes, cat="Binary")
status_vector = np.array(DV_variables)
print("Decision Variable/Allocation Matrix: ")
print(status_vector)

obj_func = lpSum(status_vector * co2_vector)
print(obj_func)
model += obj_func


temp_variables = LpVariable.matrix("temp", variable_suffixes, cat="Continuous")
cooling_rate_per_min = -(10/60)
warming_rate_per_min = (5/60)

for i in range(co2_vector.size - 1):
    #print(temp_variables[i + 1] = temp_variables[i] + cooling_rate_per_min * status_vector[i] + warming_rate_per_min * status_vector[i])
    model += temp_variables[i + 1] == temp_variables[i] + cooling_rate_per_min * status_vector[i] + warming_rate_per_min * (1 - status_vector[i])
print(model)