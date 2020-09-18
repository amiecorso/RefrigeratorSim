from pulp import *
import pandas as pd
import numpy as np

path_to_moer_data = "MOER_data/MOERS.csv"

all_moer_data = pd.read_csv(path_to_moer_data)
initial_historical_data = all_moer_data[:288]
data = all_moer_data[288:-1].reset_index(drop=True).head(300)

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
print("Decision Variable/status Matrix: ")
print(status_vector)

obj_func = lpSum(status_vector * co2_vector)
print(obj_func)
model += obj_func


temp_variables = LpVariable.matrix("temp", variable_suffixes, cat="Continuous")
cooling_rate_per_min = -(10/60) * 5
warming_rate_per_min = (5/60) * 5

for i in range(co2_vector.size - 1):
    model += temp_variables[i + 1] == temp_variables[i] + cooling_rate_per_min * status_vector[i] + warming_rate_per_min * (1 - status_vector[i])
    model += temp_variables[i + 1] >= 33
    model += temp_variables[i + 1] <= 43
model += temp_variables[0] == 33  # starting temp of fridge
print(model)

#model.solve()
model.solve(PULP_CBC_CMD())

status = LpStatus[model.status]

print(status)


# Decision Variables

#print(model.variables())
print(model.variablesDict()['status_0'].value())
for v in model.variables():
    try:
        print(v.name, "=", v.value())
    except:
        print("error couldnt find value")

print("Total Cost:", model.objective.value())