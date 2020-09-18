# Simplified AER Modeling Challenge  

## Overview
This is a simulator that models the operation of a refrigerator connected to a smart plug, with the goal of minimizing
the associated CO2 emissions.  The simulation runs for one month, during which time the algorithm governing when to turn
the refrigerator on or off has access to a perfect 1-hour forecast of the MOER (Marginal Operating Emissions Rate) of 
the energy grid.  The algorithm may also use historical data as it becomes available.
Subject to the following constraints and parameters:
- The fridge must remain between 33 and 43 degrees F.
- The starting temperature of the fridge is 33 degrees F.
- When on, the fridge consumes 200 watts.  When off, it consumes no electricity.
- When on, the fridge cools at a rate of 10 degrees F per hour.  When off, it warms at a rate of 5 degrees F per hour.
- There is no penalty for the number of times the fridge turns on or off.

## Run the simulation

Python Dependencies:
- pandas
- numpy
- PuLP
- matplotlib

From the top-level directory:
```bash
python3 refrigerator-sim.py
```

The simulation will take several minutes to complete.
The result of the simulation can be found in `./output_data`, and includes a CSV file and corresponding plot with time
on the x-axis.
The plot displays:
- The temperature of the refrigerator at each timestep, color-coded by whether the fridge was on or off at that time.
- The current MOER of the grid at each time. 
- Cumulative pounds of CO2 associated with the refrigerator’s energy consumption since the start of the simulation.

## Model Description
At a high level, the model chooses what to do (turn the fridge on or off) at each timestep by formulating the one-hour
forecast window as a linear programming problem and finding an optimal solution within the forecast window.  The first 
step of this optimal solution is then taken, at which time the simulation (and forecast horizon) has advanced one step.
The process is repeated with the new forecast. 

Objective function to minimize:

    co2_emissions = (MOER_data_of_forecast)*(adjusted_for_fridge_wattage)*(bitmap_of_fridge_status)
    
Subject to the following constraints, where `X_t` is the temperature of the fridge at time t and `S_t` is the on/off 
status of the fridge at time t:

    X_t+1 = X_t + cooling_rate * S_t + warming_rate * (1 - S_t)
    X_t >= 33
    X_t <= 43
   For all t.
    
Historical average data is gradually taken into account as it becomes available. As the simulation progresses, the
dataframe is updated with new historical averages for each timestep *based on that timestep's time of day*.  In other 
words, each timestep is associated with a particular time of day, and the historical averages for that "slot" in the day
are constructed from historical data occurring at the same time of day.

The historical average is used as the predicted MOER in order to extend the length of the forecast window in the linear 
programming problem, placing more emphasis on these historical predictions as the number of datapoints included in the 
average increases. 


## Future improvements

Given more time, there are several improvements I would make to the simulator:

The model:
- Experiment with greater timestamp granularity for decision-making.  I used the timestep size provided in the MOER data
(5 minutes).  However, a more granular dataset could be created and would potentially provide better results, though the
larger number of decision variables in the one-hour forecast window could quickly make this computationally infeasible.
- Experiment with the number of predicted MOERs included in the forecast window by way of historical averages.
- Incorporate the first day’s pre-simulation data to populate the historical averages

The software:
- Parameterize the model with a config file, making it easier to adjust parameters in one place, such as:
    - The input data path
    - The number of timestep iterations to perform (shorten for quicker testing cycles)
    - The size of a timestep
    - Which model(s) to run
    - The fridge params (valid range, starting temp, cooling and heating rates, wattage)
    - Lookahead duration
    - How far into the future to use historical average data
- Use this config design to create a driver program capable of running the simulation under many different parameter
assignments, searching for a more optimal model.
- Add timestamps to output filenames to avoid overwriting.
- Unit tests!!
- Always room for organizational refactors.