"""Script for generating shorter dataset for development process."""

OUTPUT_FILE = "./ONE_DAY_MOERS.csv"
print("Generating short dataset at {}.".format(OUTPUT_FILE))

with open("./MOERS.csv") as moerdata:
    moerdata.readline()  # headers
    data = moerdata.readlines()

    # decide how much data to grab
    num_timesteps = len(data)
    mins_per_step = 5
    total_duration_mins = mins_per_step * num_timesteps  # 5 mins per timestep
    total_duration_days = total_duration_mins / 60 / 24  # 33 days of total duration in full data
    timesteps_per_day = 24 * 60 / mins_per_step  # 288 timesteps per day

    # cut it down to one day of data (288 timesteps)
    with open(OUTPUT_FILE, "w") as short_moerdata:
        # Using the second day in the dataset, as the first day is "training data" and not part of the final sim
        short_data = data[288:288 + 288]
        short_moerdata.writelines(short_data)