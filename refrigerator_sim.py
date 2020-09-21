import argparse
import os
import pandas as pd
import subprocess
import time
from simulator import Simulator


def parse_args():
    """ Parses command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', default=False, help='Run all three models.')
    parser.add_argument('--no_data', action='store_true', default=False, help='Run model using no data.')
    parser.add_argument('--forecast_only', action='store_true', default=False,
                        help='Run model using 1-hr forecast window.')
    parser.add_argument('--forecast_and_history', action='store_true', default=False,
                        help='Run model using 1-hr forecast and historical avgs.')
    parser.add_argument('--moer_avgs', action='store_true', default=False,
                        help='Produce plot of average MOER data.')
    parser.add_argument('--data_path', action='store', default='MOER_data/MOERS.csv', help='Path to dataset.')
    parser.add_argument('--timesteps', action='store', default='all',
                        help='The number of timesteps to run for this simulation, defaults to size of dataset.')
    parser.add_argument('--clean', default=False, action='store_true',
                        help='Delete the current output data directory before starting simulations.')
    return parser.parse_args()


def end_timer(start_time, sim_id):
    """ Prints a formatted message indicating time elapsed since start_time for the simulation identified by sim_id.

    :param start_time: simulation start time, in seconds
    :param sim_id: a human-readable string identifying the simulation being timed
    """
    end_time = time.time()
    total_seconds = round(end_time - start_time, 2)
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    print("Simulation '{}' duration: {} min {} sec".format(sim_id, minutes, seconds))


if __name__ == '__main__':
    args = parse_args()

    # Create output dir for artifact collection
    output_dir = "./output_data/"
    if args.clean:
        subprocess.run(["rm", "-rf", output_dir])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read initial MOER data and cut off first day pre 3-1
    all_moer_data = pd.read_csv(args.data_path)
    initial_historical_data = all_moer_data[:288]
    sim_moer_data = all_moer_data[288:-1].reset_index(drop=True)

    # Number of timesteps (data rows) defaults to full dataset (shrink via command line args for shorter testing cycles)
    if args.timesteps == 'all':
        args.timesteps = sim_moer_data.shape[0]
    else:
        args.timesteps = int(args.timesteps)
    simulator = Simulator(sim_moer_data, output_dir, args.timesteps)

    # Run simulations based on arguments supplied at command line.
    # No args defaults to run_with_forecast_and_historical.
    if not args.no_data and not args.forecast_only and not args.forecast_and_history:
        args.forecast_and_history = True

    if args.moer_avgs:
        simulator.plot_moer_avgs()
        exit(1)

    if args.no_data or args.all:
        start_time = time.time()
        simulator.run_without_data()
        end_timer(start_time, 'no_data')

    if args.forecast_only or args.all:
        start_time = time.time()
        simulator.run_with_forecast()
        end_timer(start_time, 'forecast_only')

    if args.forecast_and_history or args.all:
        start_time = time.time()
        simulator.run_with_forecast_and_historical()
        end_timer(start_time, 'forecast_and_history')

