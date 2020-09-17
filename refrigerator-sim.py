import pandas as pd

path_to_moer_data = "MOER_Data/MOERS.csv"

moer_data = pd.read_csv(path_to_moer_data)

print(moer_data.head())
