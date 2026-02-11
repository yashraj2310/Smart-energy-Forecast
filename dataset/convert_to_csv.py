import pandas as pd

df = pd.read_csv("household_power_consumption.txt", sep=";")
df.to_csv("energy_data.csv", index=False)

print("✔ CSV created successfully!")
