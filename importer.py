import sqlite3
import pandas as pd

# 1. pulling in the fresh dataset.
print("loading new dataset...")
new_df = pd.read_csv('ncr_ride_bookings.csv')

# 2. hooking into the local sqlite database.
conn = sqlite3.connect('uber_operations.db')

print("swapping old data for new data in the database...")
new_df.to_sql('rides', conn, if_exists='replace', index=False)

conn.close()
print("database updated!")