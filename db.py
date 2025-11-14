import pandas as pd
import pymysql
from getpass import getpass
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

password = getpass("Enter MySQL password: ")

# Connect to MySQL
conn = pymysql.connect(
    host="localhost",
    user="root",
    password=password,
    database="healthcareproject"
)

cursor = conn.cursor()

# 1️⃣ Show all tables
cursor.execute("SHOW TABLES")
print("Tables in database:")
for table in cursor.fetchall():
    print(table)

# 2️⃣ Load patients into pandas
patients_df = pd.read_sql("SELECT * FROM patients", conn)
print("\nPatients DataFrame preview:")
print(patients_df.head())

# Load medications into pandas
medications_df = pd.read_sql("SELECT * FROM medications", conn)
print("\nMedications DataFrame preview:")
print(medications_df.head())

# Load conditions into pandas
conditions_df = pd.read_sql("SELECT * FROM conditions", conn)
print("\nConditions DataFrame preview:")
print(conditions_df.head())

# Merge DataFrames
merged_df = pd.merge(patients_df, medications_df, on="patient_id", how="left")
merged_df = pd.merge(merged_df, conditions_df, on="patient_id", how="left")
print("\nMerged DataFrame preview:")
print(merged_df.head())

# Save merged DataFrame to CSV
merged_df.to_csv("merged_data.csv", index=False)

# Close connection
cursor.close()
conn.close()