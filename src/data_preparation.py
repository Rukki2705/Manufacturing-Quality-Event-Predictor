import pandas as pd
import numpy as np
import random
from faker import Faker
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

# Number of records
n_records = 50000

# Simulated fields
batch_ids = [f"BATCH_{i:05}" for i in range(n_records)]
product_types = np.random.choice(['Type A', 'Type B', 'Type C'], size=n_records, p=[0.4, 0.35, 0.25])
machine_ids = np.random.choice(['M1', 'M2', 'M3', 'M4'], size=n_records)
operator_ids = np.random.choice(['OP001', 'OP002', 'OP003', 'OP004', 'OP005'], size=n_records)
shifts = np.random.choice(['Morning', 'Evening', 'Night'], size=n_records)

batch_sizes = np.random.randint(100, 1000, size=n_records)
start_times = [fake.date_time_between(start_date='-6M', end_date='now') for _ in range(n_records)]
end_times = [start + pd.Timedelta(hours=random.uniform(1, 8)) for start in start_times]

inspection_durations = [round(random.uniform(5.0, 60.0), 2) for _ in range(n_records)]
defect_counts = [np.random.poisson(lam=max(1, size//200)) for size in batch_sizes]

# Introduce label noise in rework_flag (5% flipped)
rework_flags = []
for count in defect_counts:
    flag = int(count > 5)
    if random.random() < 0.05:
        flag = 1 - flag  # flip flag
    rework_flags.append(flag)

# Generate base label using logic, then add 5% noise
quality_events = []
for defects, size, rework in zip(defect_counts, batch_sizes, rework_flags):
    label = int((defects / size) > 0.05 or rework)
    if random.random() < 0.05:
        label = 1 - label  # flip label
    quality_events.append(label)

# Assemble DataFrame
df = pd.DataFrame({
    'batch_id': batch_ids,
    'product_type': product_types,
    'machine_id': machine_ids,
    'operator_id': operator_ids,
    'shift': shifts,
    'batch_size': batch_sizes,
    'start_time': start_times,
    'end_time': end_times,
    'inspection_duration_min': inspection_durations,
    'defect_count': defect_counts,
    'rework_flag': rework_flags,
    'quality_event': quality_events
})

df['processing_time_min'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

# Reorder columns
df = df[[
    'batch_id', 'product_type', 'machine_id', 'operator_id', 'shift',
    'batch_size', 'start_time', 'end_time', 'processing_time_min',
    'inspection_duration_min', 'defect_count', 'rework_flag', 'quality_event'
]]

# Save to updated file
output_path = "data/raw/noisy_manufacturing_quality_dataset.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)


