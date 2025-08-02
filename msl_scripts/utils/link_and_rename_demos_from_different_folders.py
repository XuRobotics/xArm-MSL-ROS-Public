import os
from pathlib import Path

# Define directories
first_dir = Path('./first_90')
second_dir = Path('./second_30')

# Get sorted list of .bag files
first_bags = sorted(first_dir.glob('*.bag'))
second_bags = sorted(second_dir.glob('*.bag'))

assert len(first_bags) % 3 == 0, "first_90 should have a multiple of 3 files"
assert len(first_bags) // 3 == len(second_bags), "Mismatch between first_90 and second_30 demo count"

num_demos = len(second_bags)

for i in range(num_demos):
    demo_prefix = f'demo_{i+1:02d}_'
    
    # Rename 3 files in first_90
    for j in range(3):
        file = first_bags[i * 3 + j]
        new_name = file.parent / f'{demo_prefix}{file.name}'
        file.rename(new_name)
    
    # Rename 1 file in second_30
    file = second_bags[i]
    new_name = file.parent / f'{demo_prefix}{file.name}'
    file.rename(new_name)

print(f"Renamed files in place (with prefix) for {num_demos} demos.")
