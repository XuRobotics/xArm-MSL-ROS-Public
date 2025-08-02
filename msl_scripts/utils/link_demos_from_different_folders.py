import os
from pathlib import Path

# Define source directories
first_dir = Path('./first_90')
second_dir = Path('./second_30')
output_dir = Path('.')

# Get sorted list of .bag files
first_bags = sorted(first_dir.glob('*.bag'))
second_bags = sorted(second_dir.glob('*.bag'))

assert len(first_bags) % 3 == 0, "first_90 should have a multiple of 3 files"
assert len(first_bags) // 3 == len(second_bags), "Mismatch between first_90 and second_30 demo count"

num_demos = len(second_bags)

for i in range(num_demos):
    demo_prefix = f'demo_{i+1:02d}_'
    
    # Link the 3 corresponding files from first_90
    for j in range(3):
        src = first_bags[i * 3 + j]
        dst = output_dir / f'{demo_prefix}{src.name}'
        if not dst.exists():
            os.symlink(src.resolve(), dst)
    
    # Link the 1 corresponding file from second_30
    src = second_bags[i]
    dst = output_dir / f'{demo_prefix}{src.name}'
    if not dst.exists():
        os.symlink(src.resolve(), dst)

print(f"Created symlinks for {num_demos} demos.")
