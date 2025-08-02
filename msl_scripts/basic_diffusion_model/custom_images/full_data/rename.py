import os
import re

# Adjust to your actual directory
image_dir = "./"

# Get all matching filenames
files = [f for f in os.listdir(image_dir) if re.match(r'IMG_\d{4}\.JPG$', f)]
files = sorted(files, key=lambda x: int(x[4:8]) if int(x[4:8]) >= 9981 else int(x[4:8]) + 10000)

# Flatten the list to start from IMG_9981 and wrap around
starting_index = next(i for i, f in enumerate(files) if f.startswith("IMG_9981"))
ordered_files = files[starting_index:] + files[:starting_index]

# Prepare renaming
view_count = 1
i = 0
while i + 2 < len(ordered_files):
    mapping = {
        ordered_files[i]:     f"view{view_count}_left.JPG",
        ordered_files[i + 1]: f"view{view_count}_right.JPG",
        ordered_files[i + 2]: f"view{view_count}_forward.JPG"
    }
    for old_name, new_name in mapping.items():
        src = os.path.join(image_dir, old_name)
        dst = os.path.join(image_dir, new_name)
        os.rename(src, dst)
        print(f"Renamed {old_name} → {new_name}")
    view_count += 1
    i += 3
