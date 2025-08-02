import zarr
# Load replay_buffer.zarr
dataset_path = "/home/sam/bags/msl_bags/rosbag_parsed_data/data.zarr"
replay_buffer = zarr.open(dataset_path, mode='r')
# Print available datasets
print(replay_buffer.tree())
# # Print timestamps
# timestamps = replay_buffer['data/timestamp'][:]
# print(timestamps[:10])  # Show first 10 timestamps
# List all components inside replay_buffer.zarr
for key in replay_buffer['data']:
    data = replay_buffer[f'data/{key}']
    print(f"Component: {key}")
    print(f"  Shape: {data.shape}")
    print(f"  Data Type: {data.dtype}")
    print(f"  First 5 values:\n{data[:5]}")
    print("-" * 40)
# check what is in meta file
if 'meta' in replay_buffer:
    for key in replay_buffer['meta']:
        metadata = replay_buffer[f'meta/{key}'][:]
        print(f"Metadata: {key}")
        # print(metadata)
# check chunk size
print("-" * 40)
print(replay_buffer['data/action'].chunks)