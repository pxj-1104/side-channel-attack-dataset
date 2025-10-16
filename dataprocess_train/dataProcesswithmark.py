import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Progress bar display


# Function to load and process a single file
def load_and_process_data(file_path, save_dir):
    # Get the parent directory name, used as the CSV filename
    parent_dir_name = os.path.basename(os.path.dirname(file_path))
    countskip = 0

    # Define the target columns to be parsed
    columns = ['cache-references', 'cache-misses', 'L1-dcache-loads',
               'L1-dcache-load-misses', "L1-icache-load-misses", 'LLC-loads', 'LLC-load-misses', 'LLC-store-misses']

    # List to store parsed data
    parsed_data = []

    # Temporary dictionary to store events at each timestamp
    event_row = {}

    # Current timestamp to distinguish different event groups
    current_time = None

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        skipflie = False
        for line in file:

            # Remove leading and trailing whitespace
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith("#") or line == "":
                continue

            # Skip files containing invalid counters
            if 'not count' in line:
                countskip += 1
                skipflie = True
                break

            # Split by whitespace; the data format is fixed and values are at defined positions
            parts = line.split()

            # Extract timestamp, count value, and event name
            time = parts[0]
            count = float(parts[1].replace(',', ''))  # Convert count to float and remove commas
            event = parts[2]

            # When the timestamp changes, store the previous group of events
            if current_time != time:
                if event_row:
                    # Store event data according to the defined columns; missing events default to 0
                    parsed_data.append([event_row.get(col, 0.0) for col in columns])
                # Reset the event dictionary and update the timestamp
                event_row = {}
                current_time = time

            # Map the count value to the corresponding event name
            if event == 'cache-references':
                event_row['cache-references'] = count
            elif event == 'cache-misses':
                event_row['cache-misses'] = count
            elif event == 'L1-dcache-loads':
                event_row['L1-dcache-loads'] = count
            elif event == 'L1-dcache-load-misses':
                event_row['L1-dcache-load-misses'] = count
            elif event == 'L1-icache-load-misses':
                event_row['L1-icache-load-misses'] = count
            elif event == 'LLC-loads':
                event_row['LLC-loads'] = count
            elif event == 'LLC-load-misses':
                event_row['LLC-load-misses'] = count
            elif event == 'LLC-store-misses':
                event_row['LLC-store-misses'] = count

        # Don’t forget to handle the last group of events
        if event_row:
            parsed_data.append([event_row.get(col, 0.0) for col in columns])

    if not skipflie:
        # Convert parsed data to a DataFrame
        df = pd.DataFrame(parsed_data, columns=columns)

        # Construct the output directory path
        relative_path = os.path.relpath(os.path.dirname(file_path), base_dir)  # Relative path
        output_dir = os.path.join(save_dir, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

        saveid = 0
        # Construct output CSV path; filename based on the parent folder name
        if int(parent_dir_name) > 518500:
            saveid = int(parent_dir_name) - 518500 + 52000
        else:
            saveid = int(parent_dir_name) - 26500
        output_csv_path = os.path.join(output_dir, f'{saveid}.csv')

        # Save DataFrame to CSV file
        df.to_csv(output_csv_path, index=False)
        return output_csv_path, countskip  # Return output path and countskip value
    else:
        return None, countskip  # Return None if the file was skipped


# Recursively get all target files under the base directory
def get_all_files(base_dir):
    all_files = []

    # Traverse directories to locate files named 'hardware_events.txt'
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'hardware_events.txt':
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    return all_files


# Process multiple files in parallel using threads
def process_files_in_parallel(base_dir, save_dir):
    # Retrieve all file paths
    all_files = get_all_files(base_dir)

    total_countskip = 0  # Aggregate countskip across all files

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Display progress bar using tqdm
        with tqdm(total=len(all_files), desc="Processing files", ncols=100) as pbar:
            futures = []
            # Submit tasks
            for file in all_files:
                future = executor.submit(load_and_process_data, file, save_dir)
                futures.append(future)

            # Wait for all tasks to complete and update progress
            for future in futures:
                result, countskip = future.result()  # Capture return values
                total_countskip += countskip  # Accumulate skipped file count
                pbar.update(1)  # Update progress bar

    print(f"Total countskip: {total_countskip}")  # Print total skipped file count


# Example: process the entire dataset
base_dir = '../data/initial_data/add_data-11-25'
save_dir = '../data/csv_data/add_data-12-18'

process_files_in_parallel(base_dir, save_dir)
