#!/bin/bash

# Check number of arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <n_batches>"
    exit 1
fi

# Extract arguments
n_batches=$1

# Main directories
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
data_dir="$(dirname "$code_dir")/Data/Processed"
results_dir="$(dirname "$code_dir")/Results"

# Reusable directories
logs_dir="$results_dir/logs" # reusable directories for individual batch outputs
batch_datasets_dir="$results_dir/batch_datasets" # reusable directories for individual batch outputs

# Timestamped output directory for this run
timestamp=$(date +"%Y-%m-%d")
run_dir="$results_dir/run_${timestamp}"
# run_dir="$results_dir/run_name" # insert custom run name

# Files for this run
config_file="$run_dir/config.yaml"
dataset_file="$run_dir/dataset.h5" # full datasets

# Ask user if they want to delete old reusable directories and files
if [ -d "$logs_dir" ] || [ -d "$batch_datasets_dir" ] || [ -d "$dataset_file" ]; then
    echo "⚠️  Existing directories/files detected:"
    [ -d "$logs_dir" ] && echo " - $logs_dir/"
    [ -d "$batch_datasets_dir" ] && echo " - $batch_datasets_dir/"
    [ -f "$dataset_file" ] && echo " - $dataset_file/"
    read -p "Do you want to delete them before proceeding? (y/n): " choice

    if [[ "$choice" == [Yy] ]]; then
        rm -rf "$logs_dir" "$batch_datasets_dir" "$dataset_file"
        mkdir -p "$logs_dir" "$batch_datasets_dir"
        echo "New reusable directories/files created."
    else
        echo "Proceeding without deletion. Existing files may cause conflicts."
    fi
fi
echo

mkdir -p "$run_dir" # creates directories (-p: no errors if they already exist)

# Write configuration to config.yaml (> creates file if doesn't exist, otherwise overwrites)
cat <<EOF > "$config_file"
code_dir: "$code_dir"
data_dir: "$data_dir"
results_dir: "$results_dir"

batch_datasets_dir: "$batch_datasets_dir"
run_dir: "$run_dir"
dataset_file: "$dataset_file"

serogroup: 'B'
mod_type: 'SCS_2stage'
A: 101
year_groups:
    - '2014-2019'
    - '2020'
    - '2021'
    - '2022'
    - '2023'
pandemic_years:
    - 2020
    - 2021
risk: 'Trotter'
INCLUDE_VAX: true

n_batches: $n_batches
EOF

# Define priors and save in config file (>> appends to file)
cat <<EOF >> "$config_file"
priors:
  beta: [0, 0.3] # instead of [0,1]
  delta_duration: [0.020833, 2.0]
  zeta2020: [0, 1]
  zeta2021: [0, 1]
EOF

echo "Created config file at $config_file"
echo

#################### START PIPELINE

pipeline_start=$(date +%s)

########## SIMULATE ALL
# Run full simulation until 2023
echo "Running simulation until 2023 over $n_batches batches..."
for ((i = 1; i <= n_batches; i++)); do
    nohup python3 -u "$code_dir/simulate.py" "$i" --config "$config_file" &> "$logs_dir/log_batch_${i}.txt" & # -u option: unbuffered, everything is written immediately
done
wait # wait for all background jobs to finish (needed because n_batches jobs are run concurrently) 
echo "All batches completed. Elapsed time: $(( $(date +%s) - pipeline_start )) seconds"
echo

# Run aggregation script
echo "Running aggregation on simulation results..."
nohup python3 -u "$code_dir/aggregate.py" "all" --config "$config_file" > "$logs_dir/log_aggregate.txt" 2>&1 # so that shell waits for script to finish
echo "Aggregation completed. Elapsed time: $(( $(date +%s) - pipeline_start )) seconds"
echo

rm -rf "$batch_datasets_dir"
mkdir -p "$batch_datasets_dir"

# Run risk parameter selection (selection based on IMD likelihood maximization over all years 2019-2023)
echo "Running risk parameter selection based on IMD likelihood maximization..."
for ((i = 1; i <= n_batches; i++)); do
    nohup python3 -u "$code_dir/find_risk_params.py" "$i" --config "$config_file" &> "$logs_dir/log_risk_batch_${i}.txt" &
done
wait
echo "All batches completed. Elapsed time: $(( $(date +%s) - pipeline_start )) seconds"
echo

# Aggregate risk parameters
echo "Running risk parameters aggregation..."
nohup python3 "$code_dir/aggregate.py" "risk" --config "$config_file" > "$logs_dir/log_risk_aggregate.txt" 2>&1
echo "Aggregation completed. Elapsed time: $(( $(date +%s) - pipeline_start )) seconds"
echo

# Filter risk parameters
echo "Running risk parameter filter on full dataset..."
nohup python3 -u "$code_dir/filter.py" "risk" --config "$config_file" > "$logs_dir/log_risk_filter.txt" 2>&1
echo "Filtering completed. Elapsed time: $(( $(date +%s) - pipeline_start )) seconds"
echo
