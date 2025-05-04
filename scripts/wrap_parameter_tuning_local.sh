#!/bin/bash

# Local base paths
project="/c/Users/satej_5nayuru/CS 598/CS598-Final-Project/mimic-iv-note-di-bhc"
code="/c/Users/satej_5nayuru/CS 598/CS598-Final-Project"
model_name_dir="led-large-16384"
run_dir="mimic-iv-note-di-bhc_led-large-16384_8000_600_chars_100_valid"
output_path="${project}/models/${model_name_dir}/${run_dir}"

# Loop over parameter combinations
for dropout in 0.05 0.075 0.1; do
  for learning_rate in 1e-5 1e-6 3e-6; do
    folder_name="dropout_${dropout}_learning_rate_${learning_rate}"
    experiment_path="${output_path}/${folder_name}"

    if [ ! -d "$experiment_path" ]; then
      echo "Starting experiment: $experiment_path"
      mkdir -p "$experiment_path"
      bash "${code}/scripts/train_local_led.sh" $dropout $learning_rate
    else
      echo "Experiment already exists: $experiment_path"
    fi
  done
done