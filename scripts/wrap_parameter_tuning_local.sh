#!/bin/bash

# Local base paths
project="/path/to/your/project/mimic-iv-note-di-bhc"
code="/path/to/your/clone/of/patient_summaries_with_llms"
model_name_dir="led-large-16384"
run_dir="mimic-iv-note-di-bhc_led-large-16384_4000_600_chars_100_valid"
output_path="${project}/models/${model_name_dir}/${run_dir}"

# Activate environment
source ~/.bashrc
conda activate ps_llm

# Loop over parameter combinations
for dropout in 0.05 0.1 0.2; do
  for learning_rate in 5e-4 1e-5 5e-5 1e-6 5e-6; do
    folder_name="dropout_${dropout}_learning_rate_${learning_rate}"
    experiment_path="${output_path}/${folder_name}"

    if [ ! -d "$experiment_path" ]; then
      echo "▶ Starting experiment: $experiment_path"
      mkdir -p "$experiment_path"
      bash ${code}/scripts/train_led_local.sh $dropout $learning_rate
    else
      echo "⏩ Experiment already exists: $experiment_path"
    fi
  done
done
