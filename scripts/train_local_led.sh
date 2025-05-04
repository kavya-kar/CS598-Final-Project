#!/bin/bash

# Set paths
model="allenai/led-large-16384"
project="/c/Users/satej_5nayuru/CS 598/CS598-Final-Project/mimic-iv-note-di-bhc"
data_path="${project}/dataset"
output_path="${project}/models/led-large-16384/dropout_${1}_lr_${2}"

# Create output dir
mkdir -p "$output_path"

# Train using conda run
conda run -n ps_llms python ../summarization/run_summarization.py \
  --model_name_or_path "${model}" \
  --do_train --do_eval --do_predict \
  --train_file "${data_path}/train.json" \
  --validation_file "${data_path}/valid.json" \
  --test_file "${data_path}/test.json" \
  --output_dir "${output_path}" \
  --max_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 1 \
  --save_steps 1 \
  --load_best_model_at_end \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --dropout "${1}" \
  --learning_rate "${2}" \
  --predict_with_generate \
  --max_source_length 512 \
  --max_target_length 128
