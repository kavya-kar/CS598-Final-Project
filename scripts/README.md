# Parameter Tuning Scripts

They utilized an SLURM cluster for multiple training jobs on both an LED transformer and Llama 2 models with both 7 and 70 billion parameters. The LED transformer is a baseline for this study.

Our code is contained within `train_local_led.sh` and `wrap_parameter_tuning_local.sh`

We did not use a cluster and trained only the LED transformer locally.

## Adjusting Paths

In both `train_local_led.sh` and `wrap_parameter_tuning_local.sh`, the `project` paths need to be updated locally.

In `wrap_parameter_tuning_local.sh`, the `code` path also needs to be updated locally.

## After Data Processing

This code should be run after processing the data.

In order to run our scripts, run the following:

```
cd scripts/
bash wrap_parameter_tuning_local.sh
```

This will start parameter tuning of the LED transformer and will output ROUGE scores and training times afterwards.

## Adjusting Training Paradigms

We were unable to run an exact simulation of the authors' experiments due to time and processing constraints.

To match the authors' specifications, replace the corresponding sections in each of the files as such:

`wrap_parameter_tuning_local.sh`:

```
for dropout in 0.05 0.1 0.2; do 
    for learning_rate in 5e-4 1e-5 5e-5 1e-6 5e-6; do
```

Create a file with the last 100 entries of `valid.json` in mimic-iv-note-di-bhc called `valid_last_100.json`.

`train_local_led.sh`:

```
conda run -n ps_llms python ../summarization/run_summarization.py \
	--model_name_or_path ${model} \
	--do_train --do_eval --do_predict \
	--train_file ${data_path}/train.json \
	--validation_file ${data_path}/valid_last_100.json \
	--test_file ${data_path}/valid_last_100.json \
	--output_dir ${output_path} \
	--max_steps 200000 \
	--evaluation_strategy steps \
	--eval_steps 20000 \
	--save_steps 20000 \
	--load_best_model_at_end \
	--per_device_train_batch_size=${batch_size} \
	--per_device_eval_batch_size=${batch_size} \
	--dropout ${1} \
	--learning_rate ${2} \
	--predict_with_generate \
	--max_source_length 4096 \
	--max_target_length 350
```