# Run summarization with GPT-4 

## Setup 

```bash
pip install openai==0.27.0 guidance==0.0.64
```

1. Create OpenAI API account
2. In a config.yaml file add your OpenAI API key

## Run Summarization 

```bash
python3 run_summarization.py --task_id 1 --prompt_id 3.1 --model_name gpt-4.1 --n_shot 0 --verbose
python3 run_summarization.py --task_id 1 --prompt_id 3 --model_name gpt-4.1 --n_shot 2 --verbose
python3 run_summarization.py --task_id 2 --prompt_id 3.1 --model_name gpt-4.1 --n_shot 0 --verbose
python3 run_summarization.py --task_id 2 --prompt_id 3 --model_name gpt-4.1 --n_shot 5 --verbose
```

The following commands were used to generate the data, please adjust the paths to match your system:

```bash
tail -n 10 ./data/train.json > /Users/kavyakarthi/Documents/MCS/CS598/Final_Project/Code/final_proj/gpt-4/summarization_data/prompt_train.json
tail -n 10 ./data/valid.json > ~/Documents/MCS/CS598/Final_Project/Code/final_proj/gpt-4/summarization_data/prompt_valid.json

tail -n 10 ./data/train.json > ~/Documents/MCS/CS598/Final_Project/Code/final_proj/gpt-4/summarization_data/exp_1_in-context.json
tail -n 100 ./data/test.json > ~/Documents/MCS/CS598/Final_Project/Code/final_proj/gpt-4/summarization_data/exp_1_test.json
tail -n 10 ./data/train_8000_600_chars.json > ~/Documents/MCS/CS598/Final_Project/Code/final_proj/gpt-4/summarization_data/exp_2_in-context.json
tail -n 100 ./data/test_8000_600_chars.json > ~/Documents/MCS/CS598/Final_Project/Code/final_proj/gpt-4/summarization_data/exp_2_test.json
```
