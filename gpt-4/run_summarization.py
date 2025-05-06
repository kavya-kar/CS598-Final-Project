import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import fire
import guidance
import yaml
from tqdm import tqdm

ALL_PROMPTS = {
    "prompt_1": """
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
You will be given a doctor's note and you will need to summarize the patient's brief hospital course.

Let's do a practice round.
{{~/user}}

{{#assistant~}}
Sounds great!
{{~/assistant}}

{{#each icl_examples}}
{{#user}}Here is the doctor's note on a patient's brief hospital course:

{{this.text}}

Summarize for the patient what happened during the hospital stay based on this doctor's note. Please make it short and concise and only include key events and findings. 
{{/user}}
{{#assistant}}
{{this.summary}}
{{/assistant}}
{{/each}}


{{#user~}}
Here is the doctor's note on a patient's brief hospital course:

{{final_text}}

Summarize for the patient what happened during the hospital stay based on this doctor's note. Please make it short and concise and only include key events and findings. 
{{~/user}}

{{#assistant~}}
{{gen 'summary' max_tokens=600 temperature=0}}
{{~/assistant}}
""",
    "prompt_2": """
{{#system~}}
You are helping with a resident working at a large urban academic medical center.
{{~/system}}

{{#user~}}
You task is to help summarize a patient's brief hospital course based on the doctor's note. Please make it short and concise and only include key events and findings. 

Here are some examples:

{{#each icl_examples}}
DOCUMENT: 
{{this.text}}

SUMMARY: 
{{this.summary}}
{{/each}}

Here is another doctor note on a patient's brief hospital course:

DOCUMENT: {{final_text}}
{{~/user}}

{{#assistant~}}
{{gen 'summary' max_tokens=600 temperature=0}}
{{~/assistant}}
""",
    "prompt_3": """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
You will be given some doctor's notes and you will need to summarize the patient's brief hospital course in one paragraph. Please only include key events and findings and avoid using medical jargons, and you MUST start the summary with "You were admitted".

{{#if icl_examples}}
Here are some examples:

{{#each icl_examples}}
DOCUMENT: 
{{this.text}}

SUMMARY: 
{{this.summary}}
{{/each}}
{{/if}}

DOCUMENT: {{final_text}}
{{~/user}}

{{#assistant~}}
{{gen 'summary' max_tokens=600 temperature=0}}
{{~/assistant}}
    """,
    "prompt_3.1": """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
You will be given some doctor's notes and you will need to summarize the patient's brief hospital course in ONE paragraph with a few sentences. Please only include key events and findings and avoid using medical jargons, and you MUST start the summary with "You were admitted".

DOCUMENT: {{final_text}}
{{~/user}}

{{#assistant~}}
{{gen 'summary' max_tokens=600 temperature=0}}
{{~/assistant}}
    """,
}


def read_jsonl(file_name):
    with open(file_name, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(file_name, data):
    with open(file_name, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def load_oai_model(model_name, max_calls_per_min=4):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    
    model_kwargs = {
        "max_calls_per_min": max_calls_per_min,
    }
    
    # Only add organization if explicitly set in config
    if "openai_organization" in config and config["openai_organization"].strip():
        model_kwargs["organization"] = config["openai_organization"]
    
    # Create and return the model
    model = guidance.llms.OpenAI(model_name, **model_kwargs)
    
    return model


def run_summarization(
    task_id: int,
    prompt_id: int,
    model_name: str = "gpt-4-0125-preview",
    n_shot: int = 3,
    save_path: Optional[str] = None,
    what_for: str = "exp",
    verbose: bool = False,
    debug: bool = False,
):
    demonstrations = read_jsonl(
        f"summarization_data/{what_for}_{task_id}_in-context.json"
    )
    test_examples = read_jsonl(f"summarization_data/{what_for}_{task_id}_test.json")

    bad_demonstration_ids = []
    for i, demonstration in enumerate(demonstrations):
        if demonstration["summary"].startswith("He came to the"):
            bad_demonstration_ids.append(i)

    assert len(demonstrations) >= n_shot
    if n_shot < len(demonstrations):
        random.seed(32)
        indices = list(range(len(demonstrations)))
        random.shuffle(indices)
        indices = [i for i in indices if i not in bad_demonstration_ids]
        icl_examples = [demonstrations[i] for i in indices[:n_shot]]
    else:
        icl_examples = demonstrations

    # Print available models if verbose
    if verbose:
        print(f"Attempting to use model: {model_name}")
        try:
            import openai
            models = openai.Model.list()
            print("Available models:")
            for model in models.data:
                print(f"- {model.id}")
        except Exception as e:
            print(f"Could not list models: {e}")
    
    # Try to load the model
    try:
        llm = load_oai_model(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to gpt-3.5-turbo...")
        llm = load_oai_model("gpt-3.5-turbo")

    used_prompt = ALL_PROMPTS[f"prompt_{prompt_id}"]
    summarization_program_nshot = guidance(used_prompt)

    if verbose:
        print(f"Using {len(icl_examples)} ICL examples")
        print(icl_examples)

    if debug:
        test_examples = test_examples[:10]

    # Create a unique directory for results if needed
    results_dir = Path("summarization_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique filename for each run based on parameters
    if save_path is None:
        save_path = f"summarization_results/{model_name}_{what_for}{task_id}_results_prompt{prompt_id}_{n_shot}shot.jsonl"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(save_path.replace(".jsonl", "_icl.jsonl"), icl_examples)

    with open(save_path.replace(".jsonl", "_prompt.txt"), "w") as f:
        f.write(used_prompt)

    failure_indices = []
    all_results = []

    for example_idx in tqdm(range(len(test_examples))):
        example = test_examples[example_idx]

        try:
            gen_answer = summarization_program_nshot(
                icl_examples=icl_examples,
                final_text=example["text"],
                llm=llm,
                verbose=verbose,
            )
            
            summary = gen_answer.get("summary", "")
        except Exception as e:
            print(f"Failed to generate answer for example {example_idx}: {e}")
            summary = ""
            failure_indices.append(example_idx)

        result = {
            "index": example_idx,
            "question": example["text"],
            "summary": summary,
        }
        
        all_results.append(result)
        
    
        if verbose:
            print(f"Text: {example['text']}")
            print(f"Summary: {summary}")
            print("=====================================")

    # Write all results to the main output file
    write_jsonl(save_path, all_results)

    with open(save_path.replace(".jsonl", "_failures.json"), "w") as f:
        json.dump(failure_indices, f, indent=2)
        
    print(f"All results written to: {save_path}")



if __name__ == "__main__":
    # Define all available OpenAI models to try
    AVAILABLE_MODELS = [
        "gpt-4-1106-preview",  # GPT-4 Turbo
        "gpt-4-0125-preview",  # GPT-4 Turbo
        "gpt-4-turbo-preview",  # Latest GPT-4 Turbo
        "gpt-4.1",               # Base GPT-4
        "gpt-3.5-turbo",       # GPT-3.5 Turbo
        "gpt-3.5-turbo-16k",   # GPT-3.5 Turbo with extended context
    ]
    
    # Add GPT-4.1 mini if specified
    if os.environ.get("USE_GPT4_MINI"):
        AVAILABLE_MODELS.insert(0, "gpt-4.1-mini")  # Add to the front of the list
    
    fire.Fire(run_summarization)
