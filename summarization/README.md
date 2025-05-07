# Summarization Models

This folder contains the scripts to run summarization model LED and the results from running them.

## Results from LED

### Dropout = 0.05, Learning Rate = 3e-6

```
{
    "epoch": 0.01,
    "eval_gen_len": 128.0,
    "eval_loss": 3.4683637619018555,
    "eval_rouge1": 18.1452,
    "eval_rouge2": 2.7252,
    "eval_rougeL": 11.3885,
    "eval_rougeLsum": 16.3425,
    "eval_runtime": 2869.7096,
    "eval_samples": 232,
    "eval_samples_per_second": 0.081,
    "eval_steps_per_second": 0.081,
    "test_gen_len": 127.691,
    "test_loss": 3.4945228099823,
    "test_rouge1": 18.5505,
    "test_rouge2": 3.0848,
    "test_rougeL": 11.0659,
    "test_rougeLsum": 16.6495,
    "test_runtime": 2864.8155,
    "test_samples": 233,
    "test_samples_per_second": 0.081,
    "test_steps_per_second": 0.081,
    "train_loss": 4.7792926788330075,
    "train_runtime": 16751.7375,
    "train_samples": 1860,
    "train_samples_per_second": 0.001,
    "train_steps_per_second": 0.001
}
```

### Dropout = 0.05, Learning Rate = 1e-5

```
{
    "epoch": 0.01,
    "eval_gen_len": 128.0,
    "eval_loss": 2.8753442764282227,
    "eval_rouge1": 23.4841,
    "eval_rouge2": 4.1543,
    "eval_rougeL": 13.4987,
    "eval_rougeLsum": 21.432,
    "eval_runtime": 5007.6507,
    "eval_samples": 232,
    "eval_samples_per_second": 0.046,
    "eval_steps_per_second": 0.046,
    "test_gen_len": 128.0,
    "test_loss": 2.8893251419067383,
    "test_rouge1": 23.5844,
    "test_rouge2": 4.5681,
    "test_rougeL": 13.63,
    "test_rougeLsum": 21.6026,
    "test_runtime": 3541.2244,
    "test_samples": 233,
    "test_samples_per_second": 0.066,
    "test_steps_per_second": 0.066,
    "train_loss": 4.194908523559571,
    "train_runtime": 22816.2096,
    "train_samples": 1860,
    "train_samples_per_second": 0.0,
    "train_steps_per_second": 0.0
}
```

### Dropout = 0.075, Learning Rate = 3e-6

```
{
    "epoch": 0.01,
    "eval_gen_len": 127.7328,
    "eval_loss": 3.5448336601257324,
    "eval_rouge1": 17.019,
    "eval_rouge2": 2.5872,
    "eval_rougeL": 10.682,
    "eval_rougeLsum": 15.3803,
    "eval_runtime": 4397.6448,
    "eval_samples": 232,
    "eval_samples_per_second": 0.053,
    "eval_steps_per_second": 0.053,
    "test_gen_len": 127.8197,
    "test_loss": 3.5745017528533936,
    "test_rouge1": 17.0368,
    "test_rouge2": 2.872,
    "test_rougeL": 10.6262,
    "test_rougeLsum": 15.3524,
    "test_runtime": 3729.5672,
    "test_samples": 233,
    "test_samples_per_second": 0.062,
    "test_steps_per_second": 0.062,
    "train_loss": 4.798184204101562,
    "train_runtime": 16583.234,
    "train_samples": 1860,
    "train_samples_per_second": 0.001,
    "train_steps_per_second": 0.001
}
```

### Dropout = 0.075, Learning Rate = 1e-5

```
{
    "epoch": 0.01,
    "eval_gen_len": 128.0,
    "eval_loss": 2.9064178466796875,
    "eval_rouge1": 21.9883,
    "eval_rouge2": 4.0338,
    "eval_rougeL": 13.0871,
    "eval_rougeLsum": 20.3379,
    "eval_runtime": 3509.3221,
    "eval_samples": 232,
    "eval_samples_per_second": 0.066,
    "eval_steps_per_second": 0.066,
    "test_gen_len": 128.0,
    "test_loss": 2.9225668907165527,
    "test_rouge1": 22.5875,
    "test_rouge2": 4.3756,
    "test_rougeL": 13.4189,
    "test_rougeLsum": 20.8281,
    "test_runtime": 3391.7353,
    "test_samples": 233,
    "test_samples_per_second": 0.069,
    "test_steps_per_second": 0.069,
    "train_loss": 4.373289489746094,
    "train_runtime": 21560.4554,
    "train_samples": 1860,
    "train_samples_per_second": 0.0,
    "train_steps_per_second": 0.0
}
```

### Dropout = 0.1, Learning Rate = 3e-6

```
{
    "epoch": 0.01,
    "eval_gen_len": 127.8276,
    "eval_loss": 3.586576223373413,
    "eval_rouge1": 16.2973,
    "eval_rouge2": 2.4734,
    "eval_rougeL": 10.3455,
    "eval_rougeLsum": 14.7998,
    "eval_runtime": 5489.8942,
    "eval_samples": 232,
    "eval_samples_per_second": 0.042,
    "eval_steps_per_second": 0.042,
    "test_gen_len": 127.97,
    "test_loss": 3.614063024520874,
    "test_rouge1": 17.7568,
    "test_rouge2": 2.9847,
    "test_rougeL": 10.8504,
    "test_rougeLsum": 15.8725,
    "test_runtime": 3168.4191,
    "test_samples": 233,
    "test_samples_per_second": 0.074,
    "test_steps_per_second": 0.074,
    "train_loss": 4.685829544067383,
    "train_runtime": 21596.6628,
    "train_samples": 1860,
    "train_samples_per_second": 0.0,
    "train_steps_per_second": 0.0
}
```

### Dropout = 0.1, Learning Rate = 1e-5

```
{
    "epoch": 0.01,
    "eval_gen_len": 128.0,
    "eval_loss": 2.902143955230713,
    "eval_rouge1": 22.2417,
    "eval_rouge2": 3.9459,
    "eval_rougeL": 13.2529,
    "eval_rougeLsum": 20.3599,
    "eval_runtime": 4795.299,
    "eval_samples": 232,
    "eval_samples_per_second": 0.048,
    "eval_steps_per_second": 0.048,
    "test_gen_len": 128.0,
    "test_loss": 2.9194791316986084,
    "test_rouge1": 22.6285,
    "test_rouge2": 4.409,
    "test_rougeL": 13.4966,
    "test_rougeLsum": 20.7576,
    "test_runtime": 3128.0668,
    "test_samples": 233,
    "test_samples_per_second": 0.074,
    "test_steps_per_second": 0.074,
    "train_loss": 4.199145889282226,
    "train_runtime": 21986.1391,
    "train_samples": 1860,
    "train_samples_per_second": 0.0,
    "train_steps_per_second": 0.0
}
```