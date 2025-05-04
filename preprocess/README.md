# Create MIMIC-IV-Note-DI Dataset

## Create Initial File Structure

Create a folder to store the processed data inside of `/root/CS598-Final-Project`:

```
mkdir mimic-iv-note-di
cd mimic-iv-note-di
mkdir dataset
```

Create another folder to store the processed data where we use brief hospital course as a reference inside of `/root/CS598-Final-Project`:

```
mkdir mimic-iv-note-di-bhc
cd mimic-iv-note-di-bhc
mkdir dataset
```

## Process the MIMIC-IV Summaries

The processing script goes through several steps.
Run the following command inside of `/root/CS598-Final-Project` to create `mimic_processed_summaries.csv`
You will need to adjust the paths for the input file and output directory to match your system.

```
python -m preprocess.process_mimic_summaries --start_from_step 1 --input_file "C:\Users\satej_5nayuru\CS 598\CS598-Final-Project\note\discharge.csv" --output_dir "C:\Users\satej_5nayuru\CS 598\CS598-Final-Project\mimic-iv-note-di\dataset"
```

The resulting file still contains all comlumns of the MIMIC-IV-Note database and should be stored in the `output_dir`.
Based on this we can create different datasets using the complete hopsital course or only the brief hospital course as a reference.
We will create both here.

## Select Dataset Columns and Create Splits

To select the relevant columns and create dataset splits, we use a separate script `split_dataset.py`.
The following command will use the full `hospital course` as a reference and the preprocessed `summary` column as summary.
You will need to adjust the paths for the input file and output directory to match your system.

```
python -m preprocess.split_dataset --input_file "C:\Users\satej_5nayuru\CS 598\CS598-Final-Project\mimic-iv-note-di\dataset\mimic_processed_summaries.csv" --output_dir "C:\Users\satej_5nayuru\CS 598\CS598-Final-Project\mimic-iv-note-di\dataset" --hospital_course_column hospital_course --summary_column summary
```

Run the following command to create a separate version of the dataset using the shorter `brief_hospital_course` as a reference.
You will need to adjust the paths for the input file and output directory to match your system.

```
python -m preprocess.split_dataset --input_file "C:\Users\satej_5nayuru\CS 598\CS598-Final-Project\mimic-iv-note-di\dataset\mimic_processed_summaries.csv" --output_dir "C:\Users\satej_5nayuru\CS 598\CS598-Final-Project\mimic-iv-note-di-bhc\dataset" --hospital_course_column brief_hospital_course --summary_column summary
```

As a consequence, we have the jsonl files `all.json`, `train.json`, `valid.json`, `test.json` in the directories `/root/mimic-iv-note-di/dataset` and `/root/mimic-iv-note-di-bhc/dataset` for the full hospital course and the brief hospital course as references, respectively.
The authors focused on the brief hospital course data for this study.
