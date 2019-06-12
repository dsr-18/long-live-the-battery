# long-live-the-battery
Predicting total battery cycle life time with machine learning

## Install
Use the following commands to install dependencies for this project:
```
conda install --file requirements_conda.txt
pip install -r requirements_pip.txt
```

## Generate Local Data
Before running the model, generate local data by running:
1. Create a directory named /data in the base project directory with the three seed batch files (available for download ...)
2. Remove data/processed_data.pkl file if exists.  Run
```
python -m trainer.data_preprocessing
```
to create processed_data.pkl.
3. Remove data/tfrecords dir if exists.  Run
```
python -m trainer.data_pipeline
```
to recreate the tfrecords files.


## Run
To run the model locally, use the following command from the base directory:
```
python -m trainer.task
```

To run the model in Google Cloud Platform (Team Members Only):

1. Make sure you have access to the ion-age project
2. Install GCloud SDK (https://cloud.google.com/sdk/docs/)
3. Run from project dir (run with -h to see configurable options):
```
./train.sh
```
  .  Follow output URL to stream logs.