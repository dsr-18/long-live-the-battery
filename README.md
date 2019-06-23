# long-live-the-battery
Predicting total battery cycle life time with machine learning [...]

This project is based on the work done in the paper ['Data driven prediciton of battery cycle life before capacity degradation' by K.A. Severson, P.M. Attia, et al.](https://www.nature.com/articles/s41560-019-0356-8), and uses the corresponding data set.  The original instructions for how to load the data can be found [here.](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)


## Install
Use the following commands to install dependencies for this project:
```
conda install --file requirements_conda.txt
pip install -r requirements_pip.txt
```

## Generate Local Data
Before running the model, generate local data:

0. Download the original batch files by following the link above.
1. Create a directory named /data in the base project directory with the three seed batch files.

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
Follow output URL to stream logs.


## Predict
Every run should save at least one checkpoint automatically. Local runs save checkpoints in the "Graph" directory. To start a local server that serves predictions, copy any "saved_model" in "Graph" to the "server" directory. Then go into the server directory and start the server from there:
```
cd server
python server.py
```
Now visit "localhost:5000" in your browser and you should see the start page with a prompt to upload battery data. You can download a sample data file here: LINK
