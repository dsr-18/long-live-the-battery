# long-live-the-battery
Predicting total battery cycle life time with machine learning

## Install
Use the following commands to install dependencies for this project:
```
conda install --file requirements_conda.txt
pip install -r requirements_pip.txt
```

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