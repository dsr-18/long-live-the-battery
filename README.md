# long-live-the-battery

Predicting total battery cycle life time with [TensorFlow 2](https://www.tensorflow.org/beta). We're going to publish a blog post describing the project in-depth soon.

This project is based on the work done in the paper ['Data driven prediciton of battery cycle life before capacity degradation'](https://www.nature.com/articles/s41560-019-0356-8) by K.A. Severson, P.M. Attia, et al., and uses the corresponding data set. The original instructions for how to load the data can be found [here](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation).


## Setup

We recommend to set up a virtual environment using a tool like [Virtualenv](https://virtualenv.pypa.io/en/latest/).

Clone this repo
```
git clone https://github.com/dsr-18/long-live-the-battery
```
and install dependencies.
```
pip install -r requirements.txt
```

You can download the processed dataset [here](https://github.com/dsr-18/long-live-the-battery-dataset) and jump to *Train Model*. If you want to reproduce the data preprocessing step, go ahead with *Generate Local Data*.


## Generate Local Data

Before running the model, generate local data:

1. Download the original three batch files [here](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) into a data directory like this:
```
long-live-the-battery
├── data
|   ├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
|   ├── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
|   └── 2017-06-30_batchdata_updated_struct_errorcorrect.mat
```
2. Make sure *data* is empty otherwise. Then from the base directory run
```
python -m trainer.data_preprocessing
```
to create a *processed_data.pkl* file.

3. Then run
```
python -m trainer.data_pipeline
```
to recreate the *.tfrecord* files.


## Train Model

Make sure the *.tfrecord* files are saved in this structure:
```
long-live-the-battery
├── data
|   ├── tfrecords
|   |   ├── scaling_factors.csv
|   |   ├── train
|   |   ├── test
|   |   └── secondary_test
```
To start training the model, run this command from the base directory:
```
python -m trainer.task
```
The default is set to three epochs which is okay for testing, but too short to train a reasonably fit model. Use the above command with the *--num-epochs* flag to set a higher number. To get a list of other parameters, use the *--help* flag.

To run the model in Google Cloud Platform (team members only):
1. Make sure you have access to the ion-age project.
2. Install [GCloud SDK](https://cloud.google.com/sdk/docs/). 
3. Run from base directory (with -h to see configurable options):
```
./train.sh
```
Follow the output URL to stream logs.


## Predict

Every training run saves a TensorBoard logfile and at least one model checkpoint by default in the *Graph* directory. One way to test your model's performance without writing your own [TensorFlow Keras code](https://www.tensorflow.org/beta/guide/keras/training_and_evaluation) is to start a local Flask server that serves predictions:
1. Copy any model from a *checkpoints* directory within *Graph* to the *server* directory.
2. Rename that model folder to *saved_model*.
3. Create sample data to predict on:
```
python generate_json_samples.py
```
4. Go into the server and start it from there:
```
cd server
python server.py
```
5. Now visit "localhost:5000" in your browser and you should see the start page with a prompt to upload battery data in json-format. The site also lets you select the  sample data randomly.
