import json
import os
import tensorflow as tf
import trainer.constants as cst
from trainer.data_pipeline import create_dataset
from server.constants import NUM_SAMPLES, SAMPLES_DIR

"""Create sample files in json format from test data and save it in the server module.
These can be used by the 'load random sample' button as examples on the website.
"""

samples_fullpath = os.path.join('server',SAMPLES_DIR)

if not os.path.exists(samples_fullpath):
    os.makedirs(samples_fullpath)
    
dataset = create_dataset(cst.SECONDARY_TEST_SET,
                         window_size=20,
                         shift=1,
                         stride=1,
                         batch_size=1)
rows = dataset.take(NUM_SAMPLES)
for i, row in enumerate(rows):
    sample = {key: str(value.numpy().tolist()) for key, value in row[0].items()}
    with open(os.path.join(samples_fullpath, 'sample_input_{}.json'.format(i+1)), 'w') as outfile:
        json.dump(sample, outfile)
print("Created {} sample files in server/static/samples".format(NUM_SAMPLES))
    