import tensorflow as tf
from google.cloud import storage
import numpy as np
import trainer.evaluation as ev
import os

from trainer import constants as cst
from trainer import data_pipeline as dp


class CustomCheckpoints(tf.keras.callbacks.Callback):
    """
    Custom callback function with ability to save the model to GCP.
    The SavedModel contains:

    1) a checkpoint containing the model weights. (variables/)
    2) a SavedModel proto containing the Tensorflow backend 
    graph. (saved_model.pb)
    3) the model's json config. (assets/)

    For big models too many checkpoints can blow up the size of the
    log directory. To reduce the number of checkpoints, use the
    parameters below.
    
    log_dir: The base directory of all log files. Checkpoints
    will be saved in a "checkpoints" directory within this directory.
    
    dataset_path: data that is used for plotting the validation results.
    
    dataset_config: Same config file that is used for creating the training and
        validation datasets in train_and_evaluate(). This is needed to make the
        validation plot compareable.
    
    start_epoch: The epoch after which checkpoints are saved.
    
    save_best_only: Only save a model if it has a lower validation loss
    than all previously saved models.
    
    period: Save model only for every n-th epoch.
    """
    def __init__(self, log_dir, dataset_path, dataset_config, start_epoch=0, 
                 save_best_only=False, save_last_only=False, save_eval_plot=True, period=1):
        self.log_dir = log_dir
        self.start_epoch = start_epoch
        self.save_best_only = save_best_only
        self.save_last_only = save_last_only
        self.save_eval_plot = save_eval_plot
        self.period = period
        self.cloud_run = cst.BUCKET_NAME in log_dir
        if self.cloud_run:
            self.client = storage.Client()
            self.bucket = self.client.get_bucket(cst.BUCKET_NAME)  # Only used for saving evaluation plots
        if self.save_eval_plot:
            self.validation_dataset = dp.create_dataset(data_dir=dataset_path,
                                                        window_size=dataset_config["window_size"],
                                                        shift=dataset_config["shift"],
                                                        stride=dataset_config["stride"],
                                                        batch_size=dataset_config["batch_size"],
                                                        cycle_length=1,  # Has to be set for plotting
                                                        num_parallel_calls=1,  # Has to be set for plotting
                                                        shuffle=False,  # Has to be set for plotting
                                                        repeat=False)  # Has to be set for plotting
    
    def on_train_begin(self, logs=None):
        self.last_saved_epoch = None
        self.lowest_loss = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        self.current_loss = logs.get('val_loss')
        if (epoch % self.period == 0) and (epoch >= self.start_epoch) and not self.save_last_only:
            self.checkpoint_dir = os.path.join(self.log_dir,
                                               "checkpoints", "epoch_{}_loss_{}".format(epoch, self.current_loss))
            if self.save_best_only:
                if self.current_loss < self.lowest_loss:
                    tf.keras.experimental.export_saved_model(self.model, self.checkpoint_dir)
                    if self.save_eval_plot:
                        self._save_evaluation_plot(self.model, self.checkpoint_dir, self.validation_dataset)
                    self.lowest_loss = self.current_loss
                    self.last_saved_epoch = epoch
            else:
                tf.keras.experimental.export_saved_model(self.model, self.checkpoint_dir)
                if self.save_eval_plot:
                    self._save_evaluation_plot(self.model, self.checkpoint_dir, self.validation_dataset)

    def on_train_end(self, logs=None):
        last_epoch_dir = os.path.join(self.log_dir, "checkpoints", "last_epoch_loss_{}".format(self.current_loss))
        tf.keras.experimental.export_saved_model(self.model, last_epoch_dir)
        if self.save_eval_plot:
            self._save_evaluation_plot(self.model, last_epoch_dir, self.validation_dataset)
        
    def _save_evaluation_plot(self, model, checkpoint_dir, dataset, file_name='validation_plot.html'):
        html_dir = os.path.join(checkpoint_dir, file_name)
        
        if self.cloud_run:
            scaling_factors = dp.load_scaling_factors(gcloud_bucket=self.bucket)
        else:
            scaling_factors = dp.load_scaling_factors()
            
        # Make a forward pass over the whole validation dataset and get the results as a dataframe
        val_results = ev.get_predictions_results(model, dataset, scaling_factors)
        
        # Plot the resutls with plotly and wrap the resulting <div> as a html string
        plot_div = ev.plot_predictions_and_errors(val_results)
        plot_html = "<html><body>{}</body></html>".format(plot_div)
        
        # Save the html either in google cloud or locally
        if self.cloud_run:
            # Splitting path and only taking the tail, because self.bucket already knows about the bucket location
            blob = self.bucket.blob(html_dir.split(cst.BUCKET_NAME + "/")[-1])
            blob.upload_from_string(plot_html, content_type="text/html")
        else:
            with open(html_dir, 'w') as f:
                f.write(plot_html)