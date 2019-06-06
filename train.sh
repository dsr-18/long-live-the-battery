# standard args

BUCKET='ion_age_bucket'
REGION='europe-west1'

PACKAGE_PATH='trainer/'
MODULE_NAME='trainer.task'
CONFIG_FILE='config.yaml'

JOB_DIR="gs://${BUCKET}"
PACKAGE_STAGING_PATH="gs://${BUCKET}"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="ion_age_$now"

# user-specified args

TFRECORDS_DIR_TRAIN="gs://${BUCKET}/data/tfrecords/train/*tfrecord"
TFRECORDS_DIR_VALIDATE="gs://${BUCKET}/data/tfrecords/test/*tfrecord"

# put TensorBoard logs, saved models in individual run dirs
JOB_RUN_DIR="${PACKAGE_STAGING_PATH}/${JOB_NAME}"
SAVED_MODEL_DIR="${JOB_RUN_DIR}/saved_model"

# user-defined args go after the open '--'
gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --package-path $PACKAGE_PATH \
    --module-name $MODULE_NAME \
    --region $REGION \
    --config $CONFIG_FILE \
    --stream-logs \
    -- \
    --data-dir-train $TFRECORDS_DIR_TRAIN \
    --data-dir-validate $TFRECORDS_DIR_VALIDATE \
    --tboard-dir $JOB_RUN_DIR \
    --saved-model-dir $SAVED_MODEL_DIR

