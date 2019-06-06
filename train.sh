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

TFRECORDS_DIR="gs://${BUCKET}/data/tfrecords/train/*tfrecord"

# trying to put TensorBoard logs in individual run dirs
TBOARD_LOGS_DIR="${PACKAGE_STAGING_PATH}${JOB_NAME}"


gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --package-path $PACKAGE_PATH \
    --module-name $MODULE_NAME \
    --region $REGION \
    --config $CONFIG_FILE \
    --stream-logs \
    -- \
    --tfrecords-dir $TFRECORDS_DIR \
    --tboard-dir $TBOARD_LOGS_DIR \

    # user-defined args go after the open '--'

    # TODO add validation set

