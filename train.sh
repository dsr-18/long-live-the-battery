#!/bin/bash

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
JOB_RUN_DIR="${PACKAGE_STAGING_PATH}/jobs/${JOB_NAME}"


# parse command-line args
params=()
while getopts ":hw:e:b:s:t:l:o:v:z:f:m:" opt; do
    case $opt in
        h)
            printf "Options:\n\t -w window-size\
                            \n\t -e num-epochs\
                            \n\t -b batch-size\
                            \n\t -s shift\
                            \n\t -t stride\
                            \n\t -l loss\
                            \n\t -o optimizer\
                            \n\t -v verbosity\
                            \n\t -f save-from\
                            \n\t -m model\
                            \n\t -z shuffle-buffer\n" >&2
            exit 1
            ;;
        w)
            params+=(--window-size $OPTARG)
            ;;
        e)
            params+=(--num-epochs $OPTARG)
            ;;
        b)
            params+=(--batch-size $OPTARG)
            ;;
        s)
            params+=(--shift $OPTARG)
            ;;
        t)
            params+=(--stride $OPTARG)
            ;;
        l)
            params+=(--loss $OPTARG)
            ;;
        v)
            params+=(--verbosity $OPTARG)
            ;;
        z)
            params+=(--shuffle-buffer $OPTARG)
            ;;
        f)
            params+=(--save-from $OPTARG)
            ;;
        m)
            params+=(--model $OPTARG)
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

echo "PARAMS ${params[@]}"


# issue train command to gcloud
# user-defined args go after the open '--'

gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --package-path $PACKAGE_PATH \
    --module-name $MODULE_NAME \
    --region $REGION \
    --python-version 3.5 \
    --runtime-version 1.13 \
    --config $CONFIG_FILE \
    --stream-logs \
    -- \
    --data-dir-train $TFRECORDS_DIR_TRAIN \
    --data-dir-validate $TFRECORDS_DIR_VALIDATE \
    --tboard-dir $JOB_RUN_DIR \
    "${params[@]}"

