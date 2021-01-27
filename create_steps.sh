# Addind the commands here for future reference.

dvc add --external gs://$DATA_BUCKET/dvc/chicago-crime.csv

dvc run -n fetch_data \
        -d gs://gofind-datalake/dvc/chicago-crime.csv \
          -o data/chicago-crime.csv \
          gsutil cp gs://gofind-datalake/dvc/chicago-crime.csv data/chicago-crime.csv

dvc run -n prepare \
        -d src/prepare.py -d data/chicago-crime.csv \
        -o data/chicago-test.csv -o data/chicago-train.csv -o data/chicago-val.csv \
        -p prepare.split,prepare.validate \
        python src/prepare.py data/chicago-crime.csv

dvc run -n train \
        -d src/train.py -d data/chicago-train.csv  -d data/chicago-val.csv -d data/chicago-test.csv \
        -o model/ \                
        -p train.ds_epoch,train.epoch,train.batch_size,train.label \
        -M metrics.json \
        python src/train.py data/chicago-train.csv data/chicago-test.csv data/chicago-val.csv