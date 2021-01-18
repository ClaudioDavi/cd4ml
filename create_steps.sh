dvc add --external gs://$DATA_BUCKET/dvc/chicago-crime.csv

dvc run -n fetch_data \
        -d gs://gofind-datalake/dvc/chicago-crime.csv \
          -o data/chicago-crime.csv \
          gsutil cp gs://gofind-datalake/dvc/chicago-crime.csv data/chicago-crime.csv