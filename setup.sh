echo setting up environment...

rm -rf env
rm -rf .dvc

python3 -m venv env/

ls -lah env/

source env/bin/activate

which python

echo installing dvc...

pip install 'dvc[gs,aws]'

echo initializing dvc...

dvc init

pip install -r requirements.txt

dvc remote add gscache gs://$CACHE_BUCKET/cache
dvc config cache.gs gscache
dvc add --external gs://$DATA_BUCKET/dvc/chicago-crime.csv

dvc run -n fetch_data \
        -d gs://gofind-datalake/dvc/chicago-crime.csv \
          --external \
          -o data/chicago.csv \
          gsutil cp gs://gofind-datalake/dvc/chicago-crime.csv data/chicago-crime.csv 