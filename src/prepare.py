#
# Native Train Test Split without external libraries
# params:
#   FILE_NAME: str
#   SPLIT_SIZE: float
#   VALIDATE: bool
#

import os
import csv
import sys
import yaml

params = yaml.safe_load(open("params.yaml"))["prepare"]

FILE_NAME = sys.argv[1]


SPLIT_SIZE = params["split"]
VALIDATE = params["validate"]

COLUMNS = [
    "primary_type",
    "description",
    "location_description",
    "arrest",
    "domestic",
    "beat",
    "district",
    "ward",
    "community_area",
    "year",
]

file_size = sum(1 for row in csv.reader(open(FILE_NAME)))

train_size = int(file_size * SPLIT_SIZE)
test_size = (
    int(file_size * (1 - SPLIT_SIZE))
    if not VALIDATE
    else int(file_size * ((1 - SPLIT_SIZE) / 2))
)


train = open("data/chicago-train.csv", "w")
test = open("data/chicago-test.csv", "w")
val = open("data/chicago-val.csv", "w")

with open(FILE_NAME) as f:
    train_writer = csv.DictWriter(train, COLUMNS)
    train_writer.writeheader()
    test_writer = csv.DictWriter(test, COLUMNS)
    test_writer.writeheader()
    val_writer = csv.DictWriter(val, COLUMNS)
    val_writer.writeheader()
    for i, row in enumerate(csv.DictReader(f, COLUMNS)):
        # skip header
        if i == 0:
            continue

        row["arrest"] = 1 if row["arrest"] == "true" else 0

        if i <= train_size:
            train_writer.writerow(row)
        if i > train_size and i <= train_size + test_size:
            test_writer.writerow(row)
        if VALIDATE and i > train_size + test_size and i <= file_size:
            val_writer.writerow(row)

train.close()
test.close()
val.close()