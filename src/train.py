#
# Tensorflow Pipeline without external libraries - Tensorflow and python std only
#

import csv
import os
import sys
import json
from functools import partial
import time

import tensorflow as tf
import yaml

params = yaml.safe_load(open("params.yaml"))["train"]

TRAIN_FILE_NAME = sys.argv[1]
TEST_FILE_NAME = sys.argv[2]
VAL_FILE_NAME = sys.argv[3]


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

primary_type_unique = set()
year_unique = set()
## load single column unique
with open(TRAIN_FILE_NAME, "r") as f:
    for i, row in enumerate(csv.DictReader(f, COLUMNS)):
        # Skip Header
        if i == 0:
            continue
        primary_type_unique.add(row["primary_type"])
        year_unique.add(int(row["year"]))

## Feature Columns:

primary_type = tf.feature_column.categorical_column_with_vocabulary_list(
    key="primary_type", vocabulary_list=list(primary_type_unique), num_oov_buckets=5
)

domestic = tf.feature_column.categorical_column_with_vocabulary_list(
    key="domestic", vocabulary_list=["True", "False"], num_oov_buckets=1
)
year = tf.feature_column.categorical_column_with_vocabulary_list(
    key="year",
    vocabulary_list=list(year_unique),
    num_oov_buckets=10,
    dtype=tf.dtypes.int32,
)
location_description = tf.feature_column.categorical_column_with_hash_bucket(
    key="location_description", hash_bucket_size=250
)
description = tf.feature_column.categorical_column_with_hash_bucket(
    key="description", hash_bucket_size=250
)
community_area = tf.feature_column.categorical_column_with_hash_bucket(
    key="community_area", dtype=tf.dtypes.int32, hash_bucket_size=250
)
ward = tf.feature_column.categorical_column_with_hash_bucket(
    key="ward", dtype=tf.dtypes.int32, hash_bucket_size=250
)
beat = tf.feature_column.categorical_column_with_hash_bucket(
    key="beat", hash_bucket_size=250, dtype=tf.dtypes.int32
)
district = tf.feature_column.categorical_column_with_hash_bucket(
    key="district", dtype=tf.dtypes.int32, hash_bucket_size=250
)

primary_one_hot = tf.feature_column.indicator_column(primary_type)
year_one_hot = tf.feature_column.indicator_column(year)

location_description_embbed = tf.feature_column.embedding_column(
    location_description, 5
)
description_embbed = tf.feature_column.embedding_column(description, 5)
beat_embbed = tf.feature_column.embedding_column(beat, 5)
community_area_embbed = tf.feature_column.embedding_column(community_area, 5)
ward_embbed = tf.feature_column.embedding_column(ward, 5)
district_embbed = tf.feature_column.embedding_column(district, 5)


feature_columns = [
    primary_one_hot,
    year_one_hot,
    location_description_embbed,
    description_embbed,
    beat_embbed,
    community_area_embbed,
    ward_embbed,
    district_embbed,
]

batch_size = params["batch_size"]

train_ds = tf.data.experimental.make_csv_dataset(
    TRAIN_FILE_NAME,
    batch_size=batch_size,
    label_name=params["label"],
    num_epochs=1,
    header=True,
    shuffle=True,
    prefetch_buffer_size=1,
)

test_ds = tf.data.experimental.make_csv_dataset(
    TEST_FILE_NAME,
    num_epochs=1,
    batch_size=batch_size,
    label_name=params["label"],
    header=True,
    shuffle=True,
)


val_ds = tf.data.experimental.make_csv_dataset(
    VAL_FILE_NAME,
    batch_size=batch_size,
    num_epochs=1,
    label_name=params["label"],
    header=True,
    shuffle=True,
    prefetch_buffer_size=1,
)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

## Building inputs
in_primary_type = tf.keras.Input(
    name="primary_type", dtype=tf.dtypes.string, shape=(1,)
)
in_description = tf.keras.Input(name="description", dtype=tf.dtypes.string, shape=(1,))
in_location_description = tf.keras.Input(
    name="location_description", dtype=tf.dtypes.string, shape=(1,)
)
in_domestic = tf.keras.Input(name="domestic", dtype=tf.dtypes.string, shape=(1,))
in_beat = tf.keras.Input(name="beat", dtype=tf.dtypes.int32, shape=(1,))
in_district = tf.keras.Input(name="district", dtype=tf.dtypes.int32, shape=(1,))
in_ward = tf.keras.Input(name="ward", dtype=tf.dtypes.int32, shape=(1,))
in_community_area = tf.keras.Input(
    name="community_area", dtype=tf.dtypes.int32, shape=(1,)
)
in_year = tf.keras.Input(name="year", dtype=tf.dtypes.int32, shape=(1,))

model_input = {
    "primary_type": in_primary_type,
    "description": in_description,
    "location_description": in_location_description,
    "domestic": in_domestic,
    "beat": in_beat,
    "district": in_district,
    "ward": in_ward,
    "community_area": in_community_area,
    "year": in_year,
}

model = None

RegularizedDense = partial(
    tf.keras.layers.Dense,
    activation="relu",
    kernel_initializer="he_normal",
    kernel_regularizer=tf.keras.regularizers.l1_l2(),
)

model = tf.keras.layers.DenseFeatures(feature_columns)(model_input)
model = tf.keras.layers.BatchNormalization()(model)
model = RegularizedDense(100)(model)
model = RegularizedDense(75)(model)
model = RegularizedDense(10)(model)
model = tf.keras.layers.Dropout(0.1)(model)
model = tf.keras.layers.Dense(1)(model)

final_model = tf.keras.Model(model_input, model)


callback_list = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        "model",
        save_best_only=True,
    ),
]

final_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

final_model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callback_list)

loss, accuracy = final_model.evaluate(test_ds)

ts = int(time.time())

tf.saved_model.save(final_model, "model/{}".format(ts))

with open("metrics.json", "w") as f:
    json.dump({"loss": loss, "accuracy": accuracy}, f)
