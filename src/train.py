#
# Tensorflow Pipeline without external libraries - Tensorflow and python std only
#

import os
import csv
import sys
import tensorflow as tf


primary_type_unique = set()
## load single column unique
with open(FILE_NAME, "r") as f:
    for i, row in enumerate(csv.DictReader(f, COLUMNS)):
        # Skip Header
        if i == 0:
            continue
        primary_type_unique.add(row["primary_type"])


## Feature Columns:

primary_type = tf.feature_column.categorical_column_with_vocabulary_list(
    key="primary_type", vocabulary_list=list(unique_primary), num_oov_buckets=5
)
location_description = tf.feature_column.categorical_column_with_hash_bucket(
    key="location_description", hash_bucket_size=250
)
domestic = tf.feature_column.categorical_column_with_vocabulary_list(
    key="domestic", vocabulary_list=["True", "False"], num_oov_buckets=1
)
year = tf.feature_column.categorical_column_with_vocabulary_list(
    key="year", vocabulary_list=df.year.unique(), num_oov_buckets=10
)
description = tf.feature_column.categorical_column_with_hash_bucket(
    key="description", hash_bucket_size=250
)
community_area = tf.feature_column.categorical_column_with_hash_bucket(
    key="community_area", hash_bucket_size=250
)
ward = tf.feature_column.categorical_column_with_hash_bucket(
    key="ward", hash_bucket_size=250
)
beat = tf.feature_column.categorical_column_with_hash_bucket(
    key="beat", hash_bucket_size=250
)
district = tf.feature_column.categorical_column_with_hash_bucket(
    key="district", hash_bucket_size=250
)
