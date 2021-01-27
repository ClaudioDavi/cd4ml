import tensorflow as tf
import json
import yaml
import sys

TEST_FILE_NAME = sys.argv[1]
MODEL_DIR = sys.argv[2]
METRICS = sys.argv[3]

params = yaml.safe_load(open("params.yaml"))["evaluate"]

test_ds = tf.data.experimental.make_csv_dataset(
    TEST_FILE_NAME,
    num_epochs=2,
    batch_size=params["batch_size"],
    label_name=params["label"],
    header=True,
    shuffle=True,
)

model = tf.keras.models.load_model(MODEL_DIR)

loss, accuracy = model.evaluate(test_ds)

metrics = json.load(open(METRICS, "r"))

if accuracy < metrics.get("accuracy"):
    raise ValueError(
        "New Model did not perform better: Old acc: {} / new Acc: {}".format(
            metrics["accuracy"], accuracy
        )
    )
else:
    json.dump({"accuracy": accuracy, "loss": loss}, open("baseline.json", "w"))
    tf.saved_model.save(model, "serving/")