# DVC + Github Actions

_Note: this is a very experimental project, if you have any information and tips that can improve this, feel free to reach out and open an issue._

---

When creating machine learning projects we usually stumble upon a few very challenging problems:

1. Data and Model Versioning
2. Managing Experiments.
3. Model Deployment
4. Validation and Monitoring

This project is my attempt to solve at least the first three problems listed above.

## Data and Model Versioning

This is done by [DVC (Data Version Control)](https://dvc.org/).
The steps in which I used to create this DVC pipeline can be found on `create_step.sh`

## Managing Experiments

DVC also takes care of the experiments. Which at least in my book is a huge win (one less system to learn and manage).
The way I usually approach this problem is creating several branches each modifying a few lines and parameters and running `dvc repro`. Then comparing the results of that branch with other branches to see which will be promoted to `main`.

## Model deployment

The model is evaluated agains the metrics saved on metrics.json. If it detects an improvement it should save the model into serving, using that as a trigger to deployment
