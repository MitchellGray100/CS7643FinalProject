# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

import csv
import itertools
import os
import random
import time

from train import train

log_dir = "log"
search_log_path = os.path.join(log_dir, "search_log.csv")

csv_field_names = [
    "model_name",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "learning_rate_decay_step",
    "learning_rate_decay_factor",
    "min_learning_rate",
    "regularization_loss_weight",
    "dropout_prob",
    "adam_weight_decay",
    "augment_training_data",
    "num_points",
    "batch_norm_init_decay",
    "batch_norm_decay_rate",
    "batch_norm_decay_step",
    "batch_norm_decay_clip",
]

default_param_grid = {
    "model_name": ["ModelNet40"],
    "batch_size": [32],
    "num_epochs": [250],
    "learning_rate": [0.001],
    "learning_rate_decay_step": [200000],
    "learning_rate_decay_factor": [0.7],
    "min_learning_rate": [0],
    "regularization_loss_weight": [0.001],
    "dropout_prob": [0.3],
    "adam_weight_decay": [0.0],
    "augment_training_data": [True],
    "num_points": [1024],
    "batch_norm_init_decay": [0.5],
    "batch_norm_decay_rate": [0.5],
    "batch_norm_decay_step": [200000],
    "batch_norm_decay_clip": [0.99],
}

param_grid = {
    "model_name": ["ModelNet40"],
    "batch_size": [16, 32, 64],
    "num_epochs": [250],
    "learning_rate": [0.01, 0.001, 0.0001],
    "learning_rate_decay_step":  [200000],
    "learning_rate_decay_factor": [0.7],
    "min_learning_rate": [0],
    "regularization_loss_weight": [0.001],
    "dropout_prob": [0, 0.3, 0.6],
    "adam_weight_decay": [0.0],
    "augment_training_data": [True, False],
    "num_points": [1024],
    "batch_norm_init_decay": [0.5],
    "batch_norm_decay_rate": [0.5],
    "batch_norm_decay_step": [200000],
    "batch_norm_decay_clip": [0.99],
}


def read_completed_configs(path=search_log_path):
    if not os.path.exists(path):
        return set()

    completed_set = set()
    with open(path, "r", newline="") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            config_key = config_to_hashkey(row)
            completed_set.add(config_key)
    return completed_set


def config_to_hashkey(config):
    return (
        str(config["model_name"]),
        int(config["batch_size"]),
        int(config["num_epochs"]),
        float(config["learning_rate"]),
        int(config["learning_rate_decay_step"]),
        float(config["learning_rate_decay_factor"]),
        float(config["min_learning_rate"]),
        float(config["regularization_loss_weight"]),
        float(config["dropout_prob"]),
        float(config["adam_weight_decay"]),
        int(config["augment_training_data"]),
        int(config["num_points"]),
        float(config["batch_norm_init_decay"]),
        float(config["batch_norm_decay_rate"]),
        int(config["batch_norm_decay_step"]),
        float(config["batch_norm_decay_clip"]),
    )


def log_config(config, path=search_log_path):
    os.makedirs(log_dir, exist_ok=True)
    file_exists = os.path.exists(path)

    row = {
        "model_name": config["model_name"],
        "batch_size": int(config["batch_size"]),
        "num_epochs": int(config["num_epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "learning_rate_decay_step": int(config["learning_rate_decay_step"]),
        "learning_rate_decay_factor": float(config["learning_rate_decay_factor"]),
        "min_learning_rate": float(config["min_learning_rate"]),
        "regularization_loss_weight": float(config["regularization_loss_weight"]),
        "dropout_prob": float(config["dropout_prob"]),
        "adam_weight_decay": float(config["adam_weight_decay"]),
        "augment_training_data": int(config["augment_training_data"]),
        "num_points": int(config["num_points"]),
        "batch_norm_init_decay": float(config["batch_norm_init_decay"]),
        "batch_norm_decay_rate": float(config["batch_norm_decay_rate"]),
        "batch_norm_decay_step": int(config["batch_norm_decay_step"]),
        "batch_norm_decay_clip": float(config["batch_norm_decay_clip"]),
    }

    with open(path, "a", newline="") as file:
        csv_writer = csv.DictWriter(file, fieldnames=csv_field_names)
        if not file_exists:
            csv_writer.writeheader()
        csv_writer.writerow(row)


def make_single_param_sweep_config_list(default_param_grid, param_grid, include_default_config=True):
    config_list = []
    default_config = {
        param_name: param_values[0]
        for param_name, param_values in default_param_grid.items()
    }

    if include_default_config:
        config_list.append(default_config.copy())

    # sweep each param
    for param_name, sweep_values in param_grid.items():
        default_value = default_config[param_name]
        for param_value in sweep_values:
            if param_value == default_value:
                continue

            config = default_config.copy()
            config[param_name] = param_value
            config_list.append(config)

    return config_list


def run_config_list(config_list):
    completed_set = read_completed_configs()

    for config in config_list:
        config_key = config_to_hashkey(config)

        if config_key in completed_set:
            print("skip")
            continue

        print("\n" + "-" * 69)
        print("running config:")
        for param_name, param_value in sorted(config.items()):
            print(f"    {param_name:33} = {param_value}")

        start_time_seconds = time.time()

        train(**config)

        elapsed_minutes = (time.time() - start_time_seconds) / 60.0
        print(f"used {elapsed_minutes} minutes")

        log_config(config)
        completed_set.add(config_key)


def single_param_sweep_search():
    config_list = make_single_param_sweep_config_list(
        default_param_grid=default_param_grid,
        param_grid=param_grid,
        include_default_config=True,
    )
    run_config_list(config_list)


if __name__ == "__main__":
    single_param_sweep_search()
