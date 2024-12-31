
from typing import Dict, Any, List, Union

import numpy as np
import matplotlib.pyplot as plt
import torch

from .base import (
    UniversalModel,
    Optimizer_Type,
    Lr_Scheduler_Type,
    Loss_Function_Type,
    Regularization_Type,
)
from vutils.print_color import print_blue as print


class UniversalRegressor(UniversalModel):
    def __init__(self, feature_map: Dict[str, Dict[str, Any]], layer_num: int = 24):
        """
        feature_map in format:
        ```csv
        name, type, values
        age, numerical, [1, 30, 99]
        sex, categorical, ["male", "female"]
        ```
        """
        super().__init__(feature_map, 1, layer_num)

    def forward(self, xs: Union[Dict[str, Any], List[Dict[str, Any]]]):
        categorical, numerical = self._get_feature_initial_representation(xs)
        output_states = self._get_feature_final_representation(categorical, numerical)
        output_states = output_states.squeeze(-1)
        return output_states

    def self_train(
            self,
            data: List[Dict[str, Any]],
            label_key: str,
            epoch_size: int = 10,
            batch_size: int = 16,
            mini_batch_size: int = 4,
            initial_lr: float = 1e-5,
            eval_data_ratio: float = 0.0,
            shuffle_data: bool = True,
            optimizer: Optimizer_Type = "sgd",
            lr_scheduler: Lr_Scheduler_Type = "none",
            loss_fct: Loss_Function_Type = "mse",
            regularization: Regularization_Type = "none",
    ):
        assert batch_size % mini_batch_size == 0, "batch_size must be divisible by mini_batch_size!"
        data, data_size, train_data_size, eval_data_size = self._preprocessing_data(data, eval_data_ratio, shuffle_data)
        labels = self._preprocessing_labels(data, label_key, data_size)
        self._inner_training_loop(
            data,
            labels,
            train_data_size,
            eval_data_size,
            epoch_size,
            batch_size,
            mini_batch_size,
            initial_lr,
            optimizer,
            lr_scheduler,
            loss_fct,
            regularization,
        )
        plt.figure("regression figure")
        plt.title("regression figure")
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for i in range(0, data_size, mini_batch_size * 2):
            pred = self(data[i: min(i + mini_batch_size * 2, data_size)])
            ground_truth = labels[i: min(i + mini_batch_size * 2, data_size)]
            for j in range(pred.size(0)):
                data_index = i + j
                if data_index < train_data_size:
                    x1.append(float(ground_truth[j]))
                    y1.append(float(pred[j]))
                else:
                    x2.append(float(ground_truth[j]))
                    y2.append(float(pred[j]))
        plt.scatter(x1, y1, label="train data")
        plt.scatter(x2, y2, label="eval data")
        min_value = min(min(x1), min(x2))
        max_value = max(max(x1), max(x2))
        plt.plot([min_value, max_value], [min_value, max_value], color='red')
        plt.xlabel("true value")
        plt.ylabel("predicted value")
        plt.legend()
        plt.show()

    def _preprocessing_labels(
            self,
            data,
            label_key,
            data_size,
    ) -> torch.Tensor:
        print("### preprocessing labels...")
        labels = [data[i][label_key] for i in range(data_size)]
        labels = torch.Tensor(labels).to(self.decoder.weight.device)
        print("### preprocessing labels finished! ")
        return labels
