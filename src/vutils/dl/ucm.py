
from typing import Dict, Any, List, Union

import torch
import numpy as np
import matplotlib.pyplot as plt

from .base import (
    UniversalModel,
    Optimizer_Type,
    Lr_Scheduler_Type,
    Loss_Function_Type,
    Regularization_Type,
)


class UniversalClassifier(UniversalModel):
    def __init__(self, feature_map: Dict[str, Dict[str, Any]], label_size: int = 2, layer_num: int = 24):
        """
        feature_map in format:
        ```csv
        name, type, values
        age, numerical, [1, 30, 99]
        sex, categorical, ["male", "female"]
        ```
        """
        assert label_size >= 2, "label_size must be greater than or equal to 2!"
        super().__init__(feature_map, label_size, layer_num)

    def forward(self, xs: Union[Dict[str, Any], List[Dict[str, Any]]]):
        categorical, numerical = self._get_feature_initial_representation(xs)
        output_states = self._get_feature_final_representation(categorical, numerical)
        output_states = torch.softmax(output_states, dim=-1)
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
            loss_fct: Loss_Function_Type = "cross_entropy",
            regularization: Regularization_Type = "none",
    ):
        assert batch_size % mini_batch_size == 0, "batch_size must be divisible by mini_batch_size!"
        data, data_size, train_data_size, eval_data_size = self._preprocessing_data(data, eval_data_ratio, shuffle_data)
        labels = self._preprocessing_outputs(data, label_key, data_size)
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
        plt.figure("confusion matrix")
        plt.title("confusion matrix")
        confusion_matrix = np.zeros((self.label_size, self.label_size), dtype=int)
        for i in range(0, data_size, mini_batch_size * 2):
            pred = torch.argmax(self(data[i: min(i + mini_batch_size * 2, data_size)]), dim=-1)
            ground_truth = labels[i: min(i + mini_batch_size * 2, data_size)]
            for j in range(pred.shape[0]):
                confusion_matrix[ground_truth[j], pred[j]] += 1
        plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black")
        plt.xlabel("true label")
        plt.ylabel("predicted label")
        plt.xticks(range(self.label_size))
        plt.yticks(range(self.label_size))
        plt.show()
