
import math
import random
from typing import Dict, Any, List, Union, Literal, Optional

import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from .mlp import UniversalMlp
from vutils.print_color import print_blue as print


Optimizer_Type = Literal["sgd", "momentum", "adagrad", "rmsprop", "adam", "adamw"]
Lr_Scheduler_Type = Literal["none", "cosine", "linear", "exponential", "polynomial", "step", "multistep", "cycle", "cosine_warm"]
Loss_Function_Type = Literal["auto", "mse", "cross_entropy"]
Regularization_Type = Literal["none", "l1", "l2"]


class UniversalModel(nn.Module):
    def __init__(self, feature_map: Dict[str, Dict[str, Any]], label_size: int = 2, layer_num: int = 24):
        """
        feature_map in format:
        ```csv
        name, type, values
        age, numerical, [1, 30, 99]
        sex, categorical, ["male", "female"]
        ```
        """
        super().__init__()
        assert label_size >= 0, "label_size must be greater than 0!"
        self.user_variables = {}
        self.feature_map = feature_map
        self.label_size = label_size
        self.categorical2id = {}
        categorical_id = 0
        self.numerical2id = {}
        numerical_id = 0
        for name, info in feature_map.items():
            if info["type"] == "categorical":
                for val in info["values"]:
                    self.categorical2id[f"{name}->{val}"] = categorical_id
                    categorical_id += 1
            elif info["type"] == "numerical":
                self.numerical2id[name] = numerical_id
                numerical_id += 1
            else:
                raise Exception("error")
        self.categorical_embed = nn.Embedding(categorical_id, categorical_id // 2)
        self.numerical_norm = nn.LayerNorm(numerical_id)
        self.numerical_embed = nn.Linear(numerical_id, numerical_id // 2, bias=False)
        hidden_size = categorical_id // 2 + numerical_id // 2
        inner_hidden_size = hidden_size * 2
        self.all_embed = nn.Linear(hidden_size, inner_hidden_size, bias=False)
        self.encoder = nn.ModuleList([UniversalMlp(inner_hidden_size) for _ in range(layer_num)])
        self.decoder = nn.Linear(inner_hidden_size, label_size, bias=False)

    @torch.inference_mode()
    def predict(self, x):
        self.eval()
        y = self(x)
        return y

    def forward(self, xs: Union[Dict[str, Any], List[Dict[str, Any]]]):
        categorical, numerical = self._get_feature_initial_representation(xs)
        output_states = self._get_feature_final_representation(categorical, numerical)
        if self.label_size >= 2:
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
            loss_fct: Loss_Function_Type = "auto",
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

    def _inner_training_loop(
            self,
            data: List[Dict[str, Any]],
            labels: torch.Tensor,
            train_data_size: int,
            eval_data_size: int,
            epoch_size: int = 10,
            batch_size: int = 16,
            mini_batch_size: int = 4,
            initial_lr: float = 1e-5,
            optimizer: Optimizer_Type = "sgd",
            lr_scheduler: Lr_Scheduler_Type = "none",
            loss_fct: Loss_Function_Type = "auto",
            regularization: Regularization_Type = "none",
    ):
        self.train()
        optimizer = self._get_optimizer(optimizer, initial_lr)
        lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)
        loss_fct = self._get_loss_fct(loss_fct)
        print("### start training...")
        self._on_train_start()
        self.train_loss_record = []
        self.eval_loss_record = [[], []]
        tbar = tqdm(total=epoch_size * math.ceil(train_data_size / batch_size))
        for epoch in range(epoch_size):
            for step in range(0, train_data_size, batch_size):
                optimizer.zero_grad()
                for mini_step in range(0, batch_size, mini_batch_size):
                    if step + mini_step >= train_data_size:
                        break
                    batch_data = data[step + mini_step: min(step + mini_step + mini_batch_size, train_data_size)]
                    batch_label = labels[step + mini_step: min(step + mini_step + mini_batch_size, train_data_size)]
                    y = self(batch_data)
                    loss = loss_fct(y, batch_label)
                    loss = self._on_calculate_loss(loss, batch_data, y)
                    loss = self._do_regularization(loss, regularization)
                    self.train_loss_record.append(float(loss))
                    loss.backward()
                    self._on_backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                self._on_train_step()
                tbar.update()
            with torch.inference_mode():
                eval_model_output = self(data[-eval_data_size:])
                eval_loss = loss_fct(eval_model_output, labels[-eval_data_size:])
                self.eval_loss_record[0].append(len(self.train_loss_record) - 1)
                self.eval_loss_record[1].append(float(eval_loss))
                self._on_eval(data[-eval_data_size:], eval_model_output)
        tbar.close()
        print("### training finished!")
        plt.figure("training log")
        plt.title("training log")
        plt.plot(self.train_loss_record, label="train loss")
        plt.plot(self.eval_loss_record[0], self.eval_loss_record[1], label="eval loss")
        plt.xlabel("step")
        plt.grid()
        plt.legend()
        plt.show()
        self._on_train_finish()

    def _on_train_start(self):
        """can be overridden"""
        pass

    def _on_calculate_loss(self, loss, batch_data, y):
        """can be overridden"""
        return loss

    def _on_backward(self):
        """can be overridden"""
        pass

    def _on_train_step(self):
        """can be overridden"""
        pass

    def _on_eval(self, eval_data, model_output):
        """can be overridden"""
        pass

    def _on_train_finish(self):
        """can be overridden"""
        pass

    def _get_feature_initial_representation(self, xs: Union[Dict[str, Any], List[Dict[str, Any]]]):
        if isinstance(xs, dict):
            xs = [xs]
        batch_size = len(xs)
        categorical = [[] for _ in range(batch_size)]
        numerical = [[0] * len(self.numerical2id) for _ in range(batch_size)]
        for i, x in enumerate(xs):
            names = list(x.keys())
            for name in names:
                if name not in self.feature_map:
                    continue
                val = x[name]
                if self.feature_map[name]["type"] == "categorical":
                    cid = self.categorical2id[f"{name}->{val}"]
                    categorical[i].append(cid)
                elif self.feature_map[name]["type"] == "numerical":
                    nid = self.numerical2id[name]
                    minimum, mean, maximum = self.feature_map[name]["values"]
                    if pd.isna(val):
                        val = mean
                    norm_val = (val - minimum) / (maximum - minimum)
                    numerical[i][nid] = norm_val
                else:
                    raise Exception("error")
        return categorical, numerical

    def _get_feature_final_representation(self, categorical: List[List[int]], numerical: List[List[float]]):
        device = self.categorical_embed.weight.device
        hidden_states_1 = self.categorical_embed(torch.LongTensor(categorical).to(device)).sum(dim=-2)
        hidden_states_2 = self.numerical_embed(self.numerical_norm(torch.Tensor(numerical).to(device)))
        hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
        hidden_states = self.all_embed(hidden_states)
        outer_residual = hidden_states
        for layer in self.encoder:
            hidden_states = layer(hidden_states)
        hidden_states = hidden_states + outer_residual
        return self.decoder(hidden_states)

    def _preprocessing_data(self, data: List[Dict[str, Any]], eval_data_ratio: float, shuffle_data: bool):
        data_size = len(data)
        eval_data_size = int(data_size * eval_data_ratio)
        train_data_size = data_size - eval_data_size
        print(f"### data split finished! train data size = {train_data_size}, eval data size = {eval_data_size}.")
        if shuffle_data:
            random.seed(369)
            random.shuffle(data)
        return data, data_size, train_data_size, eval_data_size

    def _preprocessing_outputs(
            self,
            data,
            label_key,
            data_size,
    ) -> torch.Tensor:
        print("### preprocessing outputs...")
        outputs = [data[i][label_key] for i in range(data_size)]
        outputs = torch.LongTensor(outputs).to(self.decoder.weight.device)
        print("### preprocessing outputs finished! ")
        return outputs

    def _get_optimizer(self, name: Optimizer_Type, initial_lr: float, **kwargs) -> torch.optim.Optimizer:
        if name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=initial_lr, **kwargs)
        elif name == "momentum":
            return torch.optim.SGD(self.parameters(), lr=initial_lr, momentum=0.9, **kwargs)
        elif name == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=initial_lr, **kwargs)
        elif name == "rmsprop":
            return torch.optim.RMSprop(self.parameters(), lr=initial_lr, **kwargs)
        elif name == "adam":
            return torch.optim.Adam(self.parameters(), lr=initial_lr, **kwargs)
        elif name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=initial_lr, **kwargs)
        else:
            pass

    def _get_lr_scheduler(self, name: Lr_Scheduler_Type, optimizer: torch.optim.Optimizer,
                          **kwargs) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        if name == "none":
            return None
        elif name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, **kwargs)
        elif name == "linear":
            return torch.optim.lr_scheduler.LinearLR(optimizer, **kwargs)
        elif name == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif name == "multi_step":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        elif name == "cyclic":
            return torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
        elif name == "cosine_warm":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
        else:
            pass

    def _get_loss_fct(self, name: Loss_Function_Type) -> nn.Module:
        if name == "auto":
            if self.label_size == 1:
                return nn.MSELoss()
            else:
                return nn.CrossEntropyLoss()
        elif name == "mse":
            return nn.MSELoss()
        elif name == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            pass

    def _do_regularization(self, loss, regularization: Regularization_Type):
        if regularization == "none":
            return loss
        if regularization == "l1":
            return loss + self._get_l1_regularization()
        if regularization == "l2":
            return loss + self._get_l2_regularization()

    def _get_l1_regularization(self):
        param_sum = 0
        param_num = 0
        for param in self.parameters():
            param_sum += param.data.abs().sum()
            param_num += param.numel()
        return param_sum / param_num

    def _get_l2_regularization(self):
        param_sum = 0
        param_num = 0
        for param in self.parameters():
            param_sum += param.data.pow(2).sum()
            param_num += param.numel()
        return param_sum / param_num
