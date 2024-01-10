import os
from os import PathLike
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from vutils.log import logger

class MlpClsConfig:
    def __init__(self,
            num_features: int,
            num_labels: int,
            num_hidden_size: int=50,
            num_layers: int=6,
            num_intermidiate_size: int=100,
            dropout: float=0.0
        ):
            self.num_features = num_features
            self.num_labels = num_labels
            self.num_hidden_size = num_hidden_size
            self.num_layers = num_layers
            self.num_intermidiate_size = num_intermidiate_size
            self.dropout = dropout

class ResidualBlock(nn.Module):
    def __init__(
        self, input_dim, intermidiate_dim, output_dim, dropout_rate=0.0
    ):
        super().__init__()
        self.lin_a = nn.Linear(
            input_dim,
            intermidiate_dim
        )
        self.relu = nn.ReLU()
        self.lin_b = nn.Linear(
            intermidiate_dim,
            output_dim
        )
        self.lin_res = nn.Linear(
            input_dim,
            output_dim
        )
        self.lnorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        h_state = self.relu(self.lin_a(inputs))
        out = self.lin_b(h_state)
        out = self.dropout(out)
        res = self.lin_res(inputs)
        return self.lnorm(out + res)

class MlpCls(nn.Module):
    """
    MLP分类模型，基于多层残差神经网络
    """
    def __init__(self, config: MlpClsConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.num_features, config.num_hidden_size)
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            self.blocks.add_module(f'block{i}', ResidualBlock(
                                        input_dim=config.num_hidden_size,
                                        intermidiate_dim=config.num_intermidiate_size,
                                        output_dim=config.num_hidden_size,
                                        dropout_rate=config.dropout
                                    ))
        self.decoder = nn.Linear(config.num_hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        y = self.decoder(x)
        return y

def loadMlpCls(path: PathLike) -> MlpCls:
    d = torch.load(path)
    config = d['config']
    model = MlpCls(config)
    model.load_state_dict(d['model_state_dict'])
    return model

def saveMlpCls(model: MlpCls, path: PathLike):
    torch.save({
        'config': model.config,
        'model_state_dict': model.state_dict()
    }, path)

def train(
    model: MlpCls,
    num_epochs: float,
    batch_size: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    lr: float
):
    model.train()
    dataset = TensorDataset(train_features, train_labels)
    dataLoader = DataLoader(dataset, batch_size)
    trainer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(trainer, T_0=10, T_mult=2)
    lossFunction = torch.nn.CrossEntropyLoss()
    iters = len(dataLoader)
    for epoch in tqdm(range(num_epochs)):
        for i, (x, label) in enumerate(dataLoader):
            loss = lossFunction(model(x), label)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            lr_scheduler.step(epoch + i/iters)
        l = lossFunction(model(train_features), train_labels)
        logger.info(f'整体损失为{l}')
    model.eval()

def evaluate(
    model: MlpCls,
    test_features: torch.Tensor,
    test_labels: torch.Tensor
):
    model.eval()
    y = model(test_features)
    y_result = torch.argmax(y, dim=-1)
    acc = torch.sum(y_result == test_labels) / test_features.size(0)
    return acc

def train_val_split(features: torch.Tensor, labels: torch.Tensor, group: int=1):
    val_label_count = {}
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    for i in range(features.size(0)):
        label = int(labels[i])
        if label not in val_label_count:
            val_features.append(features[i, :].tolist())
            val_labels.append(label)
            val_label_count[label] = 1
        elif val_label_count[label] < group:
            val_features.append(features[i, :].tolist())
            val_labels.append(label)
            val_label_count[label] += 1
        else:
            train_features.append(features[i, :].tolist())
            train_labels.append(label)
    train_features = torch.tensor(train_features)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_features = torch.tensor(val_features)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    return train_features, train_labels, val_features, val_labels
