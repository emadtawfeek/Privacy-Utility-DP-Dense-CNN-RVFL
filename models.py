"""Model definitions for dense, convolutional, and randomized-feature networks."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    architecture_family: str
    architecture_variant: str
    privacy: str


def _binary_output_dim(task_type: str, num_classes: int) -> int:
    return 1 if task_type == "binary" else num_classes


def _flatten_dim(input_shape: tuple[int, ...]) -> int:
    return int(prod(input_shape))


class DenseNet(nn.Module):
    """Dense network for binary tabular and multiclass image classification."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        task_type: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        input_dim = _flatten_dim(input_shape)
        if task_type == "binary":
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        else:
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )
        self.task_type = task_type

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.network(inputs)
        return outputs.squeeze(-1) if self.task_type == "binary" else outputs


class ImageCNN(nn.Module):
    """Small 2D CNN for MNIST and CIFAR-10."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
    ) -> None:
        super().__init__()
        channels, height, width = input_shape
        if channels == 1:
            conv_channels = (16, 32)
        else:
            conv_channels = (32, 64)
        self.features = nn.Sequential(
            nn.Conv2d(channels, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        pooled_height = height // 4
        pooled_width = width // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[1] * pooled_height * pooled_width, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class AbadiMNISTMLP(nn.Module):
    """MNIST network matching the Abadi et al. PCA plus 1000-hidden-unit profile."""

    def __init__(self, projection: torch.Tensor, num_classes: int = 10) -> None:
        super().__init__()
        if projection.ndim != 2:
            raise ValueError("PCA projection must have shape [components, input_dim].")
        self.register_buffer("projection", projection.float())
        self.classifier = nn.Sequential(
            nn.Linear(projection.shape[0], 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = torch.flatten(inputs, start_dim=1)
        projected = flat @ self.projection.t()
        return self.classifier(projected)


class AbadiCIFARNet(nn.Module):
    """CIFAR-10 CNN structure following the TensorFlow tutorial profile."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cropped = inputs[:, :, 4:28, 4:28]
        return self.classifier(self.features(cropped))


class FeatureCNN(nn.Module):
    """Exploratory 1D CNN over tabular feature vectors."""

    def __init__(self, input_shape: tuple[int, ...], task_type: str) -> None:
        super().__init__()
        if len(input_shape) != 1:
            raise ValueError("FeatureCNN expects one-dimensional tabular input.")
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, _binary_output_dim(task_type, 2)),
        )
        self.task_type = task_type

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        outputs = self.features(inputs)
        return outputs.squeeze(-1) if self.task_type == "binary" else outputs


class RVFLNet(nn.Module):
    """Random vector functional link network with frozen random features."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        task_type: str,
        num_classes: int,
        n_random_features: int = 200,
        activation: str = "tanh",
        seed: int = 42,
        max_input_dim: int = 1024,
    ) -> None:
        super().__init__()
        original_dim = _flatten_dim(input_shape)
        self.flatten = nn.Flatten()
        self.input_reducer: nn.Module
        if original_dim > max_input_dim:
            generator_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
            reducer = nn.Linear(original_dim, max_input_dim, bias=False)
            nn.init.normal_(reducer.weight, mean=0.0, std=1.0 / (max_input_dim**0.5))
            torch.random.set_rng_state(generator_state)
            for parameter in reducer.parameters():
                parameter.requires_grad = False
            self.input_reducer = reducer
            input_dim = max_input_dim
        else:
            self.input_reducer = nn.Identity()
            input_dim = original_dim

        generator_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.random_layer = nn.Linear(input_dim, n_random_features)
        nn.init.uniform_(self.random_layer.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.random_layer.bias, a=0.0, b=1.0)
        torch.random.set_rng_state(generator_state)
        for parameter in self.random_layer.parameters():
            parameter.requires_grad = False

        self.output_layer = nn.Linear(
            input_dim + n_random_features,
            _binary_output_dim(task_type, num_classes),
        )
        self.task_type = task_type
        self.activation_name = activation
        self.n_random_features = n_random_features
        self.reduced_input_dim = input_dim
        self.original_input_dim = original_dim

    def activation(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return torch.relu(values)
        if self.activation_name == "radbas":
            return torch.exp(-(values**2))
        if self.activation_name in {"sig", "sigmoid"}:
            return torch.sigmoid(values)
        if self.activation_name in {"sin", "sine"}:
            return torch.sin(values)
        return torch.tanh(values)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = self.flatten(inputs)
        reduced = self.input_reducer(flat)
        with torch.no_grad():
            hidden = self.activation(self.random_layer(reduced))
        concatenated = torch.cat([reduced, hidden], dim=1)
        outputs = self.output_layer(concatenated)
        return outputs.squeeze(-1) if self.task_type == "binary" else outputs


def build_model(
    *,
    model_key: str,
    dataset: str,
    input_shape: tuple[int, ...],
    task_type: str,
    num_classes: int,
    seed: int,
    privacy: str,
    n_random_features: int = 200,
    rvfl_activation: str = "tanh",
    paper_profile: str = "default",
    pca_projection: torch.Tensor | None = None,
) -> tuple[nn.Module, ModelSpec]:
    """Construct a model and its reporting metadata."""
    private_prefix = "DP-" if privacy == "private" else ""
    if paper_profile == "abadi2016" and model_key == "abadi":
        if dataset == "mnist":
            if pca_projection is None:
                raise ValueError("Abadi MNIST profile requires a PCA projection.")
            model = AbadiMNISTMLP(pca_projection, num_classes)
            family = "Abadi2016"
            variant = "MNIST-PCA60-MLP1000"
            model_name = f"{private_prefix}Abadi-MNIST-MLP"
        elif dataset == "cifar10":
            model = AbadiCIFARNet(num_classes)
            family = "Abadi2016"
            variant = "CIFAR10-Tutorial-CNN"
            model_name = f"{private_prefix}Abadi-CIFAR-CNN"
        else:
            raise ValueError("Abadi 2016 profile supports only MNIST and CIFAR-10.")
        return model, ModelSpec(
            model_name=model_name,
            architecture_family=family,
            architecture_variant=variant,
            privacy=privacy,
        )
    if model_key == "dense":
        model = DenseNet(input_shape, task_type, num_classes)
        family = "Dense"
        variant = "Dense-MLP"
    elif model_key == "cnn":
        family = "CNN"
        if dataset in {"mnist", "cifar10"}:
            model = ImageCNN(input_shape, num_classes)
            variant = "2D-CNN"
        else:
            model = FeatureCNN(input_shape, task_type)
            variant = "1D-CNN"
    elif model_key == "rvfl":
        model = RVFLNet(
            input_shape=input_shape,
            task_type=task_type,
            num_classes=num_classes,
            n_random_features=n_random_features,
            activation=rvfl_activation,
            seed=seed,
        )
        family = "RVFL"
        variant = "Frozen-random-feature RVFL"
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    return model, ModelSpec(
        model_name=f"{private_prefix}{family}",
        architecture_family=family,
        architecture_variant=variant,
        privacy=privacy,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return int(trainable), int(total)
