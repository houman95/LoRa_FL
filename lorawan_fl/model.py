"""Model definition, local training, evaluation, and dataset loading."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torchvision import transforms

from .config import BATCH_SIZE, DEVICE

DATASET_IDS = {
    "mnist": "mnist",
    "fashion_mnist": "zalando-datasets/fashion_mnist",
    "cifar10": "uoft-cs/cifar10",
}


class TinyCNN(nn.Module):
    """TinyCNN-A — 2,066 parameters (SF7/4-bit: 59 gradient entries per packet)."""
    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 28,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv1   = nn.Conv2d(in_channels, 2, 3, padding=1)
        self.conv2   = nn.Conv2d(2, 4, 3, padding=1)
        self.pool    = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        pooled_size  = image_size // 4
        self.fc      = nn.Linear(4 * pooled_size * pooled_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc(self.flatten(x))


class SmallCNN(nn.Module):
    """A harder CNN for making FL convergence require multiple rounds."""
    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 28,
        num_classes: int = 10,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        pooled_size = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * pooled_size * pooled_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _group_count(channels: int, maximum: int = 32) -> int:
    """Largest group count not exceeding ``maximum`` that divides ``channels``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ParametricCNN(nn.Module):
    """Width-parametric VGG-style CNN whose capacity is set by ``base_filters``.

    Three convolutional blocks with filter depths ``base_filters``,
    ``2 * base_filters``, and ``4 * base_filters``, each convolution followed by
    batch normalisation, with max pooling and dropout between blocks. An
    adaptive average pool fixes the spatial output at 4x4 so the same
    architecture accepts 28x28 and 32x32 inputs unchanged; ``image_size`` is
    therefore accepted for signature symmetry with the other models and is not
    used to size any layer.

    ``base_filters`` is the single capacity knob. Parameter count is dominated
    by ``fc1`` (79-97% of weights over base_filters 8-64), so widening the base
    mainly grows that fully connected layer rather than the feature extractor.

    ``norm`` selects the normalisation layer, and the choice is federated, not
    cosmetic. ``lorawan_fl.compression.flatten_update`` iterates
    ``named_parameters()``, so module *buffers* are never transmitted or
    aggregated:

    - ``"batch"`` reproduces the architecture as presented, but
      ``BatchNorm2d`` keeps ``running_mean``/``running_var`` as buffers. The
      global model therefore evaluates with its initial statistics while each
      client adapts its own locally, and that train/evaluate mismatch can
      depress accuracy independently of width or payload budget.
    - ``"group"`` uses ``GroupNorm``, whose affine terms are parameters and
      which holds no running statistics at all, so the normalisation is
      carried by the update and federates cleanly.

    Use ``"group"`` to separate a normalisation artefact from a genuine
    capacity or compression limit.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        num_classes: int = 10,
        base_filters: int = 32,
        norm: str = "batch",
    ):
        super().__init__()
        if base_filters < 1:
            raise ValueError("base_filters must be at least 1")
        if norm not in ("batch", "group"):
            raise ValueError(f"norm must be 'batch' or 'group', got {norm!r}")
        self.base_filters = base_filters
        self.num_classes = num_classes
        self.norm = norm

        def normalisation(channels: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm2d(channels)
            return nn.GroupNorm(_group_count(channels), channels)

        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.bn1 = normalisation(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.bn2 = normalisation(base_filters)

        self.conv3 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.bn3 = normalisation(base_filters * 2)
        self.conv4 = nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1)
        self.bn4 = normalisation(base_filters * 2)

        self.conv5 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1)
        self.bn5 = normalisation(base_filters * 4)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        flat_features = (base_filters * 4) * 4 * 4
        self.fc1 = nn.Linear(flat_features, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(self.pool(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(self.pool(x))

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        return self.classifier(x)


class Net(TinyCNN):
    """Backward-compatible default model."""


def dataset_shape(dataset_name: str) -> tuple[int, int]:
    if dataset_name == "cifar10":
        return 3, 32
    return 1, 28


def create_model(
    model_name: str = "tinycnn",
    dataset_name: str = "mnist",
    num_classes: int = 10,
    *,
    base_filters: int | None = None,
    norm: str | None = None,
) -> nn.Module:
    """Build one model. ``base_filters`` and ``norm`` apply to ``parametriccnn``.

    Existing callers are unaffected: both extras are keyword-only and default
    to ``None``. The fixed-width models reject them rather than silently
    ignoring a requested width, and ``parametriccnn`` requires an explicit
    ``base_filters`` so a sweep cannot silently collapse to one default width.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    in_channels, image_size = dataset_shape(dataset_name)
    if model_name in ("tinycnn", "smallcnn"):
        if base_filters is not None:
            raise ValueError(f"base_filters is not supported by {model_name}")
        if norm is not None:
            raise ValueError(f"norm is not supported by {model_name}")
        builder = TinyCNN if model_name == "tinycnn" else SmallCNN
        return builder(
            in_channels=in_channels,
            image_size=image_size,
            num_classes=num_classes,
        ).to(DEVICE)
    if model_name == "parametriccnn":
        if base_filters is None:
            raise ValueError(
                "parametriccnn requires an explicit base_filters; it is the "
                "swept capacity knob and must not fall back to a default"
            )
        return ParametricCNN(
            in_channels=in_channels,
            image_size=image_size,
            num_classes=num_classes,
            base_filters=base_filters,
            norm="batch" if norm is None else norm,
        ).to(DEVICE)
    raise ValueError(f"Unknown model_name={model_name!r}")


def set_parameters(dst: nn.Module, src: nn.Module):
    with torch.no_grad():
        for (_, dp), (_, sp) in zip(dst.named_parameters(), src.named_parameters()):
            dp.data.copy_(sp.data)
        for (_, db), (_, sb) in zip(dst.named_buffers(), src.named_buffers()):
            db.data.copy_(sb.data)

def train(net, trainloader, epochs=1, lr: float = 0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            imgs, lbls = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(imgs), lbls)
            loss.backward(); optimizer.step()


def warm_start_to_accuracy(
    net,
    trainloaders,
    testloader,
    target_accuracy: float,
    max_epochs: int = 5,
    max_batches_per_epoch: int | None = None,
    eval_interval_batches: int | None = None,
    lr: float = 0.001,
):
    """
    Centrally pretrain the initial global model to a requested accuracy.

    This is outside the FL communication budget. It creates a controlled,
    reusable starting checkpoint for experiments that should begin from a
    partially trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = []

    loss, acc, _, _ = test(net, testloader)
    history.append({
        "epoch": 0,
        "test_loss": loss,
        "test_accuracy": acc,
        "batches_trained": 0,
    })
    if acc >= target_accuracy:
        return history

    for epoch in range(1, max_epochs + 1):
        net.train()
        batches = 0
        total_batches = history[-1]["batches_trained"]
        loader_iters = [iter(trainloader) for trainloader in trainloaders]
        active = list(range(len(loader_iters)))
        while active:
            next_active = []
            for loader_idx in active:
                try:
                    batch = next(loader_iters[loader_idx])
                except StopIteration:
                    continue
                imgs, lbls = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                loss_value = criterion(net(imgs), lbls)
                loss_value.backward()
                optimizer.step()
                batches += 1
                total_batches += 1
                next_active.append(loader_idx)
                if (
                    eval_interval_batches is not None
                    and batches % eval_interval_batches == 0
                ):
                    loss, acc, _, _ = test(net, testloader)
                    history.append({
                        "epoch": epoch,
                        "test_loss": loss,
                        "test_accuracy": acc,
                        "batches_trained": total_batches,
                    })
                    if acc >= target_accuracy:
                        return history
                    net.train()
                if (
                    max_batches_per_epoch is not None
                    and batches >= max_batches_per_epoch
                ):
                    active = []
                    break
            if max_batches_per_epoch is not None and batches >= max_batches_per_epoch:
                break
            active = next_active

        loss, acc, _, _ = test(net, testloader)
        history.append({
            "epoch": epoch,
            "test_loss": loss,
            "test_accuracy": acc,
            "batches_trained": total_batches,
        })
        if acc >= target_accuracy:
            break

    return history


def test(net, testloader):
    criterion = nn.CrossEntropyLoss(); net.eval()
    loss = correct = total = 0; y_true, y_pred = [], []
    with torch.no_grad():
        for batch in testloader:
            imgs, lbls = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            out   = net(imgs)
            loss += criterion(out, lbls).item()
            preds = out.argmax(1); correct += (preds == lbls).sum().item(); total += lbls.size(0)
            y_true.extend(lbls.cpu().tolist()); y_pred.extend(preds.cpu().tolist())
    return loss / len(testloader), correct / total, y_true, y_pred

def _build_train_partitioner(
    num_clients: int,
    partition: str,
    dirichlet_alpha: float,
    seed: int,
):
    if partition == "iid":
        return IidPartitioner(num_partitions=num_clients)
    if partition == "dirichlet":
        return DirichletPartitioner(
            num_partitions=num_clients,
            partition_by="label",
            alpha=dirichlet_alpha,
            min_partition_size=10,
            self_balancing=True,
            seed=seed,
        )
    raise ValueError(f"Unknown partition={partition!r}")


def _build_transform(dataset_name: str):
    if dataset_name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    if dataset_name == "fashion_mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def _normalize_class_subset(class_subset) -> tuple[int, ...] | None:
    if class_subset is None:
        return None
    normalized = tuple(int(label) for label in class_subset)
    if len(normalized) < 2:
        raise ValueError("class_subset must contain at least two classes")
    if len(set(normalized)) != len(normalized):
        raise ValueError("class_subset classes must be unique")
    if any(label < 0 or label > 9 for label in normalized):
        raise ValueError("class_subset labels must be in [0, 9]")
    return normalized


def _filter_and_remap_classes(dataset_dict, class_subset):
    """Normalize image naming, then filter/remap selected labels if requested."""
    normalized = _normalize_class_subset(class_subset)
    selected = set(normalized) if normalized is not None else None
    label_map = (
        {original: remapped for remapped, original in enumerate(normalized)}
        if normalized is not None
        else None
    )
    processed = {}
    for split_name, split in dataset_dict.items():
        if "img" in split.column_names and "image" not in split.column_names:
            split = split.rename_column("img", "image")
        if normalized is None:
            processed[split_name] = split
            continue
        filtered = split.filter(
            lambda label: int(label) in selected,
            input_columns="label",
        )
        processed[split_name] = filtered.map(
            lambda label: {"label": label_map[int(label)]},
            input_columns="label",
        )
    return type(dataset_dict)(processed)


def load_datasets(
    num_clients: int,
    dataset_name: str = "mnist",
    partition: str = "iid",
    dirichlet_alpha: float = 0.3,
    partition_seed: int = 42,
    return_partition_records: bool = False,
    class_subset=None,
    dataset_cache_dir: str | None = None,
):
    dataset_id = DATASET_IDS.get(dataset_name)
    if dataset_id is None:
        raise ValueError(f"Unknown dataset_name={dataset_name!r}")
    normalized_classes = _normalize_class_subset(class_subset)
    partitioner = _build_train_partitioner(
        num_clients, partition, dirichlet_alpha, partition_seed)
    fds_kwargs = {}
    if dataset_cache_dir is not None:
        fds_kwargs["cache_dir"] = dataset_cache_dir
    fds = FederatedDataset(
        dataset=dataset_id,
        partitioners={"train": partitioner},
        preprocessor=(
            (lambda dataset: _filter_and_remap_classes(dataset, normalized_classes))
            if normalized_classes is not None or dataset_name == "cifar10"
            else None
        ),
        **fds_kwargs,
    )
    tf  = _build_transform(dataset_name)
    def apply_transforms(batch):
        batch["image"] = [tf(img) for img in batch["image"]]; return batch
    trainloaders, valloaders = [], []
    partition_records = []
    for cid in range(num_clients):
        part  = fds.load_partition(cid)
        split = part.train_test_split(test_size=0.2, seed=42).with_transform(apply_transforms)
        trainloaders.append(DataLoader(split["train"], batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(split["test"],  batch_size=BATCH_SIZE))
        if return_partition_records:
            label_counts = Counter(int(label) for label in part["label"])
            total_samples = len(part)
            record = {
                "client_id": cid,
                "total_samples": total_samples,
                "train_samples": len(split["train"]),
                "validation_samples": len(split["test"]),
                "num_classes_represented": len(label_counts),
            }
            num_classes = len(normalized_classes) if normalized_classes else 10
            for label in range(num_classes):
                count = label_counts.get(label, 0)
                record[f"class_{label}_count"] = count
                record[f"class_{label}_proportion"] = (
                    count / total_samples if total_samples else 0.0
                )
            partition_records.append(record)
    testset = fds.load_split("test").with_transform(apply_transforms)
    result = (
        trainloaders,
        valloaders,
        DataLoader(testset, batch_size=BATCH_SIZE),
    )
    if return_partition_records:
        return (*result, partition_records)
    return result

# ---------------------------------------------------------------------------
