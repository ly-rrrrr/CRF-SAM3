import copy
import os
from typing import Dict, List, Tuple

from torch.utils.data import ConcatDataset, Dataset
import yaml

from crf_sam3_attn.dataset import CRFSAM3AttnDataset


def resolve_config_path(path: str, config_dir: str = None, repo_root: str = None) -> str:
    if os.path.isabs(path):
        return path

    candidates = []
    if config_dir:
        candidates.append(os.path.join(config_dir, path))
    if repo_root:
        candidates.append(os.path.join(repo_root, path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    if candidates:
        return os.path.abspath(candidates[0])
    return os.path.abspath(path)


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_base_config(train_cfg: Dict, train_cfg_path: str, repo_root: str) -> Tuple[Dict, str]:
    train_cfg_dir = os.path.dirname(os.path.abspath(train_cfg_path))
    base_cfg_path = resolve_config_path(
        train_cfg["base_config"],
        config_dir=train_cfg_dir,
        repo_root=repo_root,
    )
    return load_yaml(base_cfg_path), base_cfg_path


def _deep_merge_dict(dst: Dict, src: Dict) -> Dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge_dict(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _dataset_name_from_entry(dataset_entry: Dict, dataset_base_cfg: Dict, index: int) -> str:
    if dataset_entry.get("name"):
        return str(dataset_entry["name"])
    base_name = dataset_base_cfg.get("dataset", {}).get("name")
    if base_name:
        return str(base_name)
    data_root = dataset_base_cfg.get("dataset", {}).get("data_root")
    if data_root:
        return os.path.basename(os.path.normpath(str(data_root)))
    return f"dataset_{index}"


class RepeatDataset(Dataset):
    def __init__(self, dataset: Dataset, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError(f"repeats must be > 0, got {repeats}")
        self.dataset = dataset
        self.repeats = int(repeats)
        self._base_len = len(dataset)
        self._total_len = self._base_len * self.repeats

        for attr in ("dataset_name", "split", "offline_cache_root"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, index: int):
        if self._base_len == 0:
            raise IndexError("RepeatDataset cannot index an empty dataset.")
        return self.dataset[index % self._base_len]

    def __repr__(self) -> str:
        return f"RepeatDataset(dataset={self.dataset!r}, repeats={self.repeats})"


def _build_dataset_base_cfg(
    global_base_cfg: Dict,
    dataset_entry: Dict,
    train_cfg_dir: str,
    repo_root: str,
) -> Dict:
    dataset_base_cfg = copy.deepcopy(global_base_cfg)

    entry_base_cfg_path = dataset_entry.get("base_config")
    if entry_base_cfg_path:
        resolved_path = resolve_config_path(
            entry_base_cfg_path,
            config_dir=train_cfg_dir,
            repo_root=repo_root,
        )
        entry_base_cfg = load_yaml(resolved_path)
        if "dataset" not in entry_base_cfg:
            raise KeyError(f"Dataset base config missing 'dataset' section: {resolved_path}")
        dataset_base_cfg["dataset"] = copy.deepcopy(entry_base_cfg["dataset"])

    if dataset_entry.get("dataset"):
        dataset_section = copy.deepcopy(dataset_base_cfg.get("dataset", {}))
        _deep_merge_dict(dataset_section, dataset_entry["dataset"])
        dataset_base_cfg["dataset"] = dataset_section

    return dataset_base_cfg


def resolve_dataset_specs(
    global_base_cfg: Dict,
    data_cfg: Dict,
    train_cfg_dir: str,
    repo_root: str,
) -> List[Tuple[Dict, Dict]]:
    common_data_cfg = copy.deepcopy(data_cfg or {})
    dataset_entries = common_data_cfg.pop("datasets", None)

    if not dataset_entries:
        single_cfg = copy.deepcopy(common_data_cfg)
        single_cfg.setdefault("dataset_name", _dataset_name_from_entry({}, global_base_cfg, 0))
        return [(copy.deepcopy(global_base_cfg), single_cfg)]

    if not isinstance(dataset_entries, list):
        raise TypeError("data.datasets must be a list when provided.")
    if not dataset_entries:
        raise ValueError("data.datasets is empty.")

    specs: List[Tuple[Dict, Dict]] = []
    for index, dataset_entry in enumerate(dataset_entries):
        if not isinstance(dataset_entry, dict):
            raise TypeError(f"data.datasets[{index}] must be a dict.")

        dataset_base_cfg = _build_dataset_base_cfg(
            global_base_cfg=global_base_cfg,
            dataset_entry=dataset_entry,
            train_cfg_dir=train_cfg_dir,
            repo_root=repo_root,
        )
        dataset_data_cfg = copy.deepcopy(common_data_cfg)
        entry_data_cfg = {
            key: value
            for key, value in dataset_entry.items()
            if key not in {"base_config", "dataset", "name"}
        }
        _deep_merge_dict(dataset_data_cfg, entry_data_cfg)
        dataset_data_cfg["dataset_name"] = _dataset_name_from_entry(
            dataset_entry,
            dataset_base_cfg,
            index,
        )
        specs.append((dataset_base_cfg, dataset_data_cfg))

    return specs


def build_training_dataset(
    global_base_cfg: Dict,
    data_cfg: Dict,
    train_cfg_dir: str,
    repo_root: str,
) -> Tuple[Dataset, List[Dict]]:
    datasets = []
    infos = []

    for dataset_base_cfg, dataset_data_cfg in resolve_dataset_specs(
        global_base_cfg=global_base_cfg,
        data_cfg=data_cfg,
        train_cfg_dir=train_cfg_dir,
        repo_root=repo_root,
    ):
        dataset_cfg_for_build = copy.deepcopy(dataset_data_cfg)
        repeats = int(dataset_cfg_for_build.pop("repeats", 1))
        dataset = CRFSAM3AttnDataset(dataset_base_cfg, dataset_cfg_for_build)
        raw_samples = len(dataset)
        effective_samples = raw_samples * repeats
        if repeats > 1:
            dataset = RepeatDataset(dataset, repeats)
        datasets.append(dataset)
        infos.append(
            {
                "name": dataset.dataset_name,
                "samples": len(dataset),
                "raw_samples": raw_samples,
                "effective_samples": effective_samples,
                "repeats": repeats,
                "split": str(dataset.split),
                "data_root": dataset_base_cfg["dataset"]["data_root"],
                "gt_masks_root": dataset_base_cfg["dataset"]["gt_masks_root"],
                "offline_cache_root": dataset.offline_cache_root,
            }
        )

    if len(datasets) == 1:
        return datasets[0], infos
    return ConcatDataset(datasets), infos


def prepare_training_offline_cache(
    global_base_cfg: Dict,
    data_cfg: Dict,
    train_cfg_dir: str,
    repo_root: str,
) -> List[Dict]:
    results = []

    for dataset_base_cfg, dataset_data_cfg in resolve_dataset_specs(
        global_base_cfg=global_base_cfg,
        data_cfg=data_cfg,
        train_cfg_dir=train_cfg_dir,
        repo_root=repo_root,
    ):
        prep_cfg = copy.deepcopy(dataset_data_cfg)
        prep_cfg["offline_mode"] = False
        prep_cfg["offline_prepare"] = False
        dataset = CRFSAM3AttnDataset(dataset_base_cfg, prep_cfg)
        cache_root = prep_cfg.get("offline_cache_root", "crf_sam3_attn/offline_cache")
        overwrite = bool(prep_cfg.get("offline_overwrite", False))
        prepared = dataset.prepare_offline_cache(cache_root=cache_root, overwrite=overwrite)
        results.append(
            {
                "name": dataset.dataset_name,
                "prepared": int(prepared),
                "split": str(dataset.split),
                "data_root": dataset_base_cfg["dataset"]["data_root"],
                "gt_masks_root": dataset_base_cfg["dataset"]["gt_masks_root"],
                "offline_cache_root": dataset.offline_cache_root,
            }
        )

    return results
