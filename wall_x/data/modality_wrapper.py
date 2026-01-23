# wall_x/data/modality_wrapper.py
"""
Modality wrapper (Level B, stable):

- Keep the dataset class as LeRobotDataset to maximize compatibility with Wall-X / LeRobot code paths.
- Only override __getitem__ to *inject* unified keys (state/action) composed from raw parquet columns,
  according to a user-provided modality.json.

Why this design:
- torch.utils.data.DataLoader only requires __len__/__getitem__, but many training pipelines also access
  LeRobotDataset attributes (meta, fps, episode_data_index, stats, etc.).
- Subclassing LeRobotDataset keeps those attributes and behaviors (timestamps sync check, video decode,
  transforms, delta_timestamps padding flags) intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class ModalityAwareLeRobotDataset(LeRobotDataset):
    """
    A LeRobotDataset that injects unified keys:
      - `state_key` (e.g. "state" or "observation.state")
      - `action_key` (e.g. "action" or "actions")

    composed from raw columns specified by modality.json.

    Notes:
    - We do NOT remove or rename any existing keys returned by LeRobotDataset.
    - If you enabled delta_timestamps for original action keys, LeRobotDataset will already return
      horizon tensors [H, D] for those keys, and we will compose `action_key` by concatenation.
    """

    def __init__(
        self,
        *args,
        modality_json: str | Path,
        action_horizon: int,
        state_key: str = "state",
        action_key: str = "action",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.modality_path = Path(modality_json)
        self.action_horizon = int(action_horizon)
        self.state_key = state_key
        self.action_key = action_key

        with open(self.modality_path, "r") as f:
            self.modality = json.load(f)

        self.state_specs = self._parse_vector_specs(self.modality.get("state", {}))
        self.action_specs = self._parse_vector_specs(self.modality.get("action", {}))

    @staticmethod
    def _parse_vector_specs(block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        modality.json example:
          "joint_robot_position": { "original_key": "...", "start": 0, "end": 18 }
        """
        specs: List[Dict[str, Any]] = []
        for _, cfg in block.items():
            specs.append(
                {
                    "original_key": cfg["original_key"],
                    "start": int(cfg.get("start", 0)),
                    "end": int(cfg.get("end", -1)),
                }
            )
        return specs

    @staticmethod
    def _ensure_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    def _slice_last_dim(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        # x: [..., D]
        if end == -1:
            return x[..., start:]
        return x[..., start:end]

    def _compose_state(self, item: Dict[str, Any]) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for sp in self.state_specs:
            orig = sp["original_key"]
            x = self._ensure_tensor(item[orig])  # expected [D]
            parts.append(self._slice_last_dim(x, sp["start"], sp["end"]))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _compose_action(self, item: Dict[str, Any], idx: int) -> torch.Tensor:
        """
        Compose action horizon tensor [H, action_dim].

        Priority:
        1) If LeRobotDataset delta_timestamps already queried `orig` -> item[orig] is [H, D], use it.
        2) Otherwise, do a local horizon query from the parquet table (safe fallback).
        """
        H = self.action_horizon

        # Need episode range to clamp within episode
        ep_idx = int(item["episode_index"].item()) if isinstance(item["episode_index"], torch.Tensor) else int(item["episode_index"])
        ep_start = int(self.episode_data_index["from"][ep_idx].item())
        ep_end = int(self.episode_data_index["to"][ep_idx].item())
        indices = [min(max(idx + t, ep_start), ep_end - 1) for t in range(H)]

        parts: List[torch.Tensor] = []
        for sp in self.action_specs:
            orig = sp["original_key"]

            if orig in item and isinstance(item[orig], torch.Tensor) and item[orig].ndim >= 2:
                # Usually [H, D] when delta_timestamps includes this key
                x = item[orig]
            else:
                # Fallback: query hf_dataset manually
                col_list = self.hf_dataset.select(indices)[orig]  # list[Tensor[D]]
                x = torch.stack(col_list, dim=0)  # [H, D]

            x = self._slice_last_dim(x, sp["start"], sp["end"])  # [H, d_i]
            parts.append(x)

        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)

        # Inject unified keys expected by Wall-X config / KEY_MAPPINGS
        item[self.state_key] = self._compose_state(item)     # [state_dim]
        item[self.action_key] = self._compose_action(item, idx)  # [H, action_dim]
        return item