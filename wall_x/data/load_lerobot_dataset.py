"""
LeRobot Dataset Loader - Distributed Version
"""

import numpy as np
import torch
import random
from torch.utils.data import DistributedSampler, random_split
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from typing import Protocol, SupportsIndex, TypeVar
from qwen_vl_utils.vision_process import smart_resize
from wall_x.data.config import X2RDataProcessingConfig
from wall_x.data.utils import (
    process_grounding_points,
    get_wallx_normal_text,
    replace_action_token,
    preprocesser_call,
)

from transformers import AutoProcessor
from .utils import KEY_MAPPINGS



T_co = TypeVar("T_co", covariant=True)


import json
from pathlib import Path


## ---- subtask / high-level task helpers ----
def _load_subtask_id2name(dataset_root: str):
    """subtasks.parquet in your case: index is subtask string, column is subtask_index."""
    try:
        import pandas as pd
        root = Path(dataset_root) if dataset_root is not None else None
        if root is None:
            return None
        p = root / "meta" / "subtasks.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if "subtask_index" not in df.columns:
            return None
        # label is stored in index named 'subtask'
        return {int(v): str(k) for k, v in df["subtask_index"].items()}
    except Exception:
        return None

def _load_high_level_id2text(dataset_root: str, prefer_col: str = "user_prompt"):
    """tasks_high_level.parquet: key is task_index; text can come from a column (default user_prompt) or index."""
    try:
        import pandas as pd
        root = Path(dataset_root) if dataset_root is not None else None
        if root is None:
            return None
        p = root / "meta" / "tasks_high_level.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if "task_index" not in df.columns:
            return None
        if prefer_col in df.columns:
            return dict(zip(df["task_index"].astype(int).tolist(), df[prefer_col].astype(str).tolist()))
        # fallback: use index string (in your file it's the 'task' index)
        return dict(zip(df["task_index"].astype(int).tolist(), df.index.astype(str).tolist()))
    except Exception:
        return None

def _load_high_level_cot_map(dataset_root: str):
    """tasks_high_level.parquet: map task_index -> cot fields (question/answer/instruction)."""
    try:
        import pandas as pd
        root = Path(dataset_root) if dataset_root is not None else None
        if root is None:
            return None
        p = root / "meta" / "tasks_high_level.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if "task_index" not in df.columns:
            return None
        if "scenario_type" in df.columns:
            df = df[df["scenario_type"].astype(str) == "cot"]
        if "response_type" in df.columns:
            df = df[df["response_type"].astype(str) == "answer"]
        cot_map = {}
        for _, row in df.iterrows():
            task_index = int(row["task_index"])
            cot_map[task_index] = {
                "instruction": str(row.get("task", "")),
                "question": str(row.get("user_prompt", "")),
                "answer": str(row.get("robot_utterance", "")),
            }
        return cot_map
    except Exception:
        return None


def _load_vqa_map(dataset_root: str):
    """qa_labels.parquet: map (episode_index, frame_idx) -> list of QA dicts."""
    try:
        import pandas as pd
        root = Path(dataset_root) if dataset_root is not None else None
        if root is None:
            return None
        p = root / "meta" / "qa_labels.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        required_cols = {"episode_index", "frame_idx", "question", "answer"}
        if not required_cols.issubset(set(df.columns)):
            return None
        vqa_map = {}
        for (episode_index, frame_idx), sub in df.groupby(
            ["episode_index", "frame_idx"]
        ):
            qa_list = []
            for _, row in sub.iterrows():
                qa_list.append(
                    {
                        "question": str(row.get("question", "")),
                        "answer": str(row.get("answer", "")),
                        "type": str(row.get("type", "")),
                    }
                )
            vqa_map[(int(episode_index), int(frame_idx))] = qa_list
        return vqa_map
    except Exception:
        return None

def _load_meta_mappings(dataset_root: str):
    """Load subtask and high-level mappings from <root>/meta."""
    subtask_id2name = _load_subtask_id2name(dataset_root)
    high_level_id2text = _load_high_level_id2text(dataset_root, prefer_col="user_prompt")
    return subtask_id2name, high_level_id2text
# ------------------------------------------

def _resolve_modality_json_path(lerobot_config):
    # 兼容不同命名：modality_json / modality_path
    return lerobot_config.get("modality_json", None) or lerobot_config.get("modality_path", None)

def _build_delta_timestamps(repo_id, dataset_fps, action_horizon, modality_json_path=None):
    """
    action_horizon: 训练侧希望的 horizon 长度，例如 32
    return: dict[str, list[float]]
    """
    # 1) 有 modality.json：用 original_key 列表生成 delta_timestamps
    if modality_json_path is not None and len(str(modality_json_path)) > 0:
        modality = json.loads(Path(modality_json_path).read_text())
        action_orig_keys = [cfg["original_key"] for cfg in modality[KEY_MAPPINGS[repo_id]["action"]].values()]
        return {k: [t / dataset_fps for t in range(action_horizon)] for k in action_orig_keys}

    # 2) 没有 modality.json：保持旧行为
    return {
        KEY_MAPPINGS[repo_id]["action"]: [t / dataset_fps for t in range(action_horizon)]
    }


# Abstract class for dataset
class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class PreprocessedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
        seed=42,
        rank=0,
        world_size=1,
        test_only=False,
    ):
        self.hf_dataset = dataset

        if test_only:
            self._dataset = dataset
        else:
            self._dataset = None
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [0.95, 0.05],
                torch.Generator().manual_seed(seed) if seed is not None else None,
            ) # 再次分成 两部分；区别于之前 分出 0.95 0.05
            self._train()

        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # init configs
        self.config = config
        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", False)
        self.dataload_config = dataload_config
        self.normalizer_action = (normalizer_action,)
        self.normalizer_propri = normalizer_propri
        # self.norm_stats = norm_stats
        self.lerobot_config = lerobot_config

        self.data_config = X2RDataProcessingConfig()
        self.data_config.update(
            train_test_split=self.dataload_config["train_test_split"],
            split_seed=self.dataload_config["split_seed"],
            predict_action_keys=self.dataload_config["predict_action_keys"],
            obs_action_keys=self.dataload_config["obs_action_keys"],
            resolution=self.dataload_config.get("resolution", None),
            priority_order=self.dataload_config.get("priority_order", None),
            generate_subtask_ratio=self.dataload_config.get(
                "generate_subtask_ratio", self.data_config.generate_subtask_ratio
            ),
            generate_vqa_ratio=self.dataload_config.get(
                "generate_vqa_ratio", self.data_config.generate_vqa_ratio
            ),
            generate_cot_ratio=self.dataload_config.get(
                "generate_cot_ratio", self.data_config.generate_cot_ratio
            ),
            vqa_types=self.dataload_config.get(
                "vqa_types", self.data_config.vqa_types
            ),
        )

        self._cam_key_mapping = KEY_MAPPINGS[self.hf_dataset.meta.repo_id]["camera"]
        self._state_key_mapping = KEY_MAPPINGS[self.hf_dataset.meta.repo_id]["state"]
        self._action_key_mapping = KEY_MAPPINGS[self.hf_dataset.meta.repo_id]["action"]
        self._subtask_id2name, self._high_level_id2text = _load_meta_mappings(
            self.lerobot_config.get("root", None)
        )
        self._high_level_cot_map = _load_high_level_cot_map(
            self.lerobot_config.get("root", None)
        )
        self._vqa_map = _load_vqa_map(self.lerobot_config.get("root", None))


    def _vision_preprocess(self, frames):
        processed_frames = []
        for key in self.hf_dataset.meta.camera_keys:
            from PIL import Image

            current_obs = frames[key].clone().permute(1, 2, 0) # CHW -> HWC，因为 PIL / numpy 图像常用 HWC。

            img_pil = Image.fromarray((current_obs * 255).to(torch.uint8).cpu().numpy()) # 把 [0,1] 映射到 [0,255]。
            orig_width, orig_height = img_pil.size
            # 2. Apply resolution constraints (if config is not -1) 保持纵横比，并把“较长边”缩放到 target_size。
            target_size = self.data_config.resolution.get(
                self._cam_key_mapping[key], -1
            )
            if target_size != -1:
                # Maintain aspect ratio logic
                if orig_width > orig_height:  # Landscape image
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:  # Portrait image
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            # 3. Apply smart scaling (qwen logic) 第二层 resize：不是你指定某个边长，而是按模型/实现的约束把尺寸调整到“合法尺寸”。
            current_width, current_height = img_pil.size
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=self.data_config.image_factor,
                min_pixels=self.data_config.min_pixels,
                max_pixels=self.data_config.max_pixels,
            )
            resized_img = img_pil.resize((resized_width, resized_height))
            processed_frames.append(resized_img)

        return processed_frames, orig_height, orig_width, resized_height, resized_width

    def __getitem__(self, index):
        data = self._dataset[index]
        image_inputs, h, w, resize_h, resize_w = self._vision_preprocess(data)
        agent_pos = data[self._state_key_mapping]
        action = data[self._action_key_mapping]
        frame_index = data["frame_index"]
        instruction_info = {"instruction": data["task"]}
        
        # 新增标记 If available, attach subtask label for auxiliary subtask-generation training
        if self._subtask_id2name is not None and 'subtask_index' in data:
            sid = int(data['subtask_index'])
            name = self._subtask_id2name.get(sid, '')
            if name:
                instruction_info['subtask_generation'] = name
        
        # 新增标记 Optionally replace base instruction with high-level task text
        if self._high_level_id2text is not None and "task_index_high_level" in data:
            
            hid = int(data['task_index_high_level'])
            htxt = self._high_level_id2text.get(hid, '')
            if htxt:
                instruction_info['instruction'] = htxt

        # 新增标记: attach CoT prompt/answer (from tasks_high_level.parquet)
        if self._high_level_cot_map is not None and "task_index_high_level" in data:
            hid = int(data["task_index_high_level"])
            cot_item = self._high_level_cot_map.get(hid)
            if cot_item:
                instruction_info["cot_instruction"] = cot_item.get("instruction", "")
                instruction_info["cot_question"] = cot_item.get("question", "")
                instruction_info["cot_answer"] = cot_item.get("answer", "")

        # 新增标记: attach VQA question/answer (from qa_labels.parquet)
        if (
            self._vqa_map is not None
            and "vqa" in data
            and int(data["vqa"]) == 1
        ):
            episode_index = data.get("episode_index", None)
            if episode_index is not None:
                key = (int(episode_index), int(frame_index))
                qa_list = self._vqa_map.get(key)
                if qa_list:
                    allowed_types = self.data_config.vqa_types
                    if allowed_types:
                        qa_list = [
                            qa
                            for qa in qa_list
                            if str(qa.get("type", "")) in allowed_types
                        ]
                    if qa_list:
                        qa_item = random.choice(qa_list)
                        instruction_info["vqa_question"] = qa_item.get(
                            "question", ""
                        )
                        instruction_info["vqa_answer"] = qa_item.get("answer", "")
                        instruction_info["vqa_type"] = qa_item.get("type", "")
        
        generate_subtask_ratio = self.data_config.generate_subtask_ratio
        generate_vqa_ratio = self.data_config.generate_vqa_ratio
        generate_cot_ratio = self.data_config.generate_cot_ratio

        complete_text, generate_subtask = get_wallx_normal_text(
            instruction_info,
            self.dataload_config.get("action_horizon", 33) - 1,
            frame_index,
            self.data_config.priority_order,
            self._cam_key_mapping,
            generate_subtask_ratio=generate_subtask_ratio,
            generate_vqa_ratio=generate_vqa_ratio,
            generate_cot_ratio=generate_cot_ratio,
        )
        text = process_grounding_points(
            complete_text, h, w, resize_h, resize_w, self.data_config.model_type
        )
        result = {
            "image_inputs": image_inputs, 
            "text": text,
            "action": action, # [32,19]
            "agent_pos": agent_pos, # [19]
            "frame_index": frame_index,
        }

        return result

    def __len__(self) -> int:
        return len(self._dataset)

    def _eval(self):
        self._dataset = self.val_dataset

    def _train(self):
        self._dataset = self.train_dataset

    def get_train_dataloader(self):
        """
        Get distributed training dataloader

        Args:
            rank: Current process rank
            world_size: Total number of processes
            seed: Random seed for reproducibility
        """
        self._train()

        batch_size = self.config.get("batch_size_per_gpu", 8)
        num_workers = self.config.get("num_workers", 4)

        # Create distributed sampler
        sampler = DistributedSampler(
            self,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
            drop_last=True,  # Ensure all processes have same number of batches
        )

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,  # Use distributed sampler instead of shuffle=True
            num_workers=num_workers,
            collate_fn=DataCollator(
                self.config,
                self.dataload_config,
                self.normalizer_action,
                self.normalizer_propri,
                self.lerobot_config,
            ),
            pin_memory=True,  # Enable for GPU training
            persistent_workers=num_workers > 0,  # Only if num_workers > 0
            prefetch_factor=2,  # Reduce memory usage
            drop_last=True,  # Avoid incomplete batches
        )

        return dataloader, sampler

    def get_val_dataloader(self):
        """
        Get distributed evaluation dataloader (no shuffling for consistent evaluation)
        """
        self._eval()

        batch_size = self.config.get(
            "eval_batch_size_per_gpu", self.config.get("batch_size_per_gpu", 8)
        )
        num_workers = self.config.get("num_workers", 4)

        # Create distributed sampler for evaluation (no shuffle)
        sampler = DistributedSampler(
            self,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,  # No shuffling for evaluation
            drop_last=False,  # Keep all samples for evaluation
        )

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=DataCollator(
                self.config, self.dataload_config, self.norm_stats, self.lerobot_config
            ),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
            drop_last=False,
        )

        return dataloader, sampler


class DataCollator:
    # Class-level cache for processors to avoid reloading
    _processor_cache = {}
    _action_tokenizer_cache = {}

    def __init__(
        self,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
    ):
        self.config = config
        self.dataload_config = dataload_config

        self.normalizer_action = normalizer_action[0]
        self.normalizer_propri = normalizer_propri
        self.lerobot_config = lerobot_config

        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", False)
        self.dataset_name = self.config["data"]["lerobot_config"].get("repo_id", "")
        self.dataset_name = [self.dataset_name] * self.config["batch_size_per_gpu"]
        self.load_processor()

    def load_processor(self):
        processor_path = self.config["pretrained_wallx_path"] # 区分processor 一个用于 collate，一个挂在 model 上用于处理输入与动作语义。
        action_tokenizer_path = self.config.get("action_tokenizer_path", None)

        if (
            self.use_fast_tokenizer
            and action_tokenizer_path not in self._action_tokenizer_cache
        ):
            self._action_tokenizer_cache[action_tokenizer_path] = (
                AutoProcessor.from_pretrained(
                    action_tokenizer_path, trust_remote_code=True
                )
            )

        # Use cached processors if available
        if processor_path not in self._processor_cache:
            processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
            if self.config.get("padding_side", "left") == "left":
                processor.tokenizer.padding_side = "left"

            new_tokens = ["<|propri|>", "<|action|>"]
            processor.tokenizer.add_tokens(new_tokens)
            if self.use_fast_tokenizer and self.config.get("model_type") == "qwen2_5":
                action_tokenizer = self._action_tokenizer_cache[action_tokenizer_path]
                new_tokens = [
                    f"<|action_token_{i}|>" for i in range(action_tokenizer.vocab_size)
                ]
                processor.tokenizer.add_tokens(new_tokens)
                begin_idx_token = "<|action_token_0|>"
                token_id = processor.tokenizer.convert_tokens_to_ids(begin_idx_token)
                processor.tokenizer.init_kwargs["action_token_start_index"] = token_id
                processor.tokenizer.init_kwargs["action_token_vocab_size"] = (
                    action_tokenizer.vocab_size
                )

            self._processor_cache[processor_path] = processor

        self.processor = self._processor_cache[processor_path]

        if not self.use_fast_tokenizer:
            self.train_action_tokenizer = None
        else:
            self.train_action_tokenizer = self._action_tokenizer_cache[
                action_tokenizer_path
            ]

    @classmethod
    def _normalize(cls, action, min_stat, delta):
        """
        Normalize action data using min-max normalization.
        """
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        x = (action - min_stat) / delta
        x = x * 2 - 1
        x = torch.clamp(x, -1, 1)
        return x

    def __call__(self, batch):
        additional_inputs = {}

        for key in batch[0].keys():
            if key == "agent_pos":
                agent_pos = torch.stack([item["agent_pos"] for item in batch])
                if agent_pos.dim() == 2:
                    agent_pos = agent_pos.unsqueeze(1)
                agent_pos_mask = (~torch.isnan(agent_pos)).float()
                # print("agent_pos_mask",agent_pos_mask.shape)
                agent_pos.nan_to_num_(nan=0.0)

                agent_pos = self.normalizer_propri.normalize_data(
                    agent_pos, self.dataset_name
                )
                
                if agent_pos.shape[-1] != 20:
                    agent_pos = torch.cat(
                        [
                            agent_pos,
                            torch.zeros(
                                agent_pos.shape[0],
                                agent_pos.shape[1],
                                20 - agent_pos.shape[-1],
                            ),
                        ],
                        dim=-1,
                    )
                    agent_pos_mask = torch.cat(
                        [
                            agent_pos_mask,
                            torch.zeros(
                                agent_pos_mask.shape[0],
                                agent_pos_mask.shape[1],
                                20 - agent_pos_mask.shape[-1],
                            ),
                        ],
                        dim=-1,
                    )

                additional_inputs["proprioception"] = agent_pos
                additional_inputs["agent_pos_mask"] = agent_pos_mask
            elif key == "action":
                action = torch.stack([item["action"] for item in batch]) # [8,32,19]
                if action.dim() == 2:
                    action = action.unsqueeze(1)
                dof_mask = (~torch.isnan(action)).float()
                action.nan_to_num_(nan=0.0)
                
                # 一旦补充到20维度，似乎意味着normalizer_action内容也必须有20维度 所以这部分应该要放到前面
                action = self.normalizer_action.normalize_data(
                    action, self.dataset_name
                )
                
                if action.shape[-1] != 20:
                    action = torch.cat(
                        [
                            action,
                            torch.zeros(
                                action.shape[0], action.shape[1], 20 - action.shape[-1]
                            ),
                        ],
                        dim=-1,
                    )
                    dof_mask = torch.cat(
                        [
                            dof_mask,
                            torch.zeros(
                                dof_mask.shape[0],
                                dof_mask.shape[1],
                                20 - dof_mask.shape[-1],
                            ),
                        ],
                        dim=-1,
                    ) # 1 表示该维度有效，0 表示无效。“新增维度”全部无效

                additional_inputs["action_chunk"] = action
                additional_inputs["dof_mask"] = dof_mask
            elif key == "image_inputs":
                additional_inputs["image_inputs"] = [
                    item["image_inputs"] for item in batch
                ]
            elif key == "text":
                additional_inputs["text"] = [item["text"] for item in batch]
            elif key == "frame_index":
                additional_inputs["frame_index"] = torch.stack(
                    [item["frame_index"] for item in batch]
                )
            else:
                raise NotImplementedError(
                    f"{key} input not implemented in preprocesser"
                )

        additional_inputs["text"] = replace_action_token(
            additional_inputs["text"],
            additional_inputs["action_chunk"],
            self.train_action_tokenizer if self.use_fast_tokenizer else None,
            [self.lerobot_config["repo_id"]] * additional_inputs["text"].__len__(),
            additional_inputs["dof_mask"],
        ) # 这里是结尾的一些删除之类 处理FAST的结尾

        inputs = preprocesser_call(
            processor=self.processor,
            text=additional_inputs.pop("text"),
            images=additional_inputs.pop("image_inputs"),
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.dataload_config.get("max_length", 1024),
        ) # 把“一条或一批多模态对话样本（text + images/videos，占位符如 <|image_pad|> 等）”处理成模型可直接 model(**batch) 的输入张量，并生成只在 assistant 回复段计算交叉熵的 labels。在 labels 里把 <|action|>（以及 <|propri|>）全部 mask 掉，不参与 Loss ; 动作Loss全在flow

        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")

        # Gating token types 哪些 token 位置是 “action token”（用于 MoE gating 或分支路由） 决定走哪个 expert
        additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id

        inputs.update(additional_inputs) 
        
        inputs["dataset_names"] = [self.lerobot_config["repo_id"]] * inputs[
            "action_chunk"
        ].shape[0] # ["g1custom", "g1custom", ..., "g1custom"]  # B 个

        return inputs


def load_lerobot_data(
    config,
    lerobot_config,
    normalizer_action,
    normalizer_propri,
    rank=0,
    world_size=1,
    seed=42,
):
    """
    Load LeRobot dataset with distributed support

    Args:
        config: Model configuration
        rank: Current process rank (default: 0)
        world_size: Total number of processes (default: 1)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        dataset: Training dataset
        train_num: Number of training samples per process
        sampler: Distributed sampler (None if world_size=1)
    """

    # Set seed for reproducibility
    torch.manual_seed(seed)

    dataload_config = get_data_configs(config["data"])

    repo_id = lerobot_config.get("repo_id", None)
    assert repo_id is not None, "repo id is required"
    root = lerobot_config.get("root", None)
    meta_info = LeRobotDatasetMetadata(repo_id, root=root)
    dataset_fps = meta_info.fps
    episodes_num = meta_info.total_episodes

    # norm_stats_path = config.get("norm_stats_path", None)
    # assert (
    #     norm_stats_path is not None
    # ), "norm stats is required, please refer to 'wall-x/scripts/compute_norm_stats.py' to compute stats"
    # norm_stats = load_norm_stats(norm_stats_path, repo_id)

    modality_json_path = _resolve_modality_json_path(lerobot_config)
    horizon = dataload_config.get("action_horizon", 33) - 1  # 例如 32

    delta_timestamps = _build_delta_timestamps(
        repo_id=repo_id,
        dataset_fps=dataset_fps,
        action_horizon=horizon,
        modality_json_path=modality_json_path,
    ) # len = 2

    batch_size = config.get("batch_size_per_gpu", 8)
    episodes = np.arange(episodes_num).tolist()

    train_test_split = dataload_config.get("train_test_split", 0.95)
    train_episodes = episodes[: int(episodes_num * train_test_split)]
    test_episodes = episodes[int(episodes_num * train_test_split) :]
    
    if modality_json_path is not None and len(str(modality_json_path)) > 0:
        from wall_x.data.modality_wrapper import ModalityAwareLeRobotDataset

        train_dataset = ModalityAwareLeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=train_episodes,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
            modality_json=modality_json_path,
            # action_horizon=horizon
            # state_key=KEY_MAPPINGS[repo_id]["state"],   # "state"
            # action_key=KEY_MAPPINGS[repo_id]["action"], # "action"
        )
    else:
        train_dataset = LeRobotDataset(
            repo_id,
            root=root,
            episodes=train_episodes,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )

    if rank == 0:
        print(f"Selected train episodes: {train_dataset.episodes}")
        print(f"Number of train episodes selected: {train_dataset.num_episodes}")
        print(f"Number of train frames selected: {train_dataset.num_frames}")
        print(f"Selected test episodes: {test_episodes}")

    dataset = PreprocessedDataset(
        train_dataset,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Calculate samples per process
    if world_size > 1:
        # With DistributedSampler, each process gets approximately len(dataset) // world_size samples
        samples_per_process = len(dataset) // world_size
        train_num = samples_per_process // batch_size
    else:
        train_num = len(dataset) // batch_size

    if rank == 0:
        print("\n" + "=" * 50)
        print("LeRobot Data Loading Configuration:")
        print(f"✦ RANK: {rank}")
        print(f"✦ WORLD SIZE: {world_size}")
        print(f"✦ BATCH SIZE PER GPU: {batch_size}")
        print(f"✦ REPO ID: {repo_id}")
        print(f"✦ TOTAL DATASET SIZE: {len(dataset)}")
        if world_size > 1:
            print(f"✦ SAMPLES PER PROCESS: {samples_per_process}")
            print(f"✦ BATCHES PER PROCESS: {train_num}")
            print(f"✦ TOTAL BATCHES (ALL PROCESSES): {train_num * world_size}")
        else:
            print(f"✦ TOTAL BATCHES: {train_num}")
        print(f"✦ SEED: {seed}")
        print("=" * 50 + "\n")

    return dataset, train_num


def get_distributed_dataloader(
    dataset, config, rank=0, world_size=1, seed=42, is_train=True
):
    """
    Helper function to get distributed dataloader

    Args:
        dataset: PreprocessedDataset instance
        config: Configuration dict
        rank: Current process rank
        world_size: Total number of processes
        seed: Random seed
        is_train: Whether this is for training (affects shuffling)

    Returns:
        dataloader: Distributed DataLoader
        sampler: DistributedSampler
    """
    if is_train:
        return dataset.get_train_dataloader(rank=rank, world_size=world_size, seed=seed)
    else:
        return dataset.get_val_dataloader(rank=rank, world_size=world_size)


def get_data_configs(config):
    default_data_config = {
        "train_test_split": 0.95,
        "split_seed": 42,
        "batch_size": 8,
        "action_horizon": 21,
        "action_history_length": 0,
        "image_horizon": 1,
        "image_history_length": 0,
        "left_padding": False,
        "right_padding": False,
        "return_first_obs": False,
        "return_last_obs": False,
        "randomize_obs_after": None,
        "datasets": [],
        "labeled_pathes": [],
    }
    data_config = default_data_config | config
    data_config["action_horizon"] += 1

    return data_config


class TestDataset(PreprocessedDataset):
    def __init__(
        self,
        dataset,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
        seed=42,
    ):
        super().__init__(
            dataset,
            config,
            dataload_config,
            normalizer_action,
            normalizer_propri,
            lerobot_config,
            seed=seed,
            rank=0,
            world_size=1,
            test_only=True,
        )

    def get_dataloader(self):
        """
        Get distributed evaluation dataloader (no shuffling for consistent evaluation)
        """

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=1,
            collate_fn=DataCollator(
                self.config,
                self.dataload_config,
                self.normalizer_action,
                self.normalizer_propri,
                self.lerobot_config,
            ),
        )

        return dataloader


def load_test_dataset(
    config,
    lerobot_config,
    normalizer_action,
    normalizer_propri,
    seed=42,
    episode=0,
):
    """
    Load test dataset

    Args:
        config: Model configuration
        seed: Random seed for reproducibility (default: 42)

    Returns:
        dataset: Test dataset
    """

    # Set seed for reproducibility
    torch.manual_seed(seed)

    repo_id = lerobot_config.get("repo_id", None)
    assert repo_id is not None, "repo id is required"
    root = lerobot_config.get("root", None)
    meta_info = LeRobotDatasetMetadata(repo_id, root=root)
    dataset_fps = meta_info.fps
    dataload_config = get_data_configs(config["data"])

    norm_stats_path = config.get("norm_stats_path", None)
    assert (
        norm_stats_path is not None
    ), "norm stats is required, please refer to 'wall-x/scripts/compute_norm_stats.py' to compute stats"
    # norm_stats = load_norm_stats(norm_stats_path, repo_id)

    modality_json_path = _resolve_modality_json_path(lerobot_config)
    horizon = dataload_config.get("action_horizon", 33) - 1  # 例如 32

    delta_timestamps = _build_delta_timestamps(
        repo_id=repo_id,
        dataset_fps=dataset_fps,
        action_horizon=horizon,
        modality_json_path=modality_json_path,
    )
    
    if modality_json_path is not None and len(str(modality_json_path)) > 0:
        from wall_x.data.modality_wrapper import ModalityAwareLeRobotDataset

        dataset = ModalityAwareLeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=[episode],
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
            modality_json=modality_json_path,
            # action_horizon=horizon,
            # state_key=KEY_MAPPINGS[repo_id]["state"],   # "state"
            # action_key=KEY_MAPPINGS[repo_id]["action"], # "action"
        )
    else:
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            episodes=[episode],
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )

    print(f"Selected episodes: {dataset.episodes}")
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    dataset = TestDataset(
        dataset,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
        seed=seed,
    )

    return dataset
