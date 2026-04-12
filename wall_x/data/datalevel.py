import hashlib
import random
from typing import Dict, List


def _normalize_subtask(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _stable_int_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


LEVEL2_SUBTASK_MAP: Dict[str, List[str]] = {
    "place the toy car in the brown basket.": [
        "Pick up the toy car and place it in the brown basket.",
        "Put the toy car into the brown basket.",
        "Make sure the toy car ends up in the brown basket.",
        "Please place the toy car in the brown basket.",
    ],
    "place the avocado in the black basket.": [
        "Pick up the avocado and place it in the black basket.",
        "Put the avocado into the black basket.",
        "Make sure the avocado ends up in the black basket.",
        "Please place the avocado in the black basket.",
    ],
    "place the grape in the black basket.": [
        "Pick up the grape and place it in the black basket.",
        "Put the grape into the black basket.",
        "Make sure the grape ends up in the black basket.",
        "Please place the grape in the black basket.",
    ],
    "place the kiwi in the black basket.": [
        "Pick up the kiwi and place it in the black basket.",
        "Put the kiwi into the black basket.",
        "Make sure the kiwi ends up in the black basket.",
        "Please place the kiwi in the black basket.",
    ],
    "place the apple in the black basket.": [
        "Pick up the apple and place it in the black basket.",
        "Put the apple into the black basket.",
        "Make sure the apple ends up in the black basket.",
        "Please place the apple in the black basket.",
    ],
    "place the orange in the black basket.": [
        "Pick up the orange and place it in the black basket.",
        "Put the orange into the black basket.",
        "Make sure the orange ends up in the black basket.",
        "Please place the orange in the black basket.",
    ],
    "place the pumpkin in the black basket.": [
        "Pick up the pumpkin and place it in the black basket.",
        "Put the pumpkin into the black basket.",
        "Make sure the pumpkin ends up in the black basket.",
        "Please place the pumpkin in the black basket.",
    ],
}

LEVEL2_SUBTASK_MAP_NORMALIZED = {
    _normalize_subtask(k): v for k, v in LEVEL2_SUBTASK_MAP.items()
}


def Level2ExtractReplace(
    origin_subtask: str,
    time: int,
    replace_ratio: float = 1.0,
) -> str:
    normalized = _normalize_subtask(origin_subtask)
    candidates = LEVEL2_SUBTASK_MAP_NORMALIZED.get(normalized)

    if not candidates:
        return origin_subtask

    seed = _stable_int_hash(f"{normalized}__{time}")
    rng = random.Random(seed)

    if rng.random() > replace_ratio:
        return origin_subtask

    return rng.choice(candidates)

LEVEL4_SUBTASK_MAP: Dict[str, List[str]] = {
    "place the toy car in the brown basket.": [
        "Place the toy car in the brown basket.",
        "Put the toy car into the brown basket.",
        "Move the toy car into the brown basket.",
        "Pick up the toy car and place it in the brown basket.",
        "Transfer the toy car to the brown basket.",
        "Set the toy car inside the brown basket.",
        "Take the toy car and drop it into the brown basket.",
        "Place the toy car inside the brown basket.",
        "Move the toy car to the brown basket.",
        "Put the toy car in the brown basket.",
        "Pick up the toy car and put it in the brown basket.",
        "Grab the toy car and place it in the brown basket.",
        "Grab the toy car and put it into the brown basket.",
        "Lift the toy car and place it in the brown basket.",
        "Lift the toy car and put it into the brown basket.",
        "Carry the toy car to the brown basket.",
        "Take the toy car to the brown basket and place it inside.",
        "Move the toy car over to the brown basket.",
        "Set the toy car down in the brown basket.",
        "Relocate the toy car into the brown basket.",
    ],
    "place the avocado in the black basket.": [
        "Place the avocado in the black basket.",
        "Put the avocado into the black basket.",
        "Move the avocado into the black basket.",
        "Pick up the avocado and place it in the black basket.",
        "Transfer the avocado to the black basket.",
        "Set the avocado inside the black basket.",
        "Take the avocado and drop it into the black basket.",
        "Place the avocado inside the black basket.",
        "Move the avocado to the black basket.",
        "Put the avocado in the black basket.",
        "Pick up the avocado and put it in the black basket.",
        "Grab the avocado and place it in the black basket.",
        "Grab the avocado and put it into the black basket.",
        "Lift the avocado and place it in the black basket.",
        "Lift the avocado and put it into the black basket.",
        "Carry the avocado to the black basket.",
        "Take the avocado to the black basket and place it inside.",
        "Move the avocado over to the black basket.",
        "Set the avocado down in the black basket.",
        "Relocate the avocado into the black basket.",
    ],
    "place the grape in the black basket.": [
        "Place the grape in the black basket.",
        "Put the grape into the black basket.",
        "Move the grape into the black basket.",
        "Pick up the grape and place it in the black basket.",
        "Transfer the grape to the black basket.",
        "Set the grape inside the black basket.",
        "Take the grape and drop it into the black basket.",
        "Place the grape inside the black basket.",
        "Move the grape to the black basket.",
        "Put the grape in the black basket.",
        "Pick up the grape and put it in the black basket.",
        "Grab the grape and place it in the black basket.",
        "Grab the grape and put it into the black basket.",
        "Lift the grape and place it in the black basket.",
        "Lift the grape and put it into the black basket.",
        "Carry the grape to the black basket.",
        "Take the grape to the black basket and place it inside.",
        "Move the grape over to the black basket.",
        "Set the grape down in the black basket.",
        "Relocate the grape into the black basket.",
    ],
    "place the kiwi in the black basket.": [
        "Place the kiwi in the black basket.",
        "Put the kiwi into the black basket.",
        "Move the kiwi into the black basket.",
        "Pick up the kiwi and place it in the black basket.",
        "Transfer the kiwi to the black basket.",
        "Set the kiwi inside the black basket.",
        "Take the kiwi and drop it into the black basket.",
        "Place the kiwi inside the black basket.",
        "Move the kiwi to the black basket.",
        "Put the kiwi in the black basket.",
        "Pick up the kiwi and put it in the black basket.",
        "Grab the kiwi and place it in the black basket.",
        "Grab the kiwi and put it into the black basket.",
        "Lift the kiwi and place it in the black basket.",
        "Lift the kiwi and put it into the black basket.",
        "Carry the kiwi to the black basket.",
        "Take the kiwi to the black basket and place it inside.",
        "Move the kiwi over to the black basket.",
        "Set the kiwi down in the black basket.",
        "Relocate the kiwi into the black basket.",
    ],
    "place the apple in the black basket.": [
        "Place the apple in the black basket.",
        "Put the apple into the black basket.",
        "Move the apple into the black basket.",
        "Pick up the apple and place it in the black basket.",
        "Transfer the apple to the black basket.",
        "Set the apple inside the black basket.",
        "Take the apple and drop it into the black basket.",
        "Place the apple inside the black basket.",
        "Move the apple to the black basket.",
        "Put the apple in the black basket.",
        "Pick up the apple and put it in the black basket.",
        "Grab the apple and place it in the black basket.",
        "Grab the apple and put it into the black basket.",
        "Lift the apple and place it in the black basket.",
        "Lift the apple and put it into the black basket.",
        "Carry the apple to the black basket.",
        "Take the apple to the black basket and place it inside.",
        "Move the apple over to the black basket.",
        "Set the apple down in the black basket.",
        "Relocate the apple into the black basket.",
    ],
    "place the orange in the black basket.": [
        "Place the orange in the black basket.",
        "Put the orange into the black basket.",
        "Move the orange into the black basket.",
        "Pick up the orange and place it in the black basket.",
        "Transfer the orange to the black basket.",
        "Set the orange inside the black basket.",
        "Take the orange and drop it into the black basket.",
        "Place the orange inside the black basket.",
        "Move the orange to the black basket.",
        "Put the orange in the black basket.",
        "Pick up the orange and put it in the black basket.",
        "Grab the orange and place it in the black basket.",
        "Grab the orange and put it into the black basket.",
        "Lift the orange and place it in the black basket.",
        "Lift the orange and put it into the black basket.",
        "Carry the orange to the black basket.",
        "Take the orange to the black basket and place it inside.",
        "Move the orange over to the black basket.",
        "Set the orange down in the black basket.",
        "Relocate the orange into the black basket.",
    ],
    "place the pumpkin in the black basket.": [
        "Place the pumpkin in the black basket.",
        "Put the pumpkin into the black basket.",
        "Move the pumpkin into the black basket.",
        "Pick up the pumpkin and place it in the black basket.",
        "Transfer the pumpkin to the black basket.",
        "Set the pumpkin inside the black basket.",
        "Take the pumpkin and drop it into the black basket.",
        "Place the pumpkin inside the black basket.",
        "Move the pumpkin to the black basket.",
        "Put the pumpkin in the black basket.",
        "Pick up the pumpkin and put it in the black basket.",
        "Grab the pumpkin and place it in the black basket.",
        "Grab the pumpkin and put it into the black basket.",
        "Lift the pumpkin and place it in the black basket.",
        "Lift the pumpkin and put it into the black basket.",
        "Carry the pumpkin to the black basket.",
        "Take the pumpkin to the black basket and place it inside.",
        "Move the pumpkin over to the black basket.",
        "Set the pumpkin down in the black basket.",
        "Relocate the pumpkin into the black basket.",
    ],
}

LEVEL4_SUBTASK_MAP_NORMALIZED = {
    _normalize_subtask(k): v for k, v in LEVEL4_SUBTASK_MAP.items()
}


def Level4ExtractReplace(
    origin_subtask: str,
    key: int,
    replace_ratio: float = 1.0,
) -> str:
    """
    用 level4 增强池替换原始指令。

    Args:
        origin_subtask: 原始子任务文本
        key: 用于稳定随机采样的整数标识（如 frame_idx / sample_idx / step_idx）
        replace_ratio: 替换概率
            - 1.0: 一定替换，在候选表达中稳定采样
            - 0.8: 20% 保留原句，80% 在候选表达中稳定采样

    Returns:
        替换后的指令，或原始指令
    """
    normalized = _normalize_subtask(origin_subtask)
    candidates = LEVEL4_SUBTASK_MAP_NORMALIZED.get(normalized)

    if not candidates:
        return origin_subtask

    seed = _stable_int_hash(f"{normalized}__{key}")
    rng = random.Random(seed)

    if rng.random() > replace_ratio:
        return origin_subtask

    return rng.choice(candidates)