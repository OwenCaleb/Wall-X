import hashlib
import random
from typing import Dict, List


def _normalize_subtask(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


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
}

LEVEL2_SUBTASK_MAP_NORMALIZED = {
    _normalize_subtask(k): v for k, v in LEVEL2_SUBTASK_MAP.items()
}


def _stable_int_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def Level2ExtractReplace(origin_subtask: str, time: int, replace_ratio: float = 1.0) -> str:
    normalized = _normalize_subtask(origin_subtask)
    candidates = LEVEL2_SUBTASK_MAP_NORMALIZED.get(normalized)

    if not candidates:
        return origin_subtask

    seed = _stable_int_hash(f"{normalized}__{time}")
    rng = random.Random(seed)

    if rng.random() > replace_ratio:
        return origin_subtask

    return rng.choice(candidates)


import hashlib
import random
from typing import Dict, List


def _normalize_subtask(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _stable_int_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


LEVEL4_SUBTASK_MAP: Dict[str, List[str]] = {
    "place the toy car in the brown basket.": [
        "Place the toy car in the brown basket.",
        "Put the toy car into the brown basket.",
        "Pick up the toy car and place it in the brown basket.",
        "Please place the toy car in the brown basket.",
        "Make sure the toy car ends up in the brown basket.",
        "Only place the toy car in the brown basket.",
        "Move the toy car to the brown basket, not the fruits.",
        "Begin by placing the toy car in the brown basket.",
        "Take the toy car and put it into the brown basket.",
        "Put the toy car away in the brown basket.",
        "Ensure that the toy car is placed in the brown basket.",
        "Start with the toy car and place it in the brown basket.",
    ],
    "place the avocado in the black basket.": [
        "Place the avocado in the black basket.",
        "Put the avocado into the black basket.",
        "Pick up the avocado and place it in the black basket.",
        "Please place the avocado in the black basket.",
        "Make sure the avocado ends up in the black basket.",
        "Only place the avocado in the black basket.",
        "Move the avocado to the black basket, not the toy car.",
        "Begin by placing the avocado in the black basket.",
        "Take the avocado and put it into the black basket.",
        "Put the avocado away in the black basket.",
        "Ensure that the avocado is placed in the black basket.",
        "Start with the avocado and place it in the black basket.",
    ],
    "place the grape in the black basket.": [
        "Place the grape in the black basket.",
        "Put the grape into the black basket.",
        "Pick up the grape and place it in the black basket.",
        "Please place the grape in the black basket.",
        "Make sure the grape ends up in the black basket.",
        "Only place the grape in the black basket.",
        "Move the grape to the black basket, not the toy car.",
        "Begin by placing the grape in the black basket.",
        "Take the grape and put it into the black basket.",
        "Put the grape away in the black basket.",
        "Ensure that the grape is placed in the black basket.",
        "Start with the grape and place it in the black basket.",
    ],
    "place the kiwi in the black basket.": [
        "Place the kiwi in the black basket.",
        "Put the kiwi into the black basket.",
        "Pick up the kiwi and place it in the black basket.",
        "Please place the kiwi in the black basket.",
        "Make sure the kiwi ends up in the black basket.",
        "Only place the kiwi in the black basket.",
        "Move the kiwi to the black basket, not the toy car.",
        "Begin by placing the kiwi in the black basket.",
        "Take the kiwi and put it into the black basket.",
        "Put the kiwi away in the black basket.",
        "Ensure that the kiwi is placed in the black basket.",
        "Start with the kiwi and place it in the black basket.",
    ],
    "place the apple in the black basket.": [
        "Place the apple in the black basket.",
        "Put the apple into the black basket.",
        "Pick up the apple and place it in the black basket.",
        "Please place the apple in the black basket.",
        "Make sure the apple ends up in the black basket.",
        "Only place the apple in the black basket.",
        "Move the apple to the black basket, not the toy car.",
        "Begin by placing the apple in the black basket.",
        "Take the apple and put it into the black basket.",
        "Put the apple away in the black basket.",
        "Ensure that the apple is placed in the black basket.",
        "Start with the apple and place it in the black basket.",
    ],
    "place the orange in the black basket.": [
        "Place the orange in the black basket.",
        "Put the orange into the black basket.",
        "Pick up the orange and place it in the black basket.",
        "Please place the orange in the black basket.",
        "Make sure the orange ends up in the black basket.",
        "Only place the orange in the black basket.",
        "Move the orange to the black basket, not the toy car.",
        "Begin by placing the orange in the black basket.",
        "Take the orange and put it into the black basket.",
        "Put the orange away in the black basket.",
        "Ensure that the orange is placed in the black basket.",
        "Start with the orange and place it in the black basket.",
    ],
}

LEVEL4_SUBTASK_MAP_NORMALIZED = {
    _normalize_subtask(k): v for k, v in LEVEL4_SUBTASK_MAP.items()
}


def Level4ExtractReplace(origin_subtask: str, key: int, replace_ratio: float = 1.0) -> str:
    """
    用 level4 增强池替换原始指令。

    Args:
        origin_subtask: 原始子任务文本
        key: 用于稳定随机采样的整数标识（如 frame_idx / sample_idx / step_idx）
        replace_ratio: 替换概率
            - 1.0: 一定替换，12种平均采样
            - 0.8: 20% 保留原句，80% 在12种中平均采样

    Returns:
        替换后的指令，或原始指令
    """
    normalized = _normalize_subtask(origin_subtask)
    candidates = LEVEL4_SUBTASK_MAP_NORMALIZED.get(normalized)

    if not candidates:
        return origin_subtask

    seed = _stable_int_hash(f"{normalized}__{key}")
    rng = random.Random(seed)

    # 先决定要不要替换
    if rng.random() > replace_ratio:
        return origin_subtask

    # 12种平均采样
    idx = rng.randrange(len(candidates))
    return candidates[idx]