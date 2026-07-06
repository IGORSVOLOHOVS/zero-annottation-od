"""
Pipeline: Zero-Annotation Object Detection (Qwen2.5-VL -> YOLOv8)
This script performs automatic image labeling using a Vision-Language Model (vLLM)
and then trains a YOLOv8 model on the generated dataset.
"""

import os
import glob
import random
import json
import shutil
import re
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torch
import gc
from tqdm import tqdm

# --- NUCLEAR PATCH: Fix RoPE and Tokenizer conflicts for Qwen2.5-VL in vLLM ---
import json
import transformers.tokenization_utils_base

# 1. Fix RoPE Scaling (Legacy type -> rope_type)
orig_json_load = json.load
orig_json_loads = json.loads

def fix_rope_config(data):
    if isinstance(data, dict):
        # 1. Fix RoPE Scaling (Legacy type -> rope_type)
        if "rope_scaling" in data and isinstance(data["rope_scaling"], dict):
            rs = data["rope_scaling"]
            if "type" in rs:
                rs["rope_type"] = rs.pop("type")
        
        # 2. Fix missing vocab_size for Qwen2.5-VL in older vLLM
        if data.get("model_type") == "qwen2_5_vl" and "vocab_size" not in data:
            data["vocab_size"] = 151936
            
        for v in data.values():
            if isinstance(v, (dict, list)):
                fix_rope_config(v)
    elif isinstance(data, list):
        for item in data:
            fix_rope_config(item)
    return data

def patched_json_load(*args, **kwargs):
    return fix_rope_config(orig_json_load(*args, **kwargs))

def patched_json_loads(*args, **kwargs):
    return fix_rope_config(orig_json_loads(*args, **kwargs))

json.load = patched_json_load
json.loads = patched_json_loads

# 2. Fix missing 'all_special_tokens_extended' in transformers >= 4.45.0
if not hasattr(transformers.tokenization_utils_base.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

# 3. Aggressive patch for Qwen2.5-VL Config (Fix missing vocab_size)
try:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    orig_qwen_init = Qwen2_5_VLConfig.__init__
    def patched_qwen_init(self, *args, **kwargs):
        orig_qwen_init(self, *args, **kwargs)
        text_cfg = getattr(self, "text_config", None)
        if text_cfg is not None:
            cfg_dict = text_cfg if isinstance(text_cfg, dict) else (text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else text_cfg.__dict__)
            for k, v in cfg_dict.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
        if not hasattr(self, "vocab_size"):
            self.vocab_size = 151936
            setattr(self, "vocab_size", 151936)
    Qwen2_5_VLConfig.__init__ = patched_qwen_init
    print("DEBUG: Qwen2_5_VLConfig aggressively patched.")
except ImportError:
    pass

# 4. Aggressive patch for Qwen2VLImageProcessor (Fix missing min_pixels)
try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    orig_qwen2vl_init = Qwen2VLImageProcessor.__init__
    def patched_qwen2vl_init(self, *args, **kwargs):
        orig_qwen2vl_init(self, *args, **kwargs)
        if not hasattr(self, "min_pixels"):
            self.min_pixels = 3136
        if not hasattr(self, "max_pixels"):
            self.max_pixels = 12845056
    Qwen2VLImageProcessor.__init__ = patched_qwen2vl_init
    print("DEBUG: Qwen2VLImageProcessor aggressively patched.")
except ImportError:
    pass
# --- End of Patch ---

# Constants & Config
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path.cwd()
IMAGES_DIR = PROJECT_DIR / "images"
LABELS_RAW_PATH = PROJECT_DIR / "labels_raw.json"
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# vLLM Parameters
MAX_IMAGE_TOKENS = 1024
MAX_TEXT_TOKENS = 256
MAX_MODEL_LEN = MAX_IMAGE_TOKENS + MAX_TEXT_TOKENS + 512
MAX_PIXELS = MAX_IMAGE_TOKENS * 16 ** 2
BATCH_SIZE = 4

CLASS_MAP = {"helmet": 0, "no_helmet": 1}

def setup_dirs():
    for split in ["train", "val"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

def build_messages(image_path: str):
    system_prompt = """You are a precise object detection annotator. 
For each person visible in the image, determine if they are wearing a safety helmet or not.
Return a JSON array of objects. Each object must have:
- "label": "helmet" or "no_helmet"
- "bbox": [x_min, y_min, x_max, y_max] in pixel coordinates

CRITICAL RULES:
1. The bounding box MUST cover ONLY THE HEAD / HELMET of the person. Do NOT draw the box around their entire body!
2. Coordinates must be integers within image bounds.
3. If no people are visible, return an empty array: []
4. Do NOT include objects that are not people.
5. Output ONLY valid JSON, no extra text."""

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}", "max_pixels": MAX_PIXELS},
            {"type": "text", "text": "Detect all people and classify helmet/no_helmet. Return JSON array."}
        ]}
    ]

def prepare_vllm_input(messages, processor):
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    mm_data = {}
    if image_inputs is not None: mm_data["image"] = image_inputs
    return {"prompt": text, "multi_modal_data": mm_data}

def parse_json_response(text):
    text = text.strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list): return data
        except json.JSONDecodeError: pass
    return []

def convert_to_yolo(detections, img_w, img_h):
    yolo_lines = []
    for det in detections:
        label = det.get("label", "").lower().strip()
        bbox = det.get("bbox", [])
        if label not in CLASS_MAP or len(bbox) != 4: continue
        
        x_min, y_min, x_max, y_max = map(float, bbox)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(x_max, img_w), min(y_max, img_h)
        
        if x_max <= x_min or y_max <= y_min: continue
        
        cx, cy = (x_min + x_max) / 2 / img_w, (y_min + y_max) / 2 / img_h
        w, h = (x_max - x_min) / img_w, (y_max - y_min) / img_h
        
        if 0.005 < w < 0.95 and 0.005 < h < 0.95:
            yolo_lines.append(f"{CLASS_MAP[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return yolo_lines

def main():
    image_paths = sorted(glob.glob(str(IMAGES_DIR / "hard_hat_workers*.png")))
    if not image_paths:
        print(f"Error: No images found in {IMAGES_DIR}")
        return

    print(f"Total images: {len(image_paths)}")
    
    labels_raw = {}
    if LABELS_RAW_PATH.exists():
        print(f"Found existing labels at {LABELS_RAW_PATH}, skipping vLLM inference...")
        with open(LABELS_RAW_PATH, "r", encoding="utf-8") as f:
            labels_raw = json.load(f)
    else:
        # 1. Labeling with vLLM
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        
        print(f"Loading {MODEL_NAME}...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=2,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=1,
            # Force rope_type to avoid Pydantic conflict
            hf_overrides={
                "rope_scaling": {
                    "rope_type": "mrope",
                    "mrope_section": [16, 24, 24]
                }
            },
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(temperature=0.1, max_tokens=512)

        print("Starting auto-labeling...")
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
            batch_paths = image_paths[i : i + BATCH_SIZE]
            batch_inputs = [prepare_vllm_input(build_messages(p), processor) for p in batch_paths]
            
            prompts = [{
                "prompt": inp["prompt"],
                "multi_modal_data": inp["multi_modal_data"]
            } for inp in batch_inputs]
            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            
            for img_path, out in zip(batch_paths, outputs):
                response_text = out.outputs[0].text
                detections = parse_json_response(response_text)
                labels_raw[Path(img_path).stem] = {
                    "image_path": str(img_path),
                    "detections": detections,
                }

        with open(LABELS_RAW_PATH, "w") as f:
            json.dump(labels_raw, f, indent=2, ensure_ascii=False)

        # 2. Cleanup GPU for Training
        del llm
        del processor
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU memory cleared.")

    # 3. Create YOLO Dataset
    setup_dirs()
    valid_items = []
    for name, data in labels_raw.items():
        img_path = Path(data["image_path"])
        if not data["detections"]: continue
        with Image.open(img_path) as img:
            w, h = img.size
        yolo_lines = convert_to_yolo(data["detections"], w, h)
        if yolo_lines: valid_items.append((name, img_path, yolo_lines))

    if not valid_items:
        print("No valid annotations found. Pipeline stopped.")
        return

    from sklearn.model_selection import train_test_split
    train_items, val_items = train_test_split(valid_items, test_size=0.2, random_state=SEED)

    def save_split(items, split):
        for name, path, lines in items:
            shutil.copy2(path, DATASET_DIR / split / "images" / f"{name}.png")
            with open(DATASET_DIR / split / "labels" / f"{name}.txt", "w") as f:
                f.write("\n".join(lines))

    save_split(train_items, "train")
    save_split(val_items, "val")
    
    with open(DATASET_DIR / "data.yaml", "w") as f:
        f.write(f"path: {DATASET_DIR}\ntrain: train/images\nval: val/images\nnames:\n  0: helmet\n  1: no_helmet\n")

    # 4. Train YOLO
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="runs",
        name="helmet_detection",
        exist_ok=True
    )
    print("Training pipeline finished!")

if __name__ == "__main__":
    main()
