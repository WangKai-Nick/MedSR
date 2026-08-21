import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import torch
import argparse
import logging
import random
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import build_medsr
from utils.main_utils import load_cfg_from_cfg_file, read_text

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
    "--config-file",
    required=True,
    type=str,
    help="Path to config file",
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        '--prompt_design',
        type=str,
        default="original",
        help="Text prompt design."
    )

    parser.add_argument(
        "--data_percentage", 
        type=int, 
        default=100, 
        help="Percentage of data to use.")
    
    parser.add_argument(
        "--source_dataset", 
        type=str, 
        help="source dataset name for loading trained model.")
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="", 
        help="output directory")
    
    parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="modify config options using the command-line",
        )
    
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)

    cfg.merge_from_list(args.opts)

    cfg.update({k: v for k, v in vars(args).items()})    

    return cfg

def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def synchronize_if_needed(device):
    if "cuda" in str(device).lower():
        torch.cuda.synchronize(device)

def main():
    cfg = get_arguments()

    if cfg.seed >= 0:
        print(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)
    
    cfg.DATASET.NAME = cfg.DATASET.NAME+f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
    results_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME, f"seg_results", f"seed{cfg.seed}")
    os.makedirs(results_root, exist_ok=True)
    logger = logger_config(os.path.join(results_root, "log.txt"))

    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"MedSR_{backbone_name}"

    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"
    checkpoint_path = os.path.join(
        cfg.output_dir,
        cfg.source_dataset if cfg.data_percentage == 100 else cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_{checkpoint_type}.pth"
    )

    model = build_medsr(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    model.eval().to(cfg.MODEL.DEVICE)

    test_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    test_text_file = f"Test_text_{cfg.prompt_design}.xlsx"
    test_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + test_text_file)

    test_dataset = DatasetSegmentation(cfg.DATASET.TEST_PATH, cfg.DATASET.NAME,
                                        test_text, test_tf, image_size=cfg.DATASET.SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_inference_time = 0.0
    pla_time = 0.0
    pla_per_image_time = 0.0
    sar_time = 0.0
    sar_per_image_time = 0.0
    timed_batches = 0
    timed_images = 0
    warmup_batches = min(10, max(len(test_dataloader) - 1, 0))

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            images = batch["image"].to(cfg.MODEL.DEVICE)
            count_timing = batch_idx >= warmup_batches

            synchronize_if_needed(cfg.MODEL.DEVICE)
            total_start_time = time.time()

            seg_logits = torch.sigmoid(model(image=images, text=batch["text_prompt"]))
            if seg_logits.ndim != 3:
                raise RuntimeError(f"Expected MedSR output shape [B,H,W], got {tuple(seg_logits.shape)}.")
            mask_preds = seg_logits > 0.5

            synchronize_if_needed(cfg.MODEL.DEVICE)
            elapsed_time = time.time() - total_start_time

            if count_timing:
                total_inference_time += elapsed_time
            if count_timing and hasattr(model, "last_inference_timing") and model.last_inference_timing:
                pla_time += model.last_inference_timing.get("pixel_level_adaptation", 0.0)
                pla_per_image_time += model.last_inference_timing.get("pixel_level_adaptation_per_image", 0.0)
                sar_time += model.last_inference_timing.get("structure_aware_refinement", 0.0)
                sar_per_image_time += model.last_inference_timing.get("structure_aware_refinement_per_image", 0.0)

            if count_timing:
                timed_batches += 1
                timed_images += images.shape[0]

            dataset_names = batch["dataset_name"]   # list of strings
            mask_names = batch["mask_name"]         # list of strings

            for i in range(len(dataset_names)):

                pred_mask = mask_preds[i].cpu().numpy().astype(np.uint8)

                dataset_name = dataset_names[i]
                mask_name = mask_names[i]
                
                # Then in the loop:
                binary_pred = np.uint8(pred_mask > 0)

                save_dir = os.path.join(cfg.output_dir,
                                        cfg.DATASET.NAME,
                                        f"seg_results",
                                        f"seed{cfg.seed}",
                                        results_name + f"_Prompt-{cfg.prompt_design}")
                
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, mask_name), binary_pred * 255)

    if timed_batches > 0:
        avg_runtime_per_batch = total_inference_time / timed_batches
        total_images = timed_images
        fps = total_images / total_inference_time if total_inference_time > 0 else 0.0
        time_per_image = total_inference_time / total_images if total_images > 0 else 0.0

        logger.info(f"Warmup batches skipped for timing: {warmup_batches}")
        logger.info(f"Average Total inference time: {avg_runtime_per_batch:.6f} s/batch")
        logger.info(f"Runtime (s/batch): {avg_runtime_per_batch:.6f}")
        logger.info(f"FPS: {fps:.6f}")
        logger.info(f"Time per image (s/image): {time_per_image:.6f}")
        logger.info(f"Average PLA time: {pla_time / timed_batches:.6f} s/batch")
        logger.info(f"Average PLA time per image: {pla_per_image_time / timed_batches:.6f} s/image")
        logger.info(f"Average SAR time: {sar_time / timed_batches:.6f} s/batch")
        logger.info(f"Average SAR time per image: {sar_per_image_time / timed_batches:.6f} s/image")

        timing_summary_path = os.path.join(results_root, "timing_summary.txt")
        with open(timing_summary_path, "w", encoding="utf-8") as f:
            f.write("=== Timing Summary ===\n")
            f.write(f"config_file: {cfg.config_file}\n")
            f.write(f"source_dataset: {cfg.source_dataset}\n")
            f.write(f"dataset_name: {cfg.DATASET.NAME}\n")
            f.write(f"model: {cfg.MODEL.NAME}\n")
            f.write(f"backbone: {cfg.MODEL.BACKBONE}\n")
            f.write(f"batch_size: {test_dataloader.batch_size}\n")
            f.write(f"warmup_batches_skipped: {warmup_batches}\n")
            f.write(f"timed_batches: {timed_batches}\n")
            f.write(f"timed_images: {total_images}\n")
            f.write(f"total_dataset_images: {len(test_dataset)}\n")
            f.write(f"checkpoint: {checkpoint_path}\n")
            f.write(f"Runtime (s/batch): {avg_runtime_per_batch:.6f}\n")
            f.write(f"FPS: {fps:.6f}\n")
            f.write(f"Time per image (s/image): {time_per_image:.6f}\n")
            f.write(f"Average PLA time (s/batch): {pla_time / timed_batches:.6f}\n")
            f.write(f"Average PLA time per image (s/image): {pla_per_image_time / timed_batches:.6f}\n")
            f.write(f"Average SAR time (s/batch): {sar_time / timed_batches:.6f}\n")
            f.write(f"Average SAR time per image (s/image): {sar_per_image_time / timed_batches:.6f}\n")

if __name__ == "__main__":
    main()
