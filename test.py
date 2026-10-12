import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import torch
import argparse
import logging
import random
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
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit MedSR checkpoint path; overrides the default path and TEST.USE_LATEST.",
    )
    
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

    if cfg.checkpoint is not None:
        checkpoint_path = os.path.abspath(os.path.expanduser(cfg.checkpoint))
        checkpoint_selection = "explicit --checkpoint"
    else:
        checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"
        checkpoint_path = os.path.join(
            cfg.output_dir,
            cfg.source_dataset if cfg.data_percentage == 100 else cfg.DATASET.NAME,
            "trained_models",
            f"seed{cfg.seed}",
            f"{results_name}_{checkpoint_type}.pth"
        )
        checkpoint_selection = f"default {checkpoint_type}"

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found ({checkpoint_selection}): {checkpoint_path}")
    logger.info(f"Loading checkpoint ({checkpoint_selection}): {checkpoint_path}")

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


    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            images = batch["image"].to(cfg.MODEL.DEVICE)

            seg_logits = torch.sigmoid(model(image=images, text=batch["text_prompt"]))
            if seg_logits.ndim != 3:
                raise RuntimeError(f"Expected MedSR output shape [B,H,W], got {tuple(seg_logits.shape)}.")
            mask_preds = seg_logits > 0.5


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


if __name__ == "__main__":
    main()
