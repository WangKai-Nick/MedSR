"""UniMedCLIP backbone loading for MedSR."""

import os

from huggingface_hub import hf_hub_download

from open_clip_lib import create_model_and_transforms, get_mean_std


def download_unimedclip_weights(filename: str) -> str:
    """Return a local UniMedCLIP checkpoint, downloading it when needed."""
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    local_path = os.path.join(checkpoint_dir, filename)

    if os.path.isfile(local_path):
        print(f"Found UniMedCLIP checkpoint: {local_path}")
        return local_path

    print(f"Downloading UniMedCLIP checkpoint: {filename}")
    hf_hub_download(
        repo_id="TahaKoleilat/MedCLIPSeg",
        repo_type="model",
        filename=f"checkpoints/{filename}",
        local_dir=".",
        local_dir_use_symlinks=False,
    )
    return local_path


def load_unimedclip_to_device(cfg):
    """Load the frozen UniMedCLIP backbone used by MedSR."""
    if cfg.MODEL.BACKBONE == "ViT-B/16":
        model_name = "ViT-B-16-quickgelu"
        pretrained_weights = download_unimedclip_weights("unimed_clip_vit_b16.pt")
    elif cfg.MODEL.BACKBONE == "ViT-L/14":
        model_name = "ViT-L-14-336-quickgelu"
        pretrained_weights = download_unimedclip_weights("unimed_clip_vit_l14_base_text_encoder.pt")
    else:
        raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")

    mean, std = get_mean_std()
    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained_weights,
        precision="amp",
        device=cfg.MODEL.DEVICE,
        force_quick_gelu=True,
        mean=mean,
        std=std,
        inmem=True,
        text_encoder_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    )
    return model.to(cfg.MODEL.DEVICE).eval()
