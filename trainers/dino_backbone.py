import math
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn


def _to_int_patch_size(patch_size):
    if isinstance(patch_size, tuple):
        return int(patch_size[0])
    return int(patch_size)


def _is_valid_torchhub_repo(path: Path) -> bool:
    return path.is_dir() and (path / "hubconf.py").is_file()


def _find_local_dinov2_repo() -> str | None:
    env_candidates = [
        os.environ.get("DINOV2_REPO"),
        os.environ.get("DINO_REPO"),
    ]

    repo_root = Path(__file__).resolve().parents[1]
    local_candidates = [
        repo_root / "dinov2",
        repo_root / "third_party" / "dinov2",
        repo_root / "external" / "dinov2",
        repo_root / "facebookresearch_dinov2_main",
        Path("/home/share449/.cache/torch/hub/facebookresearch_dinov2_main"),
    ]

    cache_roots = []
    if os.environ.get("TORCH_HOME"):
        cache_roots.append(Path(os.environ["TORCH_HOME"]) / "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        cache_roots.append(Path(os.environ["XDG_CACHE_HOME"]) / "torch" / "hub")
    cache_roots.append(Path.home() / ".cache" / "torch" / "hub")

    cache_candidates = []
    for cache_root in cache_roots:
        cache_candidates.extend(
            [
                cache_root / "facebookresearch_dinov2_main",
                cache_root / "dinov2_main",
                cache_root / "dinov2",
            ]
        )

    for candidate in env_candidates:
        if candidate and _is_valid_torchhub_repo(Path(candidate)):
            return str(Path(candidate))

    for candidate in [*local_candidates, *cache_candidates]:
        if _is_valid_torchhub_repo(candidate):
            return str(candidate)

    return None


class DinoBackbone(nn.Module):
    def __init__(self, model_name: str, image_size: int, device: str, layer_ids=None):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.device_name = device
        self.layer_ids = list(layer_ids) if layer_ids is not None else [7, 9, 11]

        if "dinov2" in model_name:
            self.backend = "torchhub"
            local_repo = _find_local_dinov2_repo()
            if local_repo is not None:
                print(f"Loading DINOv2 from local repo: {local_repo}")
                self.model = torch.hub.load(local_repo, model_name, source="local")
            else:
                print("Local DINOv2 repo not found, falling back to torch.hub remote download.")
                self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        elif "dinov3" in model_name:
            self.backend = "timm"
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                img_size=image_size,
            )
        else:
            raise NotImplementedError(f"Unsupported DINO model: {model_name}")

        self.model.to(device).eval()
        self.model.requires_grad_(False)

        self.embed_dim = getattr(self.model, "embed_dim", getattr(self.model, "num_features", None))
        if self.embed_dim is None:
            if "vit_large" in model_name or "vitl" in model_name:
                self.embed_dim = 1024
            elif "vit_base" in model_name or "vitb" in model_name:
                self.embed_dim = 768
            else:
                raise ValueError(f"Unable to infer embed dim for {model_name}")

        self.patch_size = self._infer_patch_size()
        self.num_global_tokens = 5 if ("reg" in model_name or "dinov3" in model_name) else 1
        self._hook_features = {}
        self._register_block_hooks()

    def _register_block_hooks(self):
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            self.layer_ids = []
            self._hook_handles = []
            return

        num_blocks = len(blocks)
        valid_layer_ids = []
        self._hook_handles = []
        for layer_id in self.layer_ids:
            if 0 <= layer_id < num_blocks:
                valid_layer_ids.append(layer_id)
        self.layer_ids = valid_layer_ids

        def _hook_factory(layer_id):
            def _hook(_, __, output):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                self._hook_features[layer_id] = output
            return _hook

        for layer_id in self.layer_ids:
            self._hook_handles.append(blocks[layer_id].register_forward_hook(_hook_factory(layer_id)))

    def _infer_patch_size(self) -> int:
        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "patch_size"):
            return _to_int_patch_size(patch_embed.patch_size)

        if "patch16" in self.model_name:
            return 16
        if "14" in self.model_name:
            return 14
        raise ValueError(f"Unable to infer patch size for {self.model_name}")

    def _forward_dinov2(self, image: torch.Tensor):
        outputs = self.model.forward_features(image)
        patch_tokens = outputs["x_norm_patchtokens"]
        cls_token = outputs["x_norm_clstoken"]
        return cls_token, patch_tokens

    def _forward_dinov3(self, image: torch.Tensor):
        outputs = self.model.forward_features(image)

        if isinstance(outputs, dict):
            if "x_norm_patchtokens" in outputs:
                patch_tokens = outputs["x_norm_patchtokens"]
                if "x_norm_clstoken" in outputs:
                    cls_token = outputs["x_norm_clstoken"]
                elif "x_norm_regtokens" in outputs:
                    cls_token = outputs["x_norm_regtokens"].mean(dim=1)
                else:
                    cls_token = patch_tokens.mean(dim=1)
                return cls_token, patch_tokens

            if "tokens" in outputs:
                outputs = outputs["tokens"]
            else:
                raise ValueError(f"Unsupported DINOv3 output keys: {list(outputs.keys())}")

        if outputs.ndim != 3:
            raise ValueError(f"Expected token tensor for DINOv3, got shape {tuple(outputs.shape)}")

        cls_token = outputs[:, : self.num_global_tokens, :].mean(dim=1)
        patch_tokens = outputs[:, self.num_global_tokens :, :]
        return cls_token, patch_tokens

    def _extract_patch_tokens(self, tokens: torch.Tensor):
        if isinstance(tokens, dict):
            if "x_norm_patchtokens" in tokens:
                return tokens["x_norm_patchtokens"]
            if "tokens" in tokens:
                tokens = tokens["tokens"]
            else:
                raise ValueError(f"Unsupported intermediate token dict keys: {list(tokens.keys())}")

        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor with 3 dims, got shape {tuple(tokens.shape)}")

        if tokens.shape[1] <= self.num_global_tokens:
            raise ValueError("Intermediate tokens do not contain patch tokens")

        return tokens[:, self.num_global_tokens :, :]

    def _grid_size_from_tokens(self, patch_tokens: torch.Tensor):
        num_patches = patch_tokens.shape[1]
        side = int(math.sqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(f"Patch token count {num_patches} is not a square number")
        return side, side

    def forward(self, image: torch.Tensor):
        self._hook_features.clear()
        with torch.no_grad():
            if self.backend == "torchhub":
                cls_token, patch_tokens = self._forward_dinov2(image)
            else:
                cls_token, patch_tokens = self._forward_dinov3(image)

        stage_tokens = []
        stage_grid_sizes = []
        for layer_id in self.layer_ids:
            layer_tokens = self._hook_features.get(layer_id)
            if layer_tokens is None:
                continue
            patch_layer_tokens = self._extract_patch_tokens(layer_tokens).float()
            stage_tokens.append(patch_layer_tokens)
            stage_grid_sizes.append(self._grid_size_from_tokens(patch_layer_tokens))

        if not stage_tokens:
            stage_tokens = [patch_tokens.float()]
            stage_grid_sizes = [self._grid_size_from_tokens(patch_tokens)]

        return {
            "cls_token": cls_token.float(),
            "patch_tokens": patch_tokens.float(),
            "grid_size": self._grid_size_from_tokens(patch_tokens),
            "stage_tokens": stage_tokens,
            "stage_grid_sizes": stage_grid_sizes,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
        }


def build_dino_backbone(model_name: str, image_size: int, device: str, layer_ids=None) -> DinoBackbone:
    return DinoBackbone(model_name=model_name, image_size=image_size, device=device, layer_ids=layer_ids)
