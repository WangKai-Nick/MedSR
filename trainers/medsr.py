import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

from open_clip_lib import HFTokenizer

from .dino_backbone import build_dino_backbone
from .layers import PVL_Adapter
from .unimedclip import load_unimedclip_to_device
from .scale_block import ScaleBlock


class SemanticStructuralFusion(nn.Module):
    def __init__(self, dino_dim: int, feat_dim: int, num_scales: int):
        super().__init__()
        self.num_scales = num_scales
        self.dino_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dino_dim, feat_dim, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            )
            for _ in range(num_scales)
        ])
        self.dino_multi_fuse = nn.Sequential(
            nn.Conv2d(feat_dim * num_scales, feat_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
        )
        self.main_fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(
        self,
        semantic_map: torch.Tensor,
        text_query: torch.Tensor,
        dino_stage_tokens,
        dino_stage_grid_sizes,
    ):
        if not isinstance(dino_stage_tokens, (list, tuple)):
            dino_stage_tokens = [dino_stage_tokens]
        if not isinstance(dino_stage_grid_sizes, (list, tuple)):
            dino_stage_grid_sizes = [dino_stage_grid_sizes]

        batch_size, _, sem_h, sem_w = semantic_map.shape
        dino_guided_scales = []

        for idx, (dino_patch_tokens, dino_grid_size) in enumerate(zip(dino_stage_tokens, dino_stage_grid_sizes)):
            dino_h, dino_w = dino_grid_size
            dino_map = dino_patch_tokens.reshape(batch_size, dino_h, dino_w, -1).permute(0, 3, 1, 2)
            dino_map = self.dino_projectors[idx](dino_map)
            if dino_map.shape[-2:] != (sem_h, sem_w):
                dino_map = F.interpolate(dino_map, size=(sem_h, sem_w), mode="bilinear", align_corners=False)
            dino_guided_scales.append(dino_map)

        if len(dino_guided_scales) == 1:
            dino_multi = dino_guided_scales[0]
        else:
            dino_multi = self.dino_multi_fuse(torch.cat(dino_guided_scales, dim=1))

        fusion_feature = self.main_fuse(torch.cat([semantic_map, dino_multi], dim=1))
        combined_lowres_logits = torch.einsum("bqc,bchw->bqhw", text_query, fusion_feature)

        return combined_lowres_logits, {
            "dino_guided_scales": dino_guided_scales,
            "dino_multi": dino_multi,
            "fusion_feature": fusion_feature,
        }


class MedSR(nn.Module):

    def __init__(self, cfg, clip_model):
        super().__init__()

        self.vision_model = clip_model.visual
        self.text_model = clip_model.text_encoder
        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = cfg.MODEL.LAYERS

        if cfg.MODEL.BACKBONE == "ViT-B/16":
            self.embed_dim = 768
            self.patch_size = 16
            self.text_proj_dim = 512
        elif cfg.MODEL.BACKBONE == "ViT-L/14":
            self.embed_dim = 1024
            self.patch_size = 14
            self.text_proj_dim = 768
            raise NotImplementedError("ViT-L/14 not implemented yet.")
        else:
            raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")

        self.dtype = self.text_model.transformer.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE

        self.tokenizer = HFTokenizer(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            context_length=256,
            **{},
        )

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.n_ctx = cfg.MODEL.get("PROMPT_TOKENS", 1)
        self.dino_layers = cfg.MODEL.get("DINO_LAYERS", [9, 10, 11])
        self.mask_threshold = cfg.MODEL.get("MASK_THRESHOLD", 0.5)
        self.mask_collapse_threshold = cfg.MODEL.get("MASK_COLLAPSE_THRESHOLD", 0.75)
        self.min_semantic_area = cfg.MODEL.get("MIN_SEMANTIC_AREA", 0.001)
        self.prompt_ctx = nn.Parameter(torch.empty(self.n_ctx, self.embed_dim))
        nn.init.normal_(self.prompt_ctx, std=0.02)

        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.text_proj_dim) for _ in range(self.num_upscale)],
        )

        self.pvl_adapters = nn.ModuleList([
            PVL_Adapter(
                in_channels_vis=self.embed_dim,
                in_channels_txt=self.embed_dim,
                adapter_channels=adapter_channels,
            )
            for _ in range(len(self.fusion_stages))
        ])

        self.dino_name = cfg.MODEL.get("DINO_NAME", "dinov2_vitb14_reg")
        self.dino_backbone = build_dino_backbone(
            self.dino_name,
            self.im_size,
            self.device,
            layer_ids=self.dino_layers,
        )
        num_dino_scales = max(1, len(self.dino_backbone.layer_ids))
        self.semantic_structural_fusion = SemanticStructuralFusion(
            dino_dim=self.dino_backbone.embed_dim,
            feat_dim=self.text_proj_dim,
            num_scales=num_dino_scales,
        )


    def build_prompt_embeddings(self, tokenized_prompts: torch.Tensor):
        pad_token_id = self.text_model.config.pad_token_id
        attention_mask = (tokenized_prompts != pad_token_id).long()

        with torch.no_grad():
            word_embeds = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)

        if self.n_ctx <= 0:
            return word_embeds, attention_mask

        batch_size, seq_len, _ = word_embeds.shape
        ctx = self.prompt_ctx.to(dtype=word_embeds.dtype).unsqueeze(0).expand(batch_size, -1, -1)

        prompted_embeds = word_embeds.clone()
        prompted_mask = attention_mask.clone()
        valid_lengths = attention_mask.sum(dim=1)

        for b in range(batch_size):
            valid_len = int(valid_lengths[b].item())
            seq = torch.cat(
                [
                    word_embeds[b, :1],
                    ctx[b],
                    word_embeds[b, 1:],
                ],
                dim=0,
            )[:seq_len]
            prompted_embeds[b] = seq
            prompted_mask[b, : min(seq_len, valid_len + self.n_ctx)] = 1

        return prompted_embeds, prompted_mask

    def encode_text_image(self, tokenized_prompts, text_prompts, image, attention_mask: Optional[torch.LongTensor] = None):
        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.text_model.config.pad_token_id).long()

        x_txt = self.text_model.transformer.embeddings(inputs_embeds=text_prompts)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        x_img = self.vision_model.conv1(image)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1)
        x_img = x_img.permute(0, 2, 1)
        x_img = torch.cat(
            [
                self.vision_model.class_embedding.to(x_img.dtype)
                + torch.zeros(x_img.shape[0], 1, x_img.shape[-1], dtype=x_img.dtype, device=x_img.device),
                x_img,
            ],
            dim=1,
        )

        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)
        x_img = self.vision_model.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)

        for i, (block, layer) in enumerate(zip(self.vision_model.transformer.resblocks, self.text_model.transformer.encoder.layer)):
            if i in self.fusion_stages:
                vis_pvl, txt_pvl = self.pvl_adapters[self.fusion_stages.index(i)](x_img.transpose(1, 0), x_txt)
                x_txt = x_txt + txt_pvl
                x_img = x_img + vis_pvl.transpose(1, 0)

            x_img = block(x_img)
            x_txt = layer(x_txt, attention_mask=extended_attention_mask)
            x_txt = x_txt[0]

        x_img = x_img.permute(1, 0, 2)
        x_img = self.vision_model.ln_post(x_img)

        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj

        pooled_out = x_txt[:, 0, :]
        projected = self.text_model.proj(pooled_out)
        x_txt = self.text_model.proj(x_txt)

        return x_img, projected

    def build_semantic_map(self, image_features, batch_size: int, height: int, width: int):
        seg_feats = image_features[:, 1:, :]
        seg_feats = seg_feats / seg_feats.norm(dim=-1, keepdim=True)

        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        seg_feats = seg_feats.reshape(batch_size, h_patch, w_patch, -1).permute(0, 3, 1, 2)
        semantic_map = self.upscale(seg_feats)
        return semantic_map

    def compute_mask_logits(self, feature_map: torch.Tensor, text_features: torch.Tensor):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_query = self.mask_head(text_features).unsqueeze(1)
        lowres_logits = torch.einsum("bqc,bchw->bqhw", text_query, feature_map)
        fullres_logits = F.interpolate(
            lowres_logits,
            self.im_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return fullres_logits, lowres_logits


    def adaptive_mask_selection(
        self,
        semantic_logits: torch.Tensor,
        refined_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Select the semantic mask when refinement causes mask collapse."""
        semantic_mask = torch.sigmoid(semantic_logits.detach()) > self.mask_threshold
        refined_mask = torch.sigmoid(refined_logits.detach()) > self.mask_threshold

        semantic_area = semantic_mask.flatten(1).float().mean(dim=1)
        retained_area = (semantic_mask & refined_mask).flatten(1).float().mean(dim=1)
        collapse_factor = 1.0 - retained_area / (semantic_area + 1e-7)
        use_semantic_mask = (
            (semantic_area > self.min_semantic_area)
            & (collapse_factor >= self.mask_collapse_threshold)
        )

        if not use_semantic_mask.any():
            return refined_logits
        selected_logits = refined_logits.clone()
        selected_logits[use_semantic_mask] = semantic_logits[use_semantic_mask]
        return selected_logits

    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss

    def forward(self, image, text):
        batch_size, _, height, width = image.shape

        tokenized_prompts = self.tokenizer(text).to(self.device)
        prompts, attention_mask = self.build_prompt_embeddings(tokenized_prompts)


        image_features, text_features = self.encode_text_image(tokenized_prompts, prompts, image, attention_mask)

        semantic_map = self.build_semantic_map(image_features, batch_size, height, width)
        semantic_logits, _ = self.compute_mask_logits(semantic_map, text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_query = self.mask_head(text_features).unsqueeze(1)


        dino_out = self.dino_backbone(image)
        refined_lowres_logits, _ = self.semantic_structural_fusion(
            semantic_map,
            text_query,
            dino_out["stage_tokens"],
            dino_out["stage_grid_sizes"],
        )
        refined_logits = F.interpolate(
            refined_lowres_logits,
            self.im_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        if not self.training:
            selected_logits = self.adaptive_mask_selection(semantic_logits, refined_logits)
            return selected_logits

        if self.training:
            patch_logits = image_features[:, 1:, :]
            patch_logits = patch_logits / patch_logits.norm(dim=-1, keepdim=True)
            patch_mean = patch_logits.mean(dim=1)

            logits_per_image = (patch_mean @ text_features.T) / self.temperature
            logits_per_text = (text_features @ patch_mean.T) / self.temperature

            with torch.no_grad():
                text_sim = (text_features @ text_features.T) / self.temperature
                text_sim = text_sim / text_sim.norm(dim=-1, keepdim=True)
                soft_targets = F.softmax(text_sim, dim=-1)

            loss_i2t = self.soft_cross_entropy(logits_per_image, soft_targets)
            loss_t2i = self.soft_cross_entropy(logits_per_text, soft_targets.T)
            contrastive_loss = (loss_i2t + loss_t2i) / 2

            return refined_logits, contrastive_loss, semantic_logits

        return refined_logits


def build_medsr(cfg):
    clip_model = load_unimedclip_to_device(cfg)
    clip_model.float()
    model = MedSR(cfg, clip_model)

    for name, param in model.named_parameters():
        if "pvl_adapters" in name:
            param.requires_grad_(True)
        elif "mask_head" in name:
            param.requires_grad_(True)
        elif "upscale" in name:
            param.requires_grad_(True)
        elif "semantic_structural_fusion" in name:
            param.requires_grad_(True)
        elif "prompt_ctx" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model
