# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1-Lite/blob/main/hf_nemotron_parse_modeling.py

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, Optional, Union

import albumentations as A
from torchvision import transforms as T
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig, TensorType
from einops import rearrange
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from transformers import BartConfig, AutoModel, BatchEncoding
from transformers.configuration_utils import PretrainedConfig
from vllm.config import VllmConfig, CacheConfig
from vllm.config.lora import LoRAConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings,
                                                   SupportsMultiModal,
                                                   SupportsV0Only)
from vllm.model_executor.models.radio import RadioModel
from vllm.model_executor.models.nemotron_parse import MBartDecoderNoPos, NemotronParsePixelInputs, NemotronParseInternVitConfig, NemotronParseProcessor, NemotronParseProcessingInfo
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              _flatten_embeddings, flatten_bn)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (EncDecMultiModalProcessor,
                                        PromptIndexTargets, PromptReplacement,
                                        PromptInsertion,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.transformers_utils.tokenizer import (
    AnyTokenizer
)

from vllm.utils.tensor_schema import TensorSchema, TensorShape

def pixel_shuffle(x, scale_factor=0.5, version=2, h=None, w=None):
    """Pixel shuffle based on InternVL but adapted for our use case.

    Args:
        x (torch.Tensor): Vision model outputs [num_tiles, img_seq_len, h_vision]
        version (int): Implementation version.

    Returns:
        Shuffled vision model outputs [num_tiles, (sq ** 2) * (scale ** 2), h_vision / (scale ** 2)]
    """
    if h is None:
        h = int(x.shape[1] ** 0.5)
    if w is None:
        w = int(x.shape[2] ** 0.5)

    x = x.reshape(x.shape[0], h, w, -1)  # [num_tiles, sq, sq, h_vision]
    x = x.permute(0,2,1,3).contiguous()
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.reshape(
        n, int(h * scale_factor), int(w*scale_factor), int(c / (scale_factor * scale_factor))) #int(w * scale_factor), int(c / (scale_factor * scale_factor))
    #)

    if version == 2:
        x = x.permute(0, 2, 1, 3).contiguous()

    x = x.reshape(x.shape[0], -1, x.shape[-1])

    return x

class RadioWithNeckLight(nn.Module):
    """Vision encoder using RADIO model with custom neck."""
    
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config.encoder

        self.model_encoder = self.get_vit_model_from_radio_config(config, quant_config=quant_config)
        
        # Neck components
        last_hidden_state = 1024
        self.conv1 = nn.Conv1d(1280, last_hidden_state, 1)
        self.layer_norm1 = nn.LayerNorm(last_hidden_state, eps=1e-06, elementwise_affine=True)
        self.conv2 = nn.Conv2d(last_hidden_state, last_hidden_state, kernel_size=(1,4), stride=(1,4), padding=0, bias=False)
        self.layer_norm2 = nn.LayerNorm(last_hidden_state, eps=1e-06, elementwise_affine=True)
        self.sum_proj = ColumnParallelLinear(3840, last_hidden_state, quant_config=quant_config, prefix=f"{prefix}.sum_proj")
        self.proj_pixshuf = ColumnParallelLinear(4096, last_hidden_state, quant_config=quant_config, prefix=f"{prefix}.sum_proj")
        self.layer_norm3 = nn.LayerNorm(last_hidden_state, eps=1e-06, elementwise_affine=True)
        

    def get_vit_model_from_radio_config(
        self,
        hf_config,
        quant_config=None,
    ):
        hf_config_vision = hf_config.encoder
        model_name = hf_config_vision.args.get("model")
        if model_name is None:
            raise ValueError(f"Unsupported vit model type: {model_name}")

        internvit_config = NemotronParseInternVitConfig(
            model_name=model_name,
            patch_size=getattr(hf_config_vision, "patch_size", 16),
        )
        hf_config_vision.internvit_config = internvit_config

        return RadioModel(config=hf_config_vision, quant_config=quant_config)
        
    def forward(self, pixel_values, **kwargs):
        radio_output = self.model_encoder(pixel_values)
        summary, feature = radio_output
        
        output = self.conv1(feature.permute(0,2,1)).permute(0,2,1)
        output = self.layer_norm1(output)
        
        patch_size = self.config.patch_size
        output = rearrange(output, 'b (h w) d -> b d h w',
                    h=pixel_values.shape[-2] // patch_size,
                    w=pixel_values.shape[-1] // patch_size)
        
        output = self.conv2(output)
        h, w = output.shape[-2:]
        output = rearrange(output, 'b d h w -> b (h w) d')
        output = pixel_shuffle(output, h=h, w=w)

        output = self.layer_norm2(self.proj_pixshuf(output)[0])
        summary = self.layer_norm3(self.sum_proj(summary)[0])
        output = torch.cat((output, summary.unsqueeze(1)), dim=1)
        
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_encoder_weights = []
        adaptor_dict = {name: param for name, param in dict(self.named_parameters()).items() if not name.startswith("model_encoder")}
        for name, w in weights:
            if name.startswith("model_encoder"):
                model_encoder_weights.append((".".join(name.split(".")[1:]), w))
            else:
                param = adaptor_dict[name]
                with torch.no_grad():
                    default_weight_loader(param, w)

        self.model_encoder.load_weights(model_encoder_weights)


class NemotronParseLightProcessingInfo(NemotronParseProcessingInfo):

    def get_num_image_tokens(self) -> int:
        config=self.get_hf_config()
        final_size = config.image_size
        patch_size = config.encoder.patch_size

        return (final_size[0] // patch_size // 2) * ((final_size[1] // patch_size // 2) // 4) + 1


class NemotronParseLightDummyInputsBuilder(BaseDummyInputsBuilder[NemotronParseLightProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_hf_config(
        ).image_size

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class NemotronParseLightMultiModalProcessor(EncDecMultiModalProcessor[NemotronParseLightProcessingInfo]):

    def create_encoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:

        return [0]

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            processed_outputs = super()._call_hf_processor(
                prompt, mm_data, mm_kwargs, tok_kwargs)
        else:
            hf_processor = self.info.get_hf_processor()
            tokenizer = hf_processor.tokenizer
            processed_outputs = tokenizer(prompt,
                                          add_special_tokens=False,
                                          return_tensors="pt")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_image_tokens = self.info.get_num_image_tokens()

        return [
            PromptReplacement(
                modality="image",
                target=[0],
                replacement=[0] * num_image_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(NemotronParseLightMultiModalProcessor,
                                         info=NemotronParseLightProcessingInfo,
                                         dummy_inputs=NemotronParseLightDummyInputsBuilder)
class NemotronParseLightForConditionalGeneration(nn.Module, SupportsMultiModal,
                                    SupportsV0Only):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.vision_config = config.encoder
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.encoder = RadioWithNeckLight(config=config, quant_config=quant_config, prefix=f"{prefix}.encoder")

        self.decoder = MBartDecoderNoPos(config.decoder,
                                         cache_config=cache_config,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.decoder")

        self.vocab_size = config.decoder.vocab_size
        self.lm_head = ParallelLMHead(config.decoder.vocab_size, config.decoder.d_model, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(self.vocab_size,
                                                config.decoder.vocab_size)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and image_embeds is not None:
            raise ValueError(
                "Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            h, w = self.config.image_size
            return NemotronParsePixelInputs(type="pixel_values",
                                         data=flatten_bn(pixel_values, concat=True),
                                         resolve_bindings={
                                             "h": h,
                                             "w": w,
                                         })

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(self, image_input: NemotronParsePixelInputs) -> torch.Tensor:
        assert image_input["type"] == "pixel_values"
        pixel_values = image_input["data"]
        dtype = next(self.encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype)
        return self.encoder(pixel_values)

    def get_language_model(self) -> torch.nn.Module:
        return self.decoder

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        multimodal_embeddings: MultiModalEmbeddings,
    ) -> torch.Tensor:
        return _flatten_embeddings(multimodal_embeddings)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        encoder_input_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                torch.Tensor of *decoder* input token ids.
            positions
                torch.Tensor of *decoder* position indices.
            encoder_input_ids
                torch.Tensor of *encoder* input token ids.
            encoder_positions
                torch.Tensor of *encoder* position indices
        Returns:
            Output torch.Tensor
        """
        inputs_embeds = None
        if encoder_input_ids.numel() > 0:
            # Only compute encoder embeddings during prefill (when encoder_input_ids is non-empty)
            # During decode, encoder KV cache is reused
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(vision_embeddings)
        hidden_states = self.decoder(decoder_input_ids=input_ids,
                                     encoder_hidden_states=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        lm_head_dict = dict(self.lm_head.named_parameters())

        def is_encoder(name: str) -> bool:
            return name.startswith("encoder")

        def is_decoder(name: str) -> bool:
            return name.startswith("decoder")

        def is_lm_head(name: str):
            return name.startswith("lm_head")

        # Separate weights by component
        encoder_weights = []
        decoder_weights = []

        for name, w in weights:
            if is_encoder(name):
                encoder_weights.append((".".join(name.split(".")[1:]), w))
            elif is_decoder(name):
                decoder_weights.append((".".join(name.split(".")[1:]), w))
            elif is_lm_head(name):
                trimmed_name = ".".join(name.split(".")[1:])
                param = lm_head_dict[trimmed_name]
                with torch.no_grad():
                    default_weight_loader(param, w)
            else:
                print("Found unexpected weight: ", name)
        
        # Load encoder weights
        self.encoder.load_weights(encoder_weights)
        # Load decoder weights
        self.decoder.load_weights(decoder_weights)
