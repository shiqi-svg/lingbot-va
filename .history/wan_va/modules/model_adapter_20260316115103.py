from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .utils import load_text_encoder, load_tokenizer, load_transformer, load_vae


@dataclass
class ModelPaths:
    """规范化后的模型组件路径。"""

    root: Path
    transformer: Path
    vae: Path
    text_encoder: Path
    tokenizer: Path


@dataclass
class ModelBundle:
    """统一返回的模型句柄。

    训练场景通常只需要 transformer。
    推理场景一般会同时使用 transformer + vae + text encoder + tokenizer。
    """

    paths: ModelPaths
    transformer: Any
    vae: Any | None = None
    text_encoder: Any | None = None
    tokenizer: Any | None = None
    transformer_config: dict[str, Any] | None = None


@dataclass
class ModelSpec:
    """用户输入规格：只要求给一个 model_path，其余为可选运行参数。"""

    model_path: str
    torch_dtype: torch.dtype = torch.bfloat16
    transformer_device: str = "cpu"
    vae_device: str = "cpu"
    text_encoder_device: str = "cpu"
    require_inference_components: bool = False


class ModelAdapterError(ValueError):
    """模型适配阶段的可读错误。"""


class ModelAdapter:
    """模型适配层：

    目标：用户仅提供一个路径，即可完成
    1) 路径解析
    2) 结构校验
    3) 配置读取
    4) 统一加载
    """

    REQUIRED_DIRS = ("transformer",)
    INFERENCE_DIRS = ("vae", "text_encoder", "tokenizer")

    @staticmethod
    def resolve_paths(model_path: str) -> ModelPaths:
        """解析用户给定路径。

        支持两种输入：
        - 模型根目录（包含 transformer/vae/text_encoder/tokenizer）
        - 仅 transformer 目录（其父目录被视作 root）
        """
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise ModelAdapterError(f"Model path does not exist: {path}")

        if path.is_dir() and path.name == "transformer":
            root = path.parent
            transformer = path
        else:
            root = path
            transformer = root / "transformer"

        return ModelPaths(
            root=root,
            transformer=transformer,
            vae=root / "vae",
            text_encoder=root / "text_encoder",
            tokenizer=root / "tokenizer",
        )

    @staticmethod
    def _validate_dir(path: Path, name: str) -> None:
        if not path.exists() or not path.is_dir():
            raise ModelAdapterError(
                f"Missing required directory '{name}': {path}"
            )

    @classmethod
    def validate_structure(
        cls,
        paths: ModelPaths,
        require_inference_components: bool,
    ) -> None:
        """校验目录结构是否满足当前使用场景。"""
        for name in cls.REQUIRED_DIRS:
            cls._validate_dir(getattr(paths, name), name)

        if require_inference_components:
            for name in cls.INFERENCE_DIRS:
                cls._validate_dir(getattr(paths, name), name)

    @staticmethod
    def load_transformer_config(paths: ModelPaths) -> dict[str, Any]:
        """读取 transformer/config.json，便于前置兼容性检查。"""
        config_path = paths.transformer / "config.json"
        if not config_path.exists():
            raise ModelAdapterError(f"Missing transformer config: {config_path}")

        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelAdapterError(
                f"Invalid transformer config json: {config_path}"
            ) from exc

    @staticmethod
    def validate_contract(
        transformer_config: dict[str, Any],
        *,
        expected_action_dim: int | None = None,
        expected_patch_size: list[int] | tuple[int, int, int] | None = None,
        required_attn_mode: str | None = None,
    ) -> None:
        """可选契约检查：避免“能加载但不能跑”。"""

        if expected_action_dim is not None:
            got = transformer_config.get("action_dim")
            if got != expected_action_dim:
                raise ModelAdapterError(
                    f"action_dim mismatch, expected={expected_action_dim}, got={got}"
                )

        if expected_patch_size is not None:
            got = transformer_config.get("patch_size")
            if list(got) != list(expected_patch_size):
                raise ModelAdapterError(
                    f"patch_size mismatch, expected={list(expected_patch_size)}, got={got}"
                )

        if required_attn_mode is not None:
            got = transformer_config.get("attn_mode")
            if got != required_attn_mode:
                raise ModelAdapterError(
                    f"attn_mode mismatch, expected='{required_attn_mode}', got='{got}'"
                )

    @classmethod
    def load_bundle(
        cls,
        spec: ModelSpec,
        *,
        expected_action_dim: int | None = None,
        expected_patch_size: list[int] | tuple[int, int, int] | None = None,
        required_attn_mode: str | None = None,
    ) -> ModelBundle:
        """统一加载入口。

        参数说明：
        - spec: 用户给定路径与设备策略。
        - expected_*: 可选契约检查，建议由项目 config 传入。
        """
        paths = cls.resolve_paths(spec.model_path)
        cls.validate_structure(paths, spec.require_inference_components)

        transformer_config = cls.load_transformer_config(paths)
        cls.validate_contract(
            transformer_config,
            expected_action_dim=expected_action_dim,
            expected_patch_size=expected_patch_size,
            required_attn_mode=required_attn_mode,
        )

        transformer = load_transformer(
            str(paths.transformer),
            torch_dtype=spec.torch_dtype,
            torch_device=spec.transformer_device,
        )

        if not spec.require_inference_components:
            return ModelBundle(
                paths=paths,
                transformer=transformer,
                transformer_config=transformer_config,
            )

        vae = load_vae(
            str(paths.vae),
            torch_dtype=spec.torch_dtype,
            torch_device=spec.vae_device,
        )
        text_encoder = load_text_encoder(
            str(paths.text_encoder),
            torch_dtype=spec.torch_dtype,
            torch_device=spec.text_encoder_device,
        )
        tokenizer = load_tokenizer(str(paths.tokenizer))

        return ModelBundle(
            paths=paths,
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer_config=transformer_config,
        )


def load_model_bundle(
    model_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    transformer_device: str = "cpu",
    vae_device: str = "cpu",
    text_encoder_device: str = "cpu",
    require_inference_components: bool = False,
    expected_action_dim: int | None = None,
    expected_patch_size: list[int] | tuple[int, int, int] | None = None,
    required_attn_mode: str | None = None,
) -> ModelBundle:
    """给外部调用的简洁函数。

    用户只需要传 model_path，即可获得可用模型对象。
    """
    spec = ModelSpec(
        model_path=model_path,
        torch_dtype=torch_dtype,
        transformer_device=transformer_device,
        vae_device=vae_device,
        text_encoder_device=text_encoder_device,
        require_inference_components=require_inference_components,
    )
    return ModelAdapter.load_bundle(
        spec,
        expected_action_dim=expected_action_dim,
        expected_patch_size=expected_patch_size,
        required_attn_mode=required_attn_mode,
    )

