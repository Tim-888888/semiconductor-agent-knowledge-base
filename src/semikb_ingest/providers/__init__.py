"""Provider identities and dependency registry."""

from semikb_ingest.providers.mineru_pdf import MinerUPdfClient, MinerUPdfConfig
from semikb_ingest.providers.qwen_vision import QwenVisionClient, QwenVisionConfig
from semikb_ingest.providers.registry import ProviderClient, ProviderRegistry
from semikb_ingest.providers.types import (
    MinerUContentItem,
    MinerUImage,
    MinerUPdfResult,
    PdfProvider,
    VisionAnalysis,
    VisionProvider,
)

__all__ = [
    "MinerUContentItem",
    "MinerUImage",
    "MinerUPdfClient",
    "MinerUPdfConfig",
    "MinerUPdfResult",
    "PdfProvider",
    "ProviderClient",
    "ProviderRegistry",
    "QwenVisionClient",
    "QwenVisionConfig",
    "VisionAnalysis",
    "VisionProvider",
]
