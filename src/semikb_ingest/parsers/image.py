"""Standalone image adapter using an explicitly configured Qwen VLM provider."""

from __future__ import annotations

import time

from semikb_ingest.assets import ProcessPayloadStore, inspect_image
from semikb_ingest.models import CaptionSource, ImageAssetDraft, SourceFormat
from semikb_ingest.parsers.common import complete_document
from semikb_ingest.parsers.registry import ParseRequest
from semikb_ingest.providers import VisionProvider
from semikb_ingest.structure import BlockKind, StructuredBlock


class ImageVlmParser:
    parser_id = "image-vlm-v1"
    parser_version = "1.0.0"
    source_format = SourceFormat.IMAGE

    def __init__(
        self,
        payload_store: ProcessPayloadStore,
        provider: VisionProvider,
    ) -> None:
        if not isinstance(provider, VisionProvider):
            raise TypeError("The image parser requires a VisionProvider.")
        self.payload_store = payload_store
        self.provider = provider

    def parse(self, request: ParseRequest):
        started = time.monotonic()
        inspection = inspect_image(request.content)
        analysis = self.provider.analyze_image(
            filename=request.filename,
            content_type=inspection.content_type,
            content=request.content,
            correlation_id=request.correlation_id,
        )
        payload = self.payload_store.put(
            request.filename,
            inspection.content_type,
            request.content,
        )
        asset = ImageAssetDraft(
            asset_id="image_0001",
            payload=payload,
            image_type=f"standalone:{inspection.format}:{inspection.width}x{inspection.height}",
            caption=analysis.caption,
            caption_source=CaptionSource.VLM,
            caption_confidence=analysis.confidence,
            ocr_text=analysis.ocr_text,
            detection_summary=analysis.detection_summary,
        )
        parts = [f"图像描述：{analysis.caption}"]
        if analysis.ocr_text.strip():
            parts.append(f"OCR文字：{analysis.ocr_text.strip()}")
        if analysis.detection_summary.strip():
            parts.append(f"检测结果：{analysis.detection_summary.strip()}")
        image_text = "\n".join(parts)
        block = StructuredBlock(
            block_id="block_0001",
            kind=BlockKind.IMAGE,
            text=image_text,
            image_asset_ids=(asset.asset_id,),
            metadata={
                "width": inspection.width,
                "height": inspection.height,
                "format": inspection.format,
            },
        )
        return complete_document(
            request=request,
            source_format=self.source_format,
            parser_name=self.parser_id,
            parser_version=self.parser_version,
            provider_name=self.provider.provider_name,
            provider_version=self.provider.provider_version,
            blocks=(block,),
            images=(asset,),
            detected_title=request.filename.rsplit(".", 1)[0],
            detected_language=analysis.detected_language,
            started_at=started,
            reference_knowhere=True,
        )
