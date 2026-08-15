"""Bounded image validation shared by embedded and standalone assets."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from semikb_ingest.errors import IngestError, IngestErrorCode


@dataclass(frozen=True, slots=True)
class ImageLimits:
    max_width: int = 20_000
    max_height: int = 20_000
    max_pixels: int = 100_000_000


@dataclass(frozen=True, slots=True)
class ImageInspection:
    width: int
    height: int
    format: str
    content_type: str


def inspect_image(content: bytes, limits: ImageLimits | None = None) -> ImageInspection:
    policy = limits or ImageLimits()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                image_format = (image.format or "").upper()
                image.verify()
            if (
                width <= 0
                or height <= 0
                or width > policy.max_width
                or height > policy.max_height
                or width * height > policy.max_pixels
            ):
                raise IngestError(
                    IngestErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                    "The image exceeds configured dimension or pixel limits.",
                )
            with Image.open(io.BytesIO(content)) as image:
                image.load()
    except IngestError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise IngestError(
            IngestErrorCode.DOCUMENT_LIMIT_EXCEEDED,
            "The image exceeds safe decompression limits.",
        ) from exc
    except (OSError, UnidentifiedImageError) as exc:
        raise IngestError(
            IngestErrorCode.CORRUPT_DOCUMENT,
            "The image file is corrupt or unsupported.",
        ) from exc
    content_type = Image.MIME.get(image_format, "application/octet-stream")
    return ImageInspection(width, height, image_format.lower(), content_type)
