"""Native Apple Silicon sheet & handwriting/text detector using macOS Vision framework (ANE)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import Quartz
import Vision


def compute_sharpness(bgr: np.ndarray) -> float:
    """Fast Laplacian variance sharpness metric (normalized to 640px max dim).

    Empirically on Pixel 11 Pro rear camera:
    - Out-of-focus / AF hunting: < 15.0 (e.g. 1.6 right after 2x zoom switch)
    - Crisp focused notebook/text: 60.0 .. 250.0+
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    h, w = bgr.shape[:2]
    scale = min(1.0, 640.0 / max(1, max(h, w)))
    small = (
        cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if scale < 0.99
        else bgr
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rotate_image(bgr: np.ndarray, rotate_deg: int) -> np.ndarray:
    """Rotate image by 0, 90 (CW), 180, or 270 (90° CCW) degrees."""
    deg = int(rotate_deg) % 360
    if deg == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def enhance_document_clarity(bgr: np.ndarray, amount: float = 0.35) -> np.ndarray:
    """Gentle unsharp mask to crisp up cursive handwriting & small print without halo artifacts."""
    if bgr is None or bgr.size == 0 or amount <= 0.0:
        return bgr
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)


def prepare_desk_reading_image(
    bgr: np.ndarray,
    *,
    rotate_deg: int = 270,
    sheet: SheetDetection | None = None,
    auto_crop: bool = True,
    max_crop_area: float = 0.36,
    crop_margin: float = 0.28,
    enhance: bool = True,
) -> tuple[np.ndarray, bool]:
    """Prepare a desk camera frame for optimal Vision LLM handwriting & card reading.

    1. On wide/1x shots where the sheet occupies a modest portion of the desk
       (area <= max_crop_area), smart-crops around the detected sheet with generous
       margin so adjacent pages & cards stay visible while stripping empty desk/laptop.
       On already-zoomed shots (>= 1.5x or area > max_crop_area), preserves full frame
       so edge words are never clipped.
    2. Rotates upright (default 270° = 90° CCW for Osmo positioned to the right of user).
    3. Applies subtle unsharp clarity boost for pen ink & small printed text.

    Returns (processed_bgr, was_cropped).
    """
    img = bgr
    cropped = False

    if (
        auto_crop
        and sheet is not None
        and sheet.found
        and 0.05 <= sheet.area_ratio <= max_crop_area
    ):
        h, w = bgr.shape[:2]
        pad_x = max(sheet.w * crop_margin, 0.09)
        pad_y = max(sheet.h * crop_margin, 0.09)
        x1 = max(0, int((sheet.x - pad_x) * w))
        y1 = max(0, int((sheet.y - pad_y) * h))
        x2 = min(w, int((sheet.x + sheet.w + pad_x) * w))
        y2 = min(h, int((sheet.y + sheet.h + pad_y) * h))
        # Ensure crop keeps enough pixels (at least 480px on major side)
        if (x2 - x1) >= 420 and (y2 - y1) >= 420:
            img = bgr[y1:y2, x1:x2]
            cropped = True

    if rotate_deg % 360 != 0:
        img = rotate_image(img, rotate_deg)

    if enhance:
        img = enhance_document_clarity(img, amount=0.35)

    return img, cropped


@dataclass
class SheetDetection:
    """Detected paper sheet and/or handwritten/printed text region on the desk."""

    found: bool = False
    has_text: bool = False
    confidence: float = 0.0
    source: str = "none"
    # Normalized bounding box (top-left origin, 0..1) in RAW camera frame coordinates
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    # Optional 4-corner polygon of detected sheet in normalized (0..1) top-left coords
    quad: list[tuple[float, float]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    text_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    vision_ms: float = 0.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area_ratio(self) -> float:
        return max(0.0, self.w * self.h)


class AppleVisionSheetDetector:
    """Detects a paper sheet/notebook (VNDetectDocumentSegmentationRequest + rectangle fallback)
    and text/handwriting (VNRecognizeTextRequest) on Apple Silicon ANE (~6-40 ms).
    """

    def __init__(self) -> None:
        self._color_space = Quartz.CGColorSpaceCreateDeviceRGB()

    def _to_cgimage(self, bgr_frame: np.ndarray) -> tuple[object | None, bytes]:
        h, w = bgr_frame.shape[:2]
        rgba = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGBA)
        raw = rgba.tobytes()
        provider = Quartz.CGDataProviderCreateWithData(None, raw, len(raw), None)
        cg = Quartz.CGImageCreate(
            w,
            h,
            8,
            32,
            w * 4,
            self._color_space,
            Quartz.kCGImageAlphaNoneSkipLast | Quartz.kCGBitmapByteOrderDefault,
            provider,
            None,
            False,
            Quartz.kCGRenderingIntentDefault,
        )
        return cg, raw

    def detect(self, bgr_frame: np.ndarray, *, run_ocr: bool = True) -> SheetDetection:
        t0 = time.monotonic()
        cg_image, _keep_alive = self._to_cgimage(bgr_frame)
        if cg_image is None:
            return SheetDetection()

        requests: list[object] = []

        # 1. Primary neural document segmentation (~6-10 ms on ANE, detects open notebooks & angled sheets!)
        doc_req = None
        if hasattr(Vision, "VNDetectDocumentSegmentationRequest"):
            doc_req = Vision.VNDetectDocumentSegmentationRequest.alloc().init()
            requests.append(doc_req)

        # 2. Fallback classic rectangle detector (useful for small cards / flat sheets)
        rect_req = Vision.VNDetectRectanglesRequest.alloc().init()
        rect_req.setMinimumConfidence_(0.45)
        rect_req.setMinimumSize_(0.14)
        rect_req.setMaximumObservations_(3)
        requests.append(rect_req)

        # 3. Optional fast OCR (handles printed + clear handwritten headings)
        text_req = None
        if run_ocr:
            text_req = Vision.VNRecognizeTextRequest.alloc().init()
            text_req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
            text_req.setRecognitionLanguages_(["fr-FR", "en-US"])
            text_req.setUsesLanguageCorrection_(True)
            requests.append(text_req)

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        ok, _err = handler.performRequests_error_(requests, None)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if not ok:
            return SheetDetection(vision_ms=round(elapsed_ms, 1))

        # Parse detected text lines & bounding boxes
        lines: list[str] = []
        text_boxes: list[tuple[float, float, float, float]] = []
        if text_req is not None:
            for obs in text_req.results() or []:
                cands = obs.topCandidates_(1)
                if not cands:
                    continue
                txt = str(cands[0].string()).strip()
                if len(txt) < 2:
                    continue
                bb = obs.boundingBox()
                tx = float(bb.origin.x)
                tw = float(bb.size.width)
                th = float(bb.size.height)
                ty = 1.0 - float(bb.origin.y) - th
                lines.append(txt)
                text_boxes.append((tx, ty, tw, th))

        det = SheetDetection(
            lines=lines[:12],
            text_boxes=text_boxes[:20],
            has_text=len(lines) > 0,
            vision_ms=round(elapsed_ms, 1),
        )

        # Prefer neural document segmentation when confident (>= 0.45) and reasonable area (>= 0.04)
        best_doc = None
        if doc_req is not None:
            for obs in doc_req.results() or []:
                bb = obs.boundingBox()
                area = float(bb.size.width) * float(bb.size.height)
                conf = float(obs.confidence())
                if conf >= 0.45 and area >= 0.04:
                    if best_doc is None or conf > float(best_doc.confidence()):
                        best_doc = obs

        best_rect = None
        if best_doc is None:
            rects = list(rect_req.results() or [])
            if rects:
                best_rect = max(
                    rects,
                    key=lambda r: float(r.boundingBox().size.width) * float(r.boundingBox().size.height),
                )

        chosen = best_doc or best_rect
        if chosen is not None:
            bb = chosen.boundingBox()
            det.x = float(bb.origin.x)
            det.w = float(bb.size.width)
            det.h = float(bb.size.height)
            det.y = 1.0 - float(bb.origin.y) - det.h
            det.confidence = round(float(chosen.confidence()), 3)
            det.source = "doc_seg" if best_doc is not None else "rect"
            det.quad = [
                (float(chosen.topLeft().x), 1.0 - float(chosen.topLeft().y)),
                (float(chosen.topRight().x), 1.0 - float(chosen.topRight().y)),
                (float(chosen.bottomRight().x), 1.0 - float(chosen.bottomRight().y)),
                (float(chosen.bottomLeft().x), 1.0 - float(chosen.bottomLeft().y)),
            ]
            det.found = True
        elif text_boxes:
            # Close-up where sheet edges are outside frame: lock onto the union of text!
            min_x = min(b[0] for b in text_boxes)
            min_y = min(b[1] for b in text_boxes)
            max_x = max(b[0] + b[2] for b in text_boxes)
            max_y = max(b[1] + b[3] for b in text_boxes)
            det.x = min_x
            det.y = min_y
            det.w = max(0.05, max_x - min_x)
            det.h = max(0.05, max_y - min_y)
            det.confidence = 0.65
            det.source = "text_union"
            det.found = True

        return det
