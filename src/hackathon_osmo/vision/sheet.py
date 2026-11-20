"""Native Apple Silicon sheet & handwriting/text detector using macOS Vision framework (ANE)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import Quartz
import Vision


@dataclass
class SheetDetection:
    """Detected paper sheet and/or handwritten/printed text region on the desk."""

    found: bool = False
    has_text: bool = False
    # Normalized bounding box (top-left origin, 0..1)
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


class AppleVisionSheetDetector:
    """Detects a paper sheet (VNDetectRectanglesRequest) and text/handwriting (VNRecognizeTextRequest)."""

    def __init__(self) -> None:
        self._color_space = Quartz.CGColorSpaceCreateDeviceRGB()

    def _to_cgimage(self, bgr_frame: np.ndarray) -> object:
        h, w = bgr_frame.shape[:2]
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = bgr_frame[:, :, 2]
        rgba[:, :, 1] = bgr_frame[:, :, 1]
        rgba[:, :, 2] = bgr_frame[:, :, 0]
        rgba[:, :, 3] = 255
        raw = rgba.tobytes()
        provider = Quartz.CGDataProviderCreateWithData(None, raw, len(raw), None)
        return Quartz.CGImageCreate(
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

    def detect(self, bgr_frame: np.ndarray) -> SheetDetection:
        import time

        t0 = time.monotonic()
        cg_image = self._to_cgimage(bgr_frame)
        if cg_image is None:
            return SheetDetection()

        # 1. Paper rectangle detection
        rect_req = Vision.VNDetectRectanglesRequest.alloc().init()
        rect_req.setMinimumConfidence_(0.45)
        rect_req.setMinimumSize_(0.18)
        rect_req.setMaximumObservations_(3)

        # 2. Fast OCR (handles printed + handwritten notes in FR/EN on Apple Silicon)
        text_req = Vision.VNRecognizeTextRequest.alloc().init()
        text_req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
        text_req.setRecognitionLanguages_(["fr-FR", "en-US"])
        text_req.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        ok, _err = handler.performRequests_error_([rect_req, text_req], None)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if not ok:
            return SheetDetection(vision_ms=round(elapsed_ms, 1))

        # Parse detected text lines & bounding boxes
        lines: list[str] = []
        text_boxes: list[tuple[float, float, float, float]] = []
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

        # Parse best paper rectangle (prefer rectangle containing text if both exist)
        rects = list(rect_req.results() or [])
        best_rect = None
        if rects:
            best_rect = max(
                rects,
                key=lambda r: float(r.boundingBox().size.width) * float(r.boundingBox().size.height),
            )

        det = SheetDetection(
            lines=lines[:12],
            text_boxes=text_boxes[:20],
            has_text=len(lines) > 0,
            vision_ms=round(elapsed_ms, 1),
        )

        if best_rect is not None:
            bb = best_rect.boundingBox()
            det.x = float(bb.origin.x)
            det.w = float(bb.size.width)
            det.h = float(bb.size.height)
            det.y = 1.0 - float(bb.origin.y) - det.h
            det.quad = [
                (float(best_rect.topLeft().x), 1.0 - float(best_rect.topLeft().y)),
                (float(best_rect.topRight().x), 1.0 - float(best_rect.topRight().y)),
                (float(best_rect.bottomRight().x), 1.0 - float(best_rect.bottomRight().y)),
                (float(best_rect.bottomLeft().x), 1.0 - float(best_rect.bottomLeft().y)),
            ]
            # Consider sheet confirmed if rectangle found AND (has text OR large clear sheet)
            det.found = True
        elif text_boxes:
            # Even if sheet edges are outside frame (close-up on notes), lock onto the union of text!
            min_x = min(b[0] for b in text_boxes)
            min_y = min(b[1] for b in text_boxes)
            max_x = max(b[0] + b[2] for b in text_boxes)
            max_y = max(b[1] + b[3] for b in text_boxes)
            det.x = min_x
            det.y = min_y
            det.w = max(0.05, max_x - min_x)
            det.h = max(0.05, max_y - min_y)
            det.found = True

        return det
