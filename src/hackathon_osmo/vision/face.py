"""Fast Apple Silicon native face detection (+ optional ANE lip-activity landmarks for multi-speaker social mode)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import Quartz
import Vision


@dataclass
class FaceBox:
    """Normalized face bounding box in image coordinates (origin top-left, 0..1)."""

    x: float
    y: float
    w: float
    h: float
    confidence: float
    # Normalized inner-lip vertical opening inside the face box (0.0 when unmeasured/closed, ~0.04..0.16 speaking)
    mouth_open: float = 0.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h


def _extract_mouth_open(obs: object) -> float:
    """Extract distance-invariant lip opening ratio from a Vision VNFaceObservation."""
    try:
        lm = obs.landmarks()  # type: ignore[attr-defined]
        if lm is None:
            return 0.0
        region = lm.innerLips() or lm.outerLips()
        if region is None:
            return 0.0
        count = int(region.pointCount())
        if count < 4:
            return 0.0
        pts = region.normalizedPoints()
        ys = [float(pts[i].y) for i in range(count)]
        span = max(ys) - min(ys)
        return round(max(0.0, min(1.0, span)), 4)
    except Exception:
        return 0.0


class AppleVisionFaceDetector:
    """Zero-download, Neural-Engine-accelerated face & multi-speaker lip detector on macOS."""

    def __init__(self) -> None:
        self._color_space = Quartz.CGColorSpaceCreateDeviceRGB()

    def detect(self, bgr_frame: np.ndarray, *, measure_mouth: bool = False) -> list[FaceBox]:
        h, w = bgr_frame.shape[:2]
        # Convert BGR -> RGBA contiguous bytes for CGImage
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = bgr_frame[:, :, 2]
        rgba[:, :, 1] = bgr_frame[:, :, 1]
        rgba[:, :, 2] = bgr_frame[:, :, 0]
        rgba[:, :, 3] = 255
        raw = rgba.tobytes()

        provider = Quartz.CGDataProviderCreateWithData(None, raw, len(raw), None)
        cg_image = Quartz.CGImageCreate(
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
        if cg_image is None:
            return []

        req = (
            Vision.VNDetectFaceLandmarksRequest.alloc().init()
            if measure_mouth
            else Vision.VNDetectFaceRectanglesRequest.alloc().init()
        )
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        ok, _err = handler.performRequests_error_([req], None)
        if not ok:
            return []

        results: list[FaceBox] = []
        for obs in req.results() or []:
            bb = obs.boundingBox()
            # Vision origin is bottom-left; flip Y to top-left image coords
            fx = float(bb.origin.x)
            fw = float(bb.size.width)
            fh = float(bb.size.height)
            fy = 1.0 - float(bb.origin.y) - fh
            mouth = _extract_mouth_open(obs) if measure_mouth else 0.0
            results.append(
                FaceBox(
                    x=fx,
                    y=fy,
                    w=fw,
                    h=fh,
                    confidence=float(obs.confidence()),
                    mouth_open=mouth,
                )
            )
        return results
