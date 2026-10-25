"""Locate symbol regions on a staff-free sheet-music page.

This stage produces image regions, not musical interpretations. In particular,
one connected region may contain several touching symbols (such as beamed
notes), while one printed symbol may occupy several disconnected regions.
Classification and association can use each region's cropped ink mask and
position to resolve those cases later.

Run ``python detect.py page.png --out symbols.json --overlay symbols.png`` to
process a page using staff_detect.py and staff_remove.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from staff_detect import Staff
    from staff_remove import StaffRemoval


@dataclass(frozen=True)
class SymbolRegion:
    """One connected ink region, with coordinates in the original page.

    ``mask`` is cropped to the bounding box; its nonzero pixels are the exact
    ink belonging to this region. ``staff_indices`` may contain more than one
    staff for a barline or other symbol spanning staves. No symbol class is
    implied by this record.
    """

    id: int
    x: int
    y: int
    width: int
    height: int
    area: int
    centroid_x: float
    centroid_y: float
    staff_indices: tuple[int, ...]
    mask: np.ndarray

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def to_dict(self) -> dict:
        """Return JSON-ready geometry; pixel masks stay in memory."""
        return {
            "id": self.id,
            "bbox": [self.x, self.y, self.width, self.height],
            "area": self.area,
            "centroid": [self.centroid_x, self.centroid_y],
            "staff_indices": list(self.staff_indices),
        }


@dataclass(frozen=True)
class SymbolProposal:
    """A compact ink patch inside a connected region.

    Morphological opening isolates thick local patches from thin stems and
    beams. A proposal is only a possible symbol part; it is not a classified
    notehead. The parent region retains all surrounding ink for later use.
    """

    id: int
    parent_region_id: int
    x: int
    y: int
    width: int
    height: int
    area: int
    mask: np.ndarray

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_region_id": self.parent_region_id,
            "bbox": [self.x, self.y, self.width, self.height],
            "area": self.area,
        }


@dataclass
class SymbolDetection:
    """The regions located in a page and the image they came from."""

    regions: list[SymbolRegion]
    proposals: list[SymbolProposal]
    cleaned: np.ndarray
    staff_count: int

    def to_dict(self) -> dict:
        height, width = self.cleaned.shape
        return {
            "image_size": [width, height],
            "staff_count": self.staff_count,
            "regions": [region.to_dict() for region in self.regions],
            "proposals": [proposal.to_dict() for proposal in self.proposals],
        }

    def save_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as output:
            json.dump(self.to_dict(), output, indent=2)

    def overlay(self) -> np.ndarray:
        """Draw region boxes and IDs on the staff-free image (BGR)."""
        canvas = cv2.cvtColor(255 - self.cleaned, cv2.COLOR_GRAY2BGR)
        for region in self.regions:
            colour = (40, 150, 30) if region.staff_indices else (30, 80, 220)
            cv2.rectangle(canvas, (region.x, region.y),
                          (region.x + region.width - 1, region.y + region.height - 1),
                          colour, 1)
            cv2.putText(canvas, str(region.id), (region.x, max(10, region.y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
        for proposal in self.proposals:
            cv2.rectangle(canvas, (proposal.x, proposal.y),
                          (proposal.x + proposal.width - 1,
                           proposal.y + proposal.height - 1), (180, 40, 180), 1)
        return canvas


class SymbolDetector:
    """Find connected ink regions after staff removal.

    ``min_area_ratio`` is relative to the square of the staff-line spacing.
    Small regions, including augmentation dots, are retained by default.
    ``staff_margin`` extends each staff's five-line band to include notes on
    nearby ledger lines. Regions outside every band are still returned with
    an empty ``staff_indices`` tuple.
    """

    def __init__(self, min_area_ratio: float = 0.015,
                 staff_margin: float = 2.5,
                 dense_proposals: bool = True):
        if min_area_ratio < 0 or staff_margin < 0:
            raise ValueError("min_area_ratio and staff_margin must be nonnegative")
        self.min_area_ratio = min_area_ratio
        self.staff_margin = staff_margin
        self.dense_proposals = dense_proposals

    def detect(self, cleaned: np.ndarray, staves: list[Staff],
               line_paths: list[np.ndarray] | None = None) -> SymbolDetection:
        """Locate regions in a binary image where ink is nonzero.

        ``staves`` are from staff_detect.detect_staves; ``line_paths`` may be
        supplied from staff_remove.StaffRemoval for curved staff lines.
        """
        if cleaned is None or cleaned.ndim != 2 or cleaned.dtype != np.uint8:
            raise ValueError("cleaned must be a two-dimensional uint8 image")
        if line_paths is not None and len(line_paths) != len(staves):
            raise ValueError("line_paths must have one array per staff")
        height, width = cleaned.shape
        if line_paths is not None:
            for paths in line_paths:
                if paths.ndim != 2 or paths.shape != (5, width):
                    raise ValueError("each line_paths array must have shape (5, image width)")

        ink = (cleaned > 0).astype(np.uint8)
        count, labels, stats, centres = cv2.connectedComponentsWithStats(
            ink, connectivity=8)
        scale = float(np.median([staff.spacing for staff in staves])) if staves else 1.0
        minimum_area = max(1, int(round(self.min_area_ratio * scale * scale)))
        labelled_regions = []
        for label in range(1, count):  # zero is the background
            x, y, w, h, area = (int(value) for value in stats[label])
            if area < minimum_area:
                continue
            cx, cy = (float(value) for value in centres[label])
            nearby = self._nearby_staves(y, h, cx, staves, line_paths)
            labelled_regions.append((label, SymbolRegion(
                id=0, x=x, y=y, width=w, height=h, area=area,
                centroid_x=cx, centroid_y=cy, staff_indices=nearby,
                mask=(labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255,
            )))

        # IDs follow reading order and remain stable regardless of OpenCV's
        # connected-component label order.
        labelled_regions.sort(key=lambda item: (item[1].y, item[1].x))
        region_ids = {label: i for i, (label, _) in enumerate(labelled_regions, start=1)}
        regions = [SymbolRegion(i, region.x, region.y, region.width,
                                region.height, region.area, region.centroid_x,
                                region.centroid_y, region.staff_indices,
                                region.mask)
                   for i, (_, region) in enumerate(labelled_regions, start=1)]
        proposals = self._dense_proposals(ink, labels, region_ids, scale) if self.dense_proposals else []
        return SymbolDetection(regions, proposals, ink * 255, len(staves))

    @staticmethod
    def _dense_proposals(ink: np.ndarray, labels: np.ndarray,
                         region_ids: dict[int, int], scale: float) -> list[SymbolProposal]:
        """Suggest local thick patches even when thin strokes join symbols."""
        def odd_size(value: float) -> int:
            size = max(3, int(round(value)))
            return size if size % 2 else size + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (odd_size(0.45 * scale), odd_size(0.30 * scale)))
        dense = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        count, seed_labels, stats, _ = cv2.connectedComponentsWithStats(
            dense, connectivity=8)
        minimum_area = max(2, int(round(0.06 * scale * scale)))
        found = []
        for label in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[label])
            if area < minimum_area or w > 3 * h or h > 3 * w:
                continue
            mask = seed_labels[y:y + h, x:x + w] == label
            first_y, first_x = np.argwhere(mask)[0]
            parent = region_ids.get(int(labels[y + first_y, x + first_x]))
            if parent is None:
                continue
            found.append((parent, x, y, w, h, area, mask.astype(np.uint8) * 255))
        found.sort(key=lambda item: (item[2], item[1]))
        return [SymbolProposal(i, *item) for i, item in enumerate(found, start=1)]

    def _nearby_staves(self, y: int, height: int, centre_x: float,
                       staves: list[Staff],
                       line_paths: list[np.ndarray] | None) -> tuple[int, ...]:
        nearby = []
        for index, staff in enumerate(staves):
            if line_paths is None:
                top, bottom = staff.top, staff.bottom
            else:
                column = min(max(int(round(centre_x)), 0), line_paths[index].shape[1] - 1)
                top, bottom = line_paths[index][0, column], line_paths[index][-1, column]
            margin = self.staff_margin * staff.spacing
            if y <= bottom + margin and y + height - 1 >= top - margin:
                nearby.append(index)
        return tuple(nearby)


def detect_symbols(staff_result: dict, removal: StaffRemoval,
                   detector: SymbolDetector | None = None) -> SymbolDetection:
    """Run symbol localization on the existing detection/removal outputs."""
    if removal.cleaned.shape != staff_result["binary"].shape:
        raise ValueError("staff detection and removal images have different sizes")
    detector = detector or SymbolDetector()
    return detector.detect(removal.cleaned, staff_result["staves"], removal.line_paths)


def main(argv: list[str] | None = None) -> SymbolDetection:
    parser = argparse.ArgumentParser(description="Locate ink regions in a music page")
    parser.add_argument("image", help="sheet-music image")
    parser.add_argument("--out", help="save region geometry as JSON")
    parser.add_argument("--overlay", help="save an annotated PNG")
    args = parser.parse_args(argv)

    from staff_detect import detect_staves
    from staff_remove import remove_from_detection

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        parser.error(f"cannot read image: {args.image}")
    staff_result = detect_staves(gray)
    symbols = detect_symbols(staff_result, remove_from_detection(staff_result))
    if args.out:
        symbols.save_json(args.out)
    if args.overlay and not cv2.imwrite(args.overlay, symbols.overlay()):
        parser.error(f"cannot save overlay: {args.overlay}")
    print(f"Found {len(symbols.regions)} ink regions and "
          f"{len(symbols.proposals)} dense proposals across {symbols.staff_count} staves")
    return symbols


if __name__ == "__main__":
    main()
