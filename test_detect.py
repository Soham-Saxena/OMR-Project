"""Focused tests for symbol region geometry and staff association."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2

from detect import SymbolDetector


class SymbolDetectorTests(unittest.TestCase):
    def test_regions_keep_exact_masks_and_staff_ownership(self):
        ink = np.zeros((70, 100), dtype=np.uint8)
        ink[12:17, 14:20] = 255
        ink[12:17, 20:24] = 255  # touching ink remains one region
        ink[55:57, 70:72] = 255
        staff = SimpleNamespace(top=10, bottom=30, spacing=5)

        result = SymbolDetector().detect(ink, [staff])

        self.assertEqual(len(result.regions), 2)
        first, second = result.regions
        self.assertEqual(first.bbox, (14, 12, 10, 5))
        self.assertEqual(first.area, 50)
        self.assertEqual(np.count_nonzero(first.mask), first.area)
        self.assertEqual(first.staff_indices, (0,))
        self.assertEqual(second.staff_indices, ())
        self.assertEqual([r.id for r in result.regions], [1, 2])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.json"
            result.save_json(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["image_size"], [100, 70])
        self.assertEqual(saved["regions"][0]["bbox"], [14, 12, 10, 5])
        self.assertEqual(result.overlay().shape, (70, 100, 3))

    def test_local_line_paths_and_multiple_staff_spans(self):
        ink = np.zeros((100, 100), dtype=np.uint8)
        ink[28:67, 50:52] = 255
        staves = [SimpleNamespace(top=10, bottom=30, spacing=5),
                  SimpleNamespace(top=60, bottom=80, spacing=5)]
        paths = [np.full((5, 100), row, dtype=float)
                 for row in (10, 60)]
        paths[0][:] = np.arange(10, 31, 5)[:, None]
        paths[1][:] = np.arange(60, 81, 5)[:, None]

        result = SymbolDetector().detect(ink, staves, paths)

        self.assertEqual(len(result.regions), 1)
        self.assertEqual(result.regions[0].staff_indices, (0, 1))

    def test_rejects_mismatched_staff_paths(self):
        ink = np.zeros((10, 10), dtype=np.uint8)
        staff = SimpleNamespace(top=1, bottom=5, spacing=1)
        with self.assertRaises(ValueError):
            SymbolDetector().detect(ink, [staff], [])

    def test_dense_patches_separate_notes_joined_by_a_beam(self):
        ink = np.zeros((80, 100), dtype=np.uint8)
        cv2.ellipse(ink, (25, 48), (7, 5), -20, 0, 360, 255, -1)
        cv2.ellipse(ink, (65, 48), (7, 5), -20, 0, 360, 255, -1)
        cv2.line(ink, (31, 47), (31, 23), 255, 2)
        cv2.line(ink, (71, 47), (71, 23), 255, 2)
        cv2.line(ink, (31, 23), (71, 23), 255, 2)
        staff = SimpleNamespace(top=31, bottom=63, spacing=8)

        result = SymbolDetector().detect(ink, [staff])

        self.assertEqual(len(result.regions), 1)
        self.assertEqual(len(result.proposals), 2)
        self.assertEqual([p.parent_region_id for p in result.proposals], [1, 1])
        self.assertLess(result.proposals[0].x, result.proposals[1].x)


if __name__ == "__main__":
    unittest.main()
