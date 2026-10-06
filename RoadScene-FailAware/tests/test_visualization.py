import ast
import sys
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_visualization import colorize_depth, make_comparison, save_visualizations


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.original = np.full((180, 320, 3), (40, 70, 100), np.uint8)
        self.overlay = self.original.copy()
        cv2.rectangle(self.overlay, (80, 70), (200, 150), (0, 255, 0), 2)
        self.depth = np.tile(np.linspace(1, 10, 320, dtype=np.float32), (180, 1))

    def test_color_order_and_no_mutation(self):
        before = self.depth.copy()
        view, info = colorize_depth(self.depth)
        np.testing.assert_array_equal(self.depth, before)
        self.assertEqual(view.shape, self.original.shape)
        self.assertGreater(int(view[90, -1, 2]), int(view[90, -1, 0]))
        self.assertGreater(int(view[90, 0, 0]), int(view[90, 0, 2]))
        self.assertEqual(info["valid_fraction"], 1)

    def test_invalid_values(self):
        depth = np.asarray([[np.nan, np.inf, 0, -1]], dtype=np.float32)
        view, info = colorize_depth(depth)
        self.assertEqual(info["valid_fraction"], 0)
        self.assertTrue(np.all(view == 24))

    def test_constant_depth(self):
        view, info = colorize_depth(np.full((12, 16), 5, np.float32))
        self.assertTrue(np.all(view == view[0, 0]))
        self.assertEqual(info["percentile_2"], info["percentile_98"])

    def test_geometry_and_original_unchanged(self):
        view, _ = colorize_depth(self.depth)
        before = self.original.copy()
        result = make_comparison(self.original, self.overlay, view, panel_width=640)
        self.assertEqual(result.shape, (496, 1968, 3))
        np.testing.assert_array_equal(self.original, before)

    def test_bad_shapes_rejected(self):
        view, _ = colorize_depth(self.depth)
        with self.assertRaises(ValueError):
            make_comparison(self.original, self.overlay[:100], view)
        with self.assertRaises(ValueError):
            colorize_depth(np.ones((1, 1, 3)))

    def test_files_and_raw_values_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = save_visualizations(self.original, self.overlay, self.depth, folder, "scene")["files"]
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
                self.assertGreater(Path(path).stat().st_size, 0)
            saved = cv2.imread(paths["original"])
            np.testing.assert_array_equal(saved, self.original)
            np.testing.assert_array_equal(np.load(paths["raw_depth"], allow_pickle=False), self.depth)

    def test_same_inference_and_audit_logic(self):
        source = ast.parse((ROOT / "provenance" / "test3_original.py").read_text(encoding="utf-8-sig"))
        updated = ast.parse((ROOT / "test3.py").read_text(encoding="utf-8-sig"))
        self.assertEqual(self.logic(source), self.logic(updated))

    @staticmethod
    def logic(tree):
        # Removing only the new presentation nodes must recover the original AST.
        class StripPresentation(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == "paper_visualization" else node

            def visit_Assign(self, node):
                if any(isinstance(target, ast.Name) and target.id in {"overlay_frame", "visualization"} for target in node.targets):
                    return None
                return self.generic_visit(node)

            def visit_Dict(self, node):
                for index in reversed(range(len(node.keys))):
                    key = node.keys[index]
                    if isinstance(key, ast.Constant) and key.value == "visualization":
                        del node.keys[index]
                        del node.values[index]
                return self.generic_visit(node)
        return ast.dump(StripPresentation().visit(tree), include_attributes=False)


if __name__ == "__main__":
    unittest.main()
