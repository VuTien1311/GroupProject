import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_paper_export", ROOT / "run_paper.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExportProfileTests(unittest.TestCase):
    def test_paper_models_and_sizes(self):
        cfg = runner.load_config()
        self.assertEqual(cfg["yolo_weight"], "yolo11x-seg.pt")
        self.assertEqual((cfg["image_size"], cfg["hybrid_input_size"], cfg["depth_size"]), (960, 640, 518))
        self.assertEqual(cfg["confidence"], 0.15)
        self.assertEqual(cfg["distance_mode"], "relative")

    def test_no_aux_detector_or_adaptive_threshold(self):
        args = runner.parse_args(["example with spaces.jpg", "--dry-run"])
        command = runner.pipeline_arguments(args)
        self.assertEqual(command[command.index("--aux-yolo-weight") + 1], "")
        self.assertIn("--no-adaptive-conf", command)
        self.assertEqual(command[command.index("--conf") + 1], "0.15")

    def test_source_layout(self):
        for rel in ("test3.py", "hybrid2/backbone.py", "hybrid2/hybridnets/model.py", "depth_anything_v2/dpt.py", "static_training/select_hard_bdd100k.py"):
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_image_list_and_limit(self):
        args = runner.parse_args(["--image-list", "list with spaces.txt", "--limit", "2", "--dry-run"])
        command = runner.pipeline_arguments(args)
        self.assertIn("--image-list", command)
        self.assertEqual(command[command.index("--limit") + 1], "2")

    def test_dry_run_does_not_load_models(self):
        self.assertEqual(runner.main(["example.jpg", "--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()
