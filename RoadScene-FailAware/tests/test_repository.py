from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_readme_local_links(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if not link.startswith(("http:", "https:", "#")):
                self.assertTrue((ROOT / link.split("#")[0]).exists(), link)

    def test_required_documents_and_licenses(self):
        for name in (".gitignore", ".gitattributes", "docs/GITHUB_UPLOAD.md", "docs/TEAM_ROLES.md", "docs/WEIGHTS.md", "LICENSE_POLICY.md", "hybrid2/LICENSE", "depth_anything_v2/LICENSE", "RUN.bat", ".github/workflows/tests.yml"):
            self.assertTrue((ROOT / name).is_file(), name)

    @unittest.skipUnless(shutil.which("git"), "Git not installed")
    def test_ignore_private_artifacts_but_keep_source(self):
        if not (ROOT / ".git").exists():
            self.skipTest("ZIP folder has not been initialized as Git")
        private = ("yolo11x-seg.pt", "hybrid2/weights/hybridnets.pth", "checkpoints/depth_anything_v2_vitl.pth", "input/scene.jpg", "outputs/paper/scene.json", "kaggle.json", ".env", ".venv/Scripts/python.exe")
        for name in private:
            result = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", name], capture_output=True)
            self.assertEqual(result.returncode, 0, name)
        for name in ("run_paper.py", "paper_config.json", "docs/assets/layout_demo.png", "input/README.md", "checkpoints/README.md"):
            result = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", name], capture_output=True)
            self.assertEqual(result.returncode, 1, name)


if __name__ == "__main__":
    unittest.main()
