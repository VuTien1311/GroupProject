"""Read-only checks before publishing; never prints secret values."""
from pathlib import Path
import ast
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(rb"\b(?:KGAT_|hf_|sk-)[A-Za-z0-9_-]{18,}")
UNSAFE_SUFFIXES = {".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".npy", ".npz", ".pkl", ".zip", ".exe"}
SKIP_PARTS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache"}


def main():
    problems, checked, syntax = [], 0, 0
    files = sorted(ROOT.rglob("*"))
    if shutil.which("git") and (ROOT / ".git").exists():
        # Check tracked files (even force-added weights), plus non-ignored new files.
        selected = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"], check=True, capture_output=True)
        files = [ROOT / name for name in selected.stdout.decode("utf-8").split("\0") if name]
    for file in files:
        if not file.is_file():
            continue
        rel = file.relative_to(ROOT)
        if SKIP_PARTS.intersection(rel.parts):
            continue
        if file.name in {"REPOSITORY_MANIFEST.json", "READINESS_REPORT.json"}:
            continue
        checked += 1
        if file.suffix.lower() in UNSAFE_SUFFIXES or file.stat().st_size > 10 * 1024 * 1024:
            problems.append(f"Binary/large artifact must not be published: {rel.as_posix()}")
            continue
        if rel.parts[0] in {"input", "outputs", "checkpoints"} and file.name != "README.md":
            problems.append(f"Local data/weight/output present: {rel.as_posix()}")
            continue
        if rel.parts[:2] == ("hybrid2", "weights") and file.name != "README.md":
            problems.append(f"Model artifact present: {rel.as_posix()}")
            continue
        data = file.read_bytes()
        if SECRET.search(data):
            problems.append(f"Potential credential in: {rel.as_posix()}")
        if file.name in {"kaggle.json", ".env"} or file.suffix.lower() in {".pem", ".key"}:
            problems.append(f"Credential/config file present: {rel.as_posix()}")
        if file.suffix == ".py":
            try:
                ast.parse(data.decode("utf-8-sig"), filename=str(file))
                syntax += 1
            except (SyntaxError, UnicodeError) as error:
                problems.append(f"Python syntax error: {rel.as_posix()} ({type(error).__name__})")
    print(json.dumps({"status": "FAIL" if problems else "PASS", "files_checked": checked, "python_syntax_checked": syntax, "problems": problems}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
