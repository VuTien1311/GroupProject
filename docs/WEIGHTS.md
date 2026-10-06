# Pretrained weight setup

This is an inference repository. No training and no API token are required by
the launcher. Three separate official pretrained checkpoint files are needed.

| File path, relative to repo | Official project | Notes |
|---|---|---|
| `yolo11x-seg.pt` | [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) | Must be x-seg, not m-seg or Detect |
| `hybrid2/weights/hybridnets.pth` | [HybridNets](https://github.com/datvuthanh/HybridNets) | D3, BDD100K road/lane pretrained |
| `checkpoints/depth_anything_v2_vitl.pth` | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | ViT-L relative-depth checkpoint |

To copy existing files from a local three-model project without overwriting:

```powershell
& .\prepare_local_weights.ps1 -SourceRoot 'D:\Paper123'
```

Missing files are reported; the script does not download or train anything.
YOLO11x-seg can be downloaded using the official installed Ultralytics package:

```powershell
& .\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11x-seg.pt')"
```

Run that command from the repo root. Then:

```powershell
& .\.venv\Scripts\python.exe run_paper.py --check
```

`weights_manifest.json` contains the expected sizes and SHA-256 hashes for
the existing HybridNets and Depth Large files. YOLO11x-seg has no locally
verified hash yet; do not treat it as fully provenance-verified until recorded.

HybridNets also initializes an EfficientNet-B3 ImageNet backbone before loading
the full checkpoint, so the first inference may need network access to populate
the Torch model cache. Later cached inference does not require that download.

Do not commit weights to ordinary Git. They are excluded by `.gitignore`.
Upstream code and model-weight terms can differ; preserve attribution and check
the original project terms before redistributing weights.
