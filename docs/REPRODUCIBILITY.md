# Reproducibility

## Fixed reference configuration

The default launcher selects YOLO11x-seg@960/confidence 0.15,
HybridNets-D3@640 and Depth Anything V2 ViT-L@518. Auxiliary detection and
adaptive confidence are disabled; displayed distance is relative, not meters.

Enhancement/TTA/filter settings retained from the recovered quality profile are
explicit in `paper_config.json`. The PDF does not fully specify those choices.

## Current evidence

- Existing dependencies import successfully in the local Python 3.11 environment.
- Profile, output round-trip, invalid-depth and three-panel geometry tests passed.
- The fusion AST matches the recovered original after removing presentation-only
  additions; the original is retained at `provenance/test3_original.py`.
- Full YOLO11x-seg inference was not run during repository setup because that
  checkpoint was not locally present. Do not describe it as tested end-to-end.
- Original hard-set IDs and per-image results were not recovered. Historical
  counts/rates in the supplied paper must not be copied as new measured results.

## Rerun procedure

1. Obtain the official checkpoints; record SHA-256 and software/GPU versions.
2. Obtain BDD100K validation data under its own terms, separately from this repo.
3. Select the hard set and retain the image-ID manifest and its hash:

```powershell
& .\.venv\Scripts\python.exe static_training\select_hard_bdd100k.py --images 'D:\BDD100K\images\100k\val' --count 100 --output-dir static_training\hard_bdd100k
& .\.venv\Scripts\python.exe run_paper.py --image-list static_training\hard_bdd100k\images.txt --output-dir outputs\paper
& .\.venv\Scripts\python.exe static_training\analyze_static_benchmark.py outputs\paper --output outputs\paper_analysis.json
```

4. Keep analysis JSON outside the per-image report directory to avoid counting
   aggregate files as images in the recovered analyzer.
5. Match configuration/image IDs across ablations, then compare paired results:

```powershell
& .\.venv\Scripts\python.exe static_training\compare_static_benchmarks.py outputs\run_a outputs\run_b --output outputs\paired_comparison.json
```

The included helpers do not implement all requested reviewer ablations or a
validated COCO-to-BDD class-mapped ground-truth evaluator. Those remain separate
work. A high internal stable ratio is not proof of detection/depth accuracy.
