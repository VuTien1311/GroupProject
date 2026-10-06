# Provenance and third-party notices

- `test3.py`: recovered pre-competition project fusion/inference source.
  Only visualization export was added; model/fusion logic is unchanged.
  The unchanged original is retained in `provenance/test3_original.py`.
  The original is retained for AST-equivalence testing; hashes are recorded
  in `REPOSITORY_MANIFEST.json`.
- `hybrid2/`: existing local HybridNets source, originally
  https://github.com/datvuthanh/HybridNets . Its original LICENSE and README
  are retained. Encoder code also carries upstream attribution in its files.
- `depth_anything_v2/`: existing local Depth Anything V2 implementation,
  originally https://github.com/DepthAnything/Depth-Anything-V2 . It uses DINOv2
  components with existing notices. Check the upstream repository for complete
  code and checkpoint license conditions; model weights are not bundled.
- Ultralytics YOLO is installed as a dependency, not vendored in this export:
  https://github.com/ultralytics/ultralytics . Review upstream licensing.
- `static_training/`: the recovered project's hard-set and report-analysis
  helpers. The directory name is historical; bundled scripts do not train.
- `run_paper.py`, `paper_config.json`, `paper_visualization.py`, preparation script and export tests:
  new packaging helpers written on 2026-10-06. They are not claimed to be
  the original paper experiment scripts.

This package does not establish any new license for the entire combined work.
Consult upstream licenses before redistribution. Do not remove provenance or
present pretrained third-party architectures as a newly trained original model.

GitHub preparation added documentation, dependency profiles, ignore rules,
Windows launch support and lightweight CI. The hard-set selector's default data
path was made repository-relative. No model/fusion behavior was changed.
