# GroupProject

## RoadScene-FailAware

The traffic-scene research project is available as unpacked source in
[RoadScene-FailAware](RoadScene-FailAware/README.md).

It combines pretrained **YOLO11x-seg**, **HybridNets-D3**, and
**Depth Anything V2 ViT-L** for conservative, fail-aware scene assessment.
The output includes **original image | annotated overlay | relative depth**.
This is an inference-only research pipeline, not a vehicle-control system.
Depth is relative, not calibrated meters. Exact paper results have not been reproduced.

- [Installation and usage](RoadScene-FailAware/README.md)
- [Three-person work allocation](RoadScene-FailAware/docs/TEAM_ROLES.md)
- [Pretrained weights](RoadScene-FailAware/docs/WEIGHTS.md)
- [Reproducibility limitations](RoadScene-FailAware/docs/REPRODUCIBILITY.md)

Weights, datasets, tokens, and local generated outputs are not included.
Start commands from the `RoadScene-FailAware` directory.
