# Three-person division of work

Replace the member labels with actual names/GitHub handles before presenting.

| Member | Technical ownership | Deliverables |
|---|---|---|
| 1: Object evidence | YOLO11x-seg, input/config/weight setup, box/mask/class confidence | Explain detection and instance-mask outputs; prepare examples; verify model profile |
| 2: Road/path reasoning | HybridNets-D3 road/lane masks, ego corridor, PATH/EDGE/SIDE, severity gates | Explain corridor construction and conservative CLEAR/MONITOR/SLOW/BRAKE/STOP decisions |
| 3: Depth/reliability/presentation | Depth Anything ViT-L, multi-region consistency, reliability states, depth-only fallback, three-panel rendering | Explain relative-depth limits and STABLE/CHECK/UNCERTAIN; validate visualization and exports |

Shared responsibilities:

- Integrate the three frozen pretrained components through the fusion layer.
- Use identical image sets and configuration for comparisons.
- Review detector-empty and degraded-scene outputs together.
- Run the tests and record model hashes/software versions.
- Separate internal consistency from ground-truth accuracy in the report.

Do not claim the members trained the three pretrained architectures. The project
contribution is integration, uncertainty-aware reasoning and auditable outputs.
This document is a proposed work split, not a claim of historical contributions.
