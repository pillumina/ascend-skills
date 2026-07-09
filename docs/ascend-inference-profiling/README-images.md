# Images for ascend-inference-profiling

Generated with [Archscribe](https://github.com/lazypay/Archscribe) (MIT).

Published as WebP (800px wide, ~40-65KB each). Each file has an `.excalidraw` editable source.
PNG originals were 1210px wide, and are preserved in excalidraw format.

## Inventory

| File | Layout | Style | Original | Content |
|------|--------|-------|----------|---------|
| `pipeline-stages` | pipeline | default | 1210×650 | 11-stage analysis pipeline from triage through report |
| `execution-model` | panorama | default | 1210×1138 | Remote (SSH) vs local (direct) dual execution paths |
| `agent-interaction` | pipeline | default | 1210×650 | Agent workflow: run → read → diagnose → present → refine → calibrate |
| `config-signatures` | panorama | terminal | 1210×1138 | 7 config signature detections, signal sources, and confidence levels |
| `hardware-architecture` | layers | blueprint | 1210×1080 | A2 (910B2) → A3 (910C) architecture comparison and profiling data mapping |
| `knowledge-architecture` | layers | blueprint | 1210×866 | Three-layer knowledge system: playbook → config guides → changelog |

## Regenerate

```bash
cd archscribe-full
python3 scripts/render_animated_diagram.py \
  --spec work/<spec>.json \
  --outdir outputs \
  --basename <name> \
  --formats png,excalidraw \
  --style <default|blueprint|terminal>

# Then resize and convert for the repo:
python3 -c "
from PIL import Image
img = Image.open('outputs/<name>.jpg')
w, h = img.size
img = img.resize((800, int(h * 800/w)), Image.LANCZOS)
img.save('<name>.jpg', format='WEBP', quality=88)
"
