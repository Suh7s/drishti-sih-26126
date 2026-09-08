"""Embed recorded experiment data so the viewer works offline with a double-click."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
data=json.loads((root/"results/replays.json").read_text())
# Downsample observed-cell deltas would reduce size, but retain full auditable frames.
text=(root/"demo/template.html").read_text().replace("__REPLAY_DATA__",json.dumps(data,separators=(",",":")))
(root/"demo/index.html").write_text(text)
print("Wrote offline demo",len(text),"bytes")
