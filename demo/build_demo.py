"""Build the offline viewer from the measured run, with no runtime network calls."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'run-data.json'
if not p.exists():
    root=HERE.parent/'results/disaster_validation_06'
    data={'score':json.loads((root/'evaluation.json').read_text()),
          'navigation':[json.loads(x) for x in (root/'navigation.jsonl').read_text().splitlines()],
          'truth':[json.loads(x) for x in (root/'ground_truth.jsonl').read_text().splitlines()]}
else:data=json.loads(p.read_text())
text=(HERE/'template.html').read_text().replace('__RUN_DATA__',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
(HERE/'index.html').write_text(text)
print('Built offline Webots evidence viewer')
