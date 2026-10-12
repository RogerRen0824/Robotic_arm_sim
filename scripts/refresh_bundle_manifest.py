"""Refresh release inventory after reviewed development artifacts are complete."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
records = []
for file in sorted(ROOT.rglob('*')):
    relative = file.relative_to(ROOT)
    if not file.is_file() or any(p in relative.parts for p in ['.git','.venv','__pycache__']):
        continue
    if file.name=='bundle_manifest.json' or file.suffix=='.pyc' or (relative.parent==Path('data') and file.suffix=='.mp4'):
        continue
    records.append({'path':relative.as_posix(),'bytes':file.stat().st_size,
                    'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
(ROOT/'bundle_manifest.json').write_text(json.dumps(records,indent=2))
print('Inventoried',len(records),'release files')
