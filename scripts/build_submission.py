"""Build a reviewable local repository bundle without changing experiment sources."""
import hashlib
import json
import shutil
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / 'human-demo-robot-pushing'
if DEST == ROOT:
    DEST = ROOT.parent / 'human-demo-robot-pushing-export'


def main():
    DEST.mkdir(exist_ok=True)
    selected = [ROOT/'README.md', ROOT/'requirements.txt', ROOT/'APPLICATION_PROJECT.md']
    selected += [ROOT/name for name in ['RANDOM_SCENES.md','.gitattributes'] if (ROOT/name).exists()]
    selected += list((ROOT/'scripts').glob('*.py'))
    selected += list((ROOT/'models').glob('*.xml'))
    selected += list((ROOT/'data').glob('*.json'))
    selected += [ROOT/'outputs/dataset/manifest.json']
    for folder in (ROOT/'outputs/dataset').iterdir():
        if folder.is_dir():
            selected += [folder/name for name in ['trajectory_prepared.csv','summary.json'] if (folder/name).exists()]
    for relative in ['outputs/rl/residual/9', 'outputs/final_test', 'outputs/world_model', 'outputs/weight_ablation', 'outputs/random_scene']:
        selected += [p for p in (ROOT/relative).rglob('*') if p.is_file() and p.suffix in {'.json','.py','.xml','.zip','.npz','.pt','.png','.svg','.jpg','.mp4'}]
    selected += [ROOT/'outputs/rl_preview_v3/validation_policy.mp4', ROOT/'outputs/rl_preview_v3/final_frame.jpg']
    # One training-video tracking example demonstrates data provenance.
    selected += [ROOT/'outputs/dataset/4adc8157380f6541c351f3929c693d36/tracking.mp4']
    inventory = []
    for src in sorted(set(selected)):
        dst = DEST/src.relative_to(ROOT)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        inventory.append({'path':src.relative_to(ROOT).as_posix(),'bytes':src.stat().st_size,
                          'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
    (DEST/'bundle_manifest.json').write_text(json.dumps(inventory,indent=2))
    (DEST/'.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\ndata/*.mp4\n')
    (DEST/'DATA_CARD.md').write_text('''# Applicant data and provenance

20 phone videos, 6446 frames; 12 training, 3 validation, 5 final test videos.
The applicant recorded fixed-view tabletop pushing around a brown notebook
(22 x 13 cm). Five test goals were independently placed black markers, confirmed
by the applicant. Raw source SHA256 hashes and calibration metadata are recorded
in outputs/dataset/manifest.json. Prepared coordinates are included so simulation
and training can run without access to the original videos. A tracking overlay
from one training video is included as an example of the collected data.

Raw recordings remain local. Re-running extraction requires all original files
under data/ with their recorded filenames, including the reference video.
Coordinate extraction is scene-specific and approximate: all frames were detected,
but this does not establish coordinate accuracy. The first 15 videos have no
independent target labels; their endpoints define training/validation tasks.

Test policy evaluation uses only the initial object position and independent goal
marker, never its subsequent demonstration trajectory. Full test trajectories are
included for audit only. Notebook coordinates normalize its camera rotation.
The state world model targets are simulated transitions; it learns no measured
real-world masses or friction. See README.md for limitations and results.
''',encoding='utf-8')
    archive = DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in DEST.rglob('*'):
            if path.is_file() and '.git' not in path.parts and '__pycache__' not in path.parts:
                z.write(path,path.relative_to(DEST))
    print(json.dumps({'directory':str(DEST),'zip':str(archive),'files':len(inventory),'bytes':sum(i['bytes'] for i in inventory)}))


if __name__ == '__main__':
    main()
