"""Refresh YOLO paths after copying a result folder to another computer."""
import csv
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
if not (root / 'metadata.csv').is_file():
    root = root / 'dataset'

with (root / 'metadata.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))

base = {'path': str(root), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
        'nc': 1, 'names': {0: 'license_plate'}}
(root / 'dataset.yaml').write_text(yaml.safe_dump(base, sort_keys=False), encoding='utf-8')

lists = root / 'lists_original'
lists.mkdir(exist_ok=True)
original = dict(base)
for split in ('train', 'val', 'test'):
    paths = [str((root / r['image_path']).resolve())
             for r in rows if r['split'] == split and float(r['scale']) == 1.0]
    if not paths or not all(Path(p).is_file() for p in paths):
        raise FileNotFoundError('Missing original images for ' + split)
    (lists / f'{split}.txt').write_text('\n'.join(paths) + '\n', encoding='utf-8')
    original[split] = f'lists_original/{split}.txt'

(root / 'dataset_original.yaml').write_text(yaml.safe_dump(original, sort_keys=False), encoding='utf-8')
print('YOLO paths refreshed:', root)
