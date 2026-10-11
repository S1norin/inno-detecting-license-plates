"""Select representative CCPD images from tar.xz; extract only the selection.

Standard library only. Existing images are never deleted or overwritten.
Run: python select_ccpd_subset.py
Preview selection only: python select_ccpd_subset.py --plan-only
"""
import argparse
import bisect
import csv
import hashlib
import json
import os
import random
import shutil
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

PROVINCES = list('皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学') + ['O']
ALPHABETS = list('ABCDEFGHJKLMNPQRSTUVWXYZ') + ['O']
ADS = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789') + ['O']
FIELDS = ['member', 'plate_height', 'tilt_horizontal', 'tilt_vertical',
          'x1', 'y1', 'x2', 'y2', 'plate_text']
INTEGER_FIELDS = FIELDS[1:-1]


def parse_name(name):
    p = PurePosixPath(name)
    if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        raise ValueError('Not an image')
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe member path')
    parts = p.stem.split('-')
    if len(parts) != 7:
        raise ValueError('Not a labeled CCPD filename')
    horizontal, vertical = map(int, parts[1].split('_'))
    a, b = parts[2].split('_')
    x1, y1 = map(int, a.split('&'))
    x2, y2 = map(int, b.split('&'))
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError('Invalid bounding box')
    indices = list(map(int, parts[4].split('_')))
    if len(indices) not in (7, 8):
        raise ValueError('Invalid plate length')
    dictionaries = [PROVINCES, ALPHABETS] + [ADS] * (len(indices) - 2)
    if any(not 0 <= i < len(d) for i, d in zip(indices, dictionaries, strict=False)):
        raise ValueError('Invalid character index')
    text = ''.join(d[i] for i, d in zip(indices, dictionaries, strict=False))
    if 'O' in text:
        raise ValueError('Missing-character placeholder in OCR annotation')
    return dict(zip(FIELDS, [name, y2 - y1, horizontal, vertical,
                             x1, y1, x2, y2, text], strict=False))


def fingerprint(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'parser_version': 1}


def members(archive):
    # Explicit next() avoids retaining hundreds of thousands of TarInfo objects.
    while True:
        member = archive.next()
        if member is None:
            break
        yield member
        archive.members.clear()


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scan(archive_path, output):
    cache, info_path = output / 'archive_metadata.csv', output / 'archive_metadata.json'
    identity = fingerprint(archive_path)
    if cache.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding='utf-8'))
        if info['archive'] != identity:
            raise ValueError('This output directory belongs to a different archive. Choose --output.')
        with cache.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in INTEGER_FIELDS:
                row[key] = int(row[key])
        if len(rows) != info['valid_images']:
            raise ValueError('Metadata cache is incomplete. Choose a new --output directory.')
        print(f'Reused metadata for {len(rows):,} eligible images.', flush=True)
        return rows, info
    rows, seen, skipped = [], set(), Counter()
    scanned = 0
    last = time.monotonic()
    print('Pass 1: reading filenames from the complete archive; extracting no images.', flush=True)
    with tarfile.open(archive_path, 'r|xz') as archive:
        for member in members(archive):
            if not member.isfile():
                continue
            scanned += 1
            try:
                row = parse_name(member.name)
            except ValueError as exc:
                skipped[str(exc)] += 1
                continue
            if member.name in seen:
                raise ValueError('Duplicate archive member: ' + member.name)
            seen.add(member.name)
            rows.append(row)
            if time.monotonic() - last >= 15:
                print(f'Read {scanned:,} file entries; eligible images: {len(rows):,}.', flush=True)
                last = time.monotonic()
    rows.sort(key=lambda r: r['member'])
    info = {'archive': identity, 'valid_images': len(rows), 'skipped': dict(skipped)}
    write_csv(cache, rows, FIELDS)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Metadata complete: {len(rows):,} eligible images; skipped {sum(skipped.values()):,} entries.', flush=True)
    return rows, info


def quantile_cuts(values):
    values = sorted(values)
    return sorted(set(values[(len(values) - 1) * i // 4] for i in (1, 2, 3)))


def select(rows, count, seed):
    if not 0 < count <= len(rows):
        raise ValueError(f'Requested {count} images; eligible population is {len(rows)}.')
    axes = ('plate_height', 'tilt_horizontal', 'tilt_vertical')
    cuts = {key: quantile_cuts([r[key] for r in rows]) for key in axes}
    groups = defaultdict(list)
    for row in rows:
        key = tuple(bisect.bisect_right(cuts[axis], row[axis]) for axis in axes)
        groups[key].append(row)
    keys = sorted(groups)
    # Integer largest-remainder allocation: exact total and no floating-point ties.
    quotas = {k: count * len(groups[k]) // len(rows) for k in keys}
    remainder_order = sorted(keys, key=lambda k: (-(count * len(groups[k]) % len(rows)), k))
    for key in remainder_order[:count - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(seed)
    selected, distribution = [], []
    for key in keys:
        for row in rng.sample(groups[key], quotas[key]):
            selected.append({**row, 'stratum': '_'.join(map(str, key)),
                             'image_relpath': 'images/' + row['member']})
        distribution.append({'stratum': '_'.join(map(str, key)),
                             'population': len(groups[key]), 'selected': quotas[key],
                             'population_share': len(groups[key]) / len(rows),
                             'selected_share': quotas[key] / count})
    selected.sort(key=lambda r: r['member'])
    assert len(selected) == count and len({r['member'] for r in selected}) == count
    return selected, cuts, distribution


def extract(archive_path, output, selected):
    wanted = {row['member']: row for row in selected}
    done, hashes, last = 0, {}, time.monotonic()
    print(f'Pass 2: extracting only {len(wanted):,} selected images.', flush=True)
    with tarfile.open(archive_path, 'r|xz') as archive:
        for member in members(archive):
            if member.name not in wanted:
                if time.monotonic() - last >= 15:
                    print(f'Reading compressed archive; selected images handled: {done:,}/{len(wanted):,}.', flush=True)
                    last = time.monotonic()
                continue
            if not member.isfile():
                raise ValueError('Selected entry is not a regular file: ' + member.name)
            target = output / wanted[member.name]['image_relpath']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError('Unsafe output path: ' + str(target))
            if shutil.disk_usage(output).free < member.size + 512 * 2**20:
                raise OSError('Less than required file size + 512 MiB free. Existing files preserved.')
            source = archive.extractfile(member)
            h = hashlib.sha256()
            if target.exists():
                # Lambda is fully consumed within this iteration; no late binding.
                for block in iter(lambda: source.read(1024 * 1024), b''):  # noqa: B023
                    h.update(block)
                source.close()
                hashes[member.name] = h.hexdigest()
                with target.open('rb') as existing:
                    previous = hashlib.sha256()
                    for block in iter(lambda: existing.read(1024 * 1024), b''):
                        previous.update(block)
                if previous.hexdigest() != hashes[member.name]:
                    raise FileExistsError(f'Existing image differs; preserved: {target}. Choose a new --output.')
                done += 1
                if done == len(wanted):
                    break
                continue
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.',
                                             suffix='.part', delete=False) as temp:
                # Lambda is fully consumed within this iteration; no late binding.
                for block in iter(lambda: source.read(1024 * 1024), b''):  # noqa: B023
                    temp.write(block)
                    h.update(block)
                temp_path = Path(temp.name)
            source.close()
            if temp_path.stat().st_size != member.size:
                raise OSError('Incomplete extracted image: ' + str(temp_path))
            hashes[member.name] = h.hexdigest()
            # On Windows rename fails if another process created target; no overwrite.
            os.rename(temp_path, target)
            done += 1
            if time.monotonic() - last >= 15:
                print(f'Selected images handled: {done:,}/{len(wanted):,}.', flush=True)
                last = time.monotonic()
            if done == len(wanted):
                break
    if done != len(wanted):
        raise OSError(f'Found only {done}/{len(wanted)} selected entries; completion marker not written.')
    return hashes


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=root / 'CCPD2019.tar.xz')
    parser.add_argument('--output', type=Path, default=root / 'CCPD_subset_5000')
    parser.add_argument('--count', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plan-only', action='store_true', help='Save metadata/selection without extracting images.')
    parser.add_argument('--selection-manifest', type=Path,
                        help='Extract an exact saved selected_images.csv instead of sampling again.')
    args = parser.parse_args()
    if args.count <= 0:
        parser.error('--count must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'selection_config.json'
    settings = {'archive': fingerprint(args.archive), 'count': args.count, 'seed': args.seed}
    if args.selection_manifest:
        settings['manifest_sha256'] = hashlib.sha256(args.selection_manifest.read_bytes()).hexdigest()
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous['settings'] != settings:
            raise ValueError('Output belongs to another selection; choose a new --output directory.')
    completion = output / 'subset_complete.json'
    if completion.exists():
        receipt = json.loads(completion.read_text(encoding='utf-8'))
        if receipt['settings'] != settings:
            raise ValueError('Completed output belongs to another selection.')
        intact = True
        for row in receipt['images']:
            path = output / row['image_relpath']
            if not path.is_file():
                intact = False
                break
            with path.open('rb') as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != row['sha256']:
                intact = False
                break
        if intact:
            (output / 'images' / '.extraction_complete.json').write_text(
                json.dumps({'kind': 'selected_subset', 'images': args.count, 'settings': settings}, indent=2),
                encoding='utf-8')
            print(f'Already complete: {args.count:,} selected images verified. {output}', flush=True)
            return
        raise ValueError('Completed subset has missing/changed files. Existing files preserved; choose a new --output.')
    if args.selection_manifest:
        with args.selection_manifest.open(encoding='utf-8-sig', newline='') as stream:
            saved = list(csv.DictReader(stream))
        selected = [{**parse_name(r['member']), 'stratum':r.get('stratum','saved'),
                     'image_relpath':'images/'+r['member']} for r in saved]
        if len(selected) != args.count or len({r['member'] for r in selected}) != args.count:
            raise ValueError('Saved manifest must contain exactly --count distinct eligible members.')
        cuts, distribution, population_size = {}, [], None
    else:
        rows, info = scan(args.archive, output)
        selected, cuts, distribution = select(rows, args.count, args.seed)
        population_size = len(rows)
    config_path.write_text(json.dumps({'settings': settings, 'population_images': population_size,
                                      'quartile_cutpoints': cuts,
                                      'method': 'proportional height x horizontal tilt x vertical tilt sampling'},
                                     ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(output / 'selected_images.csv', selected, FIELDS + ['stratum', 'image_relpath'])
    write_csv(output / 'sampling_distribution.csv', distribution,
              ['stratum', 'population', 'selected', 'population_share', 'selected_share'])
    print(f'Selected {len(selected):,} images across {len(distribution)} nonempty strata; seed={args.seed}.',
          flush=True)
    if args.plan_only:
        print('Plan saved. Rerun without --plan-only to extract selected images.', flush=True)
        return
    hashes = extract(args.archive, output, selected)
    receipt = {'settings': settings, 'images': [{'image_relpath': r['image_relpath'],
                'sha256': hashes[r['member']]} for r in selected]}
    completion.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'images' / '.extraction_complete.json').write_text(
        json.dumps({'kind': 'selected_subset', 'images': len(selected), 'settings': settings}, indent=2),
        encoding='utf-8')
    print(f'DONE: {len(selected):,} source images extracted and verified. Images root: {output / "images"}', flush=True)
    print('No existing source dataset files were changed. '
          'This script does not create scale variants or splits.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Existing files preserved. Rerun the same command to resume.', flush=True)
        raise SystemExit(130) from None
