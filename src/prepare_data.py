"""DEPRECATED — do not use.
Diverges from the canonical path (fashion_data_cleaning.ipynb) and is not read by any downstream code. See ARCHITECTURE.md §3.

Data preparation and catalog curation for Fashion CLIP search.

Preserves original data in dataset/ and outputs validated artifacts to artifacts/.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image, ImageOps

EXPECTED_SCHEMA = [
    'id', 'gender', 'masterCategory', 'subCategory', 'articleType',
    'baseColour', 'season', 'year', 'usage', 'productDisplayName'
]


def clean_text(val):
    if pd.isna(val):
        return pd.NA
    cleaned = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(val))).strip()
    return cleaned if cleaned else pd.NA


def parse_raw_csv(csv_path: Path):
    records, issues = [], []
    source_rows = 0
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader)
        assert header == EXPECTED_SCHEMA, f"Unexpected header schema: {header}"
        while True:
            line_no = reader.line_num + 1
            try:
                row = next(reader)
            except StopIteration:
                break
            source_rows += 1
            if len(row) != len(header):
                action = 'repaired_final_field' if len(row) > len(header) else 'quarantined_short_row'
                issues.append({
                    'source_line': line_no,
                    'field_count': len(row),
                    'action': action,
                    'original_fields': json.dumps(row, ensure_ascii=False)
                })
                if len(row) < len(header):
                    continue
                row = row[:9] + [','.join(row[9:])]
            records.append(dict(zip(header, row), source_line=line_no))
    df = pd.DataFrame(records)
    audit = pd.DataFrame(issues, columns=['source_line', 'field_count', 'action', 'original_fields'])
    return df, audit, source_rows


def clean_metadata(raw_df: pd.DataFrame):
    df = raw_df.copy()
    for col in EXPECTED_SCHEMA:
        df[col] = df[col].map(clean_text).astype('string')

    valid_id = df['id'].str.fullmatch(r'[0-9]{1,18}').fillna(False)
    valid_id &= pd.to_numeric(df['id'], errors='coerce').gt(0).fillna(False)
    df = df.loc[valid_id].copy()
    df['id'] = df['id'].astype('int64')

    # Deduplicate metadata
    df = df.drop_duplicates(subset=EXPECTED_SCHEMA, keep='first')
    df = df.drop_duplicates(subset=['id'], keep='first')

    # Validate year
    num_year = pd.to_numeric(df['year'], errors='coerce')
    valid_year = num_year.between(1900, 2100) & num_year.mod(1).eq(0)
    df['year'] = num_year.where(valid_year).astype('Int64')

    # Fill missing values
    cat_cols = [c for c in EXPECTED_SCHEMA if c not in ['id', 'year', 'productDisplayName']]
    for c in cat_cols:
        df[c + '_was_missing'] = df[c].isna()
        df[c] = df[c].fillna('Unknown')

    df['name_was_missing'] = df['productDisplayName'].isna()
    df['productDisplayName'] = df['productDisplayName'].fillna('Unnamed product ' + df['id'].astype(str))
    return df


def audit_images(image_dirs: list[Path], root_dir: Path, max_workers: int = 16):
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    all_files = sorted(p for d in image_dirs if d.exists() for p in d.rglob('*') if p.is_file() and p.suffix.lower() in extensions)

    def _inspect(path):
        m = re.fullmatch(r'[0-9]{1,18}', path.stem)
        pid = int(path.stem) if m and int(path.stem) > 0 else None
        res = {
            'id': pid,
            'image_path': path.relative_to(root_dir).as_posix(),
            'source_folder': path.relative_to(root_dir).parts[0],
            'bytes': None,
            'sha256': None,
            'width': None,
            'height': None,
            'mode': None,
            'readable': False,
            'error': ''
        }
        try:
            b = path.read_bytes()
            res['bytes'] = len(b)
            res['sha256'] = hashlib.sha256(b).hexdigest()
            bio = io.BytesIO(b)
            with Image.open(bio) as im:
                im.verify()
            bio.seek(0)
            with Image.open(bio) as im:
                im.load()
                res.update(width=im.width, height=im.height, mode=im.mode)
                ImageOps.exif_transpose(im).convert('RGB').load()
            res['readable'] = True
        except Exception as exc:
            res['error'] = str(exc)
        return res

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_inspect, all_files))
    img_df = pd.DataFrame(results)
    img_df['id'] = pd.array(img_df['id'], dtype='Int64')
    return img_df


def build_catalog(root_dir: Path = None, output_dir: Path = None):
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[1]
    if output_dir is None:
        output_dir = root_dir / 'artifacts'
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = root_dir / 'dataset/styles.csv'
    # archive/ does not exist in this checkout; kept dormant per CLIP_PROJECT_PLAN.md §7.
    image_dirs = [root_dir / 'dataset/images', root_dir / 'archive/images']

    print(f"Loading styles from {csv_path}...")
    raw_df, csv_audit, source_rows = parse_raw_csv(csv_path)
    clean_df = clean_metadata(raw_df)

    print(f"Auditing images from {image_dirs}...")
    img_df = audit_images(image_dirs, root_dir)

    usable = img_df.loc[img_df['readable'] & img_df['id'].notna()].copy()
    usable['source_priority'] = usable['source_folder'].map({'dataset': 0, 'archive': 1}).fillna(2)
    selected = usable.sort_values(['id', 'source_priority', 'image_path']).drop_duplicates('id', keep='first')
    selected = selected[['id', 'image_path', 'sha256', 'bytes', 'width', 'height', 'mode']]

    catalog = clean_df.merge(selected, on='id', how='inner', validate='one_to_one')
    catalog = catalog.sort_values('id').reset_index(drop=True)

    catalog['search_text'] = catalog[['productDisplayName', 'gender', 'masterCategory',
                                       'subCategory', 'articleType', 'baseColour', 'usage']].agg(' | '.join, axis=1)
    catalog['image_duplicate_group'] = catalog['sha256']
    catalog['aspect_ratio'] = (catalog['width'] / catalog['height']).round(3)
    catalog['file_size_kb'] = (catalog['bytes'] / 1024).round(1)

    catalog_path = output_dir / 'catalog.csv'
    catalog.to_csv(catalog_path, index=False, encoding='utf-8')
    print(f"Successfully created canonical catalog at {catalog_path} with {len(catalog):,} items.")

    sample = catalog.sample(n=min(500, len(catalog)), random_state=42).sort_values('id').reset_index(drop=True)
    sample.to_csv(output_dir / 'catalog_sample_500.csv', index=False, encoding='utf-8')

    return catalog


if __name__ == '__main__':
    build_catalog()
