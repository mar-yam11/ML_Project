import argparse
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


INCLUDE_PATHS = [
    'src',
    'requirements.txt',
    'README.md',
    'notebooks/evaluation.ipynb',
    'models',
    'docs/REPORT.md',
]


def add_to_zip(zf: ZipFile, root: Path, rel: Path):
    p = root / rel
    if p.is_dir():
        for child in p.rglob('*'):
            if child.is_file():
                zf.write(child, child.relative_to(root))
    elif p.exists():
        zf.write(p, rel)


def main():
    ap = argparse.ArgumentParser(description='Create submission zip')
    ap.add_argument('--ta', required=True, help='TA name, e.g., Ahmed')
    ap.add_argument('--group', required=True, help='Group name/id, e.g., G5')
    ap.add_argument('--ids', nargs='+', required=True, help='Student IDs, e.g., 123 456 789')
    args = ap.parse_args()

    name = f"{args.ta}_{args.group}_" + "_".join(args.ids)
    zip_name = f"{name}.zip"

    root = Path('.')
    with ZipFile(zip_name, 'w', compression=ZIP_DEFLATED) as zf:
        for rel in INCLUDE_PATHS:
            add_to_zip(zf, root, Path(rel))
    print(f"Created: {zip_name}")


if __name__ == '__main__':
    main()
