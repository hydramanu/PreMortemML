"""Create a source archive without local environments, datasets, or run storage."""
from pathlib import Path
import os
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
destination = root.parent / 'PreMortemML-v0.2.0-full.zip'
excluded = {'.venv', '.git', '__pycache__', '.pytest_cache', 'workspace', 'build', 'dist'}
with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
    for directory, folders, files in os.walk(root):
        folders[:] = [f for f in folders if f not in excluded and not f.endswith('.egg-info') and not f.startswith('pytest-cache-')]
        for name in files:
            file = Path(directory) / name
            if file.suffix in {'.log','.pyc'}: continue
            archive.write(file, Path(root.name) / file.relative_to(root))
print(destination)
