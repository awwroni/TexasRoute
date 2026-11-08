"""Import the standalone Colab result CSVs into the Streamlit Experiments tab.

Examples:
    python scripts/import_colab_results.py /path/to/unzipped-results
    python scripts/import_colab_results.py /path/to/Texas_RoadNet_BFS_AStar_Colab_Results.zip
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from roadlab.analysis_results import REQUIRED_FILES, load_analysis


def main(source: Path) -> None:
    target = PROJECT / 'outputs' / 'colab_results'
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        if source.is_file() and source.suffix.lower() == '.zip':
            with zipfile.ZipFile(source) as archive:
                files = {Path(item.filename).name: item for item in archive.infolist()
                         if not item.is_dir() and Path(item.filename).suffix.lower() == '.csv'}
                for name in REQUIRED_FILES.values():
                    if name not in files:
                        raise FileNotFoundError(f'CSV not found inside ZIP: {name}')
                    (staging / name).write_bytes(archive.read(files[name]))
        elif source.is_dir():
            for name in REQUIRED_FILES.values():
                matches = list(source.rglob(name))
                if len(matches) != 1:
                    raise FileNotFoundError(f'Expected exactly one {name} inside {source}; got {len(matches)}')
                shutil.copy2(matches[0], staging / name)
        else:
            raise FileNotFoundError(f'Provide a directory of exported CSVs or a results ZIP: {source}')
        result = load_analysis(staging)  # Verify BEFORE overwriting the existing published results.
        target.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_FILES.values():
            shutil.copy2(staging / name, target / name)
        print(f'Imported {len(result["pairs"])} route cases, {len(result["trials"])} trials into {target}')
        print(result['accuracy'].to_string(index=False))
        print('Restart or refresh Streamlit to see the updated results.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python scripts/import_colab_results.py path/to/results.zip')
    main(Path(sys.argv[1]).expanduser().resolve())
