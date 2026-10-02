"""Hugging Face entrypoint; seed only public development forecast data."""
import os
import shutil
from pathlib import Path
from helper.storage import Store
from helper.server import serve

if __name__ == '__main__':
    revision=Path(__file__).parent/'REVISION'
    if revision.exists():os.environ['BETCHECK_REVISION']=revision.read_text().strip()
    root=Path(os.environ.get('BETCHECK_DATA','/tmp/betcheck'))
    source=Path(__file__).parent/'seed/single-run-archive'
    if source.exists():
        shutil.copytree(source,root/'single-run-archive',dirs_exist_ok=True)
    serve(Store(root),int(os.environ.get('PORT','7860')))
