"""Use the Actions secret only for upload, never put it in the image or logs."""
import os
from pathlib import Path
from huggingface_hub import HfApi

token=os.environ.get('HF_TOKEN')
if not token:raise SystemExit('Set the HF_TOKEN Actions secret on Wuhpondiscord/kp; it must have write access to wuhp/kp.')
root=Path(__file__).resolve().parents[1]
revision=os.environ['GITHUB_SHA']
(root/'REVISION').write_text(revision+'\n')
result=HfApi(token=token).upload_folder(repo_id='wuhp/kp',repo_type='space',folder_path=root,
    allow_patterns=['README.md','Dockerfile','.dockerignore','REVISION','outputs/kalshi-helper/**'],
    ignore_patterns=['**/__pycache__/**','**/*.pyc','**/*.sqlite3*','**/.env*','**/data/**'],
    commit_message='Deploy tested GitHub revision '+revision)
print('Published GitHub revision',revision,'to wuhp/kp')
