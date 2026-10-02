"""Select only models previously trained and recorded by this local app."""
import hashlib
from pathlib import Path
from .research_store import ResearchStore


def list_models(store):
    with store.connect() as db:
        rows=db.execute("""SELECT id,created_at,
            json_extract(report,'$.selected_label') label,
            json_extract(report,'$.model_id') model_id,
            json_extract(report,'$.model_file') model_file,
            json_extract(report,'$.promotion_passed') passed,
            json_extract(report,'$.training.blend_weight') weight,
            json_extract(report,'$.holdout.log_loss') loss
            FROM experiments ORDER BY created_at DESC""")
        return [dict(r) for r in rows if r['model_file'] and Path(r['model_file']).name==r['model_file'] and (store.root/'models'/r['model_file']).is_file()]


def activate_model(store,experiment_id):
    rs=ResearchStore(store);report=rs.experiment(experiment_id)
    if not report:raise ValueError('Saved model not found')
    filename=report.get('model_file','');root=(store.root/'models').resolve();path=(root/filename).resolve()
    if not filename or path.parent!=root or Path(filename).name!=filename:raise ValueError('Invalid saved model path')
    if not path.is_file():raise ValueError('Model weights are missing; train this model again')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report.get('model_sha256'):raise ValueError('Model checksum failed; train this model again')
    active=dict(experiment_id=report['id'],model_id=report['model_id'],model_file=filename,
        sha256=report['model_sha256'],passed=bool(report.get('promotion_passed',False)),
        selected=report['selected_model'],experimental=report['experimental_model'],trained_through=report['training_last_settlement'])
    if report.get('coverage'):active['coverage']=report['coverage']
    rs.set('active_model',active)
    return active
