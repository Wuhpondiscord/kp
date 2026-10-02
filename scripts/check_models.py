import json,shutil,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'outputs/kalshi-helper'
sys.path.insert(0,str(root))
from helper.storage import Store
from helper import named_models
import numpy as np
with tempfile.TemporaryDirectory() as folder:
    shutil.copytree(root/'seed/single-run-archive',Path(folder)/'single-run-archive')
    store=Store(folder)
    rows,paths=named_models.dataset(store,'validation')
    results={}
    for name in named_models.NAMES:
        config=named_models.configuration({'name':name})
        predictions=named_models.load(store,config).predict(rows,paths)
        assert len(predictions)==len(rows)>0 and np.isfinite(predictions).all()
        assert ((predictions>=0)&(predictions<=1)).all()
        results[name]=named_models.evaluate(store,config)['metrics']
    checkpoint=named_models.train(store,{'name':'ConsensusBlend'},epochs=1)
    named_models.load(store,{'name':'ConsensusBlend','checkpoint':checkpoint['id']})
    print(json.dumps({'validation_rows':len(rows),'models':results,'training_checkpoint_loaded':True},indent=2))
