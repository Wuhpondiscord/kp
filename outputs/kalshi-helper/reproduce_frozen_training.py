"""Run from an extracted professor package; no network, DB or pickle required."""
from pathlib import Path
import json
from helper.training import split_dataset,fit_epochs
from helper.adaptive import select_adaptive
from helper.evaluation import score
from helper.research_store import digest

def main():
    root=Path(__file__).resolve().parent/'training-run'
    rows=[json.loads(x) for x in (root/'features.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest=json.loads((root/'split-manifest.json').read_text(encoding='utf-8'))
    assert digest(rows)==manifest['data_hash'],'Frozen dataset hash mismatch'
    config=json.loads((root/'config.json').read_text(encoding='utf-8'));train,val,test=split_dataset(rows)
    model,history,epoch=fit_epochs(train,val,config,callback=lambda r,*_:print('Epoch',r['epoch'],flush=True))
    if config['method']=='adaptive':model,_=select_adaptive(train,val,model)
    else:
        from helper.training import choose_weight
        choose_weight(model,val)
    result=score(test,model.predict(test));expected=json.loads((root/'reference-metrics.json').read_text(encoding='utf-8'))
    differences={k:result[k]-expected[k] for k in ('log_loss','brier')}
    print(json.dumps(dict(best_epoch=epoch,epochs=len(history),metrics=result,differences=differences,matched=all(abs(v)<1e-8 for v in differences.values())),indent=2))
    if any(abs(v)>=1e-8 for v in differences.values()):raise SystemExit('Numerical reproduction differs; inspect environment and implementation')

if __name__=='__main__':main()
