from concurrent.futures import ThreadPoolExecutor, as_completed
from .resolve import resolve_markets
from .research_store import ResearchStore
from .weather import rain_context


def analyze(store, reference, progress=None):
    result = resolve_markets(store,reference)
    result['status']='loading_forecasts'
    rs=ResearchStore(store)
    rs.set('last_analysis',result)
    from .evaluation import ModelService
    model=ModelService(store)
    for card in result['markets']:
        if card['bid'] is not None:
            p,note=model.predict(card['series'],card['bid'],card['ask'],card['at'],card['close_time'],True)
            card['model_probability']=p
            card['model_note']=note
        else:
            card['model_probability']=None
            card['model_note']='No valid market quote'
    def weather(card):
        try:
            return rain_context(store,card)
        except Exception as exc:
            return dict(status='unavailable',probability=None,trade_eligible=False,periods=[],note=f'Weather source unavailable: {exc}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(weather,card):card for card in result['markets']}
        for index,future in enumerate(as_completed(futures)):
            futures[future]['forecast']=future.result()
            if progress:
                progress('Checking weather sources',index+1,len(futures))
    result['status']='ready'
    rs.set('last_analysis',result)
    return result
