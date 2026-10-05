'use strict';
(() => {
  const el=id=>document.getElementById(id);
  const money=x=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(x);
  const percent=x=>x==null?'—':(x*100).toFixed(2)+'%';
  let result=null;
  function node(tag,text,cls){const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;return n;}
  async function request(path,body){const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');return data;}
  function metric(label,value){const card=node('article','','panel');card.append(node('p',label,'eyebrow'),node('h3',value));return card;}
  request('/api/btc/catalog').then(data=>{
    el('btc-model').replaceChildren();
    for(const model of [...(data.experimental_models||[]),...data.models]){const option=node('option',model.name);option.value=model.id;el('btc-model').append(option);const card=node('article','','panel');const select=node('button','Try '+model.name);select.type='button';select.addEventListener('click',()=>{el('btc-model').value=model.id;el('btc-run').focus();});card.append(node('h3',model.name),node('p',model.description),node('p',model.period||'Legacy September 2026 cohort','fine'),node('h2',money(model.summary.net_pnl)),node('p','After spread, fees and 2¢ assumed slippage','fine'),node('p',model.summary.entries+' simulated bets · no proven edge'),select);el('btc-cards').append(card);}
    el('btc-status').textContent='Ready. Choose a model, then replay its historical cohort.';el('btc-run').disabled=false;
  }).catch(error=>{el('btc-status').textContent='BTC data unavailable: '+error.message;});
  el('btc-form').addEventListener('submit',async event=>{
    event.preventDefault();el('btc-run').disabled=true;el('btc-status').textContent='Preparing forecasts and replaying real historical contracts…';
    try{
      const data=await request('/api/btc/replay',{model:el('btc-model').value,bankroll:Number(el('btc-bankroll').value),slippage:Number(el('btc-slip').value)});result=data;const p=data.paper;
      el('btc-result-title').textContent=data.name+' · '+(data.period||'September 2026');
      el('btc-result-note').textContent=(p.pnl>0?'Positive in this assumed-cost scenario; not evidence of a proven edge. ':p.entries===0?'No bets passed the cost and risk rules. ':'This scenario lost money after costs. ')+data.note;
      el('btc-metrics').replaceChildren(metric('Net profit / loss',money(p.pnl)),metric('Ending play balance',money(p.starting_bankroll+p.pnl)),metric('Settled bets',String(p.entries)),metric('Return on money invested',percent(p.return_on_deployed)),metric('Net per contract',p.net_per_contract==null?'—':(p.net_per_contract*100).toFixed(2)+'¢'),metric('Assumed extra cost',(p.slippage*100).toFixed(0)+'¢ / contract'));
      el('btc-quality').replaceChildren(node('p',`Brier: model ${data.scores.brier.toFixed(5)} / market ${data.market_scores.brier.toFixed(5)}. Log loss: model ${data.scores.log_loss.toFixed(5)} / market ${data.market_scores.log_loss.toFixed(5)}. Lower is better.`),node('p',`Total invested: ${money(p.capital_deployed)}. Fees: ${money(p.fees)}. Largest balance drawdown (positions valued at cost): ${money(p.realized_cost_equity_drawdown)}. Maximum bet cost: ${money(p.max_bet_cost)}.`));
      el('btc-ledger').replaceChildren(...p.ledger.slice(-50).reverse().map(t=>{const tr=node('tr','');for(const v of [t.ticker,t.quantity,money(t.cost),money(t.pnl)])tr.append(node('td',v));return tr;}));
      el('btc-results').hidden=false;el('btc-status').textContent='Replay complete. Review the after-cost result below.';el('btc-results').scrollIntoView({behavior:'smooth',block:'start'});
    }catch(error){el('btc-status').textContent='Replay failed: '+error.message;}finally{el('btc-run').disabled=false;}
  });
  el('btc-export').addEventListener('click',()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=node('a','');a.href=url;a.download='betcheck-btc-'+result.model+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
})();
