'use strict';
const $ = id => document.getElementById(id);
let mode = 'demo', state = {runs: [], datasets: []}, selected = null, report = null;
const money = value => new Intl.NumberFormat('en-US', {style:'currency', currency:'USD'}).format(Number(value || 0));
function notice(message, error=false) { $('notice').hidden=false; $('notice').textContent=message; $('notice').classList.toggle('error', error); }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const result = await response.json(); if (!response.ok) throw Error(result.error || 'Request failed'); return result;
}
async function busy(button, job) { button.disabled=true; try { await job(); } catch(e) { notice(e.message, true); } finally { button.disabled=false; } }
function settings() { return {bankroll:$('bankroll').value, event_fraction:String(Number($('event-cap').value)/100), total_fraction:String(Number($('total-cap').value)/100), min_edge:String(Number($('edge').value)/100), depth_fraction:String(Number($('depth').value)/100)}; }
document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  mode=tab.dataset.mode;
  document.querySelectorAll('.tab').forEach(t => {t.classList.toggle('selected',t===tab);t.setAttribute('aria-selected',String(t===tab));});
  ['demo','replay','live'].forEach(m => $(m+'-fields').hidden=m!==mode);
  $('run-button').textContent={demo:'Run synthetic demo →',replay:'Replay dataset →',live:'Start paper session →'}[mode];
}));
$('run-form').addEventListener('submit', e => {e.preventDefault();busy($('run-button'),async()=>{
  const body={settings:settings()};
  if(mode==='replay') Object.assign(body,{dataset:$('dataset').value,start:$('start').value,end:$('end').value});
  if(mode==='live') Object.assign(body,{tickers:$('tickers').value.split(/[\n,]+/),hours:$('hours').value,poll_seconds:$('poll').value,fee_rate:$('fee-rate').value,predictions_file:$('predictions-file').value});
  const result=await api('/api/'+mode,body);selected=result.run_id;await refresh();notice(mode==='live'?'Paper session started. Keep this app and your computer running.':'Replay complete. Review the results and audit trail below.');
});});
$('collect-form').addEventListener('submit',e=>{e.preventDefault();busy(e.submitter,async()=>{
  notice('Collecting market data…');const r=await api('/api/collect',{series:$('series').value});
  notice(`Saved ${r.records} records for ${r.markets} markets.`+(r.errors.length?' '+r.errors.join('; '):''),Boolean(r.errors.length));await refresh();
});});
$('import-form').addEventListener('submit',e=>{e.preventDefault();busy(e.submitter,async()=>{
  const file=$('import-file').files[0];if(!file)throw Error('Select a JSONL file.');if(file.size>9000000)throw Error('Use the command line for files over 9 MB.');
  const r=await api('/api/import',{dataset:$('import-name').value,text:await file.text()});notice(`Imported ${r.records} new records.`);await refresh();
});});
$('run-select').addEventListener('change',()=>{selected=$('run-select').value;render();});
$('download').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`kalshi-helper-${selected}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('stop').addEventListener('click',()=>busy($('stop'),async()=>{const r=await api('/api/stop',{});notice(r.status);await refresh();}));
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o;}
async function refresh(){
  state=await api('/api/state');
  const dataset=$('dataset').value;
  $('dataset').replaceChildren(...state.datasets.map(d=>option(d.dataset,`${d.dataset} · ${d.records} records`)));
  if(state.datasets.some(d=>d.dataset===dataset))$('dataset').value=dataset;
  if(!selected && state.runs.length)selected=state.runs[0].id;
  $('run-select').replaceChildren(...(state.runs.length?state.runs.map(r=>option(r.id,`${r.mode} · ${r.id.slice(0,6)} · ${r.status}`)):[option('','No runs yet')]));
  if(selected)$('run-select').value=selected;render();
}
function scoreRow(label,value){const div=document.createElement('div');div.className='score-row';const name=document.createElement('span');name.textContent=label;const v=document.createElement('strong');v.textContent=value;div.append(name,v);return div;}
function render(){
  const run=state.runs.find(r=>r.id===selected);$('empty').hidden=Boolean(run);$('report').hidden=!run;if(!run)return;report=run.report;
  $('mode-badge').textContent=report.synthetic?'SYNTHETIC DATA':run.mode.toUpperCase();$('run-status').textContent=run.status.toUpperCase();$('asof').textContent=report.as_of?'As of '+new Date(report.as_of).toLocaleString():'';
  $('cash').textContent=money(report.cash);$('pnl').textContent=money(report.realized_pnl);$('pnl').style.color=Number(report.realized_pnl)<0?'#a34732':'#197b63';$('open-cost').textContent=money(report.open_cost);$('fees').textContent=money(report.fees_paid);$('fill-count').textContent=`${report.fills} simulated entries`;
  $('unpriced').textContent=report.unpriced_contracts?`${report.unpriced_contracts} contracts lack fresh liquidation depth`:`${Object.keys(report.positions).length} open positions`;
  $('equity').textContent=money(report.equity_lower_bound);$('drawdown').textContent='Largest decline in lower bound: '+money(report.max_drawdown_lower_bound);
  $('run-note').textContent=report.synthetic?'Invented forecasts, order books, and outcomes. This run exercises the software and does not measure a real trading edge.':run.mode==='live paper'?'Uses live market snapshots and your imported predictions. No automatic trained weather feed yet. Fees and fills are estimates. When collection stops, positions remain open until a later settlement is recorded.':'Replays supplied predictions in availability-time order. This does not train or validate a model, recover missing historical depth, or independently verify source timestamps.';
  const errors=report.errors||[];$('run-errors').textContent=errors.length?'Latest collection issue: '+errors[errors.length-1].message:'';
  $('stop').hidden=run.status!=='running'||run.id!==state.active_id;
  const scores=report.scores;$('scores').replaceChildren(scoreRow('Paired markets / event groups',`${scores.paired_markets} / ${scores.independent_event_groups}`));
  if(scores.paired_markets){$('scores').append(scoreRow('Brier · model / market',`${scores.model_brier.toFixed(4)} / ${scores.market_brier.toFixed(4)}`),scoreRow('Log loss · model / market',`${scores.model_log_loss.toFixed(4)} / ${scores.market_log_loss.toFixed(4)}`));}
  const rows=report.ledger.slice(-150).reverse().map(r=>{const tr=document.createElement('tr');let detail=r.reason||'';
    if(r.action==='fill')detail=`${r.side.toUpperCase()} × ${r.quantity} · debit ${money(r.cost)} · fees ${money(r.fees)}`;
    if(r.action==='settle')detail=`Proceeds ${money(r.proceeds)} · P&L ${money(r.profit)}`;
    if(r.action==='prediction')detail=`P(YES) ${(Number(r.probability)*100).toFixed(1)}% · ${r.model_id}`;
    [r.at?r.at.slice(0,19).replace('T',' '):'',r.action,r.ticker,detail].forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===1)td.className='action-'+r.action;tr.append(td);});return tr;});
  $('ledger').replaceChildren(...rows);$('ledger-count').textContent=`Latest ${rows.length} of ${report.ledger.length} records`;chart(report.curve);
}
function chart(curve){
  const svg=$('chart');svg.replaceChildren();$('chart-start').textContent='';$('chart-end').textContent='';if(!curve.length)return;
  const rows=curve.filter((_,i)=>i%Math.max(1,Math.floor(curve.length/600))===0||i===curve.length-1);
  const values=rows.map(r=>Number(r.equity_lower_bound));let lo=Math.min(...values),hi=Math.max(...values);const pad=Math.max(2,(hi-lo)*.18);lo-=pad;hi+=pad;
  const start=Date.parse(rows[0].at),end=Date.parse(rows[rows.length-1].at);const x=r=>55+(Date.parse(r.at)-start)/Math.max(1,end-start)*830;const y=v=>195-(v-lo)/(hi-lo)*165;
  function el(name,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;svg.append(n);return n;}
  for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;el('line',{x1:55,y1:y(v),x2:885,y2:y(v),stroke:'#e5ebe3','stroke-dasharray':'3 4'});el('text',{x:0,y:y(v)+4,fill:'#7b8b80','font-size':11},'$'+v.toFixed(0));}
  const line=rows.map(r=>`${x(r)},${y(Number(r.equity_lower_bound))}`).join(' ');el('polygon',{points:`55,205 ${line} ${x(rows[rows.length-1])},205`,fill:'#edf4e5'});el('polyline',{points:line,fill:'none',stroke:'#197b63','stroke-width':2.5,'stroke-linejoin':'round'});
  $('chart-start').textContent=new Date(start).toLocaleDateString();$('chart-end').textContent=new Date(end).toLocaleDateString();
}
refresh().catch(e=>notice(e.message,true));setInterval(()=>refresh().catch(()=>notice('Dashboard connection lost. Check that the local server is running.',true)),5000);
