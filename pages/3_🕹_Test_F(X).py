# 📈_Monitor.py (Async Performance - Final Corrected Version for Caching)
import streamlit as st
import numpy as np
import datetime
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
from threading import Lock
import os
from typing import List, Dict, Any
import asyncio
import aiohttp
import thingspeak

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- START: SimulationTracer Class (Unchanged and Minified for space) ---
class SimulationTracer:
    def __init__(self, e: str): self.e = str(e); self._d()
    def _d(self):
        if not self.e.isdigit(): self.al,self.mr,self.ds,self.ms,self.mrf=0,0,0,[],0.0; return
        dn,i=[],0
        try:
            while i<len(self.e): l=int(self.e[i]);i+=1;ns=self.e[i:i+l];i+=l;dn.append(int(ns))
        except: pass
        if len(dn)<3: self.al,self.mr,self.ds,self.ms,self.mrf=0,0,0,[],0.0; return
        self.al,self.mr,self.ds,self.ms,self.mrf=dn[0],dn[1],dn[2],dn[3:],self.mr/100.0
    def run(self) -> np.ndarray:
        if self.al<=0:return np.array([])
        dr,ca=np.random.default_rng(seed=self.ds),dna_rng.integers(0,2,size=self.al)
        if self.al>0:ca[0]=1
        for ms in self.ms:
            mr,mm=np.random.default_rng(seed=ms),mr.random(self.al)<self.mrf
            ca[mm]=1-ca[mm]
            if self.al>0:ca[0]=1
        return ca
# --- END: SimulationTracer Class ---

# ---------- CONFIGURATION & SETUP ----------
@st.cache_data
def load_config(fp: str='monitor_config.json')->Dict[str,Any]:
    if not os.path.exists(fp): st.error(f"Config not found: {fp}"); st.stop()
    try:
        with open(fp,'r',encoding='utf-8') as f: c=json.load(f)
        if 'assets' not in c or not c['assets']: st.error("No 'assets' in config"); st.stop()
        return c
    except Exception as e: st.error(f"Error reading config: {e}"); st.stop()

# ---------- GLOBAL CACHE & CLIENT MANAGEMENT ----------
_cl, _pc, _ct = Lock(), {}, {}

@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]):
    clients = {}
    for conf in configs:
        if 'asset_field' in conf:
            ac=conf['asset_field']; ci,ak=ac['channel_id'],ac['api_key']
            if ci not in clients:
                try: clients[ci] = thingspeak.Channel(ci,ak,fmt='json')
                except Exception as e: st.warning(f"Failed to create sync client for {ci}: {e}")
    return clients

def clear_all_caches():
    st.cache_data.clear(); st.cache_resource.clear(); sell.cache_clear(); buy.cache_clear()
    with _cl: _pc.clear(); _ct.clear()
    st.success("🗑️ All caches cleared!"); st.rerun()

# ---------- CALCULATION UTILITIES (Unchanged) ----------
@lru_cache(maxsize=128)
def sell(a: float, fc: int=1500, D: int=60):
    if a==0: return 0,0,0
    up=round((fc-D)/a,2); aq=round(abs(a*up-fc)/up) if up!=0 else 0
    return up, aq, round(a*up+aq*up,2)

@lru_cache(maxsize=128)
def buy(a: float, fc: int=1500, D: int=60):
    if a==0: return 0,0,0
    up=round((fc+D)/a,2); aq=round(abs(a*up-fc)/up) if up!=0 else 0
    return up, aq, round(a*up-aq*up,2)

def get_cached_price(t: str, age: int=30)->float:
    n=datetime.datetime.now()
    with _cl:
        if (t in _pc and (n-_ct.get(t,n)).seconds<age): return _pc[t]
    try: p=yf.Ticker(t).fast_info['lastPrice'];_pc[t],_ct[t]=p,n;return p
    except: return 0.0

# ---------- ASYNCHRONOUS DATA FETCHING LOGIC (Final Cachable Structure) ----------

# These are the innermost, non-cached, async functions
async def _internal_fetch_thingspeak_field(session: aiohttp.ClientSession, ci, ak, f) -> str:
    url=f"https://api.thingspeak.com/channels/{ci}/fields/{f}/last.json?api_key={ak}"
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status==200:
                d=await resp.json(); k=f"field{f}" if isinstance(f,int) else f
                return str(d.get(k)) if d.get(k) is not None else "0"
            return "0"
    except: return "0"

async def _internal_fetch_yfinance_history(t: str, sd: str) -> pd.DataFrame:
    loop=asyncio.get_event_loop()
    try:
        td=await loop.run_in_executor(None,lambda: yf.Ticker(t).history(period='max')[['Close']].round(3))
        td.index=td.index.tz_convert(tz='Asia/bangkok')
        if sd: td=td[td.index>=sd]
        return td
    except: return pd.DataFrame(columns=['Close'])

# These are the async WORKER functions, not cached
async def _worker_fetch_monitor_data(t, mci, mak, mf, sd):
    async with aiohttp.ClientSession() as session:
        res = await asyncio.gather(
            _internal_fetch_yfinance_history(t, sd),
            _internal_fetch_thingspeak_field(session, mci, mak, mf)
        )
    td, fx = res[0], res[1]
    td['index']=range(len(td)); df=pd.concat([td,pd.DataFrame(index=[f'+{i}' for i in range(5)])]).fillna(""); df['action']=""
    try:
        tr=SimulationTracer(fx); fa=tr.run(); na=min(len(df),len(fa))
        if na > 0: df.iloc[:na, df.columns.get_loc('action')]=fa[:na]
    except Exception as e: st.warning(f"Tracer Error for {t}: {e}")
    return df.tail(7), fx

async def _worker_fetch_asset_data(ci, ak, f):
    async with aiohttp.ClientSession() as session:
        res_str = await _internal_fetch_thingspeak_field(session, ci, ak, f)
    try: return float(res_str)
    except: return 0.0

# These are the SYNC, CACHED functions that wrap the async workers
@st.cache_data(ttl=300)
def get_monitor_data_cached(t, mci, mak, mf, sd):
    return asyncio.run(_worker_fetch_monitor_data(t, mci, mak, mf, sd))

@st.cache_data(ttl=60)
def get_asset_data_cached(ci, ak, f):
    return asyncio.run(_worker_fetch_asset_data(ci, ak, f))

# ---------- UI COMPONENTS (Unchanged) ----------
def render_asset_inputs(configs, last_assets):
    cols, asset_inputs = st.columns(len(configs)), {}
    for i, config in enumerate(configs):
        with cols[i]:
            ticker, last_asset_val = config['ticker'], last_assets.get(ticker, 0.0)
            if config.get('option_config'):
                oc = config['option_config']
                rv = st.number_input(oc['label'], step=0.001, value=last_asset_val, key=f"input_{ticker}_real", format="%.3f")
                asset_inputs[ticker] = oc['base_value'] + rv
            else:
                asset_inputs[ticker] = st.number_input(f'{ticker}_ASSET', step=0.001, value=last_asset_val, key=f"input_{ticker}_asset", format="%.3f")
    return asset_inputs

def render_asset_update_controls(configs, clients):
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            t, ac = config['ticker'], config['asset_field']
            if st.checkbox(f'@_{t}_ASSET', key=f'check_{t}'):
                nv = st.number_input(f"New Value for {t}", step=0.001, value=0.0, key=f'input_{t}')
                if st.button(f"GO_{t}", key=f'btn_{t}'):
                    try: clients[ac['channel_id']].update({ac['field']:nv}); st.success(f"Updated {t}"); clear_all_caches()
                    except Exception as e: st.error(f"Failed to update {t}: {e}")

def trading_section(ad: Dict, nex: int, nsd: int, clients: Dict):
    c, t, d = ad['config'], ad['ticker'], ad['df_data']
    try: ra=int(d.action.values[1+nex]); av=1-ra if nsd==1 else ra
    except: av=0
    if not st.checkbox(f'Limit_Order_{t}', value=bool(av), key=f'limit_order_{t}'): return
    bc, sc = ad['calculations']['buy'], ad['calculations']['sell']
    al, av_val, ac = ad['asset_last'], ad['asset_val'], c['asset_field']
    
    st.write('sell', 'A', sc[1], 'P', sc[0], 'C', sc[2])
    _,_,col3 = st.columns(3)
    if col3.checkbox(f'sell_match_{t}', key=f"sell_match_check_{t}"):
        if col3.button(f"GO_SELL_{t}", key=f"go_sell_btn_{t}"):
            try: nav=al-sc[1]; clients[ac['channel_id']].update({ac['field']:nav}); col3.success(f"Updated: {nav:.3f}"); clear_all_caches()
            except Exception as e: st.error(f"Failed to SELL {t}: {e}")
    try:
        cp = get_cached_price(t)
        if cp > 0:
            pv, fv=cp*av_val, c['fix_c']; plv,plc=pv-fv, "#a8d5a2" if pv-fv>=0 else "#fbb"
            st.markdown(f"Price: **{cp:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fv:,}) : <span style='color:{plc}; font-weight:bold;'>{plv:,.2f}</span>", unsafe_allow_html=True)
    except: pass
    st.write('buy', 'A', bc[1], 'P', bc[0], 'C', bc[2])
    _,_,col6 = st.columns(3)
    if col6.checkbox(f'buy_match_{t}', key=f"buy_match_check_{t}"):
        if col6.button(f"GO_BUY_{t}", key=f"go_buy_btn_{t}"):
            try: nav=al+bc[1]; clients[ac['channel_id']].update({ac['field']:nav}); col6.success(f"Updated: {nav:.3f}"); clear_all_caches()
            except Exception as e: st.error(f"Failed to BUY {t}: {e}")

# ---------- MAIN APPLICATION LOGIC ----------
def main():
    config_data = load_config()
    asset_configs = config_data['assets']
    global_start_date = config_data.get('global_settings', {}).get('start_date')
    thingspeak_clients = get_thingspeak_clients(asset_configs)

    # This part is now synchronous again, but the cached functions handle async inside
    with st.spinner("Fetching all data from cache or network..."):
        monitor_data_all = {}
        last_assets_all = {}
        for config in asset_configs:
            ticker = config['ticker']
            mon_conf = config['monitor_field']
            asset_conf = config['asset_field']
            
            monitor_data_all[ticker] = get_monitor_data_cached(ticker, mon_conf['channel_id'], mon_conf['api_key'], mon_conf['field'], global_start_date)
            last_assets_all[ticker] = get_asset_data_cached(asset_conf['channel_id'], asset_conf['api_key'], asset_conf['field'])

    with st.expander("⚙️ Controls & Asset Setup", expanded=True):
        nex, nsd = 0, 0
        if st.checkbox('nex_day'): nc,sc,_=st.columns([1,1,6]); nex=1 if nc.button("Nex_day") else nex; nex,nsd=(1,1) if sc.button("Nex_day_sell") else (nex,nsd)
        cc=st.columns(8); sc=cc[0].checkbox('start'); dv=cc[7].number_input('Diff',step=1,value=60)
        if sc: render_asset_update_controls(asset_configs, thingspeak_clients)
        asset_inputs = render_asset_inputs(asset_configs, last_assets_all)

    st.write("_____")
    
    processed_assets = []
    for config in asset_configs:
        t, (d,f), av = config['ticker'], monitor_data_all.get(t,(pd.DataFrame(),"0")), asset_inputs.get(t,0.0)
        ae = "⚪"
        try: ra=int(d.action.values[1+nex]); fa=1-ra if nsd==1 else ra; ae="🟢" if fa==1 else "🔴"
        except: pass
        cp=get_cached_price(t); plv=(cp*av)-config['fix_c'] if cp>0 and av>0 else 0.0
        processed_assets.append({
            "config": config, "ticker": t, "df_data": d, "fx_js_str": f, "asset_last": last_assets_all.get(t,0.0), "asset_val": av,
            "calculations": {'buy':buy(av,config['fix_c'],dv),'sell':sell(av,config['fix_c'],dv)}, "action_emoji":ae, "pl_value":plv
        })

    with st.expander("📈 Trading Dashboard", expanded=True):
        tabs=st.tabs([f"{a['ticker']} {a['action_emoji']} | P/L: {a['pl_value']:,.2f}" for a in processed_assets])
        for i, ad in enumerate(processed_assets):
            with tabs[i]:
                st.write(f"**{ad['ticker']}** (f(x): `{ad['fx_js_str']}`)")
                trading_section(ad, nex, nsd, thingspeak_clients)
                st.write("_____")
                with st.expander("Show Raw Data Action"): st.dataframe(ad['df_data'], use_container_width=True)

    if st.sidebar.button("RERUN"): clear_all_caches()

if __name__ == "__main__":
    main()
