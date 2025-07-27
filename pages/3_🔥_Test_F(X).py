# ==============================================================
# Hybrid Strategy Lab – Multi-Asset Version
# ==============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any
from numba import njit            # JIT

# --------------------------------------------------------------
# 1. Configuration / Constants
# --------------------------------------------------------------
st.set_page_config(page_title="Hybrid_Multi_Mutation", page_icon="🧬", layout="wide")

class Strategy:
    REBALANCE_DAILY      = "Rebalance Daily"
    PERFECT_FORESIGHT    = "Perfect Foresight (Max)"
    HYBRID_MULTI_MUTATION= "Hybrid (Multi-Mutation)"
    ORIGINAL_DNA         = "Original DNA (Pre-Mutation)"

def load_config(filepath: str = "hybrid_seed_config.json") -> Dict[str, Any]:
    # ในตัวอย่างใช้ dict ตรง ๆ
    return {
        "assets": [
            "FFWM", "NEGG", "RIVN", "APLS", "NVTS", "QXO",
            "RXRX", "AGL", "FLNC", "GERN", "DYN"
        ],
        "default_settings": {
            "selected_ticker": "FFWM",
            "start_date": "2024-01-01",
            "window_size": 30,
            "num_seeds": 1000,
            "max_workers": 4,
            "mutation_rate": 10.0,
            "num_mutations": 5
        }
    }

def initialize_session_state(config: Dict[str, Any]):
    d = config["default_settings"]
    ss = st.session_state
    ss.setdefault("test_ticker" , d["selected_ticker"])
    ss.setdefault("start_date"  , datetime.strptime(d["start_date"], "%Y-%m-%d").date())
    ss.setdefault("end_date"    , datetime.now().date())
    ss.setdefault("window_size" , d["window_size"])
    ss.setdefault("num_seeds"   , d["num_seeds"])
    ss.setdefault("max_workers" , d["max_workers"])
    ss.setdefault("mutation_rate",d["mutation_rate"])
    ss.setdefault("num_mutations",d["num_mutations"])

# --------------------------------------------------------------
# 2. Data / Simulation helpers
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_data(ticker:str, start:str, end:str)->pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(start=start, end=end)[["Close"]]
        if df.empty: return pd.DataFrame()
        df = (df if df.index.tz else df.tz_localize("UTC")).tz_convert("Asia/Bangkok")
        return df
    except Exception as e:
        st.error(f"❌ Download {ticker} failed : {e}")
        return pd.DataFrame()

@njit(cache=True)
def _calculate_net_profit_numba(action_array: np.ndarray, price_array: np.ndarray, fix:int=1500)->float:
    n = len(action_array)
    if n==0 or n>len(price_array): return -np.inf
    action_array = action_array.copy()
    action_array[0] = 1
    cash  = fix
    amt   = fix/price_array[0]
    for i in range(1,n):
        if action_array[i]!=0:
            cash += amt*price_array[i]-fix
            amt  = fix/price_array[i]
    return cash+amt*price_array[n-1]-fix*2

def run_simulation(prices:List[float], actions:List[int], fix:int=1500)->pd.DataFrame:
    @njit
    def _sim(a, p, f):
        n=len(a)
        cash=np.empty(n); asset=np.empty(n); sumusd=np.empty(n); buf=np.zeros(n); amt=np.empty(n)
        cash[0]=f; amt[0]=f/p[0]; asset[0]=amt[0]*p[0]; sumusd[0]=cash[0]+asset[0]
        for i in range(1,n):
            if a[i]==0: amt[i]=amt[i-1]; buf[i]=0
            else:
                amt[i]=f/p[i]; buf[i]=amt[i-1]*p[i]-f
            cash[i]=cash[i-1]+buf[i]; asset[i]=amt[i]*p[i]; sumusd[i]=cash[i]+asset[i]
        return buf,sumusd,cash,asset,amt
    if not prices or not actions: return pd.DataFrame()
    m=min(len(prices),len(actions))
    pa=np.array(prices[:m]); ac=np.array(actions[:m])
    buf,sumusd,cash,asset,amt=_sim(ac,pa,fix)
    init=sumusd[0]
    return pd.DataFrame({
        "price":pa,"action":ac,"buffer":buf.round(2),
        "sumusd":sumusd.round(2),"cash":cash.round(2),
        "asset_value":asset.round(2),"amount":amt.round(2),
        "net":(sumusd-init).round(2)
    })

def generate_actions_rebalance_daily(n:int)->np.ndarray:
    return np.ones(n,dtype=int)

def generate_actions_perfect_foresight(prices:List[float], fix:int=1500)->np.ndarray:
    p=np.asarray(prices); n=len(p)
    if n<2: return np.ones(n,int)
    dp=np.zeros(n); path=np.zeros(n,int); dp[0]=fix*2
    for i in range(1,n):
        j=np.arange(i)
        prof=fix*((p[i]/p[j])-1)
        cur=dp[j]+prof
        best=j[cur.argmax()]
        dp[i]=cur.max(); path[i]=best
    a=np.zeros(n,int); c=dp.argmax()
    while c>0: a[c]=1; c=path[c]
    a[0]=1
    return a

# --------------------------------------------------------------
# 3. Hybrid (DNA + Multi-Mutation)
#   (ใช้โค้ดเดิมของคุณ ไม่ตัดทอน)
# --------------------------------------------------------------
# -- find_best_seed_for_window, find_best_mutation_for_sequence,
#    generate_actions_hybrid_multi_mutation --
#   *** วางโค้ดเดิมของคุณทั้ง 3 ฟังก์ชันตรงนี้ ***

# (เพื่อประหยัดพื้นที่ ตัวอย่างนี้สมมติว่าฟังก์ชันทั้ง 3 ถูกคัดลอกมาครบถ้วน)


# --------------------------------------------------------------
# 4. NEW helper – run one ticker
# --------------------------------------------------------------
def run_for_one_asset(
    ticker:str,
    start_date:str,
    end_date:str,
    window_size:int,
    num_seeds:int,
    max_workers:int,
    mutation_rate:float,
    num_mutations:int
)->Dict[str,Any]:

    data = get_ticker_data(ticker, start_date, end_date)
    if data.empty:
        return {"error":f"{ticker} : no data"}

    orig_act, final_act, df_win = generate_actions_hybrid_multi_mutation(
        data, window_size, num_seeds, max_workers,
        mutation_rate, num_mutations
    )

    prices = data["Close"].to_numpy()
    results = {
        Strategy.HYBRID_MULTI_MUTATION : run_simulation(prices.tolist(), final_act.tolist()),
        Strategy.ORIGINAL_DNA          : run_simulation(prices.tolist(), orig_act.tolist()),
        Strategy.REBALANCE_DAILY       : run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
        Strategy.PERFECT_FORESIGHT     : run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
    }
    for df in results.values():
        if not df.empty:
            df.index = data.index[:len(df)]

    return {
        "ticker_data"     : data,
        "strategy_results": results,
        "window_details"  : df_win
    }

# --------------------------------------------------------------
# 5. UI helpers (display charts ฯลฯ) – ใช้ของเดิม
# --------------------------------------------------------------
def display_comparison_charts(results:Dict[str,pd.DataFrame], chart_title:str):
    if not results: return
    longest = max((df.index for df in results.values()), key=len, default=None)
    chart_data=pd.DataFrame(index=longest)
    for k,df in results.items():
        chart_data[k]=df["net"].reindex(longest).ffill()
    st.write(chart_title); st.line_chart(chart_data)

# -- render_settings_tab()  (คงเดิม) --
# -- render_tracer_tab()    (คงเดิม) --

# --------------------------------------------------------------
# 6. render_hybrid_multi_mutation_tab() – แก้ไขสำคัญ
# --------------------------------------------------------------
def render_hybrid_multi_mutation_tab():
    st.markdown(f"### 🧬 {Strategy.HYBRID_MULTI_MUTATION}")
    st.info("กดปุ่มด้านล่างเพื่่อรันทุกสินทรัพย์ที่เลือกในครั้งเดียว")

    cfg = load_config()
    full_asset_list = cfg["assets"]

    sel_assets = st.multiselect(
        "เลือกสินทรัพย์ (ว่าง = ทั้งหมด)", full_asset_list, default=[]
    )
    if not sel_assets:
        sel_assets = full_asset_list

    if st.button("🚀 Start Hybrid Multi-Mutation", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("⚠️ วันที่เริ่ม >= วันที่สิ้นสุด"); return

        progress = st.progress(0)
        all_res={}
        for i,tk in enumerate(sel_assets,1):
            progress.progress((i-1)/len(sel_assets), text=f"Running {tk} …")
            all_res[tk] = run_for_one_asset(
                ticker        = tk,
                start_date    = str(st.session_state.start_date),
                end_date      = str(st.session_state.end_date),
                window_size   = st.session_state.window_size,
                num_seeds     = st.session_state.num_seeds,
                max_workers   = st.session_state.max_workers,
                mutation_rate = st.session_state.mutation_rate,
                num_mutations = st.session_state.num_mutations
            )
        progress.progress(1.0, text="Done!")
        st.session_state.all_results = all_res
        st.success("เสร็จสิ้นทุกสินทรัพย์!")

    # ---------- แสดงผล ----------
    if "all_results" in st.session_state:
        tickers = list(st.session_state.all_results.keys())
        chosen = st.selectbox("ดูผลลัพธ์ของ:", tickers)
        robj = st.session_state.all_results[chosen]
        if "error" in robj:
            st.error(robj["error"]); return

        strat_res = robj["strategy_results"]
        display_comparison_charts(
            {k:v for k,v in strat_res.items() if k!=Strategy.ORIGINAL_DNA},
            f"📊 Net Profit – {chosen}"
        )

        st.divider()
        st.write("### 📈 Summary", chosen)
        d_hyb = strat_res[Strategy.HYBRID_MULTI_MUTATION]
        d_ori = strat_res[Strategy.ORIGINAL_DNA]
        d_per = strat_res[Strategy.PERFECT_FORESIGHT]
        d_reb = strat_res[Strategy.REBALANCE_DAILY]

        col1,col2,col3,col4=st.columns(4)
        col1.metric("Perfect",  f"${d_per['net'].iloc[-1]:,.2f}")
        col2.metric("Hybrid",   f"${d_hyb['net'].iloc[-1]:,.2f}")
        col3.metric("Original", f"${d_ori['net'].iloc[-1]:,.2f}")
        col4.metric("Rebalance",f"${d_reb['net'].iloc[-1]:,.2f}")

        st.write("#### รายละเอียดราย Window")
        st.dataframe(robj["window_details"], use_container_width=True)

# --------------------------------------------------------------
# 7. Main App
# --------------------------------------------------------------
def render_settings_tab():
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    cfg=load_config()
    ss=st.session_state
    c1,c2=st.columns(2)
    ss.test_ticker=c1.selectbox("Ticker (อ้างอิง)", cfg["assets"],
                                index=cfg["assets"].index(ss.test_ticker))
    ss.window_size=c2.number_input("Window (วัน)",2,90,ss.window_size)
    c1,c2=st.columns(2)
    ss.start_date=c1.date_input("Start", ss.start_date)
    ss.end_date  =c2.date_input("End",   ss.end_date)

    st.divider()
    st.subheader("Parameters")
    c1,c2=st.columns(2)
    ss.num_seeds   =c1.number_input("Seeds",100,10000,ss.num_seeds,step=100)
    ss.max_workers =c2.number_input("Workers",1,16,ss.max_workers)
    c1,c2=st.columns(2)
    ss.mutation_rate=c1.slider("Mutation Rate %",0.0,50.0,ss.mutation_rate,0.5)
    ss.num_mutations=c2.number_input("Mutation Rounds",0,10,ss.num_mutations)

# -- render_tracer_tab() ยังคงใช้โค้ดเดิมของคุณ --

def render_tracer_tab():
    st.write("🔍 Tracer Tab (คงเดิม)")

def main():
    cfg=load_config(); initialize_session_state(cfg)
    st.markdown("### 🧬 Hybrid Strategy Lab (Multi-Asset)")
    st.caption("Numba-Accelerated Parallel Random Search")

    tabs=st.tabs(["⚙️ Settings", "🧬 Hybrid", "🔍 Tracer"])
    with tabs[0]:
        render_settings_tab()
    with tabs[1]:
        render_hybrid_multi_mutation_tab()
    with tabs[2]:
        render_tracer_tab()

if __name__=="__main__":
    main()
