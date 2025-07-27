# ==============================================================
# Hybrid Strategy Lab – 100 % SELF-CONTAINED
# (เวอร์ชันอ้างอิง – อัลกอริทึมถูกทำให้ง่ายแต่ใช้งานได้จริง)
# ==============================================================

import pandas as pd, numpy as np, yfinance as yf, streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from datetime import datetime
from numba import njit

# --------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Hybrid_Multi_Mutation", page_icon="🧬", layout="wide")

class Strategy:
    HYBRID_MULTI_MUTATION = "Hybrid (Multi-Mutation)"
    ORIGINAL_DNA          = "Original DNA"
    PERFECT_FORESIGHT     = "Perfect Foresight"
    REBALANCE_DAILY       = "Rebalance Daily"

_CFG = dict(
    assets = ["FFWM","NEGG","RIVN","APLS","NVTS","QXO","RXRX",
              "AGL","FLNC","GERN","DYN"],
    default_settings = dict(
        selected_ticker = "FFWM",
        start_date      = "2024-01-01",
        window_size     = 30,
        num_seeds       = 400,
        max_workers     = 4,
        mutation_rate   = 10.0,   # %
        num_mutations   = 4
    )
)

def _init_ss():
    d=_CFG["default_settings"]; ss=st.session_state
    ss.setdefault("test_ticker" ,d["selected_ticker"])
    ss.setdefault("start_date"  ,datetime.strptime(d["start_date"],"%Y-%m-%d").date())
    ss.setdefault("end_date"    ,datetime.now().date())
    ss.setdefault("window_size" ,d["window_size"])
    ss.setdefault("num_seeds"   ,d["num_seeds"])
    ss.setdefault("max_workers" ,d["max_workers"])
    ss.setdefault("mutation_rate",d["mutation_rate"])
    ss.setdefault("num_mutations",d["num_mutations"])

# --------------------------------------------------------------
# 2. DATA + SIMULATION
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_data(ticker:str, start:str, end:str)->pd.DataFrame:
    try:
        df=yf.Ticker(ticker).history(start=start,end=end)[["Close"]]
        if df.empty: return df
        return df
    except Exception as e:
        st.error(f"Download error {ticker}: {e}")
        return pd.DataFrame()

@njit(cache=True)
def _sim_numba(actions, prices, fee):
    n=len(actions)
    cash=fee; amt=fee/prices[0]
    net=np.zeros(n)
    for i in range(n):
        if i>0 and actions[i]:
            cash+=amt*prices[i]-fee
            amt =fee/prices[i]
        net[i]=cash+amt*prices[i]-fee*2
    return net

def run_simulation(prices:List[float], actions:List[int],
                   fee:int=1500)->pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    m=min(len(prices),len(actions))
    p=np.asarray(prices[:m]); a=np.asarray(actions[:m],dtype=np.int8)
    net=_sim_numba(a,p,fee)
    return pd.DataFrame({"price":p,"action":a,"net":net})

def generate_actions_rebalance_daily(n:int)->np.ndarray:
    return np.ones(n,dtype=np.int8)

def generate_actions_perfect_foresight(prices:List[float])->np.ndarray:
    n=len(prices)
    a=np.zeros(n,dtype=np.int8); best_idx=np.argmax(prices)
    a[0]=1; a[best_idx]=1        # ซื้อวันแรก ขายวัน peak (ง่าย ๆ)
    return a

# --------------------------------------------------------------
# 3. SIMPLE EVOLUTIONARY SEARCH
# --------------------------------------------------------------
def _random_action_seq(n:int)->np.ndarray:
    a=np.random.randint(0,2,n,dtype=np.int8)
    a[0]=1                      # buy day-0 เสมอ
    return a

def find_best_seed_for_window(prices:np.ndarray,
                              num_seeds:int)->np.ndarray:
    best_act=None; best_net=-1e18
    for _ in range(num_seeds):
        act=_random_action_seq(len(prices))
        net=run_simulation(prices,act)["net"].iloc[-1]
        if net>best_net:
            best_net=net; best_act=act
    return best_act

def mutate_actions(actions:np.ndarray, rate:float)->np.ndarray:
    m=actions.copy()
    mask=np.random.rand(len(m))<(rate/100)
    m[mask]=1-m[mask]
    m[0]=1
    return m

def find_best_mutation_for_sequence(prices:np.ndarray,
                                    base_actions:np.ndarray,
                                    mutation_rate:float,
                                    num_rounds:int)->np.ndarray:
    best=base_actions.copy()
    best_net=run_simulation(prices,best)["net"].iloc[-1]
    for _ in range(num_rounds):
        cand=mutate_actions(best,mutation_rate)
        net =run_simulation(prices,cand)["net"].iloc[-1]
        if net>best_net: best,best_net=cand,net
    return best

def generate_actions_hybrid_multi_mutation(
        price_df:pd.DataFrame,
        window_size:int,
        num_seeds:int,
        max_workers:int,
        mutation_rate:float,
        num_mutations:int
)->tuple[np.ndarray,np.ndarray,pd.DataFrame]:
    prices=price_df["Close"].to_numpy()
    n=len(prices)

    # ---------- 3.1 หา DNA แบบ “ดีที่สุดต่อ window” ----------
    dna=np.zeros(n,dtype=np.int8)
    win_records=[]
    for start in range(0,n,window_size):
        end=min(start+window_size,n)
        window_prices=prices[start:end]
        best_seed=find_best_seed_for_window(window_prices,num_seeds)
        dna[start:end]=best_seed
        win_records.append(dict(
            start_idx=start,end_idx=end-1,profit=run_simulation(
                window_prices,best_seed)["net"].iloc[-1]
        ))
    df_windows=pd.DataFrame(win_records)

    # ---------- 3.2 Multi-Mutation ----------
    final_actions=find_best_mutation_for_sequence(
        prices,dna,mutation_rate,num_mutations
    )

    return dna, final_actions, df_windows

# --------------------------------------------------------------
# 4. CORE : run_for_one_asset
# --------------------------------------------------------------
def run_for_one_asset(ticker:str,
                      start_date:str,
                      end_date:str,
                      window_size:int,
                      num_seeds:int,
                      max_workers:int,
                      mutation_rate:float,
                      num_mutations:int)->Dict[str,Any]:
    data=get_ticker_data(ticker,start_date,end_date)
    if data.empty: return {"error":"no data"}

    orig_act,fin_act,df_win=generate_actions_hybrid_multi_mutation(
        data,window_size,num_seeds,max_workers,mutation_rate,num_mutations
    )
    prices=data["Close"].to_list()
    results={
        Strategy.HYBRID_MULTI_MUTATION:run_simulation(prices,fin_act.tolist()),
        Strategy.ORIGINAL_DNA         :run_simulation(prices,orig_act.tolist()),
        Strategy.REBALANCE_DAILY      :run_simulation(prices,
                                generate_actions_rebalance_daily(len(prices))),
        Strategy.PERFECT_FORESIGHT    :run_simulation(prices,
                                generate_actions_perfect_foresight(prices))
    }
    for df in results.values():
        if not df.empty: df.index=data.index[:len(df)]
    return dict(ticker_data=data,
                strategy_results=results,
                window_details=df_win)

# --------------------------------------------------------------
# 5. UI HELPERS
# --------------------------------------------------------------
def _metrics_row(results:Dict[str,pd.DataFrame]):
    cols=st.columns(len(results))
    for c,(k,df) in zip(cols,results.items()):
        c.metric(k,f"${df['net'].iloc[-1]:,.0f}")

def chart_compare(results:Dict[str,pd.DataFrame],title:str):
    if not results: return
    base_index=max((df.index for df in results.values()), key=len)
    chart=pd.DataFrame(index=base_index)
    for k,v in results.items():
        chart[k]=v["net"].reindex(base_index).fillna(method="ffill")
    st.write(title); st.line_chart(chart)

# --------------------------------------------------------------
# 6. STREAMLIT TABS
# --------------------------------------------------------------
def tab_settings():
    ss=st.session_state
    c1,c2=st.columns(2)
    ss.test_ticker=c1.selectbox("Reference Ticker",_CFG["assets"],
                                index=_CFG["assets"].index(ss.test_ticker))
    ss.window_size=c2.number_input("Window size",2,90,ss.window_size)
    c1,c2=st.columns(2)
    ss.start_date=c1.date_input("Start",ss.start_date)
    ss.end_date  =c2.date_input("End",ss.end_date)
    st.divider()
    ss.num_seeds=c1.number_input("Seeds",100,10000,ss.num_seeds,100)
    ss.max_workers=c2.number_input("Workers",1,16,ss.max_workers)
    ss.mutation_rate=c1.slider("Mutation %",0.0,50.0,ss.mutation_rate,0.5)
    ss.num_mutations=c2.number_input("Mutation rounds",0,10,ss.num_mutations)

def tab_hybrid():
    ss=st.session_state
    st.info("เลือกสินทรัพย์แล้วกด 🚀 เพื่อรันทั้งหมด")
    sel=st.multiselect("Assets",_CFG["assets"],[])
    if not sel: sel=_CFG["assets"]

    if st.button("🚀 Run"):
        if ss.start_date>=ss.end_date:
            st.error("Start ≥ End"); return
        prog=st.progress(0)
        all_res={}
        for i,tk in enumerate(sel,1):
            prog.progress((i-1)/len(sel),text=f"{tk} …")
            all_res[tk]=run_for_one_asset(
                tk,str(ss.start_date),str(ss.end_date),
                ss.window_size,ss.num_seeds,ss.max_workers,
                ss.mutation_rate,ss.num_mutations
            )
        prog.progress(1.0,text="Done")
        ss.all_results=all_res
        st.success("Finished!")

    if "all_results" in ss:
        tk=st.selectbox("View result of",list(ss.all_results))
        robj=ss.all_results[tk]
        if "error" in robj: st.error(robj["error"]); return
        chart_compare({k:v for k,v in robj["strategy_results"].items()
                       if k!=Strategy.ORIGINAL_DNA},
                      f"Net Profit – {tk}")
        st.divider()
        _metrics_row(robj["strategy_results"])
        st.write("Window details"); st.dataframe(robj["window_details"])

def tab_tracer():
    st.write("🔍 Tracer (placeholder)")

# --------------------------------------------------------------
# 7. MAIN
# --------------------------------------------------------------
def main():
    _init_ss()
    st.title("🧬 Hybrid Strategy Lab – Multi-Asset")
    tabs=st.tabs(["⚙️ Settings","🧬 Hybrid","🔍 Tracer"])
    with tabs[0]: tab_settings()
    with tabs[1]: tab_hybrid()
    with tabs[2]: tab_tracer()

if __name__=="__main__":
    main()
