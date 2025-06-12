import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px
from typing import Dict, List, Tuple, Any

# --- Constants --- 
CONFIG_FILEPATH = 'benchmark_fx_config.json'

# --- 1. Configuration & Data Loading ---

@st.cache_data
def load_config(filepath: str) -> Dict[str, Any]:
    """
    โหลดและแยกวิเคราะห์ไฟล์ตั้งค่า JSON
    
    Args:
        filepath (str): ที่อยู่ของไฟล์ config.json

    Returns:
        Dict[str, Any]: Dictionary ของค่าที่ตั้งไว้
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์ตั้งค่า: {filepath}")
        return {}
    except json.JSONDecodeError:
        st.error(f"ไฟล์ {filepath} มีรูปแบบ JSON ไม่ถูกต้อง")
        return {}

@st.cache_data
def fetch_price_history(ticker: str) -> pd.DataFrame:
    """
    ดึงข้อมูลราคาย้อนหลังทั้งหมดสำหรับ Ticker ที่กำหนดและแปลง Timezone
    
    Args:
        ticker (str): สัญลักษณ์ Ticker

    Returns:
        pd.DataFrame: DataFrame ของราคาปิด หรือ DataFrame ว่างถ้าไม่สำเร็จ
    """
    try:
        ticker_data = yf.Ticker(ticker)
        history = ticker_data.history(period='max')[['Close']]
        if history.empty:
            return pd.DataFrame()
        history.index = history.index.tz_convert(tz='Asia/bangkok')
        return history
    except Exception as e:
        # st.warning(f"ไม่สามารถดึงข้อมูลราคาสำหรับ {ticker} ได้: {e}")
        return pd.DataFrame()

# --- 2. Core Calculation Logic ---

def calculate_asset_dynamics(ticker: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    คำนวณค่า re และ net_pv สำหรับ asset เดียว (เดิมคือฟังก์ชัน delta2)
    
    Args:
        ticker (str): สัญลักษณ์ Ticker
        params (Dict[str, Any]): Dictionary ของพารามิเตอร์สำหรับ Ticker นี้

    Returns:
        pd.DataFrame: DataFrame ที่มีคอลัมน์ 're' และ 'net_pv'
    """
    try:
        # ดึงค่า settings จาก params dict
        pred = params['pred']
        filter_date = params['filter_date']
        step = params['step']
        fixed_asset_value = params['fixed_asset_value']
        initial_cash_balance = params['cash_balance']

        hist_data = fetch_price_history(ticker)
        if hist_data.empty:
            return pd.DataFrame()

        filtered_data = hist_data[hist_data.index >= filter_date].copy()
        if filtered_data.empty:
            return pd.DataFrame()

        entry_price = filtered_data.Close[0]
        
        # --- สร้าง Grid อ้างอิง ---
        samples = np.arange(0, np.around(entry_price, 2) * 3 + step, step)
        grid_df = pd.DataFrame({'Asset_Price': np.around(samples, 2)})
        grid_df['Fixed_Asset_Value'] = fixed_asset_value
        grid_df['Amount_Asset'] = grid_df['Fixed_Asset_Value'] / grid_df['Asset_Price']
        
        # คำนวณ Cash Balance สำหรับ Grid ราคาขึ้น (df_top)
        df_top = grid_df[grid_df.Asset_Price >= np.around(entry_price, 2)].copy()
        df_top['Cash_Balan'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
        df_top.fillna(0, inplace=True)
        df_top['Cash_Balan'] = initial_cash_balance + df_top['Cash_Balan'].cumsum()
        df_top = df_top.sort_values(by='Amount_Asset')[:-1]

        # คำนวณ Cash Balance สำหรับ Grid ราคาลง (df_down)
        df_down = grid_df[grid_df.Asset_Price <= np.around(entry_price, 2)].copy()
        df_down = df_down.sort_values(by='Asset_Price', ascending=False)
        df_down['Cash_Balan'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
        df_down.fillna(0, inplace=True)
        df_down['Cash_Balan'] = initial_cash_balance + df_down['Cash_Balan'].cumsum()
        
        simulation_grid = pd.concat([df_top, df_down], axis=0)
        production_costs = abs(simulation_grid['Cash_Balan'].iloc[-1] - initial_cash_balance)
        
        # --- จำลองการเทรดตามประวัติราคา ---
        sim_df = filtered_data
        sim_df['Close'] = np.around(sim_df['Close'].values, 2)
        sim_df['pred'] = pred
        sim_df['Fixed_Asset_Value'] = fixed_asset_value
        sim_df['Amount_Asset'] = 0.0
        sim_df.iloc[0, sim_df.columns.get_loc('Amount_Asset')] = fixed_asset_value / sim_df.iloc[0]['Close']
        sim_df['re'] = 0.0
        sim_df['Cash_Balan'] = initial_cash_balance

        # Vectorized calculation ไม่ได้ง่ายสำหรับ stateful loop แบบนี้, for-loop ยังคงจำเป็น
        amount_asset_arr = sim_df['Amount_Asset'].values
        close_arr = sim_df['Close'].values
        re_arr = sim_df['re'].values
        cash_balan_arr = sim_df['Cash_Balan'].values

        for i in range(1, len(sim_df)):
            if pred == 1: # สมมติว่า pred เป็น 1 ตลอดตามโค้ดเดิม
                amount_asset_arr[i] = fixed_asset_value / close_arr[i]
                re_arr[i] = (amount_asset_arr[i-1] * close_arr[i]) - fixed_asset_value
            else: # pred == 0
                amount_asset_arr[i] = amount_asset_arr[i-1]
                re_arr[i] = 0
            cash_balan_arr[i] = cash_balan_arr[i-1] + re_arr[i]

        sim_df['Amount_Asset'] = amount_asset_arr
        sim_df['re'] = re_arr
        sim_df['Cash_Balan'] = cash_balan_arr

        # --- คำนวณค่าสุดท้าย ---
        sim_df = sim_df.merge(simulation_grid[['Asset_Price', 'Cash_Balan']].rename(columns={'Cash_Balan': 'refer_model'}),
                              left_on='Close', right_on='Asset_Price', how='left')
        
        sim_df['Production_Costs'] = production_costs
        sim_df['pv'] = sim_df['Cash_Balan'] + (sim_df['Amount_Asset'] * sim_df['Close'])
        sim_df['refer_pv'] = sim_df['refer_model'] + fixed_asset_value
        sim_df['net_pv'] = sim_df['pv'] - sim_df['refer_pv']

        return sim_df[['re', 'net_pv']]

    except Exception as e:
        # st.warning(f"เกิดข้อผิดพลาดในการคำนวณสำหรับ Ticker '{ticker}': {e}")
        return pd.DataFrame()


@st.cache_data
def process_assets_data(tickers_to_process: Tuple[str, ...], _config_dict: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    ประมวลผลสินทรัพย์หลายตัวพร้อมกัน และรวมผลลัพธ์ (เดิมคือฟังก์ชัน Un_15)
    
    Args:
        tickers_to_process (Tuple[str, ...]): Tuple ของ Ticker ที่จะประมวลผล (จำเป็นต้องเป็น Tuple เพื่อให้ cache ทำงาน)
        _config_dict (Dict[str, Any]): Dictionary ของค่าตั้งค่าทั้งหมด

    Returns:
        Tuple: (net_pv_df, re_df, buffer_df, diff_fx)
    """
    all_results = {
        ticker: calculate_asset_dynamics(ticker, _config_dict[ticker]['params'])
        for ticker in tickers_to_process
    }
    valid_results = {t: res for t, res in all_results.items() if not res.empty}
    
    if not valid_results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), np.array([])

    re_df = pd.concat([res[['re']].rename(columns={"re": f"{t}_re"}) for t, res in valid_results.items()], axis=1)
    net_pv_df = pd.concat([res[['net_pv']].rename(columns={"net_pv": f"{t}_net_pv"}) for t, res in valid_results.items()], axis=1)

    re_df['Sum_Buffer'] = re_df.sum(axis=1, numeric_only=True).cumsum()
    net_pv_df['Sum_Delta'] = net_pv_df.sum(axis=1, numeric_only=True)

    buffer_df = pd.DataFrame()
    for ticker in valid_results.keys():
        buffer_df[f'{ticker}_Buffer'] = re_df[f'{ticker}_re'].cumsum()

    last_ticker = tickers_to_process[-1]
    diff_fx = np.array([])
    if last_ticker in valid_results:
        diff_fx = valid_results[last_ticker].net_pv.diff().fillna(0.0).values
        
    return net_pv_df, re_df, buffer_df, diff_fx

# --- 3. UI Display and Application Flow ---

def display_dashboard(selected_ticker, config, asset_configs, net_pv_df, buffer_df, diff_fx):
    """
    ฟังก์ชันสำหรับวาดส่วนแสดงผลหลักทั้งหมด
    """
    st.write(f"Dashboard for: {selected_ticker}")

    # --- ส่วนแสดงผลหลัก ---
    checkbox1 = st.checkbox('Delta_Benchmark_F(X) / Max.Sum_Buffer %', value=True)
    if checkbox1:
        delta_percent_df = pd.DataFrame()
        
        # สร้าง Delta_2 (กราฟเปอร์เซ็นต์) แบบ Dynamic
        for ticker in net_pv_df.columns[net_pv_df.columns.str.endswith('_net_pv')]:
            base_ticker = ticker.replace('_net_pv', '')
            asset_name = asset_configs[base_ticker]['name']
            net_pv_col = f'{base_ticker}_net_pv'
            buffer_col = f'{base_ticker}_Buffer'
            
            if buffer_col in buffer_df.columns:
                fixed_value = asset_configs[base_ticker]['params']['fixed_asset_value']
                survival = buffer_df[buffer_col].abs().max() + buffer_df[buffer_col].abs().min()
                # survival = (abs(np.min(buffer_df[buffer_col].values)) + abs(np.max(buffer_df[buffer_col].values)))
                delta_percent_df[asset_name] = net_pv_df[net_pv_col] / (fixed_value + survival) * 100

        # --- ดึงข้อมูลราคาและเตรียมกราฟ ---
        price_history = fetch_price_history(selected_ticker)
        filter_date_1 = config['global_settings']['filter_date_1']
        filter_date_2 = config['global_settings']['filter_date_2']
        
        price_data_long = price_history[price_history.index >= filter_date_1].reset_index(drop=True)
        price_data_short = price_history[price_history.index >= filter_date_2].reset_index(drop=True)
        
        if len(diff_fx) == len(price_data_short):
            price_data_short['Diff'] = diff_fx
        else:
            price_data_short['Diff'] = 0

        # --- แสดงข้อมูลตัวเลข ---
        st.write('____')
        add_risk = net_pv_df[f'{selected_ticker}_net_pv'].iloc[-1]
        survival_selected = buffer_df[f'{selected_ticker}_Buffer'].abs().max() + buffer_df[f'{selected_ticker}_Buffer'].abs().min()
        fixed_asset_value_selected = asset_configs[selected_ticker]['params']['fixed_asset_value']
        
        st.metric(label="Add Risk", value=f"{add_risk:,.2f}")
        st.metric(label="Survival Value", value=f"{survival_selected:,.2f}")
        st.write(f'Data > add_risk/Fixed: {fixed_asset_value_selected / add_risk:.2f} , Premium & Discount: {(fixed_asset_value_selected + survival_selected) / add_risk:.2f}')
        
        try:
            yahoo = yf.Ticker(selected_ticker)
            bs = yahoo.get_balance_sheet().iloc[:, 0]
            net_current_assets = bs.get('CurrentAssets', 0) - (bs.get('CurrentLiabilities', 0) + bs.get('LongTermDebt', 0))
            net_current_assets_pct = (net_current_assets / bs.get('CurrentAssets', 1)) * 100
            st.write(f'Finance > Net Current Assets %: {net_current_assets_pct:.2f}%')
        except Exception:
            st.write('Finance > ไม่สามารถดึงข้อมูล Balance Sheet ได้')
        st.write('____')

        # --- วาดกราฟ ---
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        nbinsy_val = config['global_settings'].get('nbinsy_default', 40)
        
        fig_price_short = px.line(price_data_short, y='Close', title=f'ราคาปิด (ตั้งแต่ {filter_date_2.split(" ")[0]})')
        fig_density = px.density_heatmap(price_data_short, x="Diff", y="Close", marginal_y="histogram", text_auto=True, nbinsy=nbinsy_val, color_continuous_scale=px.colors.sequential.Turbo, title="Price vs. Daily Net PV Change")
        fig_density.add_hline(y=price_data_short.Close.iloc[-1], line_color='Red', annotation_text="Current Price")
        
        fig_net_pv = px.line(net_pv_df, y=f'{selected_ticker}_net_pv', title=f'Net PV ของ {selected_ticker}')
        fig_price_long = px.line(price_data_long, y='Close', title=f'ราคาปิด (ตั้งแต่ {filter_date_1.split(" ")[0]})')
        fig_price_long.add_vline(x=len(price_data_long) - len(price_data_short), line_color='Red', annotation_text="Start of Recent Period")

        col1.plotly_chart(fig_price_short, use_container_width=True)
        col2.plotly_chart(fig_density, use_container_width=True)
        col3.plotly_chart(fig_net_pv, use_container_width=True)
        col4.plotly_chart(fig_price_long, use_container_width=True)

        st.markdown('##### Accumulation & Distribution -vs- Emotional Marketing Cycle')
        st.write('____')

    # --- ส่วนแสดงข้อมูลดิบ (ถ้าเลือก) ---
    if st.checkbox('Show Raw Data', value=False):
        st.subheader("Raw Data Inspector")
        st.write(f"Net PV (%) vs Benchmarks")
        st.line_chart(delta_percent_df if 'delta_percent_df' in locals() else "Please check the box above first.")
        st.write("Buffer of each Asset")
        st.line_chart(buffer_df)

def main():
    """
    ฟังก์ชันหลักในการรัน Streamlit Application
    """
    st.set_page_config(page_title="Benchmark_F(X)", page_icon="🛰️", layout="wide")
    st.write("🛰️ Benchmark F(X) Analysis")
    
    config = load_config(CONFIG_FILEPATH)
    if not config:
        st.stop()

    # --- เตรียมข้อมูลจาก config ---
    asset_configs = {asset['ticker']: asset for asset in config['assets']}
    selectable_tickers = [asset['ticker'] for asset in config['assets'] if asset.get('is_selectable', False)]
    benchmark_tickers = [asset['ticker'] for asset in config['assets'] if asset.get('is_benchmark', False)]
    
    # --- UI Widgets ---
    selected_ticker = st.selectbox('เลือก Ticker ที่ต้องการวิเคราะห์', options=selectable_tickers, help="เลือกหุ้นที่ต้องการเปรียบเทียบกับ Benchmark")

    # --- Data Processing ---
    if selected_ticker:
        try:
            tickers_to_run = benchmark_tickers + [selected_ticker]
            
            net_pv_df, re_df, buffer_df, diff_fx = process_assets_data(tuple(tickers_to_run), asset_configs)

            if net_pv_df.empty:
                st.warning(f"ไม่สามารถประมวลผลข้อมูลสำหรับ Ticker: {selected_ticker} ได้ โปรดลอง Ticker อื่น")
            else:
                display_dashboard(selected_ticker, config, asset_configs, net_pv_df, buffer_df, diff_fx)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")

    # --- ส่วนแสดงข้อมูลเพิ่มเติม ---
    if st.checkbox('Cycle_Market Explanation', value=False):
        st.info("""
        **การเกิด Cycle_Market ของระบบ**

        - **Step 1: Divergence (Timing Realize)**
          - **Condition:** `Intrinsic_Value_Cf` **หนี** `Benchmark_Cf` + `Delta/Zone` **สูง** + `Volume` ปกติ/ต่ำลง
          - **Interpretation:** เกิดการสะสม (Accumulation) หรือการกระจาย (Distribution) อย่างมีนัยสำคัญ สร้างโอกาสในการจับจังหวะ

        - **Step 2: Inefficiency (No Realize)**
          - **Condition:** `Intrinsic_Value_Cf` **หนี** `Benchmark_Cf` + `Delta/Zone` **ต่ำ** + `Volume` **สูง**
          - **Interpretation:** ตลาดไม่มีประสิทธิภาพ เกิดความเจ็บปวด (Pain) หรือความคาดหวัง (Hope) สูงสุด เป็นจุดเริ่มต้นของวัฏจักรรอบใหม่

        - **Step 3: Equilibrium (Realize)**
          - **Condition:** `Intrinsic_Value_Cf` **เท่ากับ** `Benchmark_Cf` + `Delta/Zone` **สูง** + `Volume` ปกติ/ต่ำลง
          - **Interpretation:** ไม่มี Premium หรือ Discount สินทรัพย์สะท้อนมูลค่าที่แท้จริง ไม่มีช่องว่างให้เก็งกำไร
        """)

if __name__ == "__main__":
    main()
