import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px
import numpy_financial as npf

# --------------------------------
# Page setup
# --------------------------------
st.set_page_config(page_title="Exist_F(X)", page_icon="☀", layout="wide")

# --------------------------------
# Config loader
# --------------------------------
def load_config(filename="un15_fx_config.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return {}, {}
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return {}, {}

    fallback_default = {
        "Fixed_Asset_Value": 1500.0, "Cash_Balan": 650.0, "step": 0.01,
        "filter_date": "2024-01-01 12:00:00+07:00", "pred": 1
    }
    default_config = data.pop('__DEFAULT_CONFIG__', fallback_default)
    ticker_configs = data
    return ticker_configs, default_config

# --------------------------------
# Core model
# --------------------------------
def calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan):
    if entry >= 10000 or entry <= 0:
        return pd.DataFrame()

    samples = np.arange(0, np.around(entry, 2) * 3 + step, step)
    df = pd.DataFrame({
        'Asset_Price': np.around(samples, 2),
        'Fixed_Asset_Value': Fixed_Asset_Value
    })
    df['Amount_Asset'] = df['Fixed_Asset_Value'] / df['Asset_Price']

    df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
    if not df_top.empty:
        df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
        df_top.fillna(0, inplace=True)
        np_Cash_Balan_top = df_top['Cash_Balan_top'].values
        xx = np.zeros(len(np_Cash_Balan_top))
        y_0 = Cash_Balan
        for idx, v_0 in enumerate(np_Cash_Balan_top):
            y_0 = y_0 + v_0
            xx[idx] = y_0
        df_top['Cash_Balan'] = xx
        df_top = df_top.sort_values(by='Amount_Asset')[:-1]
    else:
        df_top = pd.DataFrame(columns=['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan'])

    df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
    if not df_down.empty:
        df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
        df_down.fillna(0, inplace=True)
        df_down = df_down.sort_values(by='Asset_Price', ascending=False)
        np_Cash_Balan_down = df_down['Cash_Balan_down'].values
        xxx = np.zeros(len(np_Cash_Balan_down))
        y_1 = Cash_Balan
        for idx, v_1 in enumerate(np_Cash_Balan_down):
            y_1 = y_1 + v_1
            xxx[idx] = y_1
        df_down['Cash_Balan'] = xxx
    else:
        df_down = pd.DataFrame(columns=['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan'])

    combined_df = pd.concat([df_top, df_down], axis=0, ignore_index=True)
    return combined_df[['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan']]

# --------------------------------
# Simulation per ticker
# --------------------------------
def delta6(asset_config):
    try:
        hist = yf.Ticker(asset_config['Ticker']).history(period='max')
        if hist.empty:
            return None

        # Robust timezone convert → Asia/Bangkok
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            hist.index = hist.index.tz_convert('Asia/Bangkok')

        hist = hist[hist.index >= asset_config['filter_date']][['Close']]
        if hist.empty:
            return None

        entry = hist['Close'].iloc[0]
        df_model = calculate_cash_balance_model(entry, asset_config['step'],
                                                asset_config['Fixed_Asset_Value'], asset_config['Cash_Balan'])
        if df_model.empty:
            return None

        df = hist.copy()
        df['Close'] = np.around(df['Close'].values, 2)
        df['pred'] = asset_config['pred']
        df['Fixed_Asset_Value'] = asset_config['Fixed_Asset_Value']
        df['Amount_Asset'] = 0.0
        df['re'] = 0.0
        df['Cash_Balan'] = asset_config['Cash_Balan']
        df['Amount_Asset'].iloc[0] = df['Fixed_Asset_Value'].iloc[0] / df['Close'].iloc[0]

        close_vals = df['Close'].values
        pred_vals = df['pred'].values
        amount_asset_vals = df['Amount_Asset'].values
        re_vals = df['re'].values
        cash_balan_sim_vals = df['Cash_Balan'].values

        for idx in range(1, len(amount_asset_vals)):
            if pred_vals[idx] == 1:
                amount_asset_vals[idx] = asset_config['Fixed_Asset_Value'] / close_vals[idx]
                re_vals[idx] = (amount_asset_vals[idx-1] * close_vals[idx]) - asset_config['Fixed_Asset_Value']
            else:
                amount_asset_vals[idx] = amount_asset_vals[idx-1]
                re_vals[idx] = 0
            cash_balan_sim_vals[idx] = cash_balan_sim_vals[idx-1] + re_vals[idx]

        original_index = df.index
        df = df.merge(
            df_model[['Asset_Price', 'Cash_Balan']].rename(columns={'Cash_Balan': 'refer_model'}),
            left_on='Close', right_on='Asset_Price', how='left'
        ).drop('Asset_Price', axis=1)
        df.set_index(original_index, inplace=True)

        df['refer_model'].interpolate(method='linear', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

        df['pv'] = df['Cash_Balan'] + (df['Amount_Asset'] * df['Close'])
        df['refer_pv'] = df['refer_model'] + asset_config['Fixed_Asset_Value']
        df['net_pv'] = df['pv'] - df['refer_pv']

        return df[['net_pv', 're']]
    except Exception:
        return None

# --------------------------------
# Aggregate portfolio
# --------------------------------
def un_16(active_configs):
    all_re, all_net = [], []
    for tkr, cfg in active_configs.items():
        out = delta6(cfg)
        if out is not None and not out.empty:
            all_re.append(out[['re']].rename(columns={'re': f'{tkr}_re'}))
            all_net.append(out[['net_pv']].rename(columns={'net_pv': f'{tkr}_net_pv'}))
    if not all_re:
        return pd.DataFrame()
    df_re = pd.concat(all_re, axis=1)
    df_net = pd.concat(all_net, axis=1)
    df_re.fillna(0, inplace=True)
    df_net.fillna(0, inplace=True)
    df_re['maxcash_dd'] = df_re.sum(axis=1).cumsum()
    df_net['cf'] = df_net.sum(axis=1)
    return pd.concat([df_re, df_net], axis=1)

# --------------------------------
# UI
# --------------------------------
full_config, DEFAULT_CONFIG = load_config()

if full_config or DEFAULT_CONFIG:
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = {}

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Add New Ticker")
        new_ticker = st.text_input("Ticker (e.g., AAPL):", key="new_ticker_input").upper()
        if st.button("Add Ticker", key="add_ticker_button", use_container_width=True):
            if new_ticker and new_ticker not in full_config and new_ticker not in st.session_state.custom_tickers:
                st.session_state.custom_tickers[new_ticker] = {"Ticker": new_ticker, **DEFAULT_CONFIG}
                st.success(f"Added {new_ticker}!")
                st.rerun()
            elif new_ticker in full_config:
                st.warning(f"{new_ticker} already exists in config file.")
            elif new_ticker in st.session_state.custom_tickers:
                st.warning(f"{new_ticker} has already been added.")
            else:
                st.warning("Please enter a ticker symbol.")

    with c2:
        st.subheader("Select Tickers to Analyze")
        all_tickers = list(full_config.keys()) + list(st.session_state.custom_tickers.keys())
        default_selection = [t for t in list(full_config.keys()) if t in all_tickers]
        selected_tickers = st.multiselect(
            "Select from available tickers:",
            options=all_tickers,
            default=default_selection
        )
    st.divider()

    active_configs = {t: full_config.get(t, st.session_state.custom_tickers.get(t)) for t in selected_tickers}

    if not active_configs:
        st.warning("Please select at least one ticker to start the analysis.")
    else:
        with st.spinner('Calculating... Please wait.'):
            data = un_16(active_configs)

        if data.empty:
            st.error("Failed to generate data. This might happen if tickers have no historical data for the selected period or another error occurred.")
        else:
            df_new = data.copy()

            # Drawdown path (buffer used)
            roll_over = []
            mvals = df_new.maxcash_dd.values
            for i in range(len(mvals)):
                roll = mvals[:i]
                roll_min = np.min(roll) if len(roll) > 0 else 0
                roll_over.append(roll_min)

            cf_values = df_new.cf.values
            df_all = pd.DataFrame({'Sum_Delta': cf_values, 'Max_Sum_Buffer': roll_over}, index=df_new.index)

            # True Alpha (คงสูตรฐานเก่า: n×1500 + |buffer|)
            n_tickers = len(selected_tickers)
            initial_capital = n_tickers * 1500.0
            max_buffer_used = abs(np.min(roll_over))
            total_capital_at_risk = initial_capital + max_buffer_used if (initial_capital + max_buffer_used) != 0 else 1.0
            df_all_2 = pd.DataFrame({'True_Alpha': (df_new.cf.values / total_capital_at_risk) * 100}, index=df_new.index)

            # KPIs
            st.subheader("Key Performance Indicators")
            final_sum_delta = float(df_all.Sum_Delta.iloc[-1])
            final_max_buffer = float(df_all.Max_Sum_Buffer.iloc[-1])
            final_true_alpha = float(df_all_2.True_Alpha.iloc[-1])
            num_days = len(df_new)
            avg_cf = final_sum_delta / num_days if num_days > 0 else 0.0
            avg_burn_cash = abs(final_max_buffer) / num_days if num_days > 0 else 0.0

            # MIRR inputs
            sum_fixed_asset_value = float(sum(cfg.get('Fixed_Asset_Value', 0.0) for cfg in active_configs.values()))
            I = sum_fixed_asset_value + abs(final_max_buffer)                 # Initial Investment
            A = avg_cf * 252                                                 # Annual CF
            E = 0.5 * sum_fixed_asset_value                                  # Exit Multiple (JSON-only spec)
            finance_rate = 0.0
            reinvest_rate = 0.0

            cash_flows = [-I, A, A, A + E]
            mirr_value = npf.mirr(cash_flows, finance_rate, reinvest_rate) if I > 0 else 0.0

            # KPI widgets
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Total Net Profit (cf)", f"{final_sum_delta:,.2f}")
            k2.metric("Max Cash Buffer Used", f"{final_max_buffer:,.2f}")
            k3.metric("True Alpha (%)", f"{final_true_alpha:,.2f}%")
            k4.metric("Avg. Daily Profit", f"{avg_cf:,.2f}")
            k5.metric("Avg. Daily Buffer Used", f"{avg_burn_cash:,.2f}")
            k6.metric("MIRR (3-Year)", f"{mirr_value:.2%}")

            # ---------- JSON-only help block ----------
            help_payload = {
                "definition": "MIRR (3-Year)",
                "tickers_selected": int(n_tickers),
                "sum_fixed_asset_value": round(sum_fixed_asset_value, 2),
                "max_cash_buffer_used_abs": round(abs(final_max_buffer), 2),
                "I_initial_investment": round(I, 2),
                "A_annual_cash_flow": round(A, 2),
                "E_exit_multiple": {
                    "formula": "0.5 × Σ(Fixed_Asset_Value)",
                    "value": round(E, 2)
                },
                "rates": {
                    "finance_rate": finance_rate,
                    "reinvest_rate": reinvest_rate
                },
                "cash_flows_vector": [
                    round(-I, 2),
                    round(A, 2),
                    round(A, 2),
                    round(A + E, 2)
                ],
                "mirr_result": round(float(mirr_value), 6) if isinstance(mirr_value, (int, float, np.floating)) else None
            }

            st.subheader("MIRR (3-Year) — JSON")
            st.json(help_payload, expanded=False)
            st.download_button(
                label="⬇️ Download MIRR help JSON",
                data=json.dumps(help_payload, ensure_ascii=False, indent=2),
                file_name="mirr_help.json",
                mime="application/json",
                use_container_width=True
            )
            st.divider()

            # Charts
            st.subheader("Performance Charts")
            g1, g2 = st.columns(2)
            g1.plotly_chart(
                px.line(df_all.reset_index(drop=True), title="Cumulative Profit (Sum_Delta) vs. Max Buffer Used"),
                use_container_width=True
            )
            g2.plotly_chart(
                px.line(df_all_2, title="True Alpha (%)"),
                use_container_width=True
            )

            st.divider()
            st.subheader("Detailed Simulation Data")
            df_plot = df_new.copy()
            for t in selected_tickers:
                col = f"{t}_re"
                if col in df_plot.columns:
                    df_plot[col] = df_plot[col].cumsum()
            st.plotly_chart(px.line(df_plot, title="Portfolio Simulation Details"), use_container_width=True)

else:
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
