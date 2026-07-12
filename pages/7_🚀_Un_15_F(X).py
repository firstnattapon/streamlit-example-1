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
def simulate_constant_value_path(prices, actions, fixed_asset_value, starting_cash):
    """Simulate constant-value rebalancing against the exact log reference.

    Definitions:
    - re: gross cash flow when a rebalance is executed.
    - realized_harvest: locked convexity gain from completed rebalance intervals.
    - open_segment_excess: mark-to-market convexity gain since the last rebalance.
    - net_pv: total wealth excess; realized_harvest + open_segment_excess.

    This decomposition is equivalent to the net formula in
    Hybrid_Multi_Mutation, including rows where action == 0.
    """
    prices = np.asarray(prices, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int8)
    fixed_asset_value = float(fixed_asset_value)
    starting_cash = float(starting_cash)

    if prices.ndim != 1 or actions.ndim != 1 or len(prices) != len(actions):
        raise ValueError(
            "prices and actions must be one-dimensional arrays of equal length"
        )
    if len(prices) == 0:
        return pd.DataFrame()
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("all prices must be finite and greater than zero")
    if not np.isfinite(fixed_asset_value) or fixed_asset_value < 0:
        raise ValueError("Fixed_Asset_Value must be finite and non-negative")
    if not np.isfinite(starting_cash) or starting_cash < 0:
        raise ValueError("Cash_Balan must be finite and non-negative")

    n = len(prices)
    actions = (actions != 0).astype(np.int8)
    actions[0] = 0

    amount = np.empty(n, dtype=np.float64)
    realized_cash_flow = np.zeros(n, dtype=np.float64)
    cash_balance = np.empty(n, dtype=np.float64)
    realized_harvest_increment = np.zeros(n, dtype=np.float64)
    realized_harvest = np.zeros(n, dtype=np.float64)
    open_segment_excess = np.zeros(n, dtype=np.float64)

    amount[0] = fixed_asset_value / prices[0]
    cash_balance[0] = starting_cash
    last_rebalance_price = prices[0]

    for idx in range(1, n):
        if actions[idx] == 1:
            interval_ratio = prices[idx] / last_rebalance_price
            realized_cash_flow[idx] = (
                amount[idx - 1] * prices[idx] - fixed_asset_value
            )
            reference_interval = fixed_asset_value * np.log(interval_ratio)
            realized_harvest_increment[idx] = (
                realized_cash_flow[idx] - reference_interval
            )
            if np.isclose(
                realized_harvest_increment[idx],
                0.0,
                atol=1e-10,
                rtol=0.0,
            ):
                realized_harvest_increment[idx] = 0.0
            if realized_harvest_increment[idx] < -1e-7:
                raise ArithmeticError(
                    "realized-harvest invariant failed at a rebalance"
                )

            amount[idx] = fixed_asset_value / prices[idx]
            last_rebalance_price = prices[idx]
        else:
            amount[idx] = amount[idx - 1]

        cash_balance[idx] = cash_balance[idx - 1] + realized_cash_flow[idx]
        realized_harvest[idx] = (
            realized_harvest[idx - 1] + realized_harvest_increment[idx]
        )

        open_ratio = prices[idx] / last_rebalance_price
        open_segment_excess[idx] = fixed_asset_value * (
            open_ratio - 1.0 - np.log(open_ratio)
        )
        if np.isclose(
            open_segment_excess[idx], 0.0, atol=1e-10, rtol=0.0
        ):
            open_segment_excess[idx] = 0.0
        if open_segment_excess[idx] < -1e-7:
            raise ArithmeticError("open-segment invariant failed")

    portfolio_value = cash_balance + (amount * prices)
    reference_cash_flow = fixed_asset_value * np.log(prices / prices[0])
    reference_portfolio_value = (
        starting_cash + fixed_asset_value + reference_cash_flow
    )
    net_pv = portfolio_value - reference_portfolio_value
    decomposed_net = realized_harvest + open_segment_excess

    net_pv[np.isclose(net_pv, 0.0, atol=1e-10, rtol=0.0)] = 0.0
    if np.min(net_pv) < -1e-7:
        raise ArithmeticError("log-reference invariant failed: net_pv must be >= 0")
    if not np.allclose(net_pv, decomposed_net, atol=1e-7, rtol=1e-10):
        raise ArithmeticError(
            "wealth decomposition failed: net_pv != realized + open segment"
        )

    return pd.DataFrame({
        "re": realized_cash_flow,
        "Cash_Balan": cash_balance,
        "Amount_Asset": amount,
        "pv": portfolio_value,
        "refer_cash_flow": reference_cash_flow,
        "refer_pv": reference_portfolio_value,
        "realized_harvest_increment": realized_harvest_increment,
        "realized_harvest": realized_harvest,
        "open_segment_excess": open_segment_excess,
        "net_pv": net_pv,
    })


# --------------------------------
# Simulation per ticker
# --------------------------------
def delta6(asset_config):
    ticker = asset_config.get("Ticker", "UNKNOWN")
    try:
        hist = yf.Ticker(ticker).history(
            period="max",
            auto_adjust=True,
            actions=False,
        )
        if hist.empty:
            return None

        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC").tz_convert("Asia/Bangkok")
        else:
            hist.index = hist.index.tz_convert("Asia/Bangkok")

        filter_date = pd.Timestamp(asset_config["filter_date"])
        if filter_date.tzinfo is None:
            filter_date = filter_date.tz_localize("Asia/Bangkok")
        else:
            filter_date = filter_date.tz_convert("Asia/Bangkok")

        hist = hist.loc[hist.index >= filter_date, ["Close"]].copy()
        hist = hist[np.isfinite(hist["Close"]) & (hist["Close"] > 0)]
        if hist.empty:
            return None

        prices = hist["Close"].to_numpy(dtype=np.float64, copy=True)
        pred = asset_config.get("pred", 1)
        if np.isscalar(pred):
            actions = np.full(len(prices), int(pred), dtype=np.int8)
        else:
            actions = np.asarray(pred, dtype=np.int8)
            if len(actions) != len(prices):
                raise ValueError(
                    "pred action stream must have the same length as prices"
                )

        simulation = simulate_constant_value_path(
            prices=prices,
            actions=actions,
            fixed_asset_value=asset_config["Fixed_Asset_Value"],
            starting_cash=asset_config["Cash_Balan"],
        )
        simulation.index = hist.index
        return simulation[[
            "net_pv",
            "re",
            "realized_harvest",
            "open_segment_excess",
        ]]
    except Exception as exc:
        st.warning(f"{ticker}: simulation skipped ({exc})")
        return None


# --------------------------------
# Aggregate portfolio
# --------------------------------
def un_16(active_configs):
    all_flows, all_states = [], []

    for ticker, config in active_configs.items():
        # FAV=0 means disabled; do not fetch data or count its idle cash as capital.
        if float(config.get("Fixed_Asset_Value", 0.0)) <= 0:
            continue

        result = delta6(config)
        if result is not None and not result.empty:
            all_flows.append(
                result[["re"]].rename(columns={"re": f"{ticker}_re"})
            )
            all_states.append(result[[
                "net_pv",
                "realized_harvest",
                "open_segment_excess",
            ]].rename(columns={
                "net_pv": f"{ticker}_net_pv",
                "realized_harvest": f"{ticker}_realized_harvest",
                "open_segment_excess": f"{ticker}_open_segment_excess",
            }))

    if not all_flows:
        return pd.DataFrame()

    # Cash flow is an event: no row means zero flow at that timestamp.
    df_flow = pd.concat(all_flows, axis=1).sort_index().fillna(0.0)

    # Wealth/harvest columns are states: carry the latest known value forward.
    df_state = pd.concat(all_states, axis=1).sort_index().ffill().fillna(0.0)

    re_columns = [column for column in df_flow if column.endswith("_re")]
    net_columns = [
        column for column in df_state if column.endswith("_net_pv")
    ]
    realized_columns = [
        column for column in df_state
        if column.endswith("_realized_harvest")
    ]
    open_columns = [
        column for column in df_state
        if column.endswith("_open_segment_excess")
    ]

    df_flow["portfolio_cash_flow"] = df_flow[re_columns].sum(axis=1)
    df_flow["cumulative_cash_flow"] = (
        df_flow["portfolio_cash_flow"].cumsum()
    )

    df_state["portfolio_wealth_excess"] = df_state[net_columns].sum(axis=1)
    df_state["portfolio_realized_harvest"] = (
        df_state[realized_columns].sum(axis=1)
    )
    df_state["portfolio_open_segment_excess"] = (
        df_state[open_columns].sum(axis=1)
    )

    if not np.allclose(
        df_state["portfolio_wealth_excess"],
        (
            df_state["portfolio_realized_harvest"]
            + df_state["portfolio_open_segment_excess"]
        ),
        atol=1e-7,
        rtol=1e-10,
    ):
        raise ArithmeticError("portfolio wealth decomposition failed")

    return pd.concat([df_flow, df_state], axis=1)


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

    # Build active config from file + custom
    active_configs = {t: full_config.get(t, st.session_state.custom_tickers.get(t)).copy() for t in selected_tickers}

    # -------------------------------------------------------------
    # Per-Ticker Fixed_Asset_Value sliders (0 disables capital for that ticker)
    # -------------------------------------------------------------
    if active_configs:
        with st.expander("Per-Ticker Controls"):
            cols = st.columns(min(3, len(active_configs)))  # กระจายสไลเดอร์ให้ดูง่าย
            i = 0
            for tkr, cfg in active_configs.items():
                with cols[i % len(cols)]:
                    current_val = float(cfg.get('Fixed_Asset_Value', DEFAULT_CONFIG.get('Fixed_Asset_Value', 1500.0)))
                    # ใช้ key แยกราย ticker เพื่อให้ state คงอยู่
                    new_val = st.slider(
                        f"Fixed_Asset_Value — {tkr}",
                        min_value=0.0, max_value=5000.0, value=current_val, step=1.0,
                        key=f"fav_{tkr}"
                    )
                    # อัปเดตค่าเข้าคอนฟิกที่ใช้รัน simulation
                    active_configs[tkr]['Fixed_Asset_Value'] = float(new_val)
                i += 1

    if not active_configs:
        st.warning("Please select at least one ticker to start the analysis.")
    else:
        st.caption(
            "Gross mathematical audit using adjusted Close prices; fees, slippage, "
            "taxes and order-fill risk are not included. pred=1 means rebalance "
            "every bar. A Hybrid DNA action stream must be supplied explicitly "
            "to reproduce a sparse Hybrid Multi-Mutation path."
        )

        with st.spinner("Calculating... Please wait."):
            data = un_16(active_configs)

        if data.empty:
            st.error(
                "No active ticker produced data. A ticker with "
                "Fixed_Asset_Value=0 is treated as disabled."
            )
        else:
            successful_tickers = [
                ticker for ticker in active_configs
                if (
                    active_configs[ticker].get("Fixed_Asset_Value", 0.0) > 0
                    and f"{ticker}_net_pv" in data.columns
                )
            ]
            re_columns = [f"{ticker}_re" for ticker in successful_tickers]
            sum_fixed_asset_value = float(sum(
                active_configs[ticker].get("Fixed_Asset_Value", 0.0)
                for ticker in successful_tickers
            ))
            sum_starting_cash = float(sum(
                active_configs[ticker].get("Cash_Balan", 0.0)
                for ticker in successful_tickers
            ))

            cumulative_cash_flow = data["cumulative_cash_flow"].to_numpy(
                dtype=np.float64
            )
            running_cash_floor = np.minimum.accumulate(
                np.minimum(cumulative_cash_flow, 0.0)
            )

            wealth_excess = data["portfolio_wealth_excess"].to_numpy(
                dtype=np.float64
            )
            realized_harvest = data["portfolio_realized_harvest"].to_numpy(
                dtype=np.float64
            )
            open_segment_excess = data[
                "portfolio_open_segment_excess"
            ].to_numpy(dtype=np.float64)

            max_cash_reserve_required = abs(float(np.min(running_cash_floor)))
            cash_reserve_committed = max(
                sum_starting_cash,
                max_cash_reserve_required,
            )
            additional_funding_required = max(
                0.0,
                max_cash_reserve_required - sum_starting_cash,
            )
            total_capital_at_risk = (
                sum_fixed_asset_value + cash_reserve_committed
            )

            if total_capital_at_risk > 0:
                wealth_excess_return = (
                    wealth_excess / total_capital_at_risk
                ) * 100.0
            else:
                wealth_excess_return = np.zeros(
                    len(wealth_excess), dtype=np.float64
                )

            df_decomposition = pd.DataFrame({
                "Portfolio_Wealth_Excess": wealth_excess,
                "Realized_Harvest": realized_harvest,
                "Open_Segment_Excess": open_segment_excess,
            }, index=data.index)
            df_cash_floor = pd.DataFrame({
                "Running_Cash_Floor": running_cash_floor,
            }, index=data.index)
            df_return = pd.DataFrame({
                "Wealth_Excess_Return_on_Capital_Pct": wealth_excess_return,
            }, index=data.index)

            final_wealth_excess = float(wealth_excess[-1])
            final_realized_harvest = float(realized_harvest[-1])
            final_open_segment_excess = float(open_segment_excess[-1])
            final_wealth_return = float(wealth_excess_return[-1])
            num_intervals = max(len(data) - 1, 1)
            avg_daily_realized_harvest = (
                final_realized_harvest / num_intervals
            )

            with st.expander("Scenario MIRR assumptions", expanded=False):
                st.caption(
                    "This is a forward scenario built from average realized "
                    "harvest; it is not an observed historical MIRR."
                )
                m1, m2, m3 = st.columns(3)
                mirr_years = int(m1.number_input(
                    "Projection years",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                ))
                annual_trading_days = int(m2.number_input(
                    "Trading days/year",
                    min_value=1,
                    max_value=366,
                    value=252,
                    step=1,
                ))
                exit_recovery_pct = float(m3.slider(
                    "Capital recovered at exit (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0,
                    step=1.0,
                ))

                r1, r2 = st.columns(2)
                finance_rate_pct = float(r1.number_input(
                    "Finance rate (%)",
                    min_value=0.0,
                    value=0.0,
                    step=0.25,
                ))
                reinvest_rate_pct = float(r2.number_input(
                    "Reinvestment rate (%)",
                    min_value=0.0,
                    value=0.0,
                    step=0.25,
                ))

            projected_annual_realized_harvest = (
                avg_daily_realized_harvest * annual_trading_days
            )
            exit_recovery = (
                total_capital_at_risk * exit_recovery_pct / 100.0
            )
            finance_rate = finance_rate_pct / 100.0
            reinvest_rate = reinvest_rate_pct / 100.0
            cash_flows = (
                [-total_capital_at_risk]
                + [projected_annual_realized_harvest]
                * max(mirr_years - 1, 0)
                + [projected_annual_realized_harvest + exit_recovery]
            )
            mirr_value = (
                npf.mirr(cash_flows, finance_rate, reinvest_rate)
                if total_capital_at_risk > 0 else np.nan
            )

            st.subheader("Key Performance Indicators")
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric(
                "Wealth Excess vs Log Reference",
                f"{final_wealth_excess:,.2f}",
            )
            k2.metric(
                "Realized Harvest",
                f"{final_realized_harvest:,.2f}",
            )
            k3.metric(
                "Open-Segment Excess",
                f"{final_open_segment_excess:,.2f}",
            )
            k4.metric(
                "Cash Reserve Committed",
                f"{cash_reserve_committed:,.2f}",
                help=(
                    "max(configured starting cash, worst cumulative "
                    "rebalance cash-flow deficit)"
                ),
            )
            k5.metric(
                "Wealth Excess Return on Capital",
                f"{final_wealth_return:,.2f}%",
            )
            k6.metric(
                f"Scenario MIRR ({mirr_years}-Year)",
                f"{mirr_value:.2%}" if np.isfinite(mirr_value) else "N/A",
            )

            help_payload = {
                "mathematical_scope": {
                    "price_series": "yfinance adjusted Close",
                    "excluded": [
                        "fees",
                        "slippage",
                        "taxes",
                        "order-fill risk",
                    ],
                },
                "wealth_decomposition": {
                    "identity": (
                        "net_pv = realized_harvest + "
                        "open_segment_excess"
                    ),
                    "realized_interval": (
                        "FAV × (r - 1 - ln(r)) at executed rebalances"
                    ),
                    "open_interval": (
                        "FAV × (r_open - 1 - ln(r_open))"
                    ),
                    "invariants": [
                        "net_pv >= 0",
                        "realized_harvest is non-decreasing",
                        "open_segment_excess >= 0",
                    ],
                },
                "successful_tickers": successful_tickers,
                "capital_at_risk": {
                    "sum_fixed_asset_value": round(
                        sum_fixed_asset_value, 2
                    ),
                    "configured_starting_cash": round(
                        sum_starting_cash, 2
                    ),
                    "gross_cash_reserve_required": round(
                        max_cash_reserve_required, 2
                    ),
                    "additional_funding_required": round(
                        additional_funding_required, 2
                    ),
                    "cash_reserve_committed": round(
                        cash_reserve_committed, 2
                    ),
                    "formula": (
                        "Σ(FAV) + max(Σ(starting cash), "
                        "gross cash reserve required)"
                    ),
                    "value": round(total_capital_at_risk, 2),
                    "cash_netting_assumption": (
                        "cash is pooled across successful tickers"
                    ),
                },
                "scenario_mirr": {
                    "is_projection_not_observed_return": True,
                    "basis": "average daily realized harvest",
                    "years": mirr_years,
                    "annual_trading_days": annual_trading_days,
                    "projected_annual_realized_harvest": round(
                        projected_annual_realized_harvest, 2
                    ),
                    "exit_recovery_percent": exit_recovery_pct,
                    "finance_rate": finance_rate,
                    "reinvest_rate": reinvest_rate,
                    "cash_flows_vector": [
                        round(value, 2) for value in cash_flows
                    ],
                    "result": (
                        round(float(mirr_value), 6)
                        if np.isfinite(mirr_value) else None
                    ),
                },
            }

            st.json(help_payload, expanded=False)
            st.divider()

            st.subheader("Performance Charts")
            g1, g2 = st.columns(2)
            g1.plotly_chart(
                px.line(
                    df_decomposition,
                    y=[
                        "Portfolio_Wealth_Excess",
                        "Realized_Harvest",
                        "Open_Segment_Excess",
                    ],
                    title=(
                        "Wealth Excess = Realized Harvest "
                        "+ Open-Segment Excess"
                    ),
                ),
                use_container_width=True,
            )
            g2.plotly_chart(
                px.line(
                    df_return,
                    y=["Wealth_Excess_Return_on_Capital_Pct"],
                    title="Wealth Excess Return on Capital (%)",
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                px.line(
                    df_cash_floor,
                    y=["Running_Cash_Floor"],
                    title=(
                        "Running Gross Cash-Flow Floor "
                        "(before configured starting cash)"
                    ),
                ),
                use_container_width=True,
            )

            st.divider()
            st.subheader("Detailed Simulation Data")
            df_plot = data.copy()
            for ticker in successful_tickers:
                column = f"{ticker}_re"
                df_plot[column] = df_plot[column].cumsum()

            detail_columns = (
                [f"{ticker}_re" for ticker in successful_tickers]
                + [
                    f"{ticker}_realized_harvest"
                    for ticker in successful_tickers
                ]
                + [
                    f"{ticker}_open_segment_excess"
                    for ticker in successful_tickers
                ]
                + [f"{ticker}_net_pv" for ticker in successful_tickers]
                + [
                    "cumulative_cash_flow",
                    "portfolio_realized_harvest",
                    "portfolio_open_segment_excess",
                    "portfolio_wealth_excess",
                ]
            )
            st.plotly_chart(
                px.line(
                    df_plot[detail_columns],
                    title=(
                        "Per-Ticker Gross Cash Flow, Realized Harvest, "
                        "Open Segment and Wealth Excess"
                    ),
                ),
                use_container_width=True,
            )

else:
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
