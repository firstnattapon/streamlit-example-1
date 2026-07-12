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

    re is realized cash flow at an executed rebalance. net_pv is portfolio
    wealth above the continuous log reference used by Hybrid_Multi_Mutation.
    """
    prices = np.asarray(prices, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int8)
    fixed_asset_value = float(fixed_asset_value)
    starting_cash = float(starting_cash)

    if prices.ndim != 1 or actions.ndim != 1 or len(prices) != len(actions):
        raise ValueError("prices and actions must be one-dimensional arrays of equal length")
    if len(prices) == 0:
        return pd.DataFrame()
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("all prices must be finite and greater than zero")
    if not np.isfinite(fixed_asset_value) or fixed_asset_value < 0:
        raise ValueError("Fixed_Asset_Value must be finite and non-negative")
    if not np.isfinite(starting_cash):
        raise ValueError("Cash_Balan must be finite")

    n = len(prices)
    actions = (actions != 0).astype(np.int8)
    actions[0] = 0

    amount = np.empty(n, dtype=np.float64)
    realized_cash_flow = np.zeros(n, dtype=np.float64)
    cash_balance = np.empty(n, dtype=np.float64)

    amount[0] = fixed_asset_value / prices[0]
    cash_balance[0] = starting_cash

    for idx in range(1, n):
        if actions[idx] == 1:
            realized_cash_flow[idx] = (
                amount[idx - 1] * prices[idx] - fixed_asset_value
            )
            amount[idx] = fixed_asset_value / prices[idx]
        else:
            amount[idx] = amount[idx - 1]
        cash_balance[idx] = cash_balance[idx - 1] + realized_cash_flow[idx]

    portfolio_value = cash_balance + (amount * prices)
    reference_cash_flow = fixed_asset_value * np.log(prices / prices[0])
    reference_portfolio_value = (
        starting_cash + fixed_asset_value + reference_cash_flow
    )
    net_pv = portfolio_value - reference_portfolio_value

    # Floating-point cancellation can create tiny values such as -1e-13 at P0.
    net_pv[np.isclose(net_pv, 0.0, atol=1e-10, rtol=0.0)] = 0.0
    if np.min(net_pv) < -1e-7:
        raise ArithmeticError("log-reference invariant failed: net_pv must be >= 0")

    return pd.DataFrame({
        "re": realized_cash_flow,
        "Cash_Balan": cash_balance,
        "Amount_Asset": amount,
        "pv": portfolio_value,
        "refer_cash_flow": reference_cash_flow,
        "refer_pv": reference_portfolio_value,
        "net_pv": net_pv,
    })


# --------------------------------
# Simulation per ticker
# --------------------------------
def delta6(asset_config):
    ticker = asset_config.get("Ticker", "UNKNOWN")
    try:
        hist = yf.Ticker(ticker).history(period="max")
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
                raise ValueError("pred action stream must have the same length as prices")

        simulation = simulate_constant_value_path(
            prices=prices,
            actions=actions,
            fixed_asset_value=asset_config["Fixed_Asset_Value"],
            starting_cash=asset_config["Cash_Balan"],
        )
        simulation.index = hist.index
        return simulation[["net_pv", "re"]]
    except Exception as exc:
        st.warning(f"{ticker}: simulation skipped ({exc})")
        return None


# --------------------------------
# Aggregate portfolio
# --------------------------------
def un_16(active_configs):
    all_re, all_net = [], []
    for ticker, config in active_configs.items():
        result = delta6(config)
        if result is not None and not result.empty:
            all_re.append(result[["re"]].rename(columns={"re": f"{ticker}_re"}))
            all_net.append(
                result[["net_pv"]].rename(columns={"net_pv": f"{ticker}_net_pv"})
            )

    if not all_re:
        return pd.DataFrame()

    # re is a flow: no row means no cash flow at that timestamp.
    df_re = pd.concat(all_re, axis=1).sort_index().fillna(0.0)

    # net_pv is a cumulative state: carry the latest known value forward.
    # Filling a missing state with zero would create a false drop in portfolio profit.
    df_net = pd.concat(all_net, axis=1).sort_index().ffill().fillna(0.0)

    df_re["portfolio_cash_flow"] = df_re.sum(axis=1)
    df_re["cumulative_cash_flow"] = df_re["portfolio_cash_flow"].cumsum()
    df_net["portfolio_excess_profit"] = df_net.sum(axis=1)
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
            "This page audits the valuation engine. pred=1 means rebalance every bar; "
            "a Hybrid DNA action stream must be supplied explicitly to reproduce a "
            "Hybrid Multi-Mutation path."
        )

        with st.spinner("Calculating... Please wait."):
            data = un_16(active_configs)

        if data.empty:
            st.error(
                "Failed to generate data. This might happen if tickers have no "
                "historical data for the selected period or another error occurred."
            )
        else:
            successful_tickers = [
                ticker for ticker in active_configs
                if f"{ticker}_net_pv" in data.columns
            ]
            re_columns = [f"{ticker}_re" for ticker in successful_tickers]
            sum_fixed_asset_value = float(sum(
                active_configs[ticker].get("Fixed_Asset_Value", 0.0)
                for ticker in successful_tickers
            ))

            cumulative_cash_flow = data["cumulative_cash_flow"].to_numpy(
                dtype=np.float64
            )
            running_cash_floor = np.minimum.accumulate(
                np.minimum(cumulative_cash_flow, 0.0)
            )
            excess_profit = data["portfolio_excess_profit"].to_numpy(
                dtype=np.float64
            )

            df_all = pd.DataFrame({
                "Portfolio_Excess_Profit": excess_profit,
                "Running_Cash_Floor": running_cash_floor,
            }, index=data.index)

            max_buffer_used = abs(float(np.min(running_cash_floor)))
            total_capital_at_risk = sum_fixed_asset_value + max_buffer_used
            if total_capital_at_risk > 0:
                harvest_return = (excess_profit / total_capital_at_risk) * 100.0
            else:
                harvest_return = np.zeros(len(excess_profit), dtype=np.float64)

            df_all_2 = pd.DataFrame({
                "Harvest_Return_on_Capital_Pct": harvest_return
            }, index=data.index)

            final_excess_profit = float(df_all["Portfolio_Excess_Profit"].iloc[-1])
            final_harvest_return = float(
                df_all_2["Harvest_Return_on_Capital_Pct"].iloc[-1]
            )
            num_intervals = max(len(data) - 1, 1)
            avg_daily_excess_profit = final_excess_profit / num_intervals

            daily_portfolio_cash_flow = data[re_columns].sum(axis=1)
            avg_daily_cash_outflow = float(
                (-daily_portfolio_cash_flow.clip(upper=0.0)).mean()
            )

            with st.expander("Scenario MIRR assumptions", expanded=False):
                m1, m2, m3 = st.columns(3)
                mirr_years = int(m1.number_input(
                    "Projection years", min_value=1, max_value=10, value=3, step=1
                ))
                annual_trading_days = int(m2.number_input(
                    "Trading days/year", min_value=1, max_value=366,
                    value=252, step=1
                ))
                exit_recovery_pct = float(m3.slider(
                    "Capital recovered at exit (%)",
                    min_value=0.0, max_value=100.0, value=50.0, step=1.0
                ))

                r1, r2 = st.columns(2)
                finance_rate_pct = float(r1.number_input(
                    "Finance rate (%)", min_value=0.0, value=0.0, step=0.25
                ))
                reinvest_rate_pct = float(r2.number_input(
                    "Reinvestment rate (%)", min_value=0.0, value=0.0, step=0.25
                ))

            projected_annual_excess = (
                avg_daily_excess_profit * annual_trading_days
            )
            exit_recovery = total_capital_at_risk * exit_recovery_pct / 100.0
            finance_rate = finance_rate_pct / 100.0
            reinvest_rate = reinvest_rate_pct / 100.0

            cash_flows = (
                [-total_capital_at_risk]
                + [projected_annual_excess] * max(mirr_years - 1, 0)
                + [projected_annual_excess + exit_recovery]
            )
            mirr_value = (
                npf.mirr(cash_flows, finance_rate, reinvest_rate)
                if total_capital_at_risk > 0 else np.nan
            )

            st.subheader("Key Performance Indicators")
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric(
                "Excess Profit vs Log Reference",
                f"{final_excess_profit:,.2f}",
            )
            k2.metric("Max Cash Buffer Required", f"{max_buffer_used:,.2f}")
            k3.metric(
                "Harvest Return on Capital",
                f"{final_harvest_return:,.2f}%",
            )
            k4.metric(
                "Avg. Daily Excess Profit",
                f"{avg_daily_excess_profit:,.2f}",
            )
            k5.metric(
                "Avg. Daily Cash Outflow",
                f"{avg_daily_cash_outflow:,.2f}",
            )
            k6.metric(
                f"Scenario MIRR ({mirr_years}-Year)",
                f"{mirr_value:.2%}" if np.isfinite(mirr_value) else "N/A",
            )

            help_payload = {
                "reference_model": {
                    "formula": "R_n = FAV × ln(P_n / P_0)",
                    "anchor": "P_0 is the first valid Close in the selected window",
                    "invariant": "net_pv >= 0 (within floating-point tolerance)",
                },
                "successful_tickers": successful_tickers,
                "sum_fixed_asset_value": round(sum_fixed_asset_value, 2),
                "max_cash_buffer_required": round(max_buffer_used, 2),
                "capital_at_risk": {
                    "formula": "Σ(FAV) + max cash buffer required",
                    "value": round(total_capital_at_risk, 2),
                },
                "scenario_mirr": {
                    "is_projection_not_observed_return": True,
                    "years": mirr_years,
                    "annual_trading_days": annual_trading_days,
                    "projected_annual_excess": round(projected_annual_excess, 2),
                    "exit_recovery_percent": exit_recovery_pct,
                    "finance_rate": finance_rate,
                    "reinvest_rate": reinvest_rate,
                    "cash_flows_vector": [round(value, 2) for value in cash_flows],
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
                    df_all,
                    y=["Portfolio_Excess_Profit", "Running_Cash_Floor"],
                    title="Excess Profit vs. Running Cash Floor",
                ),
                use_container_width=True,
            )
            g2.plotly_chart(
                px.line(
                    df_all_2,
                    y=["Harvest_Return_on_Capital_Pct"],
                    title="Harvest Return on Capital (%)",
                ),
                use_container_width=True,
            )

            st.divider()
            st.subheader("Detailed Simulation Data")
            st.caption(
                "Explore portfolio-level results first, then drill down to "
                "individual tickers or compare a focused group."
            )

            df_plot = data.copy()
            per_ticker_detail = {}

            for ticker in successful_tickers:
                re_column = f"{ticker}_re"
                net_column = f"{ticker}_net_pv"
                cumulative_column = f"{ticker}_Cumulative_Cash_Flow"
                floor_column = f"{ticker}_Running_Cash_Floor"

                ticker_cumulative_cash = df_plot[re_column].cumsum()
                ticker_running_floor = np.minimum.accumulate(
                    np.minimum(
                        ticker_cumulative_cash.to_numpy(dtype=np.float64),
                        0.0,
                    )
                )

                df_plot[cumulative_column] = ticker_cumulative_cash
                df_plot[floor_column] = ticker_running_floor
                per_ticker_detail[ticker] = {
                    "re": re_column,
                    "net": net_column,
                    "cumulative": cumulative_column,
                    "floor": floor_column,
                }

            df_plot["Portfolio_Cumulative_Cash_Flow"] = (
                df_plot["cumulative_cash_flow"]
            )
            df_plot["Portfolio_Running_Cash_Floor"] = running_cash_floor
            df_plot["Portfolio_Excess_Profit"] = (
                df_plot["portfolio_excess_profit"]
            )

            (
                portfolio_tab,
                ticker_tab,
                compare_tab,
                table_tab,
            ) = st.tabs([
                "Portfolio Overview",
                "Per-Ticker Explorer",
                "Compare Tickers",
                "Data Table",
            ])

            with portfolio_tab:
                st.caption(
                    "Portfolio cash is pooled across all successful tickers. "
                    "The floor shows the worst cash level reached up to each date."
                )

                p1, p2, p3 = st.columns(3)
                p1.metric(
                    "Final Cumulative Cash Flow",
                    f"{df_plot['Portfolio_Cumulative_Cash_Flow'].iloc[-1]:,.2f}",
                )
                p2.metric(
                    "Maximum Portfolio Cash DD",
                    f"{abs(df_plot['Portfolio_Running_Cash_Floor'].min()):,.2f}",
                )
                p3.metric(
                    "Final Portfolio Excess Profit",
                    f"{df_plot['Portfolio_Excess_Profit'].iloc[-1]:,.2f}",
                )

                portfolio_left, portfolio_right = st.columns(2)

                with portfolio_left:
                    portfolio_cash_view = pd.DataFrame({
                        "Portfolio Cumulative Cash Flow": (
                            df_plot["Portfolio_Cumulative_Cash_Flow"]
                        ),
                        "Portfolio Running Cash Floor": (
                            df_plot["Portfolio_Running_Cash_Floor"]
                        ),
                    })
                    portfolio_cash_chart = px.line(
                        portfolio_cash_view,
                        title="Portfolio Cash Path and Running Floor",
                        labels={
                            "value": "USD",
                            "variable": "Series",
                        },
                    )
                    portfolio_cash_chart.update_layout(
                        hovermode="x unified",
                        legend_title_text="",
                        margin=dict(l=20, r=20, t=60, b=20),
                    )
                    st.plotly_chart(
                        portfolio_cash_chart,
                        use_container_width=True,
                    )

                with portfolio_right:
                    portfolio_profit_view = pd.DataFrame({
                        "Portfolio Excess Profit": (
                            df_plot["Portfolio_Excess_Profit"]
                        ),
                    })
                    portfolio_profit_chart = px.line(
                        portfolio_profit_view,
                        title="Portfolio Log-Reference Excess Profit",
                        labels={
                            "value": "USD",
                            "variable": "Series",
                        },
                    )
                    portfolio_profit_chart.update_layout(
                        hovermode="x unified",
                        legend_title_text="",
                        margin=dict(l=20, r=20, t=60, b=20),
                    )
                    st.plotly_chart(
                        portfolio_profit_chart,
                        use_container_width=True,
                    )

            with ticker_tab:
                selected_detail_ticker = st.selectbox(
                    "Select ticker",
                    options=successful_tickers,
                    key="detailed_ticker_selector",
                )
                ticker_columns = per_ticker_detail[selected_detail_ticker]

                ticker_cumulative = df_plot[ticker_columns["cumulative"]]
                ticker_floor = df_plot[ticker_columns["floor"]]
                ticker_net = df_plot[ticker_columns["net"]]

                t1, t2, t3 = st.columns(3)
                t1.metric(
                    f"{selected_detail_ticker} Cumulative Cash Flow",
                    f"{ticker_cumulative.iloc[-1]:,.2f}",
                )
                t2.metric(
                    f"{selected_detail_ticker} Maximum Cash DD",
                    f"{abs(ticker_floor.min()):,.2f}",
                )
                t3.metric(
                    f"{selected_detail_ticker} Excess Profit",
                    f"{ticker_net.iloc[-1]:,.2f}",
                )

                ticker_left, ticker_right = st.columns(2)

                with ticker_left:
                    ticker_cash_view = pd.DataFrame({
                        "Cumulative Cash Flow": ticker_cumulative,
                        "Running Cash Floor": ticker_floor,
                    })
                    ticker_cash_chart = px.line(
                        ticker_cash_view,
                        title=(
                            f"{selected_detail_ticker} — Cash Path "
                            "and Running Floor"
                        ),
                        labels={
                            "value": "USD",
                            "variable": "Series",
                        },
                    )
                    ticker_cash_chart.update_layout(
                        hovermode="x unified",
                        legend_title_text="",
                        margin=dict(l=20, r=20, t=60, b=20),
                    )
                    st.plotly_chart(
                        ticker_cash_chart,
                        use_container_width=True,
                    )

                with ticker_right:
                    ticker_net_view = pd.DataFrame({
                        "Log-Reference Excess Profit": ticker_net,
                    })
                    ticker_net_chart = px.line(
                        ticker_net_view,
                        title=(
                            f"{selected_detail_ticker} — "
                            "Log-Reference Excess Profit"
                        ),
                        labels={
                            "value": "USD",
                            "variable": "Series",
                        },
                    )
                    ticker_net_chart.update_layout(
                        hovermode="x unified",
                        legend_title_text="",
                        margin=dict(l=20, r=20, t=60, b=20),
                    )
                    st.plotly_chart(
                        ticker_net_chart,
                        use_container_width=True,
                    )

            with compare_tab:
                default_comparison = successful_tickers[
                    :min(4, len(successful_tickers))
                ]
                comparison_tickers = st.multiselect(
                    "Tickers to compare",
                    options=successful_tickers,
                    default=default_comparison,
                    key="detailed_comparison_tickers",
                    help=(
                        "Start with a small group for a readable chart. "
                        "You can add more tickers when needed."
                    ),
                )
                comparison_mode = st.radio(
                    "Compare by",
                    options=[
                        "Running Cash Floor",
                        "Cumulative Cash Flow",
                        "Excess Profit",
                    ],
                    horizontal=True,
                    key="detailed_comparison_mode",
                )

                if not comparison_tickers:
                    st.info("Select at least one ticker to display a comparison.")
                else:
                    comparison_view = pd.DataFrame(index=df_plot.index)
                    for ticker in comparison_tickers:
                        ticker_columns = per_ticker_detail[ticker]
                        if comparison_mode == "Running Cash Floor":
                            source_column = ticker_columns["floor"]
                        elif comparison_mode == "Cumulative Cash Flow":
                            source_column = ticker_columns["cumulative"]
                        else:
                            source_column = ticker_columns["net"]
                        comparison_view[ticker] = df_plot[source_column]

                    comparison_chart = px.line(
                        comparison_view,
                        title=f"Ticker Comparison — {comparison_mode}",
                        labels={
                            "value": "USD",
                            "variable": "Ticker",
                        },
                    )
                    comparison_chart.update_layout(
                        hovermode="x unified",
                        legend_title_text="Ticker",
                        margin=dict(l=20, r=20, t=60, b=20),
                    )
                    st.plotly_chart(
                        comparison_chart,
                        use_container_width=True,
                    )

            with table_tab:
                table_scope = st.radio(
                    "Table scope",
                    options=[
                        "Portfolio only",
                        "Selected ticker",
                        "All detailed columns",
                    ],
                    horizontal=True,
                    key="detailed_table_scope",
                )

                if table_scope == "Portfolio only":
                    table_columns = [
                        "Portfolio_Cumulative_Cash_Flow",
                        "Portfolio_Running_Cash_Floor",
                        "Portfolio_Excess_Profit",
                    ]
                    table_filename = "portfolio_detailed_simulation.csv"
                elif table_scope == "Selected ticker":
                    ticker_columns = per_ticker_detail[
                        selected_detail_ticker
                    ]
                    table_columns = [
                        ticker_columns["re"],
                        ticker_columns["cumulative"],
                        ticker_columns["floor"],
                        ticker_columns["net"],
                    ]
                    table_filename = (
                        f"{selected_detail_ticker}_detailed_simulation.csv"
                    )
                else:
                    table_columns = [
                        column
                        for ticker in successful_tickers
                        for column in [
                            per_ticker_detail[ticker]["re"],
                            per_ticker_detail[ticker]["cumulative"],
                            per_ticker_detail[ticker]["floor"],
                            per_ticker_detail[ticker]["net"],
                        ]
                    ] + [
                        "Portfolio_Cumulative_Cash_Flow",
                        "Portfolio_Running_Cash_Floor",
                        "Portfolio_Excess_Profit",
                    ]
                    table_filename = "all_detailed_simulation.csv"

                table_view = df_plot[table_columns]
                st.dataframe(
                    table_view,
                    use_container_width=True,
                    height=520,
                )
                st.download_button(
                    "Download displayed data (CSV)",
                    data=table_view.to_csv().encode("utf-8"),
                    file_name=table_filename,
                    mime="text/csv",
                    use_container_width=True,
                )

else:
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
