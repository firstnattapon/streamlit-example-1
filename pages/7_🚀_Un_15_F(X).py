def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Exist_F(X)", page_icon="☀", layout="wide")
    st.title("Exist F(X) - Portfolio Analysis")

    # 1. Load config and get user selection
    full_config = load_config()
    if not full_config:
        st.stop()

    all_tickers = list(full_config.keys())
    selected_tickers = st.multiselect(
        "Select Tickers to Analyze",
        options=all_tickers,
        default=all_tickers
    )

    active_configs = {ticker: full_config[ticker] for ticker in selected_tickers if ticker in full_config}

    if not active_configs:
        st.warning("Please select at least one ticker to start the analysis.")
        st.stop()

    # 2. Run calculation
    with st.spinner('Calculating portfolio simulation... This may take a moment.'):
        results_df = run_portfolio_simulation(active_configs)
    
    if results_df.empty:
        st.error("Failed to generate data for the selected tickers. Please check the warnings above or your configuration.")
        st.stop()

    # 3. Process final results for plotting
    try:
        for ticker in selected_tickers:
            col_name = f'{ticker}_re'
            if col_name in results_df.columns:
                results_df[col_name] = results_df[col_name].cumsum()

        cumulative_gains = results_df['maxcash_dd']
        running_max = cumulative_gains.cummax()
        roll_over = cumulative_gains - running_max
        
        min_sum_val = roll_over.min()
        min_sum_abs = abs(min_sum_val) if min_sum_val != 0 else 1.0

        df_all = pd.DataFrame({
            'Sum_Delta': results_df['net_pv_sum'],
            'Max_Sum_Buffer': roll_over
        })
        df_all_2 = pd.DataFrame({
            'True_Alpha': (results_df['net_pv_sum'] / min_sum_abs) * 100
        })

        # 4. Display results
        st.write('____')
        st.metric(
            label="Final Values (Sum_Delta, Max_Sum_Buffer)",
            value=f"{df_all.Sum_Delta.iloc[-1]:,.2f}",
            delta=f"{df_all.Max_Sum_Buffer.iloc[-1]:,.2f} (Buffer)",
            delta_color="inverse"
        )
        st.metric(label="Final True Alpha (%)", value=f"{df_all_2.True_Alpha.iloc[-1]:.2f}%")

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.line(df_all, title="Portfolio: Sum Delta vs Max Sum Buffer"), use_container_width=True)
        col2.plotly_chart(px.line(df_all_2, title="Portfolio: True Alpha (%)"), use_container_width=True)
        
        st.write('____')
        st.plotly_chart(px.line(results_df, title="Detailed Portfolio Simulation"), use_container_width=True)

    except (IndexError, KeyError) as e:
        st.error(f"An error occurred during final processing. The calculated data might be incomplete. Error: {e}")
        st.dataframe(results_df) # Display the raw dataframe for debugging
