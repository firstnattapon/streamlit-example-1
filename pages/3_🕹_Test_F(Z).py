# import math
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt

# # =============================
# # App Config
# # =============================
# st.set_page_config(page_title="b, c, t Calculator — F = b + c·ln(P/t)", page_icon="🧮", layout="centered")
# st.title("🧮 b, c, t Calculator — F = b + c·ln(P/t)")
# st.caption("Roll / Resize tracker for piecewise log-linear P&L")

# # =============================
# # Helper functions
# # =============================

# def safe_log(x: float) -> float:
#     if x <= 0 or math.isinf(x) or math.isnan(x):
#         raise ValueError("log argument must be positive and finite")
#     return math.log(x)


# def compute_F(b: float, c: float, t: float, P: float) -> float:
#     """F = b + c * ln(P / t)"""
#     return b + c * safe_log(P / t)


# def step_roll_only(b, c, t, t_prime):
#     # b += c * ln(t'/t); t = t'
#     delta_b = c * safe_log(t_prime / t)
#     return b + delta_b, c, t_prime, delta_b


# def step_resize_only(b, c, t, P, c_prime):
#     # b += (c - c') * ln(P/t); c = c'
#     delta_b = (c - c_prime) * safe_log(P / t)
#     return b + delta_b, c_prime, t, delta_b


# def step_roll_and_resize(b, c, t, P, t_prime, c_prime):
#     # b += c·ln(P/t) − c'·ln(P/t'); t = t', c = c'
#     delta_b = c * safe_log(P / t) - c_prime * safe_log(P / t_prime)
#     return b + delta_b, c_prime, t_prime, delta_b


# def validate_positive(name, value):
#     if value is None:
#         raise ValueError(f"{name} is required")
#     if float(value) <= 0:
#         raise ValueError(f"{name} must be > 0")


# def run_pipeline(b0: float, c0: float, t0: float, df_steps: pd.DataFrame):
#     """Run steps sequentially and return (final_b, final_c, final_t, log_df)."""
#     # Normalize column names
#     df = df_steps.copy()
#     required_cols = ["action", "P", "t_prime", "c_prime"]
#     for col in required_cols:
#         if col not in df.columns:
#             df[col] = np.nan

#     b, c, t = float(b0), float(c0), float(t0)

#     logs = []

#     for i, row in df.iterrows():
#         action = str(row.get("action", "")).strip().lower()
#         P = row.get("P", np.nan)
#         t_prime = row.get("t_prime", np.nan)
#         c_prime = row.get("c_prime", np.nan)

#         prev = dict(step=int(i + 1), action=action, b_before=b, c_before=c, t_before=t,
#                     P=P if pd.notna(P) else None,
#                     t_prime=t_prime if pd.notna(t_prime) else None,
#                     c_prime=c_prime if pd.notna(c_prime) else None)

#         try:
#             if action in ("roll", "roll-only", "roll_only"):
#                 validate_positive("t'", t_prime)
#                 b, c, t, delta_b = step_roll_only(b, c, t, float(t_prime))

#             elif action in ("resize", "resize-only", "resize_only"):
#                 validate_positive("P", P)
#                 validate_positive("t", t)
#                 if c_prime is None:
#                     raise ValueError("c' is required")
#                 b, c, t, delta_b = step_resize_only(b, c, t, float(P), float(c_prime))

#             elif action in ("roll_resize", "roll & resize", "roll+resize", "roll-and-resize"):
#                 validate_positive("P", P)
#                 validate_positive("t'", t_prime)
#                 if c_prime is None:
#                     raise ValueError("c' is required")
#                 b, c, t, delta_b = step_roll_and_resize(b, c, t, float(P), float(t_prime), float(c_prime))

#             else:
#                 raise ValueError("Unknown action — use 'roll', 'resize', or 'roll_resize'")

#             logs.append({
#                 **prev,
#                 "delta_b": delta_b,
#                 "b_after": b,
#                 "c_after": c,
#                 "t_after": t,
#             })
#         except Exception as e:
#             logs.append({**prev, "error": str(e), "delta_b": np.nan, "b_after": b, "c_after": c, "t_after": t})
#             raise

#     log_df = pd.DataFrame(logs)
#     return b, c, t, log_df


# # =============================
# # Sidebar — initial state
# # =============================
# st.sidebar.header("Initial State")
# col1, col2, col3 = st.sidebar.columns(3)
# with col1:
#     t0 = st.number_input("t₀ (reference_price)", value=3.05, min_value=0.0000001, step=0.01, format="%.6f")
# with col2:
#     c0 = st.number_input("c₀ (fix_capital)", value=1500.0, min_value=0.0, step=100.0)
# with col3:
#     b0 = st.number_input("b₀ (beta/offset)", value=0.0, step=100.0)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Evaluate F at price P")
# P_eval = st.sidebar.number_input("P (for F=b+c·ln(P/t))", value=5.12, min_value=0.0000001, step=0.01, format="%.6f")

# # =============================
# # Steps Editor
# # =============================

# def default_steps_df() -> pd.DataFrame:
#     # Matches the example sequence from the spec
#     return pd.DataFrame([
#         {"action": "roll_resize", "P": 8.25, "t_prime": 8.25, "c_prime": 1000},
#         {"action": "roll_resize", "P": 5.12, "t_prime": 5.12, "c_prime": 1500},
#         {"action": "resize",       "P": 5.12, "t_prime": np.nan, "c_prime": 1000},
#     ])

# st.markdown("### Steps")
# st.write("Fill in each step. Valid actions: **roll**, **resize**, **roll_resize**.")

# if "steps_df" not in st.session_state:
#     st.session_state.steps_df = default_steps_df()

# edited = st.data_editor(
#     st.session_state.steps_df,
#     num_rows="dynamic",
#     use_container_width=True,
#     column_config={
#         "action": st.column_config.SelectboxColumn(
#             "action",
#             options=["roll", "resize", "roll_resize"],
#             required=True,
#         ),
#         "P": st.column_config.NumberColumn("P", help="Execution price for the step (required for resize or roll_resize)", format="%.6f"),
#         "t_prime": st.column_config.NumberColumn("t' (new t)", help="New reference t when rolling", format="%.6f"),
#         "c_prime": st.column_config.NumberColumn("c' (new c)", help="New capital when resizing"),
#     },
# )

# st.session_state.steps_df = edited

# # =============================
# # Run calculation
# # =============================
# run = st.button("▶️ Compute")
# if run:
#     try:
#         b_final, c_final, t_final, log_df = run_pipeline(b0=b0, c0=c0, t0=t0, df_steps=st.session_state.steps_df)

#         st.success("Done! Final state below.")
#         c1, c2, c3 = st.columns(3)
#         c1.metric("t (reference_price)", f"{t_final:.6f}")
#         c2.metric("c (fix_capital)", f"{c_final:.6f}")
#         c3.metric("b (beta/offset)", f"{b_final:.6f}")

#         # Evaluate F at P_eval
#         F_eval = compute_F(b_final, c_final, t_final, P_eval)
#         st.info(f"F(P={P_eval:.6f}) = b + c·ln(P/t) = **{F_eval:.6f}**")

#         # Show log
#         with st.expander("Step log"):
#             st.dataframe(log_df, use_container_width=True)

#         # Plot F(P) around the final t
#         st.markdown("### Curve around final t")
#         left = t_final * 0.4
#         right = t_final * 2.5
#         xs = np.linspace(left, right, 200)
#         ys = [compute_F(b_final, c_final, t_final, float(x)) for x in xs]
#         fig = plt.figure()
#         plt.plot(xs, ys)
#         plt.axvline(t_final, linestyle='--')
#         plt.xlabel("Price P")
#         plt.ylabel("F(P)")
#         plt.title("F(P) = b + c·ln(P/t) — final state")
#         st.pyplot(fig)

#         # Quick check note (for the provided example)
#         st.caption("Tip: With the default steps, b ≈ 1015.548, t = 5.12, c = 1000.")

#     except Exception as e:
#         st.error(f"Error: {e}")

# # =============================
# # How it works
# # =============================
# with st.expander("Rules (quick reference)"):
#     st.markdown(r"""
# **Form**: \(F = b + c\,\ln(P/t)\)

# - **Roll only** (change reference \(t \to t'\)):
#   \( b \mathrel{+}= c\,\ln(t'/t) \), then set \(t = t'\)
# - **Resize only** (change size \(c \to c'\) at price \(P\)):
#   \( b \mathrel{+}= (c - c')\,\ln(P/t) \), then set \(c = c'\)
# - **Roll & Resize** (simultaneously at \(P\)):
#   \( b \mathrel{+}= c\,\ln(P/t) - c'\,\ln(P/t') \), then set \(t = t',\; c = c'\)

# Notes:
# - All prices used inside the log must be **positive**.
# - If \(P = t\) during a pure resize, the increment is zero.
# """)
