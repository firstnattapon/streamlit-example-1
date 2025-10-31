
# pytron_app.py
# Streamlit mini-app: parse "DD/MM/YYYY HH:MM:SS TICKER" in Asia/Bangkok
# Output: "Stock_Price @DD/MM/YYYY,HH:MM:SS,TICKER" on one line and the Yahoo Finance price on the next line.
# Dependencies: streamlit, yfinance, pandas, pytz, python-dateutil
# Update: add "Strict minute-only" mode -> error if the exact 1-minute bar for that minute is not available.

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

BKK = pytz.timezone("Asia/Bangkok")
UTC = pytz.UTC

HEADER = "pytron • Stock Price @ timestamp (Yahoo Finance)"
INPUT_HELP = (
    "รูปแบบอินพุต: `DD/MM/YYYY HH:MM:SS TICKER`  "
    "เช่น `27/10/2025 20:38:59 APLS`  "
    "(เวลาเข้าใจว่าเป็นเวลาไทย, Asia/Bangkok)"
)

# --- Parsing ---
_INPUT_RE = re.compile(
    r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*[,\s]\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})\s*[,\s]\s*([A-Za-z0-9\.\-^=]+)\s*$"
)

def parse_user_input(s: str) -> Tuple[datetime, str, str]:
    """
    Parse "DD/MM/YYYY HH:MM:SS TICKER" (or with commas between parts).
    Returns (dt_bkk, ticker_upper, header_stamp_str)
    where header_stamp_str is "DD/MM/YYYY,HH:MM:SS,TICKER".
    Raises ValueError if invalid.
    """
    m = _INPUT_RE.match(s)
    if not m:
        raise ValueError("รูปแบบอินพุตไม่ถูกต้อง (เช่น 27/10/2025 20:38:59 APLS)")
    dd, mm, yyyy, HH, MM, SS, tick = m.groups()
    day = int(dd); month = int(mm); year = int(yyyy)
    hour = int(HH); minute = int(MM); second = int(SS)
    ticker = tick.upper()

    dt_naive = datetime(year, month, day, hour, minute, second)
    dt_bkk = BKK.localize(dt_naive)

    header_stamp = f"{day:02d}/{month:02d}/{year:04d},{hour:02d}:{minute:02d}:{second:02d},{ticker}"
    return dt_bkk, ticker, header_stamp

# --- Finance helpers ---
def _fetch_history(ticker: str, start_utc: datetime, end_utc: datetime, interval: str) -> Optional[pd.DataFrame]:
    """Fetch history for [start_utc, end_utc) at given interval; return df with UTC index or None."""
    try:
        df = yf.Ticker(ticker).history(start=start_utc, end=end_utc, interval=interval, auto_adjust=False)
        if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize(UTC)
            else:
                df.index = df.index.tz_convert(UTC)
            return df
    except Exception:
        return None
    return None

def price_at_or_before(df: Optional[pd.DataFrame], target_utc: datetime) -> Optional[float]:
    """Pick the last available Close at or before target_utc from df."""
    if df is None or df.empty:
        return None
    try:
        sub = df.loc[:target_utc]
        if sub.empty:
            # If nothing <= target, fall back to earliest row in df
            first_row = df.iloc[0]
            return float(first_row["Close"])
        last_row = sub.iloc[-1]
        return float(last_row["Close"])
    except Exception:
        return None

def get_price_for_timestamp_relaxed(ticker: str, dt_bkk: datetime) -> Optional[float]:
    """
    Relaxed mode: try progressively coarser intervals if fine-grained data isn't available.
    Strategy: 1m (±45m) → 5m (±2h) → 15m (±6h) → 60m (±2d) → 1d (±7d).
    """
    target_utc = dt_bkk.astimezone(UTC)
    attempts = [
        ("1m", timedelta(minutes=45), timedelta(minutes=1)),
        ("5m", timedelta(hours=2), timedelta(hours=1)),
        ("15m", timedelta(hours=6), timedelta(hours=2)),
        ("60m", timedelta(days=2), timedelta(days=1)),
        ("1d", timedelta(days=7), timedelta(days=1)),
    ]
    for interval, before, after in attempts:
        start = target_utc - before
        end = target_utc + after
        df = _fetch_history(ticker, start, end, interval)
        price = price_at_or_before(df, target_utc) if df is not None else None
        if price is not None:
            return price
    return None

def get_price_for_timestamp_strict_minute(ticker: str, dt_bkk: datetime) -> float:
    """
    Strict minute-only: require the exact 1-minute bar that covers the given minute.
    If absent, raise ValueError.
    """
    target_utc = dt_bkk.astimezone(UTC)
    minute_start = target_utc.replace(second=0, microsecond=0)
    minute_end = minute_start + timedelta(minutes=1)

    df = _fetch_history(ticker, minute_start, minute_end, "1m")
    if df is None or df.empty:
        raise ValueError("ไม่พบแท่ง 1 นาทีครอบช่วงเวลานี้จาก Yahoo Finance")

    # Find row within [minute_start, minute_end)
    mask = (df.index >= minute_start) & (df.index < minute_end)
    sub = df.loc[mask]
    if sub.empty:
        raise ValueError("ไม่พบแท่ง 1 นาทีที่ตรงกับนาทีนี้ (ตลาดอาจปิดหรือไม่มีการซื้อขาย)")

    # Use the Close of that minute bar
    price = float(sub.iloc[-1]["Close"])
    return price

# --- Streamlit UI ---
st.set_page_config(page_title="pytron • stock_price", page_icon="📈", layout="centered")
st.title(HEADER)
st.caption(INPUT_HELP)

default_text = "27/10/2025 20:38:59 APLS"
user_text = st.text_input("ใส่ข้อความอินพุต (ตามรูปแบบด้านบน):", value=default_text)

strict = st.checkbox("Strict minute-only (error ถ้าไม่มีแท่ง 1 นาทีตรงเป๊ะแต่ละนาที)", value=True)
go = st.button("RUN")

if go:
    try:
        dt_bkk, ticker, header_stamp = parse_user_input(user_text)

        st.write(f"Stock_Price @{header_stamp}")
        if strict:
            price = get_price_for_timestamp_strict_minute(ticker, dt_bkk)
            st.code(f"{price}", language="text")
            st.caption("โหมดเข้มงวด: ใช้ Close ของแท่ง 1 นาทีที่ครอบช่วงเวลานั้นเท่านั้น")
        else:
            price = get_price_for_timestamp_relaxed(ticker, dt_bkk)
            if price is None:
                st.error("ไม่พบราคาในช่วงเวลาที่ระบุ (ลองเวลาอื่นหรือเช็คสัญลักษณ์)")
            else:
                st.code(f"{price}", language="text")
                st.caption("โหมดผ่อนคลาย: ถ้าไม่มีนาทีนี้ จะถอยไปกรานูลาริตี้ที่หยาบขึ้น")
    except Exception as e:
        st.error(f"อินพุต/การดึงข้อมูลผิดพลาด: {e}")

with st.expander("รายละเอียดทางเทคนิค / ขอบเขต"):
    st.markdown(
        """
- เวลาที่ป้อนเข้า **ตีความเป็นเวลาไทย (Asia/Bangkok)** แล้วแปลงเป็น UTC ก่อนยิง Yahoo
- **Strict minute-only**: ต้องมีแท่ง `1m` ในช่วงนาทีนั้น (เช่น 20:38:00–20:38:59) มิฉะนั้นขึ้น error
- **Relaxed**: ลองดึง `1m` ก่อน ถ้าไม่มีจะถอยไป `5m → 15m → 60m → 1d`
- เลือกค่า **Close** ของแท่งที่ตรงตามเกณฑ์
- หมายเหตุ: ข้อมูล `1m` ของ Yahoo มักมีให้เพียงช่วงหลัง ๆ (ราวหลายสัปดาห์) เท่านั้น
        """
    )
