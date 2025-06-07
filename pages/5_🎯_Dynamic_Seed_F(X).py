import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

def generate_sample_data(days=100, start_price=100, volatility=0.02):
    """
    สร้างข้อมูลราคาตัวอย่างสำหรับทดสอบ
    """
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # สร้างราคาแบบ random walk
    returns = np.random.normal(0, volatility, days)
    prices = [start_price]
    
    for i in range(1, days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1))  # ป้องกันราคาติดลบ
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, days)
    })
    
    return df

@lru_cache(maxsize=10000)
def calculate_profit_cached(price_tuple, action_tuple):
    """
    คำนวณกำไรด้วย LRU Cache เพื่อเร่งความเร็ว
    """
    prices = list(price_tuple)
    actions = list(action_tuple)
    
    if len(prices) != len(actions):
        return 0
    
    cash = 1.0
    shares = 0
    
    for i in range(len(prices)):
        if actions[i] == 1 and cash > 0:  # ซื้อ
            shares = cash / prices[i]
            cash = 0
        elif actions[i] == 0 and shares > 0:  # ขาย
            cash = shares * prices[i]
            shares = 0
    
    # ขายหุ้นที่เหลือในวันสุดท้าย
    if shares > 0:
        cash = shares * prices[-1]
    
    return cash

def evaluate_seed_performance(seed, price_window):
    """
    ประเมินประสิทธิภาพของ seed ในช่วงราคาที่กำหนด
    """
    np.random.seed(seed)
    actions = np.random.choice([0, 1], size=len(price_window))
    
    # แปลงเป็น tuple เพื่อใช้กับ cache
    price_tuple = tuple(price_window)
    action_tuple = tuple(actions)
    
    profit = calculate_profit_cached(price_tuple, action_tuple)
    return seed, profit, actions

def find_best_seed_for_window(price_window, num_seeds=1000, max_workers=4):
    """
    หา seed ที่ดีที่สุดสำหรับ window ราคาที่กำหนด
    """
    seeds_to_try = random.sample(range(1, 10000), min(num_seeds, 9999))
    
    best_seed = None
    best_profit = -float('inf')
    best_actions = None
    
    # ใช้ parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_performance, seed, price_window) 
                  for seed in seeds_to_try]
        
        for future in futures:
            try:
                seed, profit, actions = future.result()
                if profit > best_profit:
                    best_profit = profit
                    best_seed = seed
                    best_actions = actions
            except Exception as e:
                continue
    
    return best_seed, best_profit, best_actions

def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates=None, 
                                          window_size=30, num_seeds_to_try=1000, 
                                          max_workers=4, shift=0):
    """
    🧬 **DNA Best Seed Sliding Window พร้อม Shift Feature**
    
    Parameters:
    -----------
    price_list : list
        รายการราคาหุ้น
    ticker_data_with_dates : pd.DataFrame, optional
        ข้อมูลหุ้นพร้อมวันที่ (สำหรับแสดงผล)
    window_size : int, default=30
        ขนาดของ sliding window
    num_seeds_to_try : int, default=1000
        จำนวน seeds ที่จะทดสอบในแต่ละ window
    max_workers : int, default=4
        จำนวน workers สำหรับ parallel processing
    shift : int, default=0
        จำนวนรอบที่จะเลื่อน DNA actions ลง
    
    Returns:
    --------
    tuple: (final_actions, window_details, shifted_actions)
        - final_actions: DNA actions ต้นฉบับ
        - window_details: รายละเอียดของแต่ละ window
        - shifted_actions: DNA actions หลังจาก shift
    """
    
    print("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window (Optimized)**")
    
    if len(price_list) < window_size:
        print("❌ **ข้อผิดพลาด:** ขนาดข้อมูลน้อยกว่า window_size")
        return [], [], []
    
    # คำนวณจำนวน windows
    num_windows = len(price_list) // window_size
    if len(price_list) % window_size != 0:
        num_windows += 1
    
    print(f"📊 ข้อมูลทั้งหมด: {len(price_list)} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    print(f"⚡ ใช้ Parallel Processing: {max_workers} workers")
    print(f"🔄 Shift: {shift} รอบ")
    print("---")
    
    final_actions = []
    window_details = []
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(price_list))
        price_window = price_list[start_idx:end_idx]
        
        # แสดงข้อมูล window
        if ticker_data_with_dates is not None and len(ticker_data_with_dates) >= end_idx:
            start_date = ticker_data_with_dates.iloc[start_idx]['Date'].strftime('%Y-%m-%d')
            end_date = ticker_data_with_dates.iloc[end_idx-1]['Date'].strftime('%Y-%m-%d')
            date_info = f"{start_date} ถึง {end_date}"
        else:
            date_info = f"Index {start_idx} ถึง {end_idx-1}"
        
        # หา best seed สำหรับ window นี้
        best_seed, best_profit, best_actions = find_best_seed_for_window(
            price_window, num_seeds_to_try, max_workers
        )
        
        # คำนวณสถิติ
        price_change = ((price_window[-1] - price_window[0]) / price_window[0]) * 100
        num_actions = np.sum(best_actions)
        
        print(f"**🎯 Window {i+1}/{num_windows}** | {date_info}")
        print(f"Best Seed: {best_seed} | Net Profit: {best_profit:.2f} | "
              f"Price Change: {price_change:.2f}% | Actions: {num_actions}/{len(best_actions)}")
        
        # เก็บข้อมูล
        final_actions.extend(best_actions)
        window_details.append({
            'window': i+1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'date_range': date_info,
            'best_seed': best_seed,
            'profit': best_profit,
            'price_change': price_change,
            'actions': best_actions,
            'num_actions': num_actions
        })
    
    print("---")
    print(f"✅ **สำเร็จ!** สร้าง DNA Best Seed ขนาด {len(final_actions)} actions")
    
    # 🔄 **Apply Shift Feature**
    shifted_actions = final_actions.copy()
    if shift > 0:
        # เลื่อน actions ลงตามจำนวน shift
        shifted_actions = [0] * shift + final_actions[:-shift] if shift < len(final_actions) else [0] * len(final_actions)
        print(f"🔄 **Shift Applied:** เลื่อนข้อมูล {shift} รอบ")
    
    # แสดงสถิติ
    original_actions_count = np.sum(final_actions)
    shifted_actions_count = np.sum(shifted_actions)
    
    print(f"📊 **Original DNA Actions:** {original_actions_count} actions จาก {len(final_actions)} วัน")
    print(f"📊 **Shifted DNA Actions:** {shifted_actions_count} actions จาก {len(shifted_actions)} วัน")
    
    return final_actions, window_details, shifted_actions

def analyze_dna_performance(price_list, actions, title="DNA Performance"):
    """
    วิเคราะห์ประสิทธิภาพของ DNA actions
    """
    if len(price_list) != len(actions):
        print("❌ **ข้อผิดพลาด:** ขนาดข้อมูลราคาและ actions ไม่ตรงกัน")
        return None
    
    # คำนวณกำไร
    cash = 1.0
    shares = 0
    portfolio_values = []
    
    for i in range(len(price_list)):
        if actions[i] == 1 and cash > 0:  # ซื้อ
            shares = cash / price_list[i]
            cash = 0
        elif actions[i] == 0 and shares > 0:  # ขาย
            cash = shares * price_list[i]
            shares = 0
        
        # คำนวณมูลค่า portfolio
        current_value = cash + (shares * price_list[i] if shares > 0 else 0)
        portfolio_values.append(current_value)
    
    # ขายหุ้นที่เหลือในวันสุดท้าย
    if shares > 0:
        cash = shares * price_list[-1]
        portfolio_values[-1] = cash
    
    final_return = ((portfolio_values[-1] - 1.0) / 1.0) * 100
    buy_hold_return = ((price_list[-1] - price_list[0]) / price_list[0]) * 100
    
    print(f"
📈 **{title} Analysis**")
    print(f"🎯 Final Portfolio Value: {portfolio_values[-1]:.4f}")
    print(f"📊 Strategy Return: {final_return:.2f}%")
    print(f"📊 Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"🏆 Outperformance: {final_return - buy_hold_return:.2f}%")
    print(f"🔢 Total Actions: {np.sum(actions)}/{len(actions)} ({np.sum(actions)/len(actions)*100:.1f}%)")
    
    return {
        'final_value': portfolio_values[-1],
        'strategy_return': final_return,
        'buy_hold_return': buy_hold_return,
        'outperformance': final_return - buy_hold_return,
        'total_actions': np.sum(actions),
        'action_rate': np.sum(actions)/len(actions)*100,
        'portfolio_values': portfolio_values
    }

# 🎯 **ตัวอย่างการใช้งาน**
def demo_dna_best_seed_with_shift():
    """
    สาธิตการใช้งาน DNA Best Seed พร้อม Shift
    """
    print("🚀 **DEMO: DNA Best Seed with Shift Feature**")
    print("=" * 60)
    
    # สร้างข้อมูลตัวอย่าง
    sample_data = generate_sample_data(days=50, start_price=100, volatility=0.02)
    price_list = sample_data['Close'].tolist()
    
    print(f"📊 **ข้อมูลตัวอย่าง:** {len(price_list)} วัน")
    print(f"💰 ราคาเริ่มต้น: ${price_list[0]:.2f}")
    print(f"💰 ราคาสิ้นสุด: ${price_list[-1]:.2f}")
    print(f"📈 การเปลี่ยนแปลง: {((price_list[-1]-price_list[0])/price_list[0]*100):+.2f}%")
    
    # ทดสอบ DNA Best Seed พร้อม Shift
    final_actions, window_details, shifted_actions = find_best_seed_sliding_window_optimized(
        price_list=price_list,
        ticker_data_with_dates=sample_data,
        window_size=15,
        num_seeds_to_try=200,
        max_workers=2,
        shift=1  # เลื่อน 1 รอบ
    )
    
    # วิเคราะห์ประสิทธิภาพ
    print("
" + "="*60)
    analyze_dna_performance(price_list, final_actions, "Original DNA")
    analyze_dna_performance(price_list, shifted_actions, "Shifted DNA")
    
    return final_actions, shifted_actions, sample_data

# เรียกใช้ demo
if __name__ == "__main__":
    demo_dna_best_seed_with_shift()
