import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import ta  # For technical indicators
import requests
from bs4 import BeautifulSoup

# Set page configuration
st.set_page_config(
    page_title="EMA-ADX Candlestick Trading Strategy",
    page_icon="📈",
    layout="wide"
)

# Define list of top-performing stocks based on backtesting
def get_stock_list():
    """Get the list of best-performing stocks based on backtesting results"""
    # These are the top 10 stocks by risk-adjusted performance
    return [
        "JSWSTEEL-EQ",    # Best overall performance
        "BAJAJ-AUTO-EQ",  # Excellent win rate with low drawdown
        "BAJAJFINSV-EQ",  # High win rate >50%
        "TATAMOTORS-EQ",  # Highest absolute return
        "ADANIENT-EQ",    # Strong returns
        "BAJFINANCE-EQ",  # Good win rate and manageable drawdown
        "DIVISLAB-EQ",    # Solid performance
        "TECHM-EQ",       # Good risk/reward
        "LT-EQ",          # Low drawdown
        "GRASIM-EQ"       # Balanced performance
    ]

def format_symbol(symbol):
    """Convert the stock symbol to Yahoo Finance format"""
    # Remove the -EQ suffix and add .NS for Indian stocks
    if "-EQ" in symbol:
        return symbol.replace("-EQ", "") + ".NS"
    elif "-INDEX" in symbol:
        # Handle index symbols
        if "NIFTY50" in symbol:
            return "^NSEI"
        elif "FINNIFTY" in symbol:
            return "^NSEBANK"  # Using Bank Nifty as proxy for Fin Nifty
        else:
            return symbol.replace("-INDEX", "")
    else:
        return symbol

def fetch_stock_data(symbol, period="60d", interval="1d"):
    """Fetch stock data using yfinance with error handling and retries"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            yahoo_symbol = format_symbol(symbol)
            data = yf.download(yahoo_symbol, period=period, interval=interval, progress=False)
            
            if len(data) > 0:
                return data
            else:
                st.warning(f"No data returned for {symbol} (attempt {attempt+1}/{max_retries})")
                
        except Exception as e:
            st.warning(f"Error fetching data for {symbol} (attempt {attempt+1}/{max_retries}): {e}")
            
        # Wait before retrying
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    # If all attempts fail, raise an error
    raise Exception(f"Failed to fetch data for {symbol} after {max_retries} attempts")

def calculate_indicators(df):
    """Calculate necessary indicators for the strategy:
    - 10 EMA
    - ADX
    - Candle colors
    """
    # Calculate 10 EMA
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    
    # Calculate ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['PDI'] = adx.adx_pos()  # +DI line
    df['NDI'] = adx.adx_neg()  # -DI line
    
    # Identify candle colors (green = close > open, red = close < open)
    df['Candle_Color'] = np.where(df['Close'] >= df['Open'], 'green', 'red')
    
    # Calculate if candle touches or crosses EMA
    df['Touches_EMA'] = np.where(
        ((df['Candle_Color'] == 'green') & (df['Low'] <= df['EMA_10'])) | 
        ((df['Candle_Color'] == 'red') & (df['High'] >= df['EMA_10'])),
        True, False
    )
    
    return df

def check_entry_exit_signals(df):
    """Check for entry and exit signals based on the strategy"""
    # Initialize signal columns
    df['long_entry'] = 0
    df['long_exit'] = 0
    df['short_entry'] = 0
    df['short_exit'] = 0
    
    # Loop through the dataframe (starting from index 1 to check previous candle)
    for i in range(1, len(df)):
        prev = df.iloc[i-1]  # Previous candle
        curr = df.iloc[i]    # Current candle
        
        # LONG ENTRY
        # When a Red Candle forms above the 10 EMA line and red candle does not touch the 10 EMA line
        # & a green candle is formed just after the red candle and it closes above the red candle high
        # given that ADX must be above 18
        if (prev['Candle_Color'] == 'red' and  # Previous candle is red
            prev['Low'] > prev['EMA_10'] and    # Red candle is above EMA
            not prev['Touches_EMA'] and         # Red candle doesn't touch EMA
            curr['Candle_Color'] == 'green' and # Current candle is green
            curr['Close'] > prev['High'] and    # Green candle closes above red candle high
            curr['ADX'] > 18):                  # ADX is above 18
            df.at[df.index[i], 'long_entry'] = 1
        
        # LONG EXIT
        # When a red candle closes below 10 EMA & does not touch the 10 EMA
        if (curr['Candle_Color'] == 'red' and   # Current candle is red
            curr['High'] < curr['EMA_10'] and   # Closes below EMA
            not curr['Touches_EMA']):           # Doesn't touch EMA
            df.at[df.index[i], 'long_exit'] = 1
        
        # SHORT ENTRY
        # When a green Candle forms below the 10 EMA line and green candle does not touch the 10 EMA line
        # & a red candle is formed just after the green candle and red candle closes below the green candle low
        # given that ADX must be above 18
        if (prev['Candle_Color'] == 'green' and  # Previous candle is green
            prev['High'] < prev['EMA_10'] and    # Green candle is below EMA
            not prev['Touches_EMA'] and          # Green candle doesn't touch EMA
            curr['Candle_Color'] == 'red' and    # Current candle is red
            curr['Close'] < prev['Low'] and      # Red candle closes below green candle low
            curr['ADX'] > 18):                   # ADX is above 18
            df.at[df.index[i], 'short_entry'] = 1
        
        # SHORT EXIT
        # When a green candle closes above the 10 EMA & does not touch the 10 EMA
        if (curr['Candle_Color'] == 'green' and  # Current candle is green
            curr['Low'] > curr['EMA_10'] and     # Closes above EMA
            not curr['Touches_EMA']):            # Doesn't touch EMA
            df.at[df.index[i], 'short_exit'] = 1
    
    return df

def get_current_signals(df):
    """Get the current signals for the most recent data"""
    # Get the last few rows to examine recent behavior
    last_rows = df.iloc[-3:].copy()
    
    # Check signals from the latest row
    latest = last_rows.iloc[-1]
    prev = last_rows.iloc[-2] if len(last_rows) > 1 else None
    
    signals = {
        'long_entry': bool(latest['long_entry']),
        'long_exit': bool(latest['long_exit']),
        'short_entry': bool(latest['short_entry']),
        'short_exit': bool(latest['short_exit']),
        'latest_candle_color': latest['Candle_Color'],
        'adx': float(latest['ADX']),
        'pdi': float(latest['PDI']),
        'ndi': float(latest['NDI']),
        'above_ema': latest['Close'] > latest['EMA_10'],
        'touches_ema': bool(latest['Touches_EMA']),
        'ema_10': float(latest['EMA_10']),
        'latest_close': float(latest['Close']),
        'latest_high': float(latest['High']),
        'latest_low': float(latest['Low']),
    }
    
    # Check for potential signals (conditions that might lead to a signal on the next candle)
    if prev is not None:
        # Potential Long Entry
        if (latest['Candle_Color'] == 'red' and 
            latest['Low'] > latest['EMA_10'] and 
            not latest['Touches_EMA'] and
            latest['ADX'] > 18):
            signals['potential_long_entry'] = True
        else:
            signals['potential_long_entry'] = False
            
        # Potential Short Entry
        if (latest['Candle_Color'] == 'green' and 
            latest['High'] < latest['EMA_10'] and 
            not latest['Touches_EMA'] and
            latest['ADX'] > 18):
            signals['potential_short_entry'] = True
        else:
            signals['potential_short_entry'] = False
    else:
        signals['potential_long_entry'] = False
        signals['potential_short_entry'] = False
    
    return signals

def calculate_performance_metrics(df):
    """Calculate performance metrics based on signals"""
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Initialize columns
    df_copy['position'] = 0  # 1 for long, -1 for short, 0 for no position
    df_copy['trade_returns'] = 0.0
    
    position = 0
    entry_price = 0
    trade_count = 0
    winning_trades = 0
    
    # Loop through the dataframe
    for i in range(1, len(df_copy)):
        prev_row = df_copy.iloc[i-1]
        curr_row = df_copy.iloc[i]
        
        # Check for entry/exit signals
        if curr_row['long_entry'] == 1 and position == 0:
            # Enter long position
            position = 1
            entry_price = curr_row['Close']
            trade_count += 1
        elif curr_row['short_entry'] == 1 and position == 0:
            # Enter short position
            position = -1
            entry_price = curr_row['Close']
            trade_count += 1
        elif curr_row['long_exit'] == 1 and position == 1:
            # Exit long position
            trade_return = (curr_row['Close'] - entry_price) / entry_price
            df_copy.at[df_copy.index[i], 'trade_returns'] = trade_return
            position = 0
            if trade_return > 0:
                winning_trades += 1
        elif curr_row['short_exit'] == 1 and position == -1:
            # Exit short position
            trade_return = (entry_price - curr_row['Close']) / entry_price
            df_copy.at[df_copy.index[i], 'trade_returns'] = trade_return
            position = 0
            if trade_return > 0:
                winning_trades += 1
        
        df_copy.at[df_copy.index[i], 'position'] = position
    
    # Calculate performance metrics
    total_return = df_copy['trade_returns'].sum() * 100  # Convert to percentage
    win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0
    
    metrics = {
        'trade_count': trade_count,
        'win_rate': win_rate,
        'total_return': total_return,
        'current_position': position
    }
    
    return metrics

def plot_strategy_chart(df, symbol):
    """Create a Plotly chart with candlesticks, EMA, and ADX"""
    # Create subplots: 1 for candlestick with EMA, 1 for ADX
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlesticks",
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Add 10 EMA line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA_10'],
            line=dict(color='blue', width=1.5),
            name='10 EMA'
        ),
        row=1, col=1
    )
    
    # Add entry and exit signals
    # Long entry signals
    long_entries = df[df['long_entry'] == 1]
    if not long_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=long_entries.index,
                y=long_entries['Low'] * 0.99,  # Place slightly below the low
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Long Entry'
            ),
            row=1, col=1
        )
    
    # Long exit signals
    long_exits = df[df['long_exit'] == 1]
    if not long_exits.empty:
        fig.add_trace(
            go.Scatter(
                x=long_exits.index,
                y=long_exits['High'] * 1.01,  # Place slightly above the high
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Long Exit'
            ),
            row=1, col=1
        )
    
    # Short entry signals
    short_entries = df[df['short_entry'] == 1]
    if not short_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=short_entries.index,
                y=short_entries['High'] * 1.01,  # Place slightly above the high
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Short Entry'
            ),
            row=1, col=1
        )
    
    # Short exit signals
    short_exits = df[df['short_exit'] == 1]
    if not short_exits.empty:
        fig.add_trace(
            go.Scatter(
                x=short_exits.index,
                y=short_exits['Low'] * 0.99,  # Place slightly below the low
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Short Exit'
            ),
            row=1, col=1
        )
    
    # Add ADX line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ADX'],
            line=dict(color='purple', width=1.5),
            name='ADX'
        ),
        row=2, col=1
    )
    
    # Add +DI and -DI lines
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['PDI'],
            line=dict(color='green', width=1, dash='dot'),
            name='+DI'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['NDI'],
            line=dict(color='red', width=1, dash='dot'),
            name='-DI'
        ),
        row=2, col=1
    )
    
    # Add horizontal line at ADX = 18 (threshold for our strategy)
    fig.add_shape(
        type="line", line_color="gray", line_width=1, opacity=0.5, line_dash="dash",
        x0=df.index[0], x1=df.index[-1], y0=18, y1=18,
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Candlestick Chart with 10 EMA and ADX",
        xaxis_title="Date",
        yaxis_title="Price",
        height=800,
        yaxis2_title="ADX / DI",
        showlegend=True,
        xaxis_rangeslider_visible=False,
    )
    
    return fig

# Function to scan all stocks for entry signals
def scan_stocks_for_signals(stocks_list):
    """Scan all stocks to find those with active entry signals"""
    signal_stocks = []
    
    with st.spinner("Scanning stocks for trading signals..."):
        for stock in stocks_list:
            try:
                # Get stock data
                stock_data = fetch_stock_data(stock, period="60d", interval="1d")
                
                # Calculate indicators
                indicator_data = calculate_indicators(stock_data)
                
                # Check for entry and exit signals
                signal_data = check_entry_exit_signals(indicator_data)
                
                # Get current signals
                signals = get_current_signals(signal_data)
                
                # Calculate performance metrics
                metrics = calculate_performance_metrics(signal_data)
                
                # Check for active or potential signals
                signal_type = None
                if signals['long_entry']:
                    signal_type = "🟢 LONG ENTRY"
                elif signals['short_entry']:
                    signal_type = "🔴 SHORT ENTRY"
                elif signals['potential_long_entry']:
                    signal_type = "⏳ Potential LONG ENTRY"
                elif signals['potential_short_entry']:
                    signal_type = "⏳ Potential SHORT ENTRY"
                
                # Add to the list regardless of signal (for comprehensive scanning)
                signal_stocks.append({
                    'stock': stock,
                    'signal': signal_type if signal_type else "🔄 NEUTRAL",
                    'last_price': f"₹{signals['latest_close']:.2f}",
                    'adx': f"{signals['adx']:.2f}",
                    'candle_color': signals['latest_candle_color'],
                    'ema_10': f"₹{signals['ema_10']:.2f}",
                    'win_rate': f"{metrics['win_rate']:.2f}%",
                    'recent_trades': metrics['trade_count'],
                    'position': "LONG" if metrics['current_position'] > 0 else 
                               "SHORT" if metrics['current_position'] < 0 else "NONE"
                })
            except Exception as e:
                st.error(f"Error processing {stock}: {e}")
                continue
    
    return signal_stocks

def get_stock_fundamentals(symbol):
    """Get basic fundamental data for a stock"""
    try:
        # Format the symbol for Yahoo Finance
        yahoo_symbol = format_symbol(symbol)
        
        # Get stock info
        stock = yf.Ticker(yahoo_symbol)
        info = stock.info
        
        # Extract key metrics
        fundamentals = {
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'avg_volume': info.get('averageVolume', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
        }
        
        return fundamentals
    except Exception as e:
        st.warning(f"Could not fetch fundamentals for {symbol}: {e}")
        return {
            'company_name': symbol,
            'sector': 'N/A',
            'industry': 'N/A',
            'market_cap': 0,
            'pe_ratio': 0,
            '52w_high': 0,
            '52w_low': 0,
            'avg_volume': 0,
            'dividend_yield': 0,
        }

# Define the app layout
def main():
    # App title and description
    st.title("EMA-ADX Candlestick Trading Strategy Dashboard")
    
    st.markdown("""
    ### Optimized Trading Strategy Dashboard
    
    This app analyzes top-performing Indian stocks using a candlestick pattern strategy with 10 EMA and ADX indicators.
    The stock list has been curated based on extensive backtesting to select only the best-performing stocks.
    """)
    
    # Sidebar for strategy information
    with st.sidebar:
        st.header("Strategy Information")
        st.subheader("Timeframe: Daily Candles")
        st.subheader("Indicators: 10 EMA, ADX (14)")
        
        # Strategy details in expandable sections
        with st.expander("Long Entry Conditions"):
            st.write("""
            1. A red candle forms above the 10 EMA line
            2. The red candle does not touch the 10 EMA
            3. A green candle forms just after the red candle
            4. The green candle closes above the red candle's high
            5. ADX must be above 18
            """)
            
        with st.expander("Long Exit Conditions"):
            st.write("""
            Exit when a red candle closes below the 10 EMA and does not touch the 10 EMA
            """)
        
        with st.expander("Short Entry Conditions"):
            st.write("""
            1. A green candle forms below the 10 EMA line
            2. The green candle does not touch the 10 EMA
            3. A red candle forms just after the green candle
            4. The red candle closes below the green candle's low
            5. ADX must be above 18
            """)
        
        with st.expander("Short Exit Conditions"):
            st.write("""
            Exit when a green candle closes above the 10 EMA and does not touch the 10 EMA
            """)
            
        with st.expander("Strategy Performance Summary"):
            st.write("""
            This strategy has been backtested across multiple Indian stocks. The top performers are:
            
            1. **JSWSTEEL**: Highest risk-adjusted returns with excellent win rate
            2. **BAJAJ-AUTO**: Consistent performance with minimal drawdowns
            3. **BAJAJFINSV**: Win rate over 50% with good returns
            4. **TATAMOTORS**: Highest absolute returns
            5. **ADANIENT**: Strong performance in various market conditions
            """)
            
        # Add time frame selector
        st.subheader("Data Settings")
        period = st.selectbox(
            "Select Analysis Period",
            options=["30d", "60d", "90d", "180d", "1y"],
            index=1  # Default to 60 days
        )
        
        interval = st.selectbox(
            "Select Candle Timeframe",
            options=["1d", "1wk"],
            index=0  # Default to daily
        )
    
    # Load stock list
    all_stocks = get_stock_list()
    
    # Tab layout for different sections
    tabs = st.tabs(["Signal Scanner", "Stock Analysis", "Performance Dashboard"])
    
    with tabs[0]:  # Signal Scanner tab
        st.header("Stock Signal Scanner")
        
        # Button to refresh signals
        if st.button("Scan for New Signals"):
            st.session_state.signal_stocks = scan_stocks_for_signals(all_stocks)
        
        # Initialize signal_stocks in session state if not already done
        if 'signal_stocks' not in st.session_state:
            st.session_state.signal_stocks = scan_stocks_for_signals(all_stocks)
        
        # Display stocks with signals
        if st.session_state.signal_stocks:
            # Create a filtered dataframe for stocks with actual signals
            active_signals = [s for s in st.session_state.signal_stocks if s['signal'] != "🔄 NEUTRAL"]
            
            if active_signals:
                st.subheader("Active Trading Signals")
                signal_df = pd.DataFrame(active_signals)
                st.dataframe(signal_df, use_container_width=True)
            else:
                st.info("No active signals currently. Here are all monitored stocks:")
            
            # Show all stocks    
            st.subheader("All Monitored Stocks")
            all_stocks_df = pd.DataFrame(st.session_state.signal_stocks)
            
            # Add styling to highlight important signals
            def highlight_signal(val):
                if '🟢 LONG ENTRY' in str(val):
                    return 'background-color: lightgreen'
                elif '🔴 SHORT ENTRY' in str(val):
                    return 'background-color: lightcoral'
                return ''
            
            styled_df = all_stocks_df.style.applymap(highlight_signal, subset=['signal'])
            st.dataframe(styled_df, use_container_width=True)
            
        else:
            st.error("Failed to scan stocks. Please try again or check your internet connection.")
    
    with tabs[1]:  # Stock Analysis tab
        st.header("Stock Analysis")
        
        # Allow selection from all stocks
        selected_stock = st.selectbox(
            "Select a stock to analyze", 
            all_stocks, 
            key="stock_selector"
        )
        
        # Fetch and process stock data
        try:
            with st.spinner(f"Fetching {selected_stock} data..."):
                # Fetch stock data
                stock_data = fetch_stock_data(selected_stock, period=period, interval=interval)
                
                # Calculate indicators
                indicator_data = calculate_indicators(stock_data)
                
                # Check for entry and exit signals
                signal_data = check_entry_exit_signals(indicator_data)
                
                # Get current signals and metrics
                signals = get_current_signals(signal_data)
                metrics = calculate_performance_metrics(signal_data)
                fundamentals = get_stock_fundamentals(selected_stock)
            
            # Display stock information
            st.subheader(f"{fundamentals['company_name']} ({selected_stock})")
            
            # Display fundamentals in columns
            cols = st.columns(3)
            with cols[0]:
                st.metric("Sector", fundamentals['sector'])
                st.metric("Industry", fundamentals['industry'])
            
            with cols[1]:
                st.metric("Market Cap", f"₹{fundamentals['market_cap']:,}" if fundamentals['market_cap'] > 0 else "N/A")
                st.metric("P/E Ratio", f"{fundamentals['pe_ratio']:.2f}" if fundamentals['pe_ratio'] > 0 else "N/A")
            
            with cols[2]:
                st.metric("52W High", f"₹{fundamentals['52w_high']:.2f}" if fundamentals['52w_high'] > 0 else "N/A")
                st.metric("52W Low", f"₹{fundamentals['52w_low']:.2f}" if fundamentals['52w_low'] > 0 else "N/A")
            
            # Display current signals and recommendations
            st.header("Current Trading Signals")
            
            # Create columns for signal display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Stock Status")
                st.metric("Last Close Price", f"₹ {signals['latest_close']:.2f}")
                st.metric("10 EMA", f"₹ {signals['ema_10']:.2f}")
                
                # Display current candle information
                st.write(f"Latest Candle Color: **{signals['latest_candle_color'].upper()}**")
                st.write(f"Position to EMA: **{'ABOVE' if signals['above_ema'] else 'BELOW'}**")
                st.write(f"Touches EMA: **{'YES' if signals['touches_ema'] else 'NO'}**")
            
            with col2:
                st.subheader("Indicator Status")
                st.metric("ADX", f"{signals['adx']:.2f}")
                st.metric("+DI", f"{signals['pdi']:.2f}")
                st.metric("-DI", f"{signals['ndi']:.2f}")
                
                # ADX Strength interpretation
                adx_strength = "Strong" if signals['adx'] > 25 else "Moderate" if signals['adx'] > 18 else "Weak"
                st.write(f"Trend Strength: **{adx_strength}**")
                
                # Check if ADX meets the strategy requirement
                st.write(f"ADX > 18: **{'YES ✅' if signals['adx'] > 18 else 'NO ❌'}**")
            
            # Trading recommendations section
            st.header("Trading Recommendations")
            
            # Active signal recommendations
            if signals['long_entry']:
                st.success("🟢 **LONG ENTRY SIGNAL**: Take a long position now. Consider an entry price of ₹{:.2f} with a stop-loss at ₹{:.2f}.".format(
                    signals['latest_close'], 
                    min(signals['latest_low'], signals['ema_10']) * 0.99
                ))
            elif signals['long_exit']:
                st.warning("🔴 **LONG EXIT SIGNAL**: Exit your long position now.")
            elif signals['short_entry']:
                st.error("🔴 **SHORT ENTRY SIGNAL**: Take a short position now. Consider an entry price of ₹{:.2f} with a stop-loss at ₹{:.2f}.".format(
                    signals['latest_close'], 
                    max(signals['latest_high'], signals['ema_10']) * 1.01
                ))
            elif signals['short_exit']:
                st.info("🟢 **SHORT EXIT SIGNAL**: Exit your short position now.")
            
            # Potential signal recommendations
            potential_signals = []
            
            if signals['potential_long_entry']:
                potential_signals.append("Potential **LONG ENTRY** setup forming. Watch for a green candle closing above the previous red candle's high.")
            if signals['potential_short_entry']:
                potential_signals.append("Potential **SHORT ENTRY** setup forming. Watch for a red candle closing below the previous green candle's low.")
            
            if potential_signals:
                st.subheader("Potential Signals")
                for signal in potential_signals:
                    st.write(f"👀 {signal}")
            
            # Display the chart
            st.header("Candlestick Chart with Signals")
            fig = plot_strategy_chart(signal_data, selected_stock)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary of recent trades
            st.header("Recent Signals Analysis")
            
            # Get all signals from the last 20 candles
            recent_signals = signal_data.iloc[-20:].copy()
            
            long_entries = recent_signals[recent_signals['long_entry'] == 1]
            long_exits = recent_signals[recent_signals['long_exit'] == 1]
            short_entries = recent_signals[recent_signals['short_entry'] == 1]
            short_exits = recent_signals[recent_signals['short_exit'] == 1]
            
            # Create a table of recent signals
            if not (long_entries.empty and long_exits.empty and short_entries.empty and short_exits.empty):
                signals_data = []
                
                for date, row in long_entries.iterrows():
                    signals_data.append({
                        'Date': date,
                        'Signal': 'LONG ENTRY',
                        'Price': f"₹{row['Close']:.2f}",
                        'ADX': f"{row['ADX']:.2f}"
                    })
                    
                for date, row in long_exits.iterrows():
                    signals_data.append({
                        'Date': date,
                        'Signal': 'LONG EXIT',
                        'Price': f"₹{row['Close']:.2f}",
                        'ADX': f"{row['ADX']:.2f}"
                    })
                    
                for date, row in short_entries.iterrows():
                    signals_data.append({
                        'Date': date,
                        'Signal': 'SHORT ENTRY',
                        'Price': f"₹{row['Close']:.2f}",
                        'ADX': f"{row['ADX']:.2f}"
                    })
                    
                for date, row in short_exits.iterrows():
                    signals_data.append({
                        'Date': date,
                        'Signal': 'SHORT EXIT',
                        'Price': f"₹{row['Close']:.2f}",
                        'ADX': f"{row['ADX']:.2f}"
                    })
                
                if signals_data:
                    signals_df = pd.DataFrame(signals_data).sort_values('Date', ascending=False)
                    st.dataframe(signals_df, use_container_width=True)
                else:
                    st.write("No signals in the recent data period.")
            else:
                st.write("No signals in the recent data period.")
                
        except Exception as e:
            st.error(f"Error analyzing {selected_stock}: {e}")
            st.info("Try selecting a different stock or checking your internet connection.")
    
    with tabs[2]:  # Performance Dashboard tab
        st.header("Strategy Performance Dashboard")
        
        # Fetch performance data for all stocks
        if 'performance_data' not in st.session_state:
            st.session_state.performance_data = []
            
            with st.spinner("Calculating performance metrics for all stocks..."):
                for stock in all_stocks:
                    try:
                        # Fetch data
                        data = fetch_stock_data(stock, period="180d", interval="1d")
                        
                        # Process data
                        processed_data = calculate_indicators(data)
                        signal_data = check_entry_exit_signals(processed_data)
                        
                        # Calculate metrics
                        metrics = calculate_performance_metrics(signal_data)
                        
                        # Add to performance data
                        st.session_state.performance_data.append({
                            'stock': stock,
                            'trade_count': metrics['trade_count'],
                            'win_rate': metrics['win_rate'],
                            'total_return': metrics['total_return'],
                            'current_position': metrics['current_position']
                        })
                    except Exception as e:
                        st.warning(f"Could not calculate performance for {stock}: {e}")
        
        # Display performance data
        if st.session_state.performance_data:
            # Create dataframe
            perf_df = pd.DataFrame(st.session_state.performance_data)
            
            # Sort by total return
            perf_df = perf_df.sort_values('total_return', ascending=False)
            
            # Format columns
            perf_df['win_rate'] = perf_df['win_rate'].apply(lambda x: f"{x:.2f}%")
            perf_df['total_return'] = perf_df['total_return'].apply(lambda x: f"{x:.2f}%")
            perf_df['current_position'] = perf_df['current_position'].apply(
                lambda x: "LONG" if x > 0 else "SHORT" if x < 0 else "NONE"
            )
            
            # Show performance table
            st.subheader("6-Month Performance by Stock")
            st.dataframe(perf_df, use_container_width=True)
            
            # Create performance charts
            st.subheader("Performance Visualization")
            
            # Prepare data for charts
            chart_data = pd.DataFrame(st.session_state.performance_data)
            
            # Bar chart of returns
            returns_data = chart_data.sort_values('total_return', ascending=False).head(10)
            
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=returns_data['stock'],
                y=returns_data['total_return'],
                marker_color=['green' if x > 0 else 'red' for x in returns_data['total_return']],
                text=returns_data['total_return'].apply(lambda x: f"{x:.2f}%"),
                textposition='auto',
            ))
            
            fig1.update_layout(
                title="Top 10 Stocks by Total Return",
                xaxis_title="Stock",
                yaxis_title="Total Return (%)",
                height=500
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Scatter plot of win rate vs return
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=chart_data['win_rate'],
                y=chart_data['total_return'],
                mode='markers+text',
                marker=dict(
                    size=chart_data['trade_count'],
                    sizemode='area',
                    sizeref=2.*max(chart_data['trade_count'])/(40.**2),
                    sizemin=4,
                    color=chart_data['total_return'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Return %")
                ),
                text=chart_data['stock'],
                textposition="top center"
            ))
            
            fig2.update_layout(
                title="Win Rate vs Return (bubble size = number of trades)",
                xaxis_title="Win Rate (%)",
                yaxis_title="Total Return (%)",
                height=600
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.info("No performance data available. Please refresh the page or check your internet connection.")
    
    # Disclaimer
    st.divider()
    st.caption("""
    **Disclaimer**: This app is for informational purposes only and does not constitute financial advice. 
    Trading stocks involves risk, and past performance is not indicative of future results. 
    Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
    
    Data source: Yahoo Finance API
    """)

if __name__ == "__main__":
    main()