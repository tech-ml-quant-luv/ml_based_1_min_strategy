import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("Machine Learning Enhanced Strategy")

# Sidebar for controls
st.sidebar.header("Controls")

# Get list of parquet files from data folder
data_folder = Path("./data")
if data_folder.exists():
    parquet_files = [f.name for f in data_folder.glob("*.parquet")]
    parquet_files.sort()
else:
    parquet_files = []
    st.sidebar.error("'data' folder not found!")

# File selection dropdown
if parquet_files:
    selected_file = st.sidebar.selectbox(
        "Select Stock Data",
        parquet_files,
        index=0
    )
else:
    selected_file = None
    st.sidebar.warning("No parquet files found in 'data' folder")

# Chart settings
st.sidebar.subheader("Chart Settings")
chart_height = st.sidebar.slider("Chart Height", 400, 1000, 600, 50)
show_support = st.sidebar.checkbox("Show Support Line", value=True)
show_resistance = st.sidebar.checkbox("Show Resistance Line", value=True)
show_entries = st.sidebar.checkbox("Show Entry Points", value=True)
show_exits = st.sidebar.checkbox("Show Exit Points", value=True)

# Technical Indicators
st.sidebar.subheader("Technical Indicators")
show_sma_20 = st.sidebar.checkbox("Show SMA 20", value=False)
show_sma_50 = st.sidebar.checkbox("Show SMA 50", value=False)
show_sma_200 = st.sidebar.checkbox("Show SMA 200", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=False)

# Function to load data
@st.cache_data
def load_data(file_path):
    """Load parquet file and filter required columns"""
    df = pd.read_parquet(file_path)
    
    # Define required columns
    required_columns = [
        "open", "high", "low", "close", 
        "position", "position_ml", 
        "support", "resistance", "atr",
        "per_bar_pnl_ml", "bar_return_ml", "equity"
    ]
    
    # Add technical indicator columns if they exist
    optional_columns = ["sma_20", "sma_50", "sma_200", "rsi"]
    
    # Filter only required columns that exist
    available_columns = [col for col in required_columns if col in df.columns]
    available_columns += [col for col in optional_columns if col in df.columns]
    
    df = df[available_columns]
    
    return df

# Function to calculate technical indicators if not present
def calculate_indicators(df):
    """Calculate technical indicators if they don't exist"""
    df = df.copy()
    
    if 'sma_20' not in df.columns:
        df['sma_20'] = df['close'].rolling(window=20).mean()
    
    if 'sma_50' not in df.columns:
        df['sma_50'] = df['close'].rolling(window=50).mean()
    
    if 'sma_200' not in df.columns:
        df['sma_200'] = df['close'].rolling(window=200).mean()
    
    if 'rsi' not in df.columns:
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

# Function to convert date to timezone-aware if needed
def make_tz_aware(date_obj, tz):
    """Convert a date to timezone-aware datetime"""
    if tz is not None:
        dt = pd.Timestamp(date_obj).tz_localize(tz)
    else:
        dt = pd.Timestamp(date_obj)
    return dt

# Function to calculate annual metrics
def calculate_annual_metrics(df, capital=1_000_000):
    """
    Calculate performance metrics using pre-calculated equity column.
    """
    if "equity" not in df.columns:
        return None
    
    # Use the equity column directly
    equity = df["equity"].copy()
    
    # Compute returns from equity
    returns = equity.pct_change().dropna()
    
    if len(returns) < 2:
        return None
    
    # Periods per year (1-minute bars)
    BARS_PER_DAY = 375
    TRADING_DAYS = 252
    PERIODS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS  # 94,500
    
    # Sharpe Ratio
    ret_std = returns.std()
    sharpe = (
        (returns.mean() / ret_std) * np.sqrt(PERIODS_PER_YEAR)
        if ret_std > 0
        else np.nan
    )
    
    # Sortino Ratio
    target = 0
    downside_diff = np.minimum(returns - target, 0)
    downside_std = np.sqrt((downside_diff ** 2).mean())
    sortino = (
        (returns.mean() / downside_std) * np.sqrt(PERIODS_PER_YEAR)
        if downside_std > 0
        else np.nan
    )
    
    # CAGR (time-based)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (
        ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100
        if years > 0 and equity.iloc[0] > 0
        else np.nan
    )
    
    # Drawdown
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1
    max_drawdown = abs(drawdown.min()) * 100
    
    # Calmar Ratio
    calmar = cagr / max_drawdown if max_drawdown > 0 else np.nan
    
    # Total Return
    total_return = ((equity.iloc[-1] / equity.iloc[0]) - 1) * 100
    
    # Metrics output
    metrics = {
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "CAGR (%)": cagr,
        "Total Return (%)": total_return,
        "Max Drawdown (%)": max_drawdown,
        "Calmar Ratio": calmar,
        "Total Bars": int(len(equity)),
    }
    
    return metrics

def calculate_yearly_comparison(df_full, tz, years=[2023, 2024, 2025]):
    """
    Calculate metrics for multiple years and return comparison table.
    """
    comparison_data = []
    
    for year in years:
        try:
            year_start = make_tz_aware(pd.Timestamp(f"{year}-01-01"), tz)
            year_end = make_tz_aware(pd.Timestamp(f"{year}-12-31"), tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_year = df_full.loc[year_start:year_end].copy()
            
            if len(df_year) < 2:
                continue
                
            metrics = calculate_annual_metrics(df_year)
            
            if metrics:
                comparison_data.append({
                    "Year": year,
                    "Sharpe Ratio": metrics["Sharpe Ratio"],
                    "Sortino Ratio": metrics["Sortino Ratio"],
                    "CAGR (%)": metrics["CAGR (%)"],
                    "Total Return (%)": metrics["Total Return (%)"],
                    "Max Drawdown (%)": metrics["Max Drawdown (%)"],
                    "Calmar Ratio": metrics["Calmar Ratio"],
                    "Total Bars": metrics["Total Bars"]
                })
        except:
            continue
    
    if comparison_data:
        return pd.DataFrame(comparison_data)
    else:
        return None


def calculate_monthly_performance(df_full):
    """Calculate monthly PnL from equity column"""
    if 'equity' not in df_full.columns:
        return None
    
    # Resample to month-end and calculate monthly returns
    monthly_equity = df_full['equity'].resample('M').last()
    monthly_returns = monthly_equity.pct_change().dropna() * 100
    
    # Create DataFrame with year-month
    monthly_df = pd.DataFrame({
        'Month': monthly_returns.index.strftime('%Y-%m'),
        'Return (%)': monthly_returns.values
    })
    
    return monthly_df

def calculate_monthly_performance_year(df_full, tz, year=2025):
    """Calculate monthly PnL from equity column for a specific year"""
    if 'equity' not in df_full.columns:
        return None
    
    try:
        # Filter data for specific year
        year_start = make_tz_aware(pd.Timestamp(f"{year}-01-01"), tz)
        year_end = make_tz_aware(pd.Timestamp(f"{year}-12-31"), tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_year = df_full.loc[year_start:year_end].copy()
        
        if len(df_year) < 2 or 'equity' not in df_year.columns:
            return None
        
        # Resample to month-end and calculate monthly returns
        monthly_equity = df_year['equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100
        
        # Create DataFrame with year-month
        monthly_df = pd.DataFrame({
            'Month': monthly_returns.index.strftime('%Y-%m'),
            'Return (%)': monthly_returns.values
        })
        
        return monthly_df
    except:
        return None


def get_best_worst_trades(trades_df, n=5):
    """Get top N best and worst trades by PnL"""
    if trades_df.empty:
        return None, None
    
    # Sort by PnL
    sorted_trades = trades_df.sort_values('pnl', ascending=False)
    
    # Best trades
    best_trades = sorted_trades.head(n)[['entry_price', 'exit_price', 'position_ml', 'pnl', 'pnl_pct', 'exit_index']].copy()
    best_trades['Type'] = best_trades['position_ml'].map({1: 'Long', -1: 'Short'})
    best_trades = best_trades.rename(columns={
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'pnl': 'PnL',
        'pnl_pct': 'PnL %',
        'exit_index': 'Exit Time'
    })
    best_trades.index.name = 'Entry Time'
    
    # Worst trades
    worst_trades = sorted_trades.tail(n)[['entry_price', 'exit_price', 'position_ml', 'pnl', 'pnl_pct', 'exit_index']].copy()
    worst_trades['Type'] = worst_trades['position_ml'].map({1: 'Long', -1: 'Short'})
    worst_trades = worst_trades.rename(columns={
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'pnl': 'PnL',
        'pnl_pct': 'PnL %',
        'exit_index': 'Exit Time'
    })
    worst_trades.index.name = 'Entry Time'
    
    return best_trades, worst_trades

def process_trades(df):
    """Process the dataframe to mark entry and exit points"""
    df = df.copy()
    df["exit_index"] = pd.NaT
    df["entry_price"] = np.nan
    df["exit_price"] = np.nan
    df["pnl"] = np.nan
    df["pnl_pct"] = np.nan
    
    i = 0
    while i < len(df):
        idx = df.index[i]
        if df.loc[idx, "position_ml"] == 0:
            i += 1
        else:
            entry_idx = idx
            
            # Record entry price based on long or short position
            if df.loc[entry_idx, 'position_ml'] == 1:  # Long trade
                entry_price = df.loc[entry_idx, 'support']
            elif df.loc[entry_idx, 'position_ml'] == -1:  # Short trade
                entry_price = df.loc[entry_idx, 'resistance']
            
            df.loc[entry_idx, 'entry_price'] = entry_price
            
            j = i
            in_trade = True
            while in_trade and j < len(df):
                current_index = df.index[j]
                if df.loc[entry_idx, "position_ml"] == df.loc[current_index, "position_ml"]:
                    j += 1
                else:
                    in_trade = False
            
            exit_index = df.index[j-1]
            
            # Record exit price based on position type
            if df.loc[entry_idx, 'position_ml'] == 1:  # Long trade
                atr = df.loc[exit_index, 'atr']
                stop_loss = entry_price - atr
                target = df.loc[exit_index, 'resistance']
                
                if df.loc[exit_index, 'low'] <= stop_loss:
                    exit_price = stop_loss
                elif df.loc[exit_index, 'high'] >= target:
                    exit_price = target
                else:
                    exit_price = df.loc[exit_index, 'close']
                
                pnl = exit_price - entry_price
                pnl_pct = pnl / entry_price
                    
            elif df.loc[entry_idx, 'position_ml'] == -1:  # Short trade
                atr = df.loc[exit_index, 'atr']
                stop_loss = entry_price + atr
                target = df.loc[exit_index, 'support']
                
                if df.loc[exit_index, 'high'] >= stop_loss:
                    exit_price = stop_loss
                elif df.loc[exit_index, 'low'] <= target:
                    exit_price = target
                else:
                    exit_price = df.loc[exit_index, 'close']
                
                pnl = entry_price - exit_price
                pnl_pct = pnl / entry_price
            
            df.loc[entry_idx, "exit_index"] = exit_index
            df.loc[entry_idx, "exit_price"] = exit_price
            df.loc[entry_idx, "pnl"] = pnl
            df.loc[entry_idx, "pnl_pct"] = pnl_pct * 100
            
            i = j
    
    return df

# Function to create the chart
def create_chart(df, stock_name, height, show_sup, show_res, show_ent, show_ext, 
                 show_sma20, show_sma50, show_sma200, show_rsi_ind):
    """Create the plotly candlestick chart"""
    # Reset index and save datetime as column
    df_plot = df.reset_index()
    df_plot.rename(columns={'index': 'datetime'}, inplace=True)
    
    # Determine if we need subplots for RSI
    if show_rsi_ind and 'rsi' in df_plot.columns:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{stock_name} - Trading Strategy', 'RSI')
        )
        rsi_row = 2
    else:
        fig = go.Figure()
        rsi_row = None
    
    # Create the candlestick chart
    candlestick = go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='OHLC',
        increasing=dict(line=dict(color='#26a69a', width=1), fillcolor='#26a69a'),
        decreasing=dict(line=dict(color='#ef5350', width=1), fillcolor='#ef5350')
    )
    
    if rsi_row:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Plot support line
    if show_sup and 'support' in df_plot.columns:
        support_trace = go.Scatter(
            x=df_plot.index,
            y=df_plot['support'],
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='Support',
            hovertemplate='Support: %{y:.2f}<extra></extra>'
        )
        if rsi_row:
            fig.add_trace(support_trace, row=1, col=1)
        else:
            fig.add_trace(support_trace)
    
    # Plot resistance line
    if show_res and 'resistance' in df_plot.columns:
        resistance_trace = go.Scatter(
            x=df_plot.index,
            y=df_plot['resistance'],
            mode='lines',
            line=dict(color='orange', width=1, dash='dash'),
            name='Resistance',
            hovertemplate='Resistance: %{y:.2f}<extra></extra>'
        )
        if rsi_row:
            fig.add_trace(resistance_trace, row=1, col=1)
        else:
            fig.add_trace(resistance_trace)
    
    # Plot SMAs
    if show_sma20 and 'sma_20' in df_plot.columns:
        sma20_trace = go.Scatter(
            x=df_plot.index,
            y=df_plot['sma_20'],
            mode='lines',
            line=dict(color='purple', width=1),
            name='SMA 20',
            hovertemplate='SMA 20: %{y:.2f}<extra></extra>'
        )
        if rsi_row:
            fig.add_trace(sma20_trace, row=1, col=1)
        else:
            fig.add_trace(sma20_trace)
    
    if show_sma50 and 'sma_50' in df_plot.columns:
        sma50_trace = go.Scatter(
            x=df_plot.index,
            y=df_plot['sma_50'],
            mode='lines',
            line=dict(color='yellow', width=1),
            name='SMA 50',
            hovertemplate='SMA 50: %{y:.2f}<extra></extra>'
        )
        if rsi_row:
            fig.add_trace(sma50_trace, row=1, col=1)
        else:
            fig.add_trace(sma50_trace)
    
    if show_sma200 and 'sma_200' in df_plot.columns:
        sma200_trace = go.Scatter(
            x=df_plot.index,
            y=df_plot['sma_200'],
            mode='lines',
            line=dict(color='white', width=1),
            name='SMA 200',
            hovertemplate='SMA 200: %{y:.2f}<extra></extra>'
        )
        if rsi_row:
            fig.add_trace(sma200_trace, row=1, col=1)
        else:
            fig.add_trace(sma200_trace)
    
    # Get entries (where exit_index is not NaT)
    entries = df_plot[df_plot['exit_index'].notna()].copy()
    
    # Separate long and short entries
    long_entries = entries[entries['position_ml'] == 1]
    short_entries = entries[entries['position_ml'] == -1]
    
    # Plot long entry markers
    if show_ent and not long_entries.empty:
        long_entry_trace = go.Scatter(
            x=long_entries.index,
            y=long_entries['support'],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(width=1, color='darkgreen')
            ),
            name='Long Entry',
            customdata=long_entries[['datetime', 'pnl', 'pnl_pct']],
            hovertemplate='Long Entry<br>Time: %{customdata[0]}<br>Price: %{y:.2f}<br>PnL: %{customdata[1]:.2f}<br>PnL %: %{customdata[2]:.2f}%<extra></extra>'
        )
        if rsi_row:
            fig.add_trace(long_entry_trace, row=1, col=1)
        else:
            fig.add_trace(long_entry_trace)
    
    # Plot short entry markers
    if show_ent and not short_entries.empty:
        short_entry_trace = go.Scatter(
            x=short_entries.index,
            y=short_entries['resistance'],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(width=1, color='darkred')
            ),
            name='Short Entry',
            customdata=short_entries[['datetime', 'pnl', 'pnl_pct']],
            hovertemplate='Short Entry<br>Time: %{customdata[0]}<br>Price: %{y:.2f}<br>PnL: %{customdata[1]:.2f}<br>PnL %: %{customdata[2]:.2f}%<extra></extra>'
        )
        if rsi_row:
            fig.add_trace(short_entry_trace, row=1, col=1)
        else:
            fig.add_trace(short_entry_trace)
    
    # Plot long exit markers
    if show_ext and not long_entries.empty:
        exit_positions = df_plot[df_plot['datetime'].isin(long_entries['exit_index'])].index
        exit_prices = long_entries['exit_price'].values
        exit_times = df_plot.loc[exit_positions, 'datetime'].values
        pnl_values = long_entries['pnl'].values
        pnl_pct_values = long_entries['pnl_pct'].values
        
        long_exit_trace = go.Scatter(
            x=exit_positions,
            y=exit_prices,
            mode='markers',
            marker=dict(
                symbol='x',
                size=12,
                color='red',
                line=dict(width=2)
            ),
            name='Long Exit',
            customdata=list(zip(exit_times, pnl_values, pnl_pct_values)),
            hovertemplate='Long Exit<br>Time: %{customdata[0]}<br>Price: %{y:.2f}<br>PnL: %{customdata[1]:.2f}<br>PnL %: %{customdata[2]:.2f}%<extra></extra>'
        )
        if rsi_row:
            fig.add_trace(long_exit_trace, row=1, col=1)
        else:
            fig.add_trace(long_exit_trace)
    
    # Plot short exit markers
    if show_ext and not short_entries.empty:
        exit_positions = df_plot[df_plot['datetime'].isin(short_entries['exit_index'])].index
        exit_prices = short_entries['exit_price'].values
        exit_times = df_plot.loc[exit_positions, 'datetime'].values
        pnl_values = short_entries['pnl'].values
        pnl_pct_values = short_entries['pnl_pct'].values
        
        short_exit_trace = go.Scatter(
            x=exit_positions,
            y=exit_prices,
            mode='markers',
            marker=dict(
                symbol='x',
                size=12,
                color='green',
                line=dict(width=2)
            ),
            name='Short Exit',
            customdata=list(zip(exit_times, pnl_values, pnl_pct_values)),
            hovertemplate='Short Exit<br>Time: %{customdata[0]}<br>Price: %{y:.2f}<br>PnL: %{customdata[1]:.2f}<br>PnL %: %{customdata[2]:.2f}%<extra></extra>'
        )
        if rsi_row:
            fig.add_trace(short_exit_trace, row=1, col=1)
        else:
            fig.add_trace(short_exit_trace)
    
    # Plot RSI
    if show_rsi_ind and rsi_row and 'rsi' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot['rsi'],
            mode='lines',
            line=dict(color='cyan', width=1),
            name='RSI',
            hovertemplate='RSI: %{y:.2f}<extra></extra>'
        ), row=2, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3, row=2, col=1)
    
    # Create custom tick values and labels
    tick_spacing = max(1, len(df_plot) // 20)
    tick_indices = list(range(0, len(df_plot), tick_spacing))
    tick_labels = [df_plot.loc[i, 'datetime'].strftime('%Y-%m-%d %H:%M') for i in tick_indices]
    
    # Update layout
    if rsi_row:
        fig.update_xaxes(
            tickmode='array',
            tickvals=tick_indices,
            ticktext=tick_labels,
            tickangle=-45,
            row=2, col=1
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_layout(
            height=height + 200,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            showlegend=True
        )
    else:
        fig.update_layout(
            title=f'{stock_name} - Trading Strategy',
            yaxis_title='Price',
            xaxis_title='Date/Time',
            xaxis=dict(
                tickmode='array',
                tickvals=tick_indices,
                ticktext=tick_labels,
                tickangle=-45
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            height=height,
            showlegend=True
        )
    
    return fig

# Function to create equity curve chart
def create_equity_curve(df, stock_name):
    """Create equity curve vs stock returns chart"""
    df_plot = df.reset_index()
    df_plot.rename(columns={'index': 'datetime'}, inplace=True)
    
    # Calculate strategy equity curve from per_bar_pnl_ml
    if 'per_bar_pnl_ml' in df_plot.columns:
        df_plot['equity_curve'] = df_plot['per_bar_pnl_ml'].cumsum()
    else:
        df_plot['equity_curve'] = 0
    
    # Calculate buy & hold returns
    initial_price = df_plot['close'].iloc[0]
    df_plot['stock_returns'] = ((df_plot['close'] - initial_price) / initial_price) * 100
    
    # Create the chart
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['equity_curve'],
        mode='lines',
        name='Strategy Equity',
        line=dict(color='#26a69a', width=2),
        hovertemplate='Strategy: %{y:.2f}<extra></extra>'
    ))
    
    # Stock returns
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['stock_returns'],
        mode='lines',
        name='Buy & Hold',
        line=dict(color='#ff9800', width=2),
        hovertemplate='Buy & Hold: %{y:.2f}%<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Create custom tick values and labels
    tick_spacing = max(1, len(df_plot) // 20)
    tick_indices = list(range(0, len(df_plot), tick_spacing))
    tick_labels = [df_plot.loc[i, 'datetime'].strftime('%Y-%m-%d %H:%M') for i in tick_indices]
    
    fig.update_layout(
        title=f'{stock_name} - Equity Curve vs Buy & Hold',
        yaxis_title='Cumulative Returns (%)',
        xaxis_title='Date/Time',
        xaxis=dict(
            tickmode='array',
            tickvals=tick_indices,
            ticktext=tick_labels,
            tickangle=-45
        ),
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig

# Function to create drawdown chart
def create_drawdown_chart(df, stock_name):
    """Create drawdown chart for the strategy"""
    df_plot = df.reset_index()
    df_plot.rename(columns={'index': 'datetime'}, inplace=True)
    
    # Calculate strategy equity curve from per_bar_pnl_ml
    if 'per_bar_pnl_ml' in df_plot.columns:
        df_plot['equity_curve'] = df_plot['per_bar_pnl_ml'].cumsum()
        
        # Calculate drawdown
        running_max = df_plot['equity_curve'].cummax()
        df_plot['drawdown'] = df_plot['equity_curve'] - running_max
        df_plot['drawdown_pct'] = (df_plot['drawdown'] / running_max.replace(0, 1)) * 100
    else:
        df_plot['drawdown'] = 0
        df_plot['drawdown_pct'] = 0
    
    # Create the chart
    fig = go.Figure()
    
    # Drawdown area chart
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['drawdown'],
        mode='lines',
        name='Drawdown',
        line=dict(color='#ef5350', width=0),
        fill='tozeroy',
        fillcolor='rgba(239, 83, 80, 0.3)',
        hovertemplate='Drawdown: %{y:.2f}<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    # Find max drawdown point
    max_dd_idx = df_plot['drawdown'].idxmin()
    max_dd_value = df_plot.loc[max_dd_idx, 'drawdown']
    max_dd_date = df_plot.loc[max_dd_idx, 'datetime']
    
    # Mark max drawdown point
    fig.add_trace(go.Scatter(
        x=[max_dd_idx],
        y=[max_dd_value],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Max Drawdown',
        hovertemplate=f'Max DD: {max_dd_value:.2f}<br>Date: {max_dd_date}<extra></extra>'
    ))
    
    # Create custom tick values and labels
    tick_spacing = max(1, len(df_plot) // 20)
    tick_indices = list(range(0, len(df_plot), tick_spacing))
    tick_labels = [df_plot.loc[i, 'datetime'].strftime('%Y-%m-%d %H:%M') for i in tick_indices]
    
    fig.update_layout(
        title=f'{stock_name} - Drawdown',
        yaxis_title='Drawdown',
        xaxis_title='Date/Time',
        xaxis=dict(
            tickmode='array',
            tickvals=tick_indices,
            ticktext=tick_labels,
            tickangle=-45
        ),
        hovermode='x unified',
        height=300,
        showlegend=True
    )
    
    return fig

# Main app logic
if selected_file is not None:
    # Load data
    file_path = data_folder / selected_file
    
    with st.spinner(f"Loading {selected_file}..."):
        df_full = load_data(file_path)
        df_full = calculate_indicators(df_full)
    
    # Extract stock name from filename
    stock_name = selected_file.replace('.parquet', '')
    
    # Get min and max dates from the full dataset
    min_date = df_full.index.min()
    max_date = df_full.index.max()
    
    # Get timezone info from the index
    tz = df_full.index.tz
    
    # Calculate yearly comparison for 2023, 2024, 2025
    yearly_comparison_df = calculate_yearly_comparison(df_full, tz)
    
    # Initialize session state for dates if not exists
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.start_date_val = (max_date - timedelta(days=5)).date()
        st.session_state.end_date_val = max_date.date()
    
    # Date Range Filter Section
    st.subheader("Date Range Filter")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date_input = st.date_input(
            "Start Date", 
            value=st.session_state.start_date_val,
            min_value=min_date.date(), 
            max_value=max_date.date(),
            key="start_date"
        )
        st.session_state.start_date_val = start_date_input
    
    with col2:
        end_date_input = st.date_input(
            "End Date", 
            value=st.session_state.end_date_val,
            min_value=min_date.date(), 
            max_value=max_date.date(),
            key="end_date"
        )
        st.session_state.end_date_val = end_date_input
    
    # Convert dates to timezone-aware
    start_date = make_tz_aware(start_date_input, tz)
    end_date = make_tz_aware(end_date_input, tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Filter dataframe based on selected dates
    df = df_full.loc[start_date:end_date].copy()
    
    # Process trades
    with st.spinner("Processing trades..."):
        df_processed = process_trades(df)
    
    # Extract trades for analysis (DEFINE ONCE HERE)
    trades = df_processed[df_processed['exit_index'].notna()]
    
    # Create and display chart
    st.subheader(f"{stock_name}")
    
    fig = create_chart(
        df_processed,
        stock_name,
        chart_height, 
        show_support, 
        show_resistance, 
        show_entries, 
        show_exits,
        show_sma_20,
        show_sma_50,
        show_sma_200,
        show_rsi
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create and display equity curve
    st.subheader("Performance Comparison")
    equity_fig = create_equity_curve(df_processed, stock_name)
    st.plotly_chart(equity_fig, use_container_width=True)
    
    # Create and display drawdown chart
    st.subheader("Drawdown Analysis")
    drawdown_fig = create_drawdown_chart(df_processed, stock_name)
    st.plotly_chart(drawdown_fig, use_container_width=True)
    
    # Annual Performance Metrics Comparison (2023, 2024, 2025)
    st.subheader("Annual Performance Metrics (2023-2025)")
    
    if yearly_comparison_df is not None and not yearly_comparison_df.empty:
        st.dataframe(
            yearly_comparison_df.style.format({
                'Sharpe Ratio': '{:.3f}',
                'Sortino Ratio': '{:.3f}',
                'CAGR (%)': '{:.2f}',
                'Total Return (%)': '{:.2f}',
                'Max Drawdown (%)': '{:.2f}',
                'Calmar Ratio': '{:.3f}',
                'Total Bars': '{:,}'
            }),
            use_container_width=True
        )
    else:
        st.warning("No data available for year comparison.")
    
    # Trade Analysis Section
    st.subheader("Trade Analysis")
    
    if not trades.empty:
        # Get best and worst trades
        best_trades, worst_trades = get_best_worst_trades(trades, n=5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top 5 Best Trades")
            if best_trades is not None:
                st.dataframe(
                    best_trades[['Type', 'Entry Price', 'Exit Price', 'PnL', 'PnL %']].style.format({
                        'Entry Price': '{:.2f}',
                        'Exit Price': '{:.2f}',
                        'PnL': '{:.2f}',
                        'PnL %': '{:.2f}'
                    }),
                    use_container_width=True
                )
        
        with col2:
            st.markdown("#### 📉 Top 5 Worst Trades")
            if worst_trades is not None:
                st.dataframe(
                    worst_trades[['Type', 'Entry Price', 'Exit Price', 'PnL', 'PnL %']].style.format({
                        'Entry Price': '{:.2f}',
                        'Exit Price': '{:.2f}',
                        'PnL': '{:.2f}',
                        'PnL %': '{:.2f}'
                    }),
                    use_container_width=True
                )
    
    # Monthly Performance Section
    st.subheader("Monthly Performance Analysis (2025)")
    
    # Get monthly performance for 2025 only
    monthly_perf_2025 = calculate_monthly_performance_year(df_full, tz, year=2025)
    
    if monthly_perf_2025 is not None and not monthly_perf_2025.empty:
        # Sort by return to get best and worst months
        monthly_sorted = monthly_perf_2025.sort_values('Return (%)', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Best Months (2025)")
            best_months = monthly_sorted.head(5)  # Show all months, sorted best to worst
            st.dataframe(
                best_months.style.format({'Return (%)': '{:.2f}'}),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 📉 Worst Months (2025)")
            worst_months = monthly_sorted.tail(5).sort_values('Return (%)')  # Show all months, sorted worst to best
            st.dataframe(
                worst_months.style.format({'Return (%)': '{:.2f}'}),
                use_container_width=True
            )
        
        # Monthly returns heatmap (still show all years for context)
        st.markdown("#### 📊 Monthly Returns Heatmap (All Years)")
        
        # Use full dataset for heatmap
        monthly_perf_all = calculate_monthly_performance(df_full)
        
        if monthly_perf_all is not None and not monthly_perf_all.empty:
            # Create pivot table for heatmap (Year x Month)
            monthly_perf_all['Year'] = pd.to_datetime(monthly_perf_all['Month']).dt.year
            monthly_perf_all['Month_Name'] = pd.to_datetime(monthly_perf_all['Month']).dt.strftime('%b')
            
            pivot_table = monthly_perf_all.pivot_table(
                values='Return (%)',
                index='Year',
                columns='Month_Name',
                aggfunc='sum'
            )
            
            # Reorder columns to calendar order
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_table = pivot_table.reindex(columns=[m for m in month_order if m in pivot_table.columns])
            
            # Create heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='RdYlGn',
                text=pivot_table.values,
                texttemplate='%{text:.2f}%',
                textfont={"size": 10},
                colorbar=dict(title="Return %")
            ))
            
            fig_heatmap.update_layout(
                title='Monthly Returns by Year',
                xaxis_title='Month',
                yaxis_title='Year',
                height=300
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("No monthly data available for 2025.")
    
    # Trading Metrics
    st.subheader("Trading Metrics")
    
    if not trades.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(trades)
        long_trades = len(trades[trades['position_ml'] == 1])
        short_trades = len(trades[trades['position_ml'] == -1])
        winning_trades = len(trades[trades['pnl'] > 0])
        
        total_pnl = trades['pnl'].sum()
        avg_pnl_pct = trades['pnl_pct'].mean()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.2f}%")
        col3.metric("Total PnL", f"{total_pnl:.2f}")
        col4.metric("Avg PnL %", f"{avg_pnl_pct:.2f}%")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Long Trades", long_trades)
        col2.metric("Short Trades", short_trades)
        col3.metric("Winning Trades", winning_trades)
        col4.metric("Losing Trades", total_trades - winning_trades)
    else:
        st.warning("No trades found in the selected date range.")
    
    # Trade details table
    if not trades.empty:
        st.subheader("Trade Details")
        
        # Prepare trade details
        trade_details = trades[['entry_price', 'exit_price', 'position_ml', 'pnl', 'pnl_pct', 'exit_index']].copy()
        trade_details['position_type'] = trade_details['position_ml'].map({1: 'Long', -1: 'Short'})
        trade_details = trade_details.rename(columns={
            'entry_price': 'Entry Price',
            'exit_price': 'Exit Price',
            'pnl': 'PnL',
            'pnl_pct': 'PnL %',
            'exit_index': 'Exit Time',
            'position_type': 'Type'
        })
        trade_details.index.name = 'Entry Time'
        
        # Display
        st.dataframe(
            trade_details[['Type', 'Entry Price', 'Exit Price', 'PnL', 'PnL %', 'Exit Time']],
            use_container_width=True
        )
        
        # Download button
        csv = trade_details.to_csv()
        st.download_button(
            label="Download Trade Details as CSV",
            data=csv,
            file_name=f"{stock_name}_trade_details_{start_date.date()}_to_{end_date.date()}.csv",
            mime="text/csv"
        )

else:
    st.info("Please select a stock from the dropdown")
    
    st.markdown("""
    ### Instructions:
    1. Select a stock from the dropdown in the sidebar
    2. Use the date range filter to select the period you want to analyze
    3. Use the sidebar controls to customize the chart and indicators
    4. View equity curve, annual metrics, trading metrics and trade details below
    
    ### Available Stocks:
    """)
    
    if parquet_files:
        for i, file in enumerate(parquet_files, 1):
            st.write(f"{i}. {file.replace('.parquet', '')}")
    else:
        st.warning("No parquet files found in the 'data' folder")