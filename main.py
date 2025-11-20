import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import plotly.graph_objects as go
import argparse
import dask.dataframe as dd
from pyspark.sql import SparkSession
import json
import time
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# Function to fetch historical data
def fetch_historical_data(ticker, start_date, end_date, retries=0, retry_sleep=5):
    """Download historical close prices with optional retries.
    Returns a Series (Close) or empty Series if failed after retries."""
    attempt = 0
    data = pd.DataFrame()
    while attempt <= retries:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Download exception for {ticker} attempt {attempt}: {e}")
            data = pd.DataFrame()
        if data is not None and not data.empty:
            break
        attempt += 1
        if attempt <= retries:
            print(f"Retrying download for {ticker} in {retry_sleep}s (attempt {attempt}/{retries})...")
            time.sleep(retry_sleep)
    if data is None or data.empty or 'Close' not in data.columns:
        print(f"Historical data unavailable for {ticker} after {attempt} attempt(s).")
        return pd.Series(dtype=float)
    return data['Close']

# Fetch full OHLC data for candlestick visualization
def fetch_ohlc_data(ticker, start_date, end_date, include_volume=False, retries=0, retry_sleep=5):
    attempt = 0
    data = pd.DataFrame()
    while attempt <= retries:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"OHLC download exception for {ticker} attempt {attempt}: {e}")
            data = pd.DataFrame()
        if data is not None and not data.empty:
            break
        attempt += 1
        if attempt <= retries:
            print(f"Retrying OHLC download for {ticker} in {retry_sleep}s (attempt {attempt}/{retries})...")
            time.sleep(retry_sleep)
    if data is None or data.empty:
        raise ValueError(f"OHLC data unavailable for {ticker} after {attempt} attempt(s).")
    required = ['Open', 'High', 'Low', 'Close']
    if include_volume:
        required.append('Volume')
    # Flatten MultiIndex columns if present (common in some yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        flat_map = {}
        for col in data.columns:
            if len(col) >= 2:
                flat_map[col] = next((part for part in col if part in required), col[-1])
            else:
                flat_map[col] = col[-1]
        data.columns = [flat_map[c] for c in data.columns]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for candlestick after flatten: {missing}; columns={list(data.columns)}")
    return data[required]

# Function to fetch real-time data (last day's close)
def fetch_real_time_data(ticker):
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period='1d')
    if hist.empty:
        return None, None
    # return price and timestamp of the last available point
    ts = hist.index[-1]
    price = hist['Close'].iloc[-1]
    return price, ts

# Preprocess data for LSTM
def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict future prices
def predict_future(model, last_sequence, scaler, days=30):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(current_sequence.reshape(1, -1, 1))
        predictions.append(pred[0][0])
        current_sequence = np.append(current_sequence[1:], pred[0][0])

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Visualize with matplotlib
def visualize_matplotlib(data, predictions=None, ticker=None, realtime_price=None, realtime_ts=None):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data.values, label='Historical Prices')

    if predictions is not None:
        future_dates = pd.date_range(start=data.index[-1], periods=len(predictions)+1, freq='D')[1:]
        plt.plot(future_dates, predictions, label='Predicted Prices', color='red')

    # If a realtime price is provided, annotate it on the chart
    if realtime_price is not None and realtime_ts is not None:
        plt.scatter([realtime_ts], [realtime_price], color='orange', s=100, zorder=5, label='Real-time Price')
        plt.annotate(f'Real-time: {realtime_price:.2f}\n{pd.to_datetime(realtime_ts)}',
                     xy=(realtime_ts, realtime_price), xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.4))

    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


def live_update_plot(data, ticker, interval_seconds=5, marker_size=100, realtime_only=False, output_meta=None):
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 7))

    if not realtime_only:
        line, = ax.plot(data.index, data.values, label='Historical Prices')
    else:
        line = None

    scatter = ax.scatter([], [], color='orange', s=marker_size, zorder=5, label='Real-time Price')
    annot = ax.annotate('', xy=(0, 0), xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.4))
    annot.set_visible(False)

    ax.set_title(f'{ticker} Real-time Monitor')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    try:
        while True:
            try:
                price, ts = fetch_real_time_data(ticker)
                if price is None:
                    print('Could not fetch real-time data during live update')
                    time.sleep(interval_seconds)
                    continue
                scatter.set_offsets([[pd.to_datetime(ts), price]])
                annot.xy = (pd.to_datetime(ts), price)
                annot.set_text(f'Real-time: {price:.2f}\n{pd.to_datetime(ts)}')
                annot.set_visible(True)
                try:
                    ax.relim(); ax.autoscale_view()
                except Exception:
                    pass
                fig.canvas.draw(); fig.canvas.flush_events()
                if output_meta:
                    write_metadata(output_meta, ticker, price, ts, source='yfinance')
            except Exception as loop_err:
                print(f'Live loop rendering error: {loop_err}; falling back to static plot.')
                break
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print('Live update stopped by user')
    finally:
        plt.ioff()


def write_metadata(path, ticker, price, ts, source='yfinance'):
    payload = {
        'ticker': ticker,
        'price': float(price),
        'timestamp': pd.to_datetime(ts).isoformat(),
        'source': source,
        'fetched_at': datetime.utcnow().isoformat() + 'Z'
    }
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f'Failed to write metadata to {path}: {e}')


def run_daemon(ticker, interval=60, log_path='realtime.log', max_lines=0, output_meta=None):
    print(f'Daemon loop started for {ticker}; interval={interval}s; log={log_path}; max_lines={max_lines or "unlimited"}')
    lines_written = 0
    while True:
        price, ts = fetch_real_time_data(ticker)
        if price is None:
            print('Realtime fetch failed; retrying after interval')
            time.sleep(interval)
            continue
        entry = {
            'ticker': ticker,
            'price': float(price),
            'timestamp': pd.to_datetime(ts).isoformat(),
            'source': 'yfinance',
            'fetched_at': datetime.utcnow().isoformat() + 'Z'
        }
        # append JSON line
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f'Failed to append to {log_path}: {e}')
        # optional single metadata file overwrite
        if output_meta:
            write_metadata(output_meta, ticker, price, ts)
        lines_written += 1
        print(f'Daemon wrote line {lines_written}: price={price} ts={ts}')
        if max_lines and max_lines > 0 and lines_written >= max_lines:
            print('Max lines reached; exiting daemon.')
            break
        time.sleep(interval)

# Visualize with plotly
def visualize_plotly(data, predictions, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name='Historical Prices'))
    future_dates = pd.date_range(start=data.index[-1], periods=len(predictions)+1, freq='D')[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Predicted Prices', line=dict(color='red')))
    fig.update_layout(title=f'{ticker} Stock Price Prediction', xaxis_title='Date', yaxis_title='Price')
    fig.show()

# Candlestick visualization with Plotly
def generate_forecast_candles(predicted_closes, last_close):
    rows = []
    prev_close = last_close
    today = datetime.utcnow().date()
    for i, pc in enumerate(predicted_closes):
        # Simple synthetic OHLC logic around predicted close
        open_price = prev_close
        close_price = pc
        mid = (open_price + close_price) / 2.0
        # volatility band 1% around mid
        high_price = max(open_price, close_price) * 1.01
        low_price = min(open_price, close_price) * 0.99
        date = today + timedelta(days=i+1)
        rows.append({'Date': date, 'Open': open_price, 'High': high_price, 'Low': low_price, 'Close': close_price})
        prev_close = close_price
    df = pd.DataFrame(rows).set_index('Date')
    return df

def visualize_candlestick_plotly(ohlc_df, ticker, predictions=None, realtime_price=None, realtime_ts=None,
                                  save_html=None, include_volume=False, hide_realtime=False, forecast_candles=0):
    # If volume requested and present, build subplot with secondary row
    volume_available = include_volume and 'Volume' in ohlc_df.columns
    if volume_available:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_heights=[0.8, 0.2])
    else:
        fig = go.Figure()
    # Candlestick base trace
    candle_trace = go.Candlestick(
        x=ohlc_df.index,
        open=ohlc_df['Open'].values,
        high=ohlc_df['High'].values,
        low=ohlc_df['Low'].values,
        close=ohlc_df['Close'].values,
        name='OHLC'
    )
    if volume_available:
        fig.add_trace(candle_trace, row=1, col=1)
    else:
        fig.add_trace(candle_trace)
    # Volume bars
    if volume_available:
        fig.add_trace(go.Bar(x=ohlc_df.index, y=ohlc_df['Volume'].values, name='Volume', marker_color='rgba(50,150,255,0.5)'), row=2, col=1)
    # Forecast future candlestick synthetic candles if requested and predictions provided
    if forecast_candles > 0 and predictions is not None and len(predictions) >= forecast_candles:
        last_close = ohlc_df['Close'].iloc[-1]
        fc_df = generate_forecast_candles(predictions[:forecast_candles], last_close)
        fig.add_trace(go.Candlestick(
            x=fc_df.index,
            open=fc_df['Open'], high=fc_df['High'], low=fc_df['Low'], close=fc_df['Close'], name='Forecast OHLC',
            increasing_line_color='red', decreasing_line_color='red'
        ))
    # Optional predicted close line (full predictions vector)
    if predictions is not None:
        future_dates = pd.date_range(start=ohlc_df.index[-1], periods=len(predictions)+1, freq='D')[1:]
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Predicted Close', line=dict(color='red')))
    # Real-time marker unless hidden
    if not hide_realtime and realtime_price is not None and realtime_ts is not None:
        fig.add_trace(go.Scatter(x=[realtime_ts], y=[realtime_price], mode='markers',
                                 marker=dict(color='orange', size=12), name='Real-time Price'))
        try:
            fig.add_vline(x=realtime_ts, line=dict(color='orange', width=1, dash='dot'))
        except Exception:
            pass
    # Debug summary
    try:
        low_min = ohlc_df['Low'].min()
        high_max = ohlc_df['High'].max()
        print("Candlestick debug: rows=", len(ohlc_df), "low_min=", float(low_min), "high_max=", float(high_max))
    except Exception as e:
        print("Candlestick debug skipped:", e)
    fig.update_layout(title=f'{ticker} Candlestick Chart', xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    if volume_available:
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
    if save_html:
        try:
            fig.write_html(save_html)
            print(f'Saved candlestick chart to {save_html}')
        except Exception as e:
            print(f'Failed to save HTML chart: {e}')
    fig.show()

def visualize_multi_candlestick_grid(tickers, start_date, end_date, include_volume=False, hide_realtime=False, save_html=None):
    ohlc_map = {}
    for t in tickers:
        try:
            ohlc_map[t] = fetch_ohlc_data(t, start_date, end_date, include_volume=include_volume)
        except Exception as e:
            print(f'Failed OHLC for {t}: {e}')
    valid = [t for t in tickers if t in ohlc_map]
    if not valid:
        print('No valid tickers for multi-candlestick grid.')
        return
    cols = 2 if len(valid) > 1 else 1
    rows = int(np.ceil(len(valid) / cols))
    specs = [[{'secondary_y': False} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=False, subplot_titles=valid)
    r = c = 1
    for t in valid:
        df = ohlc_map[t]
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f'{t} OHLC'), row=r, col=c)
        # Add volume if requested and present
        if include_volume and 'Volume' in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name=f'{t} Vol', marker_color='rgba(50,150,255,0.4)'), row=r, col=c)
        # Real-time marker per ticker
        if not hide_realtime:
            rt_price, rt_ts = fetch_real_time_data(t)
            if rt_price is not None and rt_ts is not None:
                fig.add_trace(go.Scatter(x=[rt_ts], y=[rt_price], mode='markers', marker=dict(size=8), name=f'{t} RT'), row=r, col=c)
        c += 1
        if c > cols:
            c = 1
            r += 1
    fig.update_layout(title='Multi-Ticker Candlestick Grid', xaxis_rangeslider_visible=False, showlegend=False)
    if save_html:
        try:
            fig.write_html(save_html)
            print(f'Saved multi-candlestick grid to {save_html}')
        except Exception as e:
            print(f'Failed to save multi grid HTML: {e}')
    fig.show()

# Main function
def main():
    # Accept ticker(s) and date range from command line
    parser = argparse.ArgumentParser(description='Stock market analysis and prediction')
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Ticker for single-stock prediction (e.g. AAPL)')
    parser.add_argument('--tickers', type=str, default='AAPL,GOOGL,MSFT',
                        help='Comma-separated tickers for big-data processing')
    parser.add_argument('--start', dest='start_date', type=str, default='2020-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end', dest='end_date', type=str, default='2023-01-01',
                        help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--no-train', action='store_true', help='Skip training and prediction; only fetch and visualize historical + real-time data')
    parser.add_argument('--plot-realtime-only', action='store_true', help='When plotting, show only the real-time point (no historical line)')
    parser.add_argument('--realtime-marker-size', type=int, default=100, help='Marker size for the realtime point')
    parser.add_argument('--no-plot', action='store_true', help='Do not open any plots (useful for CI / headless runs)')
    parser.add_argument('--live', type=int, default=0, help='If >0, run live-updating plot refreshing every N seconds')
    parser.add_argument('--output-meta', type=str, default=None, help='Write JSON metadata about realtime fetch to this file')
    parser.add_argument('--candlestick', action='store_true', help='Show candlestick (OHLC) chart instead of line chart')
    parser.add_argument('--save-html', type=str, default=None, help='Save interactive Plotly chart (candlestick) to this HTML file')
    parser.add_argument('--include-volume', action='store_true', help='Include volume bars beneath candlesticks where available')
    parser.add_argument('--hide-realtime', action='store_true', help='Hide the realtime marker/guideline on charts')
    parser.add_argument('--forecast-candles', type=int, default=0, help='Overlay synthetic forecast candlesticks for N future days (requires training)')
    parser.add_argument('--multi-candle-html', type=str, default=None, help='Generate multi-ticker candlestick grid HTML to this file (uses --tickers list)')
    # Training control flags
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (default 1 for quick demo)')
    parser.add_argument('--download-retries', type=int, default=2, help='Retries for historical/ohlc downloads when rate limited')
    parser.add_argument('--download-retry-sleep', type=int, default=5, help='Seconds to sleep between download retries')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length (timesteps) for LSTM input')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training (default 1)')
    parser.add_argument('--early-stop-patience', type=int, default=0, help='Patience for EarlyStopping on loss (0 disables)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to save best model checkpoint (optional)')
    # Daemon mode flags
    parser.add_argument('--daemon', action='store_true', help='Run in daemon mode: periodic realtime fetch to log file')
    parser.add_argument('--daemon-interval', type=int, default=60, help='Interval in seconds for daemon realtime polling')
    parser.add_argument('--daemon-log', type=str, default='realtime.log', help='Path to append JSON lines with realtime data')
    parser.add_argument('--daemon-max-lines', type=int, default=0, help='If >0, stop daemon after writing this many lines')
    args = parser.parse_args()

    ticker = args.ticker
    tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]
    start_date = args.start_date
    end_date = args.end_date

    # Fetch historical data
    historical_data = fetch_historical_data(ticker, start_date, end_date,
                                           retries=args.download_retries,
                                           retry_sleep=args.download_retry_sleep)
    print(f"Fetched {len(historical_data)} days of historical data for {ticker}")

    ohlc_data = None
    if args.candlestick:
        try:
            ohlc_data = fetch_ohlc_data(ticker, start_date, end_date, include_volume=args.include_volume,
                                        retries=args.download_retries, retry_sleep=args.download_retry_sleep)
        except Exception as e:
            print(f"Failed to fetch OHLC data for candlestick: {e}")
            args.candlestick = False  # fallback to normal line plot
    # Graceful exit if no historical data (rate limit or outage)
    if historical_data.empty:
        print('No historical data retrieved; aborting before training. Please wait and retry or reduce date range.')
        return

    # Fetch real-time data (price + timestamp)
    real_time_price, real_time_ts = fetch_real_time_data(ticker)
    if real_time_price is not None:
        print(f"Real-time price for {ticker}: {real_time_price} (as of {real_time_ts})")
    else:
        print("Could not fetch real-time data")

    # If daemon mode requested early, we can still fetch big-data metrics once then enter loop
    # Big Data Concepts: Processing Multiple Stocks (tickers comes from CLI `--tickers`)
    all_data = []
    for t in tickers:
        data = yf.download(t, start=start_date, end=end_date)
        # if yfinance returns MultiIndex columns, try to flatten
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data.columns = data.columns.droplevel(1)
            except Exception:
                pass
        data = data.reset_index()
        data['Ticker'] = t
        # keep only Date and Close for aggregation
        data = data[['Date', 'Close', 'Ticker']]
        all_data.append(data)

    combined_df = pd.concat(all_data)
    print("Combined DF columns:", combined_df.columns)
    print(combined_df.head())

    # Use Dask for parallel processing
    dask_df = dd.from_pandas(combined_df, npartitions=4)
    mean_prices = dask_df.groupby('Ticker')['Close'].mean().compute()
    print("Mean close prices using Dask:")
    print(mean_prices)

    

    # Use Spark for distributed processing
    try:
        spark = SparkSession.builder.appName("StockAnalysis").getOrCreate()
        spark_df = spark.createDataFrame(combined_df)
        spark_df.createOrReplaceTempView("stocks")
        result = spark.sql("SELECT Ticker, AVG(Close) as avg_close FROM stocks GROUP BY Ticker")
        print("Average close prices using Spark:")
        result.show()
        spark.stop()
    except Exception as e:
        print(f"Spark not available: {e}")

        # If insufficient history for sequence length, force no-train mode
        if len(historical_data) < args.sequence_length + 1:
            print(f"Insufficient history ({len(historical_data)} points) for sequence length {args.sequence_length}; switching to --no-train mode.")
            args.no_train = True

    # Daemon mode (run after initial data + aggregation, skip training & plotting unless live requested separately)
    if args.daemon:
        print(f"--daemon: starting realtime polling every {args.daemon_interval}s; logging to {args.daemon_log}")
        run_daemon(ticker=ticker,
                   interval=args.daemon_interval,
                   log_path=args.daemon_log,
                   max_lines=args.daemon_max_lines,
                   output_meta=args.output_meta)
        return

    # If user requested a quick demo/no-train, skip model training and only visualize
    if args.no_train:
        print("--no-train: skipping model training and prediction; showing historical data with real-time annotation.")
        # handle no-plot
        if args.no_plot:
            if args.output_meta and real_time_price is not None:
                write_metadata(args.output_meta, ticker, real_time_price, real_time_ts)
            print('--no-plot specified; skipping plot display.')
            return

        if args.live and args.live > 0:
            live_update_plot(historical_data, ticker, interval_seconds=args.live,
                             marker_size=args.realtime_marker_size,
                             realtime_only=args.plot_realtime_only,
                             output_meta=args.output_meta)
            return
        if args.candlestick and ohlc_data is not None:
            visualize_candlestick_plotly(ohlc_data, ticker=ticker,
                                          realtime_price=real_time_price, realtime_ts=real_time_ts,
                                          save_html=args.save_html,
                                          include_volume=args.include_volume,
                                          hide_realtime=args.hide_realtime,
                                          forecast_candles=0)  # no-train => no forecast overlay
        else:
            visualize_matplotlib(historical_data, predictions=None, ticker=ticker,
                                 realtime_price=real_time_price, realtime_ts=real_time_ts)
        # Multi grid generation in no-train mode if requested
        if args.multi_candle_html:
            visualize_multi_candlestick_grid(tickers, start_date, end_date,
                                             include_volume=args.include_volume,
                                             hide_realtime=args.hide_realtime,
                                             save_html=args.multi_candle_html)
        return

    # Preprocess data
    sequence_length = args.sequence_length
    X, y, scaler = preprocess_data(historical_data, sequence_length)

    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train model
    model = build_lstm_model((X_train.shape[1], 1))
    callbacks = []
    if args.early_stop_patience and args.early_stop_patience > 0:
        callbacks.append(EarlyStopping(monitor='loss', patience=args.early_stop_patience, restore_best_weights=True))
    if args.checkpoint:
        callbacks.append(ModelCheckpoint(filepath=args.checkpoint, monitor='loss', save_best_only=True))
    print(f"Training model: epochs={args.epochs}, batch_size={args.batch_size}")
    model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks)

    # Evaluate
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    print(f"RMSE: {rmse}")

    # Predict future
    last_sequence = X[-1]
    future_predictions = predict_future(model, last_sequence, scaler, days=30)

    # Visualize (include realtime point if available)
    if args.no_plot:
        if args.output_meta and real_time_price is not None:
            write_metadata(args.output_meta, ticker, real_time_price, real_time_ts)
        print('--no-plot specified; skipping plot display.')
        return

    if args.live and args.live > 0:
        live_update_plot(historical_data, ticker, interval_seconds=args.live,
                         marker_size=args.realtime_marker_size,
                         realtime_only=args.plot_realtime_only,
                         output_meta=args.output_meta)
    else:
        if args.candlestick and ohlc_data is not None:
            visualize_candlestick_plotly(ohlc_data, ticker=ticker,
                                          predictions=future_predictions,
                                          realtime_price=real_time_price, realtime_ts=real_time_ts,
                                          save_html=args.save_html,
                                          include_volume=args.include_volume,
                                          hide_realtime=args.hide_realtime,
                                          forecast_candles=args.forecast_candles)
        else:
            visualize_matplotlib(historical_data, future_predictions, ticker,
                                 realtime_price=real_time_price, realtime_ts=real_time_ts)
        if args.multi_candle_html:
            visualize_multi_candlestick_grid(tickers, start_date, end_date,
                                             include_volume=args.include_volume,
                                             hide_realtime=args.hide_realtime,
                                             save_html=args.multi_candle_html)
    # visualize_plotly(historical_data, future_predictions, ticker)  # Uncomment for interactive plot

if __name__ == "__main__":
    main()