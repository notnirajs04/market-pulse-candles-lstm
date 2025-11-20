#Stock Market Analysis and Prediction Analyze historical and real-time stock market data 
# to predict future prices using time-series analysis or machine learning models (e.g., LSTM) and visualize trends.
import argparse
import json
from datetime import datetime
import yfinance as yf
from flask import Flask, jsonify, render_template_string

HTML_TEMPLATE = """
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <title>{{ ticker }} Real-time Price</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    .price { font-size: 2rem; color: #ff6600; }
    .meta { color: #555; margin-top: 0.5rem; }
    .container { max-width: 600px; }
    footer { margin-top: 2rem; font-size: 0.8rem; color: #888; }
  </style>
  <script>
    async function refresh() {
      const r = await fetch('/api/price');
      const data = await r.json();
      document.getElementById('price').textContent = data.price.toFixed(2);
      document.getElementById('timestamp').textContent = data.timestamp;
      document.getElementById('fetched_at').textContent = data.fetched_at;
    }
    setInterval(refresh, 10000);
    window.onload = refresh;
  </script>
</head>
<body>
  <div class='container'>
    <h1>{{ ticker }} Real-time Price</h1>
    <div class='price' id='price'>Loading...</div>
    <div class='meta'>Timestamp: <span id='timestamp'>...</span></div>
    <div class='meta'>Fetched At (UTC): <span id='fetched_at'>...</span></div>
    <button onclick='refresh()'>Refresh Now</button>
    <footer>Data source: yfinance | Auto-refresh every 10s</footer>
  </div>
</body>
</html>
"""

def fetch_realtime(ticker: str):
    t = yf.Ticker(ticker)
    hist = t.history(period='1d')
    if hist.empty:
        return None
    price = float(hist['Close'].iloc[-1])
    ts = hist.index[-1]
    return {
        'ticker': ticker,
        'price': price,
        'timestamp': ts.isoformat(),
        'source': 'yfinance',
        'fetched_at': datetime.utcnow().isoformat() + 'Z'
    }


def create_app(ticker: str):
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE, ticker=ticker)

    @app.route('/api/price')
    def api_price():
        data = fetch_realtime(ticker)
        if data is None:
            return jsonify({'error': 'No data'}), 503
        return jsonify(data)

    return app


def main():
    parser = argparse.ArgumentParser(description='Flask web UI for real-time stock price')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol')
    parser.add_argument('--port', type=int, default=5050, help='Port to serve on')
    parser.add_argument('--once', action='store_true', help='Fetch once and print JSON (do not start server)')
    parser.add_argument('--no-train', action='store_true', help='Do not train the model')
    parser.add_argument('--candlestick', action='store_true', help='Show candlestick chart')
    parser.add_argument('--include-volume', action='store_true', help='Include volume in the chart')
    parser.add_argument('--save-html', type=str, help='Save the chart as HTML file')
    parser.add_argument('--forecast-candles', type=int, help='Number of candles to forecast')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--multi-candle-html', type=str, help='HTML file for multi-candle view')
    parser.add_argument('--hide-realtime', action='store_true', help='Hide real-time price')
    args = parser.parse_args()

    if args.once:
        data = fetch_realtime(args.ticker)
        print(json.dumps(data, indent=2))
        return

    app = create_app(args.ticker)
    app.run(host='0.0.0.0', port=args.port)

if __name__ == '__main__':
    main()
