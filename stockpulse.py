# ============================================================
#  StockPulse — Time Series Stock Price Forecaster
#  Author : Vighan Raj Verma (@Vighan-coder)
#  GitHub : https://github.com/Vighan-coder/StockPulse
# ============================================================
#
#  SETUP:
#    pip install numpy pandas matplotlib scikit-learn torch yfinance
#
#  RUN:
#    python stockpulse.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────
TICKER      = "AAPL"          # Change to any ticker
SEQ_LEN     = 60              # Look-back window (days)
PRED_STEPS  = 5               # Days ahead to forecast
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-3
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
NHEAD       = 4               # Transformer attention heads
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[StockPulse] Device: {DEVICE}")


# ════════════════════════════════════════════════════════════
#  1. DATA — download or generate synthetic OHLCV
# ════════════════════════════════════════════════════════════
def fetch_data(ticker: str, years: int = 5) -> pd.DataFrame:
    try:
        import yfinance as yf
        end   = pd.Timestamp.today()
        start = end - pd.DateOffset(years=years)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False)
        if df.empty:
            raise ValueError("Empty response")
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        print(f"[StockPulse] Downloaded {len(df)} rows for {ticker}")
        return df
    except Exception as e:
        print(f"[StockPulse] yfinance failed ({e}) — generating synthetic data.")
        return _synthetic_ohlcv(1200)


def _synthetic_ohlcv(n=1200, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    dates  = pd.bdate_range("2019-01-01", periods=n)
    price  = 150.0
    closes = []
    for _ in range(n):
        price += np.random.normal(0.05, 1.8)
        price  = max(price, 10)
        closes.append(price)
    closes = np.array(closes)
    df = pd.DataFrame({
        "Open"   : closes * np.random.uniform(0.98, 1.00, n),
        "High"   : closes * np.random.uniform(1.00, 1.03, n),
        "Low"    : closes * np.random.uniform(0.97, 1.00, n),
        "Close"  : closes,
        "Volume" : np.random.randint(5_000_000, 80_000_000, n).astype(float),
    }, index=dates)
    return df


# ════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"]    = df["Close"].pct_change()
    df["MA7"]       = df["Close"].rolling(7).mean()
    df["MA21"]      = df["Close"].rolling(21).mean()
    df["MA50"]      = df["Close"].rolling(50).mean()
    df["Volatility"]= df["Return"].rolling(14).std()
    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - 100 / (1 + rs)
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df = df.dropna()
    return df


# ════════════════════════════════════════════════════════════
#  3. DATASET
# ════════════════════════════════════════════════════════════
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_sequences(scaled, seq_len, pred_steps, close_idx=3):
    X, y = [], []
    for i in range(len(scaled) - seq_len - pred_steps + 1):
        X.append(scaled[i : i + seq_len])
        y.append(scaled[i + seq_len : i + seq_len + pred_steps, close_idx])
    return np.array(X), np.array(y)


# ════════════════════════════════════════════════════════════
#  4. MODEL — LSTM + Transformer Hybrid
# ════════════════════════════════════════════════════════════
class LSTMTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead, pred_steps):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True, dropout=0.2)
        enc_layer   = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, pred_steps)

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, H)
        out     = self.transformer(out) # (B, T, H)
        out     = out[:, -1, :]        # last timestep
        return self.fc(out)             # (B, pred_steps)


# ════════════════════════════════════════════════════════════
#  5. TRAIN / EVALUATE
# ════════════════════════════════════════════════════════════
def train(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total = 0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            total += criterion(pred, yb).item()
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
    return total / len(loader), np.vstack(preds), np.vstack(trues)


# ════════════════════════════════════════════════════════════
#  6. PLOT
# ════════════════════════════════════════════════════════════
def plot_predictions(trues, preds, scaler, close_idx, n_features, title="StockPulse"):
    def inverse(arr):
        dummy = np.zeros((len(arr), n_features))
        dummy[:, close_idx] = arr
        return scaler.inverse_transform(dummy)[:, close_idx]

    t_inv = inverse(trues[:, 0])
    p_inv = inverse(preds[:, 0])

    plt.figure(figsize=(12, 5))
    plt.plot(t_inv, label="Actual", color="white", lw=1.5)
    plt.plot(p_inv, label="Predicted", color="#7cff67", lw=1.5, linestyle="--")
    plt.title(f"{title} — Next-Day Close Price Forecast")
    plt.xlabel("Trading Days"); plt.ylabel("Price ($)")
    plt.legend(); plt.tight_layout()
    plt.savefig("stockpulse_forecast.png", dpi=150)
    plt.show()
    print("[Saved] stockpulse_forecast.png")

    mae  = mean_absolute_error(t_inv, p_inv)
    rmse = math.sqrt(mean_squared_error(t_inv, p_inv))
    print(f"MAE : ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Data ────────────────────────────────────────────────
    raw  = fetch_data(TICKER)
    df   = add_features(raw)
    feat_cols  = ["Open","High","Low","Close","Volume",
                  "Return","MA7","MA21","MA50","Volatility","RSI","MACD"]
    data        = df[feat_cols].values
    n_features  = data.shape[1]
    close_idx   = feat_cols.index("Close")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y   = make_sequences(scaled, SEQ_LEN, PRED_STEPS, close_idx)
    split  = int(len(X) * 0.85)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    train_ds = StockDataset(X_tr, y_tr)
    test_ds  = StockDataset(X_te, y_te)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── Model ───────────────────────────────────────────────
    model     = LSTMTransformer(n_features, HIDDEN_DIM, NUM_LAYERS,
                                NHEAD, PRED_STEPS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = nn.HuberLoss()

    print(f"\n[StockPulse] Training for {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train(model, train_dl, optimizer, criterion)
        scheduler.step()
        if epoch % 10 == 0:
            val_loss, _, _ = evaluate(model, test_dl, criterion)
            print(f"  Epoch {epoch:>3}/{EPOCHS}  train={tr_loss:.5f}  val={val_loss:.5f}")

    # ── Evaluate ────────────────────────────────────────────
    _, preds, trues = evaluate(model, test_dl, criterion)
    plot_predictions(trues, preds, scaler, close_idx, n_features, TICKER)

    torch.save(model.state_dict(), "stockpulse_model.pt")
    print("[Saved] stockpulse_model.pt")
    print("\n[StockPulse] Done!")