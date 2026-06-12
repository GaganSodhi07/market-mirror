
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Market Mirror", page_icon="📈", layout="wide")
st.title("📈 Market Mirror")
st.caption("Find historical setups similar to today — and see what happened next.")

with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_input("Ticker symbol", value="AAPL",
                                  help="US: AAPL, TSLA | India: RELIANCE, INFY")
    period       = st.selectbox("Historical period", ["2y","3y","5y"], index=2)
    forward_days = st.selectbox("Forward horizon (days)", [5,10,20], index=1)
    k_matches    = st.slider("Similar instances (K)", 10, 40, 20)
    run_button   = st.button("Run analysis", type="primary",
                              use_container_width=True)

forward_col = f"fwd_{forward_days}d"

SECTOR_ETF_MAP = {
    "AAPL":"XLK","MSFT":"XLK","GOOGL":"XLK","NVDA":"XLK","META":"XLK",
    "AMZN":"XLY","TSLA":"XLY","JPM":"XLF","BAC":"XLF","GS":"XLF",
    "XOM":"XLE","CVX":"XLE","JNJ":"XLV","PFE":"XLV",
    "_US_DEFAULT":"SPY","_IN_DEFAULT":"^NSEI",
}

def get_sector_etf(symbol):
    base = symbol.replace(".NS","").replace(".BO","")
    if base in SECTOR_ETF_MAP:
        return SECTOR_ETF_MAP[base]
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return SECTOR_ETF_MAP["_IN_DEFAULT"]
    return SECTOR_ETF_MAP["_US_DEFAULT"]

def resolve_ticker(user_input):
    symbol = user_input.strip().upper()
    if "." in symbol:
        return symbol
    for suffix in ["", ".NS", ".BO"]:
        try:
            t = yf.Ticker(symbol + suffix)
            h = t.history(period="5d")
            if len(h) > 0:
                return symbol + suffix
        except Exception:
            continue
    raise ValueError(f"Cannot resolve ticker: {user_input}")

def fetch_data(symbol, period="5y"):
    sector_etf = get_sector_etf(symbol)
    stock = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    stock.index = stock.index.tz_localize(None)
    etf   = yf.Ticker(sector_etf).history(period=period, auto_adjust=True)
    etf.index = etf.index.tz_localize(None)
    common = stock.index.intersection(etf.index)
    return stock.loc[common], etf.loc[common]

# ── Pure pandas/numpy indicator functions ──────────────────────────────
def calc_rsi(close, length=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(length).mean()
    loss  = (-delta.clip(upper=0)).rolling(length).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_ema(close, length):
    return close.ewm(span=length, adjust=False).mean()

def calc_sma(close, length):
    return close.rolling(length).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast   = calc_ema(close, fast)
    ema_slow   = calc_ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line= calc_ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

ddef calc_bbands(close, length=20, std=2):
    mid   = calc_sma(close, length)
    sigma = close.rolling(length).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    return upper, mid, lower

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    d = k.rolling(d_period).mean()
    return k, d

def calc_adx(high, low, close, length=14):
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional movement
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm,  index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    # Smoothed averages (Wilder)
    atr      = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/length,  adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()

    return adx, plus_di, minus_di

def build_features(stock_df, etf_df, fwd_days=10):
    df     = stock_df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # RSI
    df["rsi"]       = calc_rsi(close, 14)
    df["rsi_slope"] = df["rsi"].diff(3)
    df["rsi_zone"]  = pd.cut(df["rsi"],
                              bins=[0,30,45,55,70,100],
                              labels=[0,1,2,3,4]).astype(float)

    # MACD
    macd_line, macd_sig, macd_hist = calc_macd(close)
    df["macd_line"]       = macd_line
    df["macd_signal"]     = macd_sig
    df["macd_hist"]       = macd_hist
    df["macd_hist_slope"] = df["macd_hist"].diff(2)
    df["macd_cross"]      = 0
    df.loc[(macd_line > macd_sig) &
           (macd_line.shift(1) <= macd_sig.shift(1)), "macd_cross"] =  1
    df.loc[(macd_line < macd_sig) &
           (macd_line.shift(1) >= macd_sig.shift(1)), "macd_cross"] = -1

    # EMA / SMA
    df["ema20"]  = calc_ema(close, 20)
    df["ema50"]  = calc_ema(close, 50)
    df["ema200"] = calc_ema(close, 200)
    df["sma20"]  = calc_sma(close, 20)
    df["sma50"]  = calc_sma(close, 50)

    df["price_vs_ema20"]  = (close - df["ema20"])  / df["ema20"]  * 100
    df["price_vs_ema50"]  = (close - df["ema50"])  / df["ema50"]  * 100
    df["price_vs_ema200"] = (close - df["ema200"]) / df["ema200"] * 100

    df["ema_cross"] = 0
    df.loc[(df["ema20"] > df["ema50"]) &
           (df["ema20"].shift(1) <= df["ema50"].shift(1)), "ema_cross"] =  1
    df.loc[(df["ema20"] < df["ema50"]) &
           (df["ema20"].shift(1) >= df["ema50"].shift(1)), "ema_cross"] = -1

    df["sma_cross"] = 0
    df.loc[(df["sma20"] > df["sma50"]) &
           (df["sma20"].shift(1) <= df["sma50"].shift(1)), "sma_cross"] =  1
    df.loc[(df["sma20"] < df["sma50"]) &
           (df["sma20"].shift(1) >= df["sma50"].shift(1)), "sma_cross"] = -1

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calc_bbands(close, 20, 2)
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_mid"]   = bb_mid
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid * 100
    df["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # Volume
    df["vol_sma20"]  = calc_sma(volume.astype(float), 20)
    df["vol_zscore"] = (volume - df["vol_sma20"]) / (
        volume.rolling(20).std().replace(0, np.nan))
    df["vol_ratio"]  = volume / df["vol_sma20"]

    # Candlestick patterns
    body         = (close - df["Open"]).abs()
    candle_range = (high - low).replace(0, np.nan)
    df["doji"]   = (body / candle_range < 0.1).astype(int)
    lower_wick   = df[["Open","Close"]].min(axis=1) - low
    upper_wick   = high - df[["Open","Close"]].max(axis=1)
    df["hammer"] = (
        (lower_wick > 2 * body) &
        (upper_wick < body) &
        (candle_range > 0)
    ).astype(int)
    df["engulfing_bull"] = (
        (df["Open"] < close.shift(1)) &
        (close > df["Open"].shift(1)) &
        (close.shift(1) < df["Open"].shift(1))
    ).astype(int)
    df["engulfing_bear"] = (
        (df["Open"] > close.shift(1)) &
        (close < df["Open"].shift(1)) &
        (close.shift(1) > df["Open"].shift(1))
    ).astype(int)

    # Stochastic Oscillator
    stoch_k, stoch_d    = calc_stochastic(high, low, close)
    df["stoch_k"]       = stoch_k
    df["stoch_d"]       = stoch_d
    df["stoch_kd_diff"] = stoch_k - stoch_d   # positive = bullish cross
    df["stoch_zone"]    = pd.cut(stoch_k,
                                  bins=[0, 20, 45, 55, 80, 100],
                                  labels=[0, 1, 2, 3, 4]).astype(float)

    # ADX
    adx, plus_di, minus_di = calc_adx(high, low, close)
    df["adx"]           = adx
    df["adx_plus_di"]   = plus_di
    df["adx_minus_di"]  = minus_di
    df["adx_trend"]     = pd.cut(adx,
                                  bins=[0, 20, 25, 40, 100],
                                  labels=[0, 1, 2, 3]).astype(float)
    # DI crossover: +1 when +DI crosses above -DI (bullish), -1 when below
    df["di_cross"]      = 0
    df.loc[(plus_di > minus_di) &
           (plus_di.shift(1) <= minus_di.shift(1)), "di_cross"] =  1
    df.loc[(plus_di < minus_di) &
           (plus_di.shift(1) >= minus_di.shift(1)), "di_cross"] = -1

    # Sector correlation
    df["sector_corr"] = (
        close.pct_change().rolling(21).corr(etf_df["Close"].pct_change())
    )

    # Forward returns
    df[f"fwd_{fwd_days}d"] = close.pct_change(fwd_days).shift(-fwd_days) * 100

    feature_cols = [
        "rsi","rsi_slope","rsi_zone",
        "macd_hist","macd_hist_slope","macd_cross",
        "price_vs_ema20","price_vs_ema50","price_vs_ema200",
        "ema_cross","sma_cross",
        "bb_width","bb_pct_b",
        "vol_zscore","vol_ratio",
        "doji","hammer","engulfing_bull","engulfing_bear",
        "sector_corr",
    ]
    df = df.dropna(subset=feature_cols)
    return df, feature_cols

def run_clustering(X_scaled, max_k=8):
    best_k, best_score = 2, -1
    for k in range(2, max_k+1):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        s   = silhouette_score(X_scaled, lbl)
        if s > best_score:
            best_k, best_score = k, s
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    return km_final.fit_predict(X_scaled), best_k

def find_similar(features_df, feature_cols, scaler, k=20, forward_col="fwd_10d"):
    X  = scaler.transform(features_df[feature_cols])
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X[[-1]])
    dists, idxs = dists[0][1:k+1], idxs[0][1:k+1]
    rows = []
    for d, i in zip(dists, idxs):
        r   = features_df.iloc[i]
        fwd = r.get(forward_col, np.nan)
        rows.append({
            "date"      : features_df.index[i],
            "similarity": round((1-d)*100, 1),
            "rsi"       : round(r["rsi"], 1),
            "macd_hist" : round(r["macd_hist"], 3),
            "bb_pct_b"  : round(r["bb_pct_b"], 3),
            "cluster"   : int(r["cluster"]),
            forward_col : round(fwd, 2) if not np.isnan(fwd) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("similarity", ascending=False)

def train_rf(features_df, feature_cols, scaler, forward_col):
    rf_df = features_df[feature_cols+[forward_col]].dropna().copy()
    rf_df["label"] = (rf_df[forward_col] > 0).astype(int)
    X     = scaler.transform(rf_df[feature_cols])
    y     = rf_df["label"]
    split = int(len(X)*0.8)
    rf    = RandomForestClassifier(n_estimators=200, max_depth=6,
                                    min_samples_leaf=20, random_state=42,
                                    class_weight="balanced")
    rf.fit(X[:split], y.iloc[:split])
    return rf, rf.score(X[split:], y.iloc[split:])*100

def describe_setup(today, similar_df, symbol, forward_col):
    fwd    = similar_df[forward_col].dropna()
    up_pct = (fwd > 0).mean() * 100
    avg    = fwd.mean()
    rsi    = today["rsi"]
    rsi_txt = ("oversold"      if rsi < 30 else
               "weakening"     if rsi < 45 else
               "neutral"       if rsi < 55 else
               "strengthening" if rsi < 70 else "overbought")
    hist    = today["macd_hist"]
    slope   = today["macd_hist_slope"]
    macd_txt = ("MACD bullish and accelerating"     if hist>0 and slope>0 else
                "MACD bullish but losing momentum"  if hist>0 else
                "MACD bearish but momentum slowing" if slope>0 else
                "MACD bearish and accelerating down")
    bb      = today["bb_pct_b"]
    bb_txt  = ("near lower band — oversold zone"    if bb < 0.2 else
               "near upper band — overbought zone"  if bb > 0.8 else
               f"mid-band (BB %B: {bb:.2f})")
    vol_txt = ("above-average volume" if today["vol_zscore"] > 0.5 else
               "below-average volume" if today["vol_zscore"] < -0.5 else
               "average volume")
    d       = today["price_vs_ema50"]
    ma_txt  = (f"price {d:.1f}% above EMA50"       if d > 2 else
               f"price {abs(d):.1f}% below EMA50"  if d < -2 else
               "price hugging EMA50")
    bias    = (f"historically bullish — {up_pct:.0f}% of similar setups rose"
               if up_pct >= 60 else
               f"historically bearish — {100-up_pct:.0f}% of similar setups fell"
               if up_pct <= 40 else
               f"historically mixed — {up_pct:.0f}% up / {100-up_pct:.0f}% down")
    return (
        f"{symbol} is currently **{rsi_txt}** (RSI {rsi:.0f}). "
        f"{macd_txt}, with price {bb_txt}. "
        f"Volume is {vol_txt} and {ma_txt}. "
        f"Among the {len(fwd)} most similar historical setups, "
        f"the average {forward_days}-day return was **{avg:.1f}%**. "
        f"The setup is {bias}."
    )

# ── Main ──────────────────────────────────────────────────────────────
if run_button:
    try:
        symbol = resolve_ticker(ticker_input)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner(f"Fetching data for {symbol}..."):
        stock_df, etf_df = fetch_data(symbol, period)

    with st.spinner("Building feature matrix..."):
        features_df, feature_cols = build_features(stock_df, etf_df, forward_days)

    with st.spinner("Running clustering and similarity engine..."):
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(features_df[feature_cols])
        clusters, best_k = run_clustering(X_scaled)
        features_df["cluster"] = clusters

        pca   = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        features_df["pca1"] = X_pca[:,0]
        features_df["pca2"] = X_pca[:,1]

        similar_df       = find_similar(features_df, feature_cols, scaler,
                                         k=k_matches, forward_col=forward_col)
        rf, test_acc     = train_rf(features_df, feature_cols, scaler, forward_col)
        today            = features_df.iloc[-1]
        today_vec        = scaler.transform(features_df[feature_cols].iloc[[-1]])
        proba            = rf.predict_proba(today_vec)[0]

    fwd_returns = similar_df[forward_col].dropna()
    up_pct      = (fwd_returns > 0).mean() * 100
    avg_ret     = fwd_returns.mean()

    st.subheader(f"{symbol}  —  {features_df.index[-1].strftime('%d %b %Y')}")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Current regime",   f"Regime {int(today['cluster'])}")
    c2.metric(f"Avg {forward_days}d return", f"{avg_ret:.1f}%", delta=f"{avg_ret:.1f}%")
    c3.metric("Bullish instances", f"{up_pct:.0f}%")
    c4.metric("RF bullish prob",   f"{proba[1]*100:.0f}%")
    c5.metric("RF test accuracy",  f"{test_acc:.0f}%")

    st.info(describe_setup(today, similar_df, symbol, forward_col))

    st.subheader("Current indicator state")
    b1,b2,b3,b4,b5,b6 = st.columns(6)
    b1.metric("RSI",         f"{today['rsi']:.1f}")
    b2.metric("MACD hist",   f"{today['macd_hist']:.3f}")
    b3.metric("BB %B",       f"{today['bb_pct_b']:.2f}")
    b4.metric("Vol z-score", f"{today['vol_zscore']:.2f}")
    b5.metric("vs EMA50",    f"{today['price_vs_ema50']:.1f}%")
    b6.metric("Sector corr", f"{today['sector_corr']:.2f}")

    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "Similar instances","Return distribution",
        "Regime clusters","Feature importance","Regression",
        "Monte Carlo"
    ])

    with tab1:
        st.subheader(f"Top {k_matches} similar historical instances")
        st.dataframe(similar_df, use_container_width=True)

        st.subheader("Price behaviour — top 5 matches (normalised to 100)")
        top5 = similar_df.head(5)
        fig  = make_subplots(rows=1, cols=5,
                              subplot_titles=[
                                  f"{r['date'].strftime('%Y-%m-%d')} | {r['similarity']}%"
                                  for _,r in top5.iterrows()
                              ])
        for i,(_,match) in enumerate(top5.iterrows(), 1):
            pos   = features_df.index.get_loc(match["date"])
            start = max(0, pos-30)
            end   = min(len(stock_df), pos+30)
            chunk = stock_df.loc[features_df.index[start:end]]["Close"]
            base  = chunk.iloc[min(30, pos-start)]
            chunk = chunk / base * 100
            fig.add_trace(go.Scatter(x=list(range(len(chunk))), y=chunk.values,
                                      mode="lines",
                                      line=dict(width=1.5, color="#7F77DD"),
                                      showlegend=False), row=1, col=i)
            fig.add_vline(x=min(30, pos-start), line_dash="dash",
                          line_color="#E24B4A", row=1, col=i)
        fig.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"{forward_days}-day forward return distribution")
        fig = go.Figure(go.Histogram(
            x=fwd_returns, nbinsx=15,
            marker_color=["#5DCAA5" if v>0 else "#E24B4A" for v in fwd_returns],
            opacity=0.85
        ))
        fig.add_vline(x=0,       line_dash="dash",  line_color="#888780")
        fig.add_vline(x=avg_ret, line_dash="dot",   line_color="#7F77DD",
                      annotation_text=f"Avg {avg_ret:.1f}%",
                      annotation_position="top right")
        fig.update_layout(xaxis_title=f"{forward_days}-day return (%)",
                           yaxis_title="Count", height=400,
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader(f"Market regimes — PCA (K={best_k})")
        var      = pca.explained_variance_ratio_ * 100
        colors_c = ["#5DCAA5","#7F77DD","#D85A30","#EF9F27","#378ADD","#D4537E"]
        fig = go.Figure()
        for c in sorted(features_df["cluster"].unique()):
            mask = features_df["cluster"] == c
            sub  = features_df[mask]
            fig.add_trace(go.Scatter(
                x=sub["pca1"], y=sub["pca2"], mode="markers",
                marker=dict(size=5, color=colors_c[c], opacity=0.6),
                name=f"Regime {c}",
                text=sub.index.strftime("%Y-%m-%d"),
                hovertemplate=f"<b>Regime {c}</b><br>Date: %{{text}}<extra></extra>"
            ))
        fig.add_trace(go.Scatter(
            x=[today["pca1"]], y=[today["pca2"]], mode="markers",
            marker=dict(size=16, color="#E24B4A", symbol="star",
                        line=dict(width=1, color="white")),
            name="Today"
        ))
        fig.update_layout(
            title=f"PC1={var[0]:.1f}%  PC2={var[1]:.1f}% variance explained",
            xaxis_title=f"PC1 ({var[0]:.1f}%)",
            yaxis_title=f"PC2 ({var[1]:.1f}%)",
            height=500, plot_bgcolor="white", paper_bgcolor="white"
        )
        col_a, col_b = st.columns([2,1])
        col_a.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.markdown("**Cluster means**")
            st.dataframe(
                features_df.groupby("cluster")[
                    ["rsi","macd_hist","bb_pct_b","price_vs_ema50","sector_corr"]
                ].mean().round(2),
                use_container_width=True
            )
            st.markdown("**Silhouette scores**")
            ks, silhs = [], []
            for k_ in range(2,9):
                km_ = KMeans(n_clusters=k_, random_state=42, n_init=5)
                lb_ = km_.fit_predict(X_scaled)
                ks.append(k_)
                silhs.append(silhouette_score(X_scaled, lb_))
            fig_e = go.Figure(go.Scatter(
                x=ks, y=silhs, mode="lines+markers",
                line=dict(color="#7F77DD")
            ))
            fig_e.add_vline(x=best_k, line_dash="dash", line_color="#E24B4A",
                            annotation_text=f"K={best_k}")
            fig_e.update_layout(height=220, margin=dict(t=20,b=30),
                                  plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_e, use_container_width=True)

    with tab4:
        st.subheader("Random Forest — feature importance")
        imp_df = pd.DataFrame({
            "feature"   : feature_cols,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=True)
        colors_i = ["#7F77DD" if i >= len(imp_df)-5 else "#D3D1C7"
                    for i in range(len(imp_df))]
        fig = go.Figure(go.Bar(
            x=imp_df["importance"], y=imp_df["feature"],
            orientation="h", marker_color=colors_i
        ))
        fig.update_layout(xaxis_title="Importance", height=560,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=160))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"RF test accuracy: {test_acc:.1f}% | "
                   f"Low accuracy is expected — markets are noisy.")

    with tab5:
        st.subheader("Regression within bearish regime")
        reg_df = features_df[
            features_df["cluster"]==0
        ][["rsi","bb_pct_b",forward_col]].dropna()
        fig = make_subplots(rows=1, cols=2,
                             subplot_titles=("RSI vs forward return",
                                             "BB %B vs forward return"))
        for col_n, col_i in [("rsi",1),("bb_pct_b",2)]:
            Xr = reg_df[[col_n]].values
            yr = reg_df[forward_col].values
            lr = LinearRegression().fit(Xr, yr)
            xl = np.linspace(Xr.min(), Xr.max(), 100)
            yl = lr.predict(xl.reshape(-1,1))
            fig.add_trace(go.Scatter(x=reg_df[col_n], y=reg_df[forward_col],
                                      mode="markers",
                                      marker=dict(size=4, color="#B5D4F4", opacity=0.5),
                                      showlegend=False), row=1, col=col_i)
            fig.add_trace(go.Scatter(x=xl, y=yl, mode="lines",
                                      line=dict(color="#E24B4A", width=2),
                                      name=f"R²={lr.score(Xr,yr):.3f}"),
                           row=1, col=col_i)
        fig.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Low R² confirms no single indicator predicts returns reliably "
                   "— justifying the multi-feature similarity approach.")
                   with tab6:
        st.subheader("Monte Carlo simulation — forward price paths")
        st.caption(
            "Simulates 500 possible price paths by sampling from the "
            "forward returns of similar historical instances."
        )

        fwd_clean  = similar_df[forward_col].dropna().values
        spot_price = stock_df["Close"].iloc[-1]
        n_sims     = 500
        n_days     = forward_days

        if len(fwd_clean) < 5:
            st.warning("Not enough similar instances for simulation.")
        else:
            # Convert percentage returns to daily step returns
            daily_returns = fwd_clean / 100 / n_days

            # Run simulations
            np.random.seed(42)
            simulations = np.zeros((n_sims, n_days + 1))
            simulations[:, 0] = spot_price

            for day in range(1, n_days + 1):
                sampled = np.random.choice(daily_returns, size=n_sims, replace=True)
                simulations[:, day] = simulations[:, day-1] * (1 + sampled)

            final_prices = simulations[:, -1]
            pct_up       = (final_prices > spot_price).mean() * 100
            median_price = np.median(final_prices)
            p10          = np.percentile(final_prices, 10)
            p90          = np.percentile(final_prices, 90)

            # Metrics
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Current price",  f"${spot_price:.2f}")
            m2.metric("Median outcome", f"${median_price:.2f}",
                      delta=f"{((median_price/spot_price)-1)*100:.1f}%")
            m3.metric("10th percentile (bear)", f"${p10:.2f}")
            m4.metric("90th percentile (bull)", f"${p90:.2f}")

            # Fan chart — plot all paths
            fig = go.Figure()

            # All simulation paths (faint)
            for i in range(min(200, n_sims)):
                fig.add_trace(go.Scatter(
                    x=list(range(n_days + 1)),
                    y=simulations[i],
                    mode="lines",
                    line=dict(width=0.4, color="rgba(127,119,221,0.15)"),
                    showlegend=False,
                    hoverinfo="skip"
                ))

            # Percentile bands
            p10_path = np.percentile(simulations, 10, axis=0)
            p50_path = np.percentile(simulations, 50, axis=0)
            p90_path = np.percentile(simulations, 90, axis=0)

            fig.add_trace(go.Scatter(
                x=list(range(n_days + 1)), y=p90_path,
                mode="lines", name="90th percentile",
                line=dict(width=2, color="#5DCAA5", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=list(range(n_days + 1)), y=p50_path,
                mode="lines", name="Median path",
                line=dict(width=2.5, color="#7F77DD")
            ))
            fig.add_trace(go.Scatter(
                x=list(range(n_days + 1)), y=p10_path,
                mode="lines", name="10th percentile",
                line=dict(width=2, color="#E24B4A", dash="dash")
            ))

            # Current price line
            fig.add_hline(y=spot_price, line_dash="dot",
                          line_color="#888780", line_width=1,
                          annotation_text="Current price",
                          annotation_position="right")

            fig.update_layout(
                title=f"{n_sims} simulated price paths over {n_days} days "
                      f"(sampled from {len(fwd_clean)} similar historical instances)",
                xaxis_title="Trading days forward",
                yaxis_title="Simulated price",
                height=500,
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Final price distribution
            fig2 = go.Figure(go.Histogram(
                x=final_prices, nbinsx=30,
                marker_color=["#5DCAA5" if v > spot_price else "#E24B4A"
                              for v in final_prices],
                opacity=0.85
            ))
            fig2.add_vline(x=spot_price,    line_dash="dash",
                           line_color="#888780",
                           annotation_text="Current price",
                           annotation_position="top left")
            fig2.add_vline(x=median_price,  line_dash="dot",
                           line_color="#7F77DD",
                           annotation_text=f"Median ${median_price:.2f}",
                           annotation_position="top right")
            fig2.update_layout(
                title=f"Distribution of simulated prices after {n_days} days",
                xaxis_title="Simulated price",
                yaxis_title="Count",
                height=350,
                plot_bgcolor="white",
                paper_bgcolor="white"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.caption(
                f"{pct_up:.0f}% of simulated paths ended above current price. "
                f"Simulation is based on forward returns from the {len(fwd_clean)} "
                f"most similar historical setups — not random walk assumptions."
            )

else:
    st.info("Enter a ticker and click **Run analysis** to begin.")
    st.markdown("""
    **How it works:**
    - Computes 20 technical features from price history using pure pandas/numpy
    - K-Means clustering identifies market regimes (elbow method selects K)
    - Finds K most similar historical setups via cosine distance (k-NN)
    - Shows distribution of what happened next
    - Random Forest cross-validates the signal

    **Supported:** Any US stock (AAPL, TSLA, NVDA) or Indian stock (RELIANCE, INFY, TCS)
    """)
st.divider()
st.caption("Prepared by Gagan Sodhi | HTW Berlin, Germany")

