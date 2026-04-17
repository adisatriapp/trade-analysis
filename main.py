"""
╔══════════════════════════════════════════════════════════╗
║         STOCK ANALYZER PRO  v2.0                        ║
║         Analisa Saham Teknikal + Fundamental             ║
╚══════════════════════════════════════════════════════════╝
Dibuat dengan pendekatan modular, robust, dan profesional.
Cocok untuk saham BEI (.JK) maupun global.
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import ta

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

# ─────────────────────────────────────────────
# KONFIGURASI GLOBAL
# ─────────────────────────────────────────────
console = Console()

WEIGHTS = {
    "technical": 0.55,   # 55% bobot teknikal
    "fundamental": 0.45, # 45% bobot fundamental
}

THRESHOLDS = {
    "bullish": 62,
    "bearish": 38,
}


# ═══════════════════════════════════════════════
# SECTION 1: DATA FETCHER
# ═══════════════════════════════════════════════

def fetch_data(symbol: str, period: str = "120d") -> tuple[pd.DataFrame, dict]:
    """Ambil data historis dan info fundamental dari yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            console.print(f"[bold red]❌ Data untuk '{symbol}' tidak ditemukan. Periksa kode saham.[/]")
            sys.exit(1)

        if len(df) < 30:
            console.print(f"[bold yellow]⚠ Data terlalu sedikit ({len(df)} hari). Hasil mungkin kurang akurat.[/]")

        info = ticker.info
        return df, info

    except Exception as e:
        console.print(f"[bold red]❌ Error mengambil data: {e}[/]")
        sys.exit(1)


# ═══════════════════════════════════════════════
# SECTION 2: TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung semua indikator teknikal ke dalam DataFrame."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend ──
    df["EMA9"]  = ta.trend.ema_indicator(close, window=9)
    df["EMA21"] = ta.trend.ema_indicator(close, window=21)
    df["EMA50"] = ta.trend.ema_indicator(close, window=50)
    df["SMA200"] = close.rolling(window=200).mean()

    # ── Momentum ──
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["STOCH_K"] = ta.momentum.StochasticOscillator(high, low, close).stoch()
    df["STOCH_D"] = ta.momentum.StochasticOscillator(high, low, close).stoch_signal()

    # ── MACD ──
    macd_obj     = ta.trend.MACD(close)
    df["MACD"]   = macd_obj.macd()
    df["MACD_sig"] = macd_obj.macd_signal()
    df["MACD_hist"] = macd_obj.macd_diff()

    # ── Bollinger Bands ──
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_mid"]   = bb.bollinger_mavg()

    # ── Volume ──
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["Vol_MA20"] = vol.rolling(window=20).mean()

    # ── ATR (Volatilitas) ──
    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    return df


def score_technical(df: pd.DataFrame) -> tuple[float, dict]:
    """
    Hitung skor teknikal dengan sistem poin berbobot.
    Return: (skor 0-10, dict detail sinyal)
    """
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    score = 0.0
    signals = {}

    # ── 1. Golden/Death Cross EMA9/EMA21 (0-2 poin) ──
    if last["EMA9"] > last["EMA21"]:
        score += 2.0
        signals["EMA Cross"] = ("✅ Golden Cross (EMA9 > EMA21)", "green")
    else:
        signals["EMA Cross"] = ("❌ Death Cross (EMA9 < EMA21)", "red")

    # ── 2. Posisi vs EMA50 (0-1 poin) ──
    if last["Close"] > last["EMA50"]:
        score += 1.0
        signals["EMA50 Trend"] = ("✅ Harga di atas EMA50", "green")
    else:
        signals["EMA50 Trend"] = ("⚠ Harga di bawah EMA50", "yellow")

    # ── 3. MACD Signal (0-2 poin, dengan penalti) ──
    macd_cross_up = last["MACD"] > last["MACD_sig"] and prev["MACD"] <= prev["MACD_sig"]
    if macd_cross_up:
        score += 2.0
        signals["MACD"] = ("✅ MACD Bullish Crossover (Fresh)", "green")
    elif last["MACD"] > last["MACD_sig"]:
        score += 1.2
        signals["MACD"] = ("✅ MACD di atas Signal Line", "green")
    elif last["MACD_hist"] > prev["MACD_hist"]:
        score += 0.5
        signals["MACD"] = ("⚠ Histogram MACD Menguat (Divergen Bullish)", "yellow")
    else:
        signals["MACD"] = ("❌ MACD Bearish", "red")

    # ── 4. RSI (0-2 poin, zona ideal 45-65) ──
    rsi = last["RSI"]
    if 45 < rsi < 65:
        score += 2.0
        signals["RSI"] = (f"✅ RSI Ideal: {rsi:.1f} (Momentum Bullish)", "green")
    elif 30 <= rsi <= 45:
        score += 1.0
        signals["RSI"] = (f"⚠ RSI Oversold Area: {rsi:.1f} (Potensi Rebound)", "yellow")
    elif 65 <= rsi <= 75:
        score += 0.5
        signals["RSI"] = (f"⚠ RSI Overbought Warning: {rsi:.1f}", "yellow")
    elif rsi > 75:
        score -= 0.5
        signals["RSI"] = (f"❌ RSI Overbought: {rsi:.1f} (Risiko Koreksi)", "red")
    else:  # rsi < 30
        score += 0.5
        signals["RSI"] = (f"⚠ RSI Sangat Oversold: {rsi:.1f} (Potensi Rebound Kuat)", "yellow")

    # ── 5. Bollinger Bands (0-1 poin) ──
    bb_pos = (last["Close"] - last["BB_lower"]) / (last["BB_upper"] - last["BB_lower"]) * 100
    if 20 < bb_pos < 80:
        score += 0.5
        signals["Bollinger"] = (f"✅ Harga di zona tengah BB ({bb_pos:.0f}%)", "green")
    elif bb_pos <= 20:
        score += 1.0
        signals["Bollinger"] = (f"✅ Harga dekat Lower Band ({bb_pos:.0f}%) — Potensi Naik", "green")
    else:
        signals["Bollinger"] = (f"⚠ Harga dekat Upper Band ({bb_pos:.0f}%) — Hati-hati", "yellow")

    # ── 6. Volume Konfirmasi (0-1 poin) ──
    if last["Volume"] > last["Vol_MA20"] * 1.2:
        score += 1.0
        signals["Volume"] = (f"✅ Volume Tinggi ({last['Volume'] / last['Vol_MA20']:.1f}x rata-rata)", "green")
    elif last["Volume"] > last["Vol_MA20"]:
        score += 0.5
        signals["Volume"] = (f"⚠ Volume Cukup ({last['Volume'] / last['Vol_MA20']:.1f}x rata-rata)", "yellow")
    else:
        signals["Volume"] = (f"❌ Volume Lemah ({last['Volume'] / last['Vol_MA20']:.1f}x rata-rata)", "red")

    # Normalisasi ke 0-10
    max_possible = 9.0
    score_normalized = min(max(score, 0), max_possible) / max_possible * 10

    return round(score_normalized, 2), signals


# ═══════════════════════════════════════════════
# SECTION 3: FUNDAMENTAL ANALYSIS
# ═══════════════════════════════════════════════

def score_fundamental(info: dict) -> tuple[float, dict]:
    """
    Hitung skor fundamental berbobot.
    Return: (skor 0-10, dict detail metrik)
    """
    score = 0.0
    metrics = {}

    def safe_get(key, fallback=None):
        val = info.get(key, fallback)
        return val if val is not None else fallback

    # ── 1. Profitabilitas: ROE (0-2.5 poin) ──
    roe = safe_get("returnOnEquity", 0) * 100
    if roe >= 20:
        score += 2.5
        metrics["ROE"] = (f"{roe:.1f}%", "✅ Sangat Profitabel (>20%)", "green")
    elif roe >= 15:
        score += 2.0
        metrics["ROE"] = (f"{roe:.1f}%", "✅ Profitabel Baik (15-20%)", "green")
    elif roe >= 8:
        score += 1.0
        metrics["ROE"] = (f"{roe:.1f}%", "⚠ Cukup (8-15%)", "yellow")
    else:
        metrics["ROE"] = (f"{roe:.1f}%", "❌ Lemah (<8%)", "red")

    # ── 2. Valuasi: P/E Ratio (0-2 poin) ──
    pe = safe_get("trailingPE", 0)
    if 0 < pe < 12:
        score += 2.0
        metrics["P/E Ratio"] = (f"{pe:.1f}x", "✅ Murah Sekali (<12x)", "green")
    elif 12 <= pe < 20:
        score += 1.5
        metrics["P/E Ratio"] = (f"{pe:.1f}x", "✅ Valuasi Wajar (12-20x)", "green")
    elif 20 <= pe < 30:
        score += 0.5
        metrics["P/E Ratio"] = (f"{pe:.1f}x", "⚠ Mahal (20-30x)", "yellow")
    elif pe >= 30:
        metrics["P/E Ratio"] = (f"{pe:.1f}x", "❌ Sangat Mahal (>30x)", "red")
    else:
        metrics["P/E Ratio"] = ("N/A", "⚠ Data tidak tersedia", "dim")

    # ── 3. Valuasi: P/BV (0-1.5 poin) ──
    pbv = safe_get("priceToBook", 0)
    if 0 < pbv < 1:
        score += 1.5
        metrics["P/BV"] = (f"{pbv:.2f}x", "✅ Di bawah Book Value (<1x)", "green")
    elif 1 <= pbv < 2.5:
        score += 1.0
        metrics["P/BV"] = (f"{pbv:.2f}x", "✅ Wajar (1-2.5x)", "green")
    elif 2.5 <= pbv < 4:
        score += 0.3
        metrics["P/BV"] = (f"{pbv:.2f}x", "⚠ Cukup Mahal (2.5-4x)", "yellow")
    else:
        metrics["P/BV"] = (f"{pbv:.2f}x" if pbv else "N/A", "❌ Mahal atau N/A", "red")

    # ── 4. Kesehatan: DER (0-2 poin) ──
    der = safe_get("debtToEquity", 0)
    if 0 <= der < 50:
        score += 2.0
        metrics["DER"] = (f"{der:.0f}%", "✅ Utang Sangat Rendah (<50%)", "green")
    elif 50 <= der < 100:
        score += 1.5
        metrics["DER"] = (f"{der:.0f}%", "✅ Utang Aman (50-100%)", "green")
    elif 100 <= der < 200:
        score += 0.5
        metrics["DER"] = (f"{der:.0f}%", "⚠ Utang Moderat (100-200%)", "yellow")
    else:
        metrics["DER"] = (f"{der:.0f}%", "❌ Utang Tinggi (>200%)", "red")

    # ── 5. Dividend Yield (0-1 poin) ──
    div_yield = safe_get("dividendYield", 0)
    div_yield_pct = div_yield * 100 if div_yield else 0
    if div_yield_pct >= 4:
        score += 1.0
        metrics["Div. Yield"] = (f"{div_yield_pct:.2f}%", "✅ Tinggi (>4%)", "green")
    elif div_yield_pct > 0:
        score += 0.5
        metrics["Div. Yield"] = (f"{div_yield_pct:.2f}%", "⚠ Ada Dividen", "yellow")
    else:
        metrics["Div. Yield"] = ("0%", "❌ Tidak Ada Dividen", "dim")

    # ── Bonus Metrik Tambahan (tidak diskor, hanya info) ──
    eps = safe_get("trailingEps", 0)
    metrics["EPS"] = (f"{eps:,.2f}" if eps else "N/A", "Laba per Saham", "cyan")

    revenue_growth = safe_get("revenueGrowth", None)
    if revenue_growth is not None:
        rg_pct = revenue_growth * 100
        color = "green" if rg_pct > 10 else "yellow" if rg_pct > 0 else "red"
        metrics["Revenue Growth"] = (f"{rg_pct:.1f}%", "Pertumbuhan Pendapatan YoY", color)

    # Normalisasi ke 0-10
    max_possible = 9.0
    score_normalized = min(max(score, 0), max_possible) / max_possible * 10

    return round(score_normalized, 2), metrics


# ═══════════════════════════════════════════════
# SECTION 4: PRICE FORECASTING
# ═══════════════════════════════════════════════

def forecast_price(df: pd.DataFrame, days: int = 3) -> dict:
    """
    Proyeksi harga menggunakan volatilitas historis + ATR.
    Menggunakan pendekatan statistik (1 standar deviasi).
    """
    close    = df["Close"]
    returns  = close.pct_change().dropna()
    last_price = close.iloc[-1]

    # Volatilitas historis 20 hari (annualized, lalu daily)
    hist_vol_daily = returns.tail(20).std()
    expected_move  = last_price * hist_vol_daily * np.sqrt(days)

    # ATR sebagai penyesuaian tambahan (buffer realistis)
    atr = df["ATR"].iloc[-1]
    atr_buffer = atr * 0.5

    # Support & Resistance sederhana (swing high/low 20 hari)
    recent = df.tail(20)
    support    = recent["Low"].min()
    resistance = recent["High"].max()

    # Level Stop Loss & Take Profit berdasarkan ATR
    stop_loss   = last_price - (atr * 2)
    take_profit = last_price + (atr * 3)

    return {
        "last_price": last_price,
        "max_3d": last_price + expected_move + atr_buffer,
        "min_3d": last_price - expected_move - atr_buffer,
        "support": support,
        "resistance": resistance,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": atr,
        "volatility_daily": hist_vol_daily * 100,  # dalam %
    }


# ═══════════════════════════════════════════════
# SECTION 5: SCORING ENGINE & KEPUTUSAN
# ═══════════════════════════════════════════════

def compute_composite_score(tech_score: float, funda_score: float) -> dict:
    """Gabungkan skor teknikal dan fundamental menjadi skor komposit."""
    composite = (
        tech_score * WEIGHTS["technical"] +
        funda_score * WEIGHTS["fundamental"]
    )
    prob_bullish = composite * 10  # ke skala 0-100%

    if prob_bullish >= THRESHOLDS["bullish"]:
        sentiment = "BULLISH"
        color = "green"
        emoji = "🚀"
        recommendation = "BELI / KOLEKSI"
    elif prob_bullish <= THRESHOLDS["bearish"]:
        sentiment = "BEARISH"
        color = "red"
        emoji = "⚠️"
        recommendation = "HINDARI / JUAL"
    else:
        sentiment = "SIDEWAYS / NETRAL"
        color = "yellow"
        emoji = "⚖️"
        recommendation = "WAIT & SEE"

    return {
        "composite": composite,
        "prob": round(prob_bullish, 1),
        "sentiment": sentiment,
        "color": color,
        "emoji": emoji,
        "recommendation": recommendation,
    }


def get_investment_style(funda_score: float, tech_score: float) -> str:
    """Tentukan gaya investasi yang paling cocok."""
    if funda_score >= 7 and tech_score >= 7:
        return "💎 STRONG BUY — Kuat secara teknikal DAN fundamental"
    elif funda_score >= 7 and tech_score < 5:
        return "📦 VALUE INVESTING — Fundamental bagus, tunggu momen teknikal"
    elif tech_score >= 7 and funda_score < 5:
        return "⚡ SHORT-TERM TRADE — Teknikal kuat, fundamental lemah (risiko lebih tinggi)"
    elif funda_score < 4 and tech_score < 4:
        return "🚫 AVOID — Lemah di semua aspek"
    else:
        return "🔍 SELECTIVE — Perlu konfirmasi lebih lanjut sebelum masuk"


# ═══════════════════════════════════════════════
# SECTION 6: OUTPUT / DISPLAY
# ═══════════════════════════════════════════════

def display_header(info: dict, symbol: str):
    name   = info.get("longName", symbol)
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    market = info.get("exchange", "N/A")

    header = Text()
    header.append(f"  {name}\n", style="bold cyan")
    header.append(f"  {symbol}  |  {market}  |  {sector}\n", style="dim")
    header.append(f"  Industri: {industry}", style="dim")
    console.print(Panel(header, border_style="cyan", padding=(0, 1)))


def display_sentiment(result: dict):
    color = result["color"]
    panel_text = (
        f"[bold {color}]{result['emoji']}  {result['sentiment']}[/bold {color}]\n\n"
        f"Probabilitas Kenaikan : [bold {color}]{result['prob']:.1f}%[/bold {color}]\n"
        f"Rekomendasi           : [bold]{result['recommendation']}[/bold]"
    )
    console.print(Panel(panel_text, title="[bold]SENTIMEN KESELURUHAN[/bold]", border_style=color))


def display_score_summary(tech_score: float, funda_score: float):
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Aspek Analisa", style="cyan", width=28)
    table.add_column("Skor (0-10)", justify="center", width=14)
    table.add_column("Rating", justify="center", width=12)

    def rating(s):
        if s >= 7: return "[green]★★★[/green]"
        elif s >= 5: return "[yellow]★★☆[/yellow]"
        else: return "[red]★☆☆[/red]"

    table.add_row("📈 Kekuatan Teknikal",   f"[bold]{tech_score:.1f}[/bold]",   rating(tech_score))
    table.add_row("🏦 Kesehatan Fundamental", f"[bold]{funda_score:.1f}[/bold]", rating(funda_score))
    console.print(table)


def display_technical_signals(signals: dict):
    table = Table(title="SINYAL TEKNIKAL", box=box.SIMPLE_HEAD, padding=(0, 1))
    table.add_column("Indikator",  style="dim cyan", width=18)
    table.add_column("Keterangan", width=50)

    for indicator, (description, color) in signals.items():
        table.add_row(indicator, f"[{color}]{description}[/{color}]")
    console.print(table)


def display_fundamental_metrics(metrics: dict):
    table = Table(title="DATA FUNDAMENTAL", box=box.SIMPLE_HEAD, padding=(0, 1))
    table.add_column("Indikator",  style="dim cyan", width=18)
    table.add_column("Nilai",      justify="right",  width=14)
    table.add_column("Keterangan", width=40)

    for name, data in metrics.items():
        val, desc, color = data
        table.add_row(name, f"[bold]{val}[/bold]", f"[{color}]{desc}[/{color}]")
    console.print(table)


def display_forecast(fc: dict, currency: str = ""):
    lp = fc["last_price"]

    def fmt(x):
        return f"{currency}{x:,.0f}"

    forecast_text = (
        f"Harga Penutupan Terakhir : [bold white]{fmt(lp)}[/bold white]\n\n"
        f"[dim]── Proyeksi {3} Hari ──[/dim]\n"
        f"  Estimasi Tertinggi  : [bold green]{fmt(fc['max_3d'])}[/bold green]\n"
        f"  Estimasi Terendah   : [bold red]{fmt(fc['min_3d'])}[/bold red]\n\n"
        f"[dim]── Level Kunci ──[/dim]\n"
        f"  Resistance 20H      : [yellow]{fmt(fc['resistance'])}[/yellow]\n"
        f"  Support 20H         : [yellow]{fmt(fc['support'])}[/yellow]\n\n"
        f"[dim]── Risk Management ──[/dim]\n"
        f"  Take Profit (3x ATR): [green]{fmt(fc['take_profit'])}[/green]\n"
        f"  Stop Loss (2x ATR)  : [red]{fmt(fc['stop_loss'])}[/red]\n\n"
        f"  ATR (14D)           : {fmt(fc['atr'])}\n"
        f"  Volatilitas Harian  : {fc['volatility_daily']:.2f}%"
    )
    console.print(Panel(forecast_text, title="PROYEKSI HARGA & MANAJEMEN RISIKO", border_style="blue"))


def display_investment_style(style: str, result: dict):
    console.print(Panel(
        f"[bold]{style}[/bold]",
        title="KESIMPULAN & STRATEGI",
        border_style=result["color"],
        padding=(0, 2)
    ))


def display_disclaimer():
    console.print(
        "\n[dim italic]⚠ DISCLAIMER: Analisa ini bersifat edukatif dan tidak merupakan saran investasi.\n"
        "Keputusan investasi sepenuhnya ada di tangan Anda. Selalu lakukan riset mandiri (DYOR).[/dim italic]\n"
    )


# ═══════════════════════════════════════════════
# SECTION 7: MAIN ENTRYPOINT
# ═══════════════════════════════════════════════

def main():
    console.rule("[bold cyan]STOCK ANALYZER PRO v2.0[/bold cyan]")
    console.print()

    # Input
    raw = input("Masukkan kode saham (contoh: BBCA.JK, AAPL): ").strip()
    symbol = raw.upper() if raw else "BBCA.JK"

    console.print(f"\n[dim]⏳ Mengambil data untuk [bold]{symbol}[/bold] ...[/dim]\n")

    # ── Fetch Data ──
    df, info = fetch_data(symbol)

    # ── Technical ──
    df = calculate_technical_indicators(df)
    tech_score, tech_signals = score_technical(df)

    # ── Fundamental ──
    funda_score, funda_metrics = score_fundamental(info)

    # ── Scoring & Decision ──
    result = compute_composite_score(tech_score, funda_score)
    style  = get_investment_style(funda_score, tech_score)

    # ── Forecast ──
    currency = "Rp " if symbol.endswith(".JK") else ""
    forecast = forecast_price(df)

    # ─────────────────────── DISPLAY ───────────────────────
    console.rule("[bold]LAPORAN ANALISA[/bold]")

    display_header(info, symbol)
    display_sentiment(result)
    display_score_summary(tech_score, funda_score)

    console.print()
    display_technical_signals(tech_signals)

    console.print()
    display_fundamental_metrics(funda_metrics)

    console.print()
    display_forecast(forecast, currency)

    console.print()
    display_investment_style(style, result)

    display_disclaimer()
    console.rule()


if __name__ == "__main__":
    main()