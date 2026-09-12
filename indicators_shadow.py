"""
Shadow-логирование технических индикаторов (finta) рядом с торговыми сигналами.
Не влияет на решения бота — только собирает данные для будущего анализа
confluence-факторов (RSI, ATR, ADX) в дополнение к SMC-сигналам.
"""
import csv
import os
from datetime import datetime, timezone

import pandas as pd
from finta import TA

INDICATORS_LOG_PATH = os.path.join(os.path.dirname(__file__), "indicators_shadow_log.csv")


def _candles_to_df(candles):
    """Конвертирует list[Candle] в DataFrame с колонками, которые ожидает finta."""
    data = {
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
        "volume": [c.volume for c in candles],
    }
    return pd.DataFrame(data)


def compute_indicators(candles):
    """
    Считает RSI(14), ATR(14), ADX(14) на последней свече.
    Возвращает dict с последними значениями или None при ошибке/недостатке данных.
    """
    if not candles or len(candles) < 20:
        return None
    try:
        df = _candles_to_df(candles)
        rsi = TA.RSI(df, period=14)
        atr = TA.ATR(df, period=14)
        adx = TA.ADX(df, period=14)
        return {
            "rsi": float(rsi.iloc[-1]) if not rsi.empty else None,
            "atr": float(atr.iloc[-1]) if not atr.empty else None,
            "adx": float(adx.iloc[-1]) if not adx.empty else None,
        }
    except Exception:
        return None


def log_indicators(symbol, side, candles, order_id=""):
    """
    Считает индикаторы и дописывает строку в indicators_shadow_log.csv.
    Полностью изолирована исключениями — вызывающий код может звать её
    без try/except, но рекомендуется всё равно оборачивать на всякий случай.
    """
    values = compute_indicators(candles)
    try:
        file_exists = os.path.exists(INDICATORS_LOG_PATH)
        with open(INDICATORS_LOG_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "symbol", "side", "order_id", "rsi", "atr", "adx"])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                symbol, side, order_id,
                f"{values['rsi']:.2f}" if values and values.get("rsi") is not None else "",
                f"{values['atr']:.6f}" if values and values.get("atr") is not None else "",
                f"{values['adx']:.2f}" if values and values.get("adx") is not None else "",
            ])
    except Exception:
        pass
    return values
