"""
classifier_adapter.py — SHADOW MODE адаптер для ML-классификатора.
Логирует предсказания, не влияет на решения бота.
"""

import os
import csv
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MODEL_DIR = "/sdcard/bybit_bot_model"
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.txt")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")
SHADOW_LOG_PATH = "ml_shadow_log.csv"

_model = None
_feature_cols = None
_load_attempted = False


def _try_load_model():
    global _model, _feature_cols, _load_attempted
    _load_attempted = True
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        logger.info("ML classifier: модель не найдена, shadow predict отключён.")
        return
    try:
        import lightgbm as lgb
        _model = lgb.Booster(model_file=MODEL_PATH)
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            _feature_cols = json.load(f)["features"]
        logger.info(f"ML classifier: модель загружена, признаки: {_feature_cols}")
    except Exception as e:
        logger.warning(f"ML classifier: не удалось загрузить модель: {e}")
        _model = None
        _feature_cols = None


def _symbol_to_code(symbol):
    known = sorted(["ATOMUSDT", "BNBUSDT", "BTCUSDT", "CRVUSDT", "DOTUSDT", "ETHUSDT",
                     "HYPEUSDT", "LINKUSDT", "ONDOUSDT", "SOLUSDT", "UNIUSDT", "WLDUSDT", "XRPUSDT"])
    try:
        return known.index(symbol)
    except ValueError:
        return -1


def _session_to_code(session):
    known = sorted(["Asia", "London", "New York"])
    try:
        return known.index(session)
    except ValueError:
        return -1


def predict_proba(symbol, side, session, rr, entry, sl, tp):
    global _model, _feature_cols
    if not _load_attempted:
        _try_load_model()
    if _model is None or not entry:
        return None
    try:
        sl_dist_pct = abs(entry - sl) / entry * 100
        tp_dist_pct = abs(tp - entry) / entry * 100
        feature_values = {
            "symbol_code": _symbol_to_code(symbol),
            "side_code": 1 if side == "Buy" else 0,
            "session_code": _session_to_code(session),
            "rr": rr or 0.0,
            "sl_dist_pct": sl_dist_pct,
            "tp_dist_pct": tp_dist_pct,
        }
        row = [[feature_values[col] for col in _feature_cols]]
        proba = _model.predict(row)[0]
        return float(proba)
    except Exception as e:
        logger.warning(f"ML classifier: ошибка predict: {e}")
        return None


def predict_and_log(symbol, side, session, rr, entry, sl, tp, order_id=""):
    proba = predict_proba(symbol, side, session, rr, entry, sl, tp)
    try:
        file_exists = os.path.exists(SHADOW_LOG_PATH)
        with open(SHADOW_LOG_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "symbol", "side", "session", "rr",
                                  "entry", "sl", "tp", "order_id", "ml_proba"])
            writer.writerow([datetime.now(timezone.utc).isoformat(), symbol, side, session,
                              rr, entry, sl, tp, order_id,
                              f"{proba:.4f}" if proba is not None else ""])
    except Exception as e:
        logger.warning(f"ML shadow log: не удалось записать строку: {e}")

    if proba is not None:
        logger.info(f"🧪 ML shadow predict: {symbol} {side} conf_ml={proba*100:.1f}%")

    return proba
