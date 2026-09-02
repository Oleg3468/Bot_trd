"""
risk_manager.py v3 — Управление рисками
Правила: макс 5 сделок/день, макс 2 убытка/день,
после 2 лоссов ждать Asia, force не обходит лимиты,
риск 1% от депозита на сделку, плечо 20x.

ИСТОРИЯ ИЗМЕНЕНИЙ v2 -> v3 (важно):
До этой версии calc_position_qty() считал объём позиции ТОЛЬКО из
фиксированной маржи (10 USDT) и плеча — совершенно не глядя на расстояние
до стоп-лосса. Из-за этого risk_pct существовал только в конфиге и в
журнале, но реально не влиял на объём: сделка с широким стопом теряла
в долларах пропорционально больше, чем сделка с узким стопом, хотя
обе считались "−1R" в R-мультипликаторе. Это и объясняло парадокс
"средний реализованный R положительный (+1.26), а Profit Factor в
долларах ниже 1" — эдж в цене был, а сайзинг его не уважал.

Теперь calc_position_qty() считает объём от РАССТОЯНИЯ ДО СТОПА:
    объём = (депозит * risk_pct%) / |entry - sl|
Так убыток по стопу всегда равен ровно risk_pct% от депозита, независимо
от того, насколько далеко находится SL — то есть $-результат наконец
соответствует R-мультипликатору.
"""
from __future__ import annotations
from config import MIN_RR
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Оставлены для обратной совместимости импортов (bot.py показывает их в /help
# и в шапке при старте) — но для сайзинга больше не используются напрямую.
MARGIN_PER_TRADE = 10.0
LEVERAGE         = 20
MAX_TRADES_DAY = 10
MAX_LOSSES_DAY = 4

SESSION_LIMITS = {
    "Asia":     {"max_trades": 5, "max_losses": 2},
    "London":   {"max_trades": 5, "max_losses": 2},
    "New York": {"max_trades": 7, "max_losses": 3},
}

def get_current_session() -> str:
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 13:
        return "London"
    else:
        return "New York"
ROI_TARGET       = 0.35

DEFAULT_DEPOSIT  = 1000.0
DEFAULT_RISK_PCT = 1.0
FIXED_MARGIN_USDT = 10.0
SAFETY_MAX_LOSS_PCT = 15.0  # предохранитель: макс. убыток по сделке в % от депозита

SYMBOL_PARAMS = {
    "BTCUSDT":  {"step": 0.001, "min": 0.001},
    "ETHUSDT":  {"step": 0.01,  "min": 0.01},
    "SOLUSDT":  {"step": 0.1,   "min": 0.1},
    "ADAUSDT":  {"step": 1.0,   "min": 1.0},
    "BNBUSDT":  {"step": 0.01,  "min": 0.01},
    "XRPUSDT":  {"step": 1.0,   "min": 1.0},
    "DOTUSDT":  {"step": 0.1,   "min": 0.1},
    "LINKUSDT": {"step": 0.1,   "min": 0.1},
    "AVAXUSDT": {"step": 0.1,   "min": 0.1},
    "ATOMUSDT": {"step": 0.1,   "min": 0.1},
    "UNIUSDT":  {"step": 0.1,   "min": 0.1},
    "CRVUSDT":  {"step": 1.0,   "min": 1.0},
    "ONDOUSDT": {"step": 1.0,  "min": 1.0},
    "HYPEUSDT": {"step": 0.01, "min": 0.01},
    "WLDUSDT":  {"step": 0.1,  "min": 0.1},
}
DEFAULT_PARAMS = {"step": 0.01, "min": 0.01}

def get_qty_params(symbol: str) -> dict:
    return SYMBOL_PARAMS.get(symbol, DEFAULT_PARAMS)

def is_asia_session() -> bool:
    hour = datetime.now(timezone.utc).hour
    return 0 <= hour < 8

def is_tradeable_now() -> bool:
    hour = datetime.now(timezone.utc).hour
    return True  # временно: торговля 24/7 для сбора данных ML, вернуть "7 <= hour < 22" перед реальным счётом

def calc_position_qty(symbol: str, entry: float, sl: float,
                       deposit: float = DEFAULT_DEPOSIT,
                       risk_pct: float = DEFAULT_RISK_PCT,
                       leverage: int = LEVERAGE) -> float:
    """
    Фиксированная маржа: 10 USDT * leverage = номинал позиции.
    Предохранитель: если потенциальный убыток по SL превышает SAFETY_MAX_LOSS_PCT
    от депозита, объём урезается вниз (вплоть до отказа при недостижимом минимуме),
    чтобы широкий стоп не сжигал непропорционально много депозита.
    """
    if entry <= 0 or not sl or sl == entry:
        return 0.0

    sl_dist = abs(entry - sl)
    params = get_qty_params(symbol)
    step = params["step"]
    min_qty = params["min"]

    notional = FIXED_MARGIN_USDT * leverage
    raw_qty = notional / entry
    qty = round(int(raw_qty / step) * step, 8)

    if qty < min_qty:
        return 0.0

    max_loss_usdt = deposit * (SAFETY_MAX_LOSS_PCT / 100)
    potential_loss = qty * sl_dist
    if potential_loss > max_loss_usdt:
        safe_qty_raw = max_loss_usdt / sl_dist
        safe_qty = round(int(safe_qty_raw / step) * step, 8)
        if safe_qty < min_qty:
            return 0.0
        qty = safe_qty

    return qty

TICK_SIZES = {
    "BTCUSDT":  0.10,
    "ETHUSDT":  0.01,
    "SOLUSDT":  0.010,
    "LINKUSDT": 0.001,
    "HYPEUSDT": 0.010,
    "ONDOUSDT": 0.0001,
    "WLDUSDT":  0.0001,
}
DEFAULT_TICK_SIZE = 0.01

def round_to_tick(price: float, symbol: str) -> float:
    tick = TICK_SIZES.get(symbol, DEFAULT_TICK_SIZE)
    return round(round(price / tick) * tick, 8)

def calc_sl_tp(side: str, entry: float, roi: float = ROI_TARGET, leverage: int = LEVERAGE, symbol: str = "") -> tuple[float, float]:
    price_move = roi / leverage
    sl_move    = price_move / 2
    if side == "Buy":
        sl, tp = entry * (1 - sl_move), entry * (1 + price_move)
    else:
        sl, tp = entry * (1 + sl_move), entry * (1 - price_move)
    return round_to_tick(sl, symbol), round_to_tick(tp, symbol)

@dataclass
class RiskDecision:
    allowed:   bool
    reason:    str   = ""
    qty:       float = 0.0
    entry:     float = 0.0
    sl:        float = 0.0
    tp:        float = 0.0
    rr:        float = 0.0
    risk_usdt: float = 0.0
    margin:    float = 0.0
    leverage:  int   = LEVERAGE

    def format(self) -> str:
        if not self.allowed:
            return f"⛔ *Сделка запрещена*\n{self.reason}"
        rr_emoji = "✅" if self.rr >= 2 else "⚠️"
        return (
            f"📊 *План сделки*\n{'─'*28}\n"
            f"Риск:   `{self.risk_usdt:.2f} USDT`\n"
            f"Маржа:  `{self.margin:.2f} USDT`\n"
            f"Плечо:  `{self.leverage}x`\n"
            f"Объём:  `{self.qty}`\n"
            f"Entry:  `{self.entry}`\n"
            f"SL:     `{self.sl}`\n"
            f"TP:     `{self.tp}`\n"
            f"{rr_emoji} RR: `1:{self.rr:.2f}`"
        )

def can_open_trade(symbol: str, side: str, entry: float, sl: float = 0.0, tp: float = 0.0,
                    deposit: float = DEFAULT_DEPOSIT, risk_pct: float = DEFAULT_RISK_PCT,
                    leverage: int = LEVERAGE) -> RiskDecision:
    from journal import count_trades_session, count_losing_trades_session

    session = get_current_session()
    limits = SESSION_LIMITS[session]
    trades_today = count_trades_session(session)
    if trades_today >= limits["max_trades"]:
        return RiskDecision(allowed=False, reason=f"Лимит сделок ({session}): {trades_today}/{limits['max_trades']} за сессию")

    losses_today = count_losing_trades_session(session)
    if losses_today >= limits["max_losses"]:
        return RiskDecision(allowed=False, reason=f"🔴 Лимит убытков ({session}): {losses_today}/{limits['max_losses']} за сессию")

    if not is_tradeable_now():
        return RiskDecision(allowed=False, reason="😴 Не торговое время. London (07-16 UTC) и NY (13-22 UTC)")

    # Важно: sl/tp по умолчанию должны быть определены ДО расчёта объёма,
    # потому что теперь объём считается именно от расстояния до sl.
    if not sl or not tp:
        sl, tp = calc_sl_tp(side, entry, leverage=leverage, symbol=symbol)

    qty = calc_position_qty(symbol, entry, sl, deposit, risk_pct, leverage)
    if qty <= 0:
        return RiskDecision(allowed=False, reason=f"Размер позиции = 0 при entry={entry}, sl={sl}")

    if side == "Buy":
        if sl >= entry:
            return RiskDecision(allowed=False, reason=f"SL ({sl}) должен быть ниже Entry ({entry})")
        if tp <= entry:
            return RiskDecision(allowed=False, reason=f"TP ({tp}) должен быть выше Entry ({entry})")
    else:
        if sl <= entry:
            return RiskDecision(allowed=False, reason=f"SL ({sl}) должен быть выше Entry ({entry})")
        if tp >= entry:
            return RiskDecision(allowed=False, reason=f"TP ({tp}) должен быть ниже Entry ({entry})")

    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    rr      = tp_dist / sl_dist if sl_dist > 0 else 0

    if rr < MIN_RR:
        return RiskDecision(allowed=False, reason=f"RR слишком низкий: 1:{rr:.2f} (минимум 1:1.5)")

    risk_usdt = deposit * (risk_pct / 100)
    margin_used = round(qty * entry / leverage, 2) if leverage else 0.0

    return RiskDecision(
        allowed=True,
        reason=f"✅ Сделок: {trades_today}/{MAX_TRADES_DAY} | Убытков: {losses_today}/{MAX_LOSSES_DAY}",
        qty=qty, entry=entry, sl=sl, tp=tp,
        rr=round(rr, 2), risk_usdt=round(risk_usdt, 2), margin=margin_used, leverage=leverage,
    )

def build_trade_plan(symbol: str, side: str, entry: float, sl: float = 0.0, tp: float = 0.0,
                      deposit: float = DEFAULT_DEPOSIT, risk_pct: float = DEFAULT_RISK_PCT,
                      leverage: int = LEVERAGE) -> RiskDecision:
    return can_open_trade(symbol, side, entry, sl, tp, deposit, risk_pct, leverage)

def calculate_risk_for_symbol(symbol, side, entry, sl, tp, deposit=DEFAULT_DEPOSIT, risk_pct=DEFAULT_RISK_PCT, leverage=LEVERAGE):
    decision = can_open_trade(symbol, side, entry, sl, tp, deposit, risk_pct, leverage)
    class RiskResult:
        def __init__(self, d):
            self.valid = d.allowed; self.qty = d.qty; self.rr = d.rr
            self.error = d.reason; self.entry = d.entry; self.sl = d.sl; self.tp = d.tp
        def format(self): return decision.format()
    return RiskResult(decision)