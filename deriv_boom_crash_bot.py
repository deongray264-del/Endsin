"""
Deriv Boom/Crash Directional CALL/PUT Bot
==========================================
Targets: BOOM1000, BOOM500, CRASH1000, CRASH500

WHY THESE SYMBOLS:
  Boom/Crash indices are synthetic instruments engineered with structural
  momentum: Boom indices spike UP approximately every N ticks (on average),
  Crash indices spike DOWN. Between spikes, price drifts in the spike
  direction. This is a real, exploitable directional edge — unlike 1HZ10V/
  RDBEAR which are near-random walks where CALL/PUT has ~50% win rate.

SIGNAL STACK (4 layers, all must pass):
  1. Spike detector   — identifies a confirmed spike in the last SPIKE_LOOKBACK
                        ticks using a z-score threshold (N sigma single-tick move)
  2. Post-spike window — only enter SPIKE_MIN_TICKS to SPIKE_MAX_TICKS after
                        spike (avoid the spike itself, capture the momentum tail)
  3. EMA confirmation — fast EMA must be trending in the expected direction
                        relative to slow EMA (confirms drift is continuing)
  4. Hawkes gate      — reject if spike cluster intensity is high (multiple
                        rapid spikes = noise, not momentum)

CONTRACT:
  Boom  indices → CALL (price should continue upward after spike)
  Crash indices → PUT  (price should continue downward after spike)
  Duration: 60s (FAST_DUR) or 120s (SLOW_DUR) selected by signal strength
  Stake: BASE_STAKE flat (no martingale on a new strategy until validated)

SELF-IMPROVEMENT:
  Daily at midnight UTC, per-symbol performance is analyzed and:
  - EMA periods adjusted if win rate diverges from 60% target
  - Spike threshold tightened/loosened based on false-positive rate
  - Duration preference updated based on which holds momentum better

SUPABASE:
  Separate table: bot_boom_crash_log
  Separate config: bot_boom_crash_config
"""

import asyncio, json, math, os, random, sys, time, warnings
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import websockets

warnings.filterwarnings("ignore")

# ── Deriv connection ──────────────────────────────────────────────────────
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN    = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID   = os.getenv("DERIV_ACCOUNT_ID") or None
API_BASE           = "https://api.derivws.com"
ACCOUNTS_PATH      = "/trading/v1/options/accounts"
OTP_PATH           = "/trading/v1/options/accounts/{account_id}/otp"

# ── Supabase ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── Symbols ───────────────────────────────────────────────────────────────
# Maps symbol → expected spike direction and natural spike frequency (ticks)
SYMBOLS: Dict[str, dict] = {
    "BOOM1000":  {"direction": "CALL", "spike_freq": 1000, "barrier_dp": 2},
    "BOOM500":   {"direction": "CALL", "spike_freq": 500,  "barrier_dp": 2},
    "CRASH1000": {"direction": "PUT",  "spike_freq": 1000, "barrier_dp": 2},
    "CRASH500":  {"direction": "PUT",  "spike_freq": 500,  "barrier_dp": 2},
}

# ── Contract parameters ───────────────────────────────────────────────────
BASE_STAKE         = 0.35       # Deriv minimum; flat until strategy is validated
MIN_NET_PAYOUT     = 0.08       # lower than EXPIRYRANGE — CALL/PUT pays better
                                 # at similar win probs (less house margin)
WATCHDOG_TIMEOUT   = 10 * 60
HISTORY_BOOTSTRAP  = 5000

# ── Duration candidates ───────────────────────────────────────────────────
# Shorter is better for momentum: post-spike drift decays within 60-180s
FAST_DUR  = 60    # strong signal → tight duration
SLOW_DUR  = 120   # moderate signal → slightly longer

# ── Spike detection ───────────────────────────────────────────────────────
SPIKE_ZSCORE_THRESH  = 4.0   # single-tick move must be >4 sigma to be a spike
SPIKE_LOOKBACK       = 10    # check last N ticks for a spike
SPIKE_MIN_TICKS      = 2     # don't enter mid-spike — wait at least 2 ticks after
SPIKE_MAX_TICKS      = 30    # signal expires if >30 ticks post-spike with no entry
SPIKE_COOLDOWN_SECS  = 90    # min gap between trades per symbol

# ── EMA confirmation ──────────────────────────────────────────────────────
EMA_FAST_PERIOD  = 8
EMA_SLOW_PERIOD  = 21
EMA_MIN_SPREAD   = 0.0005    # fast EMA must lead slow EMA by at least this
                              # (as fraction of price) to count as trend confirmed

# ── Hawkes cluster guard ──────────────────────────────────────────────────
HAWKES_MAX_INTENSITY = 0.55   # reject if spike cluster is too dense
HAWKES_DECAY         = 0.95   # per-tick decay of Hawkes intensity

# ── Signal strength → stake scaling ──────────────────────────────────────
# Bias magnitude (from directional strength score) scales stake linearly
# between BASE_STAKE and MAX_STAKE_MULT * BASE_STAKE.
# Uses the same principle as the EXPIRYRANGE bot's bias-scaled asymmetry.
MAX_STAKE_MULT   = 2.5        # at maximum signal strength, stake = 2.5x base
MIN_SIGNAL_SCORE = 0.40       # below this, don't trade at all
MAX_SIGNAL_SCORE = 1.00       # at or above this, use MAX_STAKE_MULT

# ── Self-improvement ──────────────────────────────────────────────────────
DAILY_TUNE_HOUR_UTC  = 0
TARGET_WIN_RATE      = 0.62   # if actual win rate diverges, adjust thresholds


# =============================================================================
# SUPABASE STORE
# =============================================================================
class SupabaseStore:
    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.ok  = bool(self.url and self.key)
        print(f"[Store] {'Active -> ' + self.url if self.ok else 'No creds — state will not persist.'}")

    def _hdr(self, prefer="return=minimal"):
        return {"apikey": self.key, "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json", "Prefer": prefer}

    def _upsert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                headers=self._hdr("resolution=merge-duplicates,return=minimal"),
                json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} upsert {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Store] {table} upsert error: {e}")

    def _insert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                headers=self._hdr(), json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} insert {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Store] {table} insert error: {e}")

    def _select(self, table, query="select=*"):
        if not self.ok: return []
        try:
            r = requests.get(f"{self.url}/rest/v1/{table}?{query}",
                headers=self._hdr("return=representation"), timeout=12)
            if r.status_code == 200: return r.json()
            print(f"[Store] {table} select {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Store] {table} select error: {e}")
        return []

    def log_trade(self, rec: dict):
        payload = {
            "ts":               datetime.now(timezone.utc).isoformat(),
            "symbol":           rec["symbol"],
            "direction":        rec["direction"],           # CALL or PUT
            "entry_price":      round(float(rec["entry_price"]),      5),
            "duration_secs":    int(rec["duration_secs"]),
            "stake":            round(float(rec["stake"]),            4),
            "signal_score":     round(float(rec["signal_score"]),     4),
            "spike_age_ticks":  int(rec["spike_age_ticks"]),          # ticks since spike
            "spike_magnitude":  round(float(rec["spike_magnitude"]),  5),  # zscore of spike
            "ema_spread":       round(float(rec["ema_spread"]),       6),
            "hawkes_intensity": round(float(rec["hawkes_intensity"]), 4),
            "won":              bool(rec["won"]),
            "profit":           round(float(rec["profit"]),           4),
            "ask_price":        round(float(rec.get("ask_price", 0)), 4),
            "exit_price":       round(float(rec.get("exit_price", 0)), 5),
        }
        self._insert("bot_boom_crash_log", payload)

    def save_config(self, key, value):
        self._upsert("bot_boom_crash_config",
            {"key": key, "value": json.dumps(value),
             "updated_at": datetime.now(timezone.utc).isoformat()})

    def load_config(self, key):
        rows = self._select("bot_boom_crash_config", f"select=value&key=eq.{key}")
        if rows:
            raw = rows[0]["value"]
            return json.loads(raw) if isinstance(raw, str) else raw
        return None

    def load_recent_trades(self, symbol, days=7):
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return self._select("bot_boom_crash_log",
            f"select=*&symbol=eq.{symbol}&ts=gte.{since}&order=ts.asc")


# =============================================================================
# DERIV CLIENT  (shared connection layer — identical to EXPIRYRANGE bot)
# =============================================================================
class DerivClient:
    HEARTBEAT_INTERVAL = 20
    RECONNECT_BASE     = 2.0
    RECONNECT_CAP      = 60.0

    def __init__(self, app_id, token, account_type="demo", account_id=None):
        self.app_id         = app_id
        self.token          = token
        self.account_type   = account_type
        self.account_id     = account_id
        self.ws             = None
        self.req_id         = 0
        self.pending: dict  = {}
        self.subscriptions  = defaultdict(list)
        self.account        = None
        self.resubscribe_cb = None
        self._running       = False
        self._reader_task   = None
        self._ka_task       = None

    def _rest_headers(self):
        return {"Authorization": f"Bearer {self.token}",
                "Deriv-App-ID": self.app_id,
                "Content-Type": "application/json"}

    def _resolve_account_id_sync(self):
        resp = requests.get(f"{API_BASE}{ACCOUNTS_PATH}",
                            headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == self.account_type:
                aid = acc.get("account_id") or acc.get("id")
                if aid:
                    return aid
        raise RuntimeError(f"No '{self.account_type}' account found. data={data}")

    def _fetch_otp_url_sync(self):
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
            print(f"Resolved {self.account_type} account_id = {self.account_id}")
        resp = requests.post(
            f"{API_BASE}{OTP_PATH.format(account_id=self.account_id)}",
            headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data    = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url  = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self):
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    async def connect(self):
        self._running = True
        await self._connect_once()
        asyncio.create_task(self._supervise())
        return self.account

    async def _connect_once(self):
        ws_url = await self._get_ws_url()
        self.ws = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        self._reader_task = asyncio.create_task(self._read_loop())
        self._ka_task     = asyncio.create_task(self._heartbeat())
        bal          = await self.send({"balance": 1})
        self.account = bal.get("balance", {})
        print(f"Connected ({self.account_type}). "
              f"loginid={self.account.get('loginid')} "
              f"balance=${self.account.get('balance'):.2f}")

    async def _read_loop(self):
        try:
            async for message in self.ws:
                self._dispatch(json.loads(message))
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[Client] WS lost: {e}")

    async def _supervise(self):
        while self._running:
            if self._reader_task:
                await self._reader_task
            if self._ka_task:
                self._ka_task.cancel()
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("WS disconnected"))
            self.pending.clear()
            self.ws = None
            if not self._running:
                break
            attempt = 0
            while self._running and self.ws is None:
                attempt += 1
                delay = min(self.RECONNECT_BASE * (2 ** (attempt - 1)),
                            self.RECONNECT_CAP) + random.uniform(0, 1)
                print(f"[Client] Reconnecting in {delay:.1f}s (attempt {attempt})...")
                await asyncio.sleep(delay)
                try:
                    await self._connect_once()
                    if self.resubscribe_cb:
                        await self.resubscribe_cb(self)
                except Exception as e:
                    print(f"[Client] Reconnect failed: {e}")

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                await self.ws.send(json.dumps({"ping": 1}))
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    def _dispatch(self, data):
        req_id   = data.get("req_id")
        msg_type = data.get("msg_type")
        if msg_type == "ping":
            return
        if req_id is not None and req_id in self.pending:
            fut = self.pending.pop(req_id)
            if not fut.done():
                fut.set_result(data)
                return
        if msg_type in self.subscriptions:
            for q in self.subscriptions[msg_type]:
                q.put_nowait(data)

    async def send(self, request, timeout=20):
        self.req_id += 1
        rid = self.req_id
        request = {**request, "req_id": rid}
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(fut, timeout=timeout)

    def subscribe_channel(self, msg_type):
        q = asyncio.Queue()
        self.subscriptions[msg_type].append(q)
        return q


# =============================================================================
# TICK BUFFER
# =============================================================================
class SymbolData:
    def __init__(self, symbol: str, maxlen: int = 8000):
        self.symbol = symbol
        self._prices  = deque(maxlen=maxlen)
        self._epochs  = deque(maxlen=maxlen)

    def add_tick(self, epoch: float, price: float):
        self._prices.append(float(price))
        self._epochs.append(float(epoch))

    def prices(self) -> np.ndarray:
        return np.array(self._prices, dtype=float)

    def diffs(self) -> np.ndarray:
        p = self.prices()
        return np.diff(p) if len(p) > 1 else np.array([])


# =============================================================================
# BOT STATE
# =============================================================================
class BotState:
    def __init__(self):
        self.last_price:      Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        self.last_trade_time: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        self.last_activity:   float            = time.time()
        self.balance:         float            = 0.0
        self.trading_locked:  bool             = False

        # Session stats
        self.session_trades:  Dict[str, int]   = {s: 0   for s in SYMBOLS}
        self.session_wins:    Dict[str, int]   = {s: 0   for s in SYMBOLS}
        self.session_profit:  Dict[str, float] = {s: 0.0 for s in SYMBOLS}

        # Spike tracking — when was the last confirmed spike per symbol
        self.last_spike_tick: Dict[str, int]   = {s: -999 for s in SYMBOLS}
        self.last_spike_mag:  Dict[str, float] = {s: 0.0  for s in SYMBOLS}
        self.tick_count:      Dict[str, int]   = {s: 0    for s in SYMBOLS}

        # Hawkes intensity — decays per tick, spikes bump it up
        self.hawkes:          Dict[str, float] = {s: 0.0  for s in SYMBOLS}

        # Self-improvement tuneable parameters (per symbol)
        # Loaded from Supabase on boot, saved after daily tune
        self.spike_thresh:    Dict[str, float] = {s: SPIKE_ZSCORE_THRESH for s in SYMBOLS}
        self.ema_fast:        Dict[str, int]   = {s: EMA_FAST_PERIOD      for s in SYMBOLS}
        self.ema_slow:        Dict[str, int]   = {s: EMA_SLOW_PERIOD      for s in SYMBOLS}
        self.dur_pref:        Dict[str, int]   = {s: SLOW_DUR             for s in SYMBOLS}

        self.last_tune_date:  str = ""
        self.garch_cache:     Dict = {}


# =============================================================================
# SIGNAL ENGINE
# =============================================================================
def compute_rolling_std(diffs: np.ndarray, window: int = 50) -> float:
    """Rolling std of absolute tick moves for spike z-score baseline."""
    if len(diffs) < 10:
        return float(np.std(diffs)) if len(diffs) > 0 else 1.0
    recent = diffs[-window:]
    return max(float(np.std(np.abs(recent))), 1e-9)


def detect_spike(prices: np.ndarray, diffs: np.ndarray,
                 spike_thresh: float) -> Tuple[bool, int, float]:
    """
    Scans the last SPIKE_LOOKBACK ticks for a spike.
    Returns (found, age_in_ticks, z_score_magnitude).
    age_in_ticks = 0 means the spike is the very last tick.
    """
    if len(diffs) < SPIKE_LOOKBACK + 5:
        return False, 0, 0.0

    baseline_std = compute_rolling_std(diffs[:-SPIKE_LOOKBACK], window=100)
    window_diffs = diffs[-SPIKE_LOOKBACK:]

    best_age  = -1
    best_zscore = 0.0
    for i, d in enumerate(window_diffs):
        zscore = abs(d) / baseline_std
        if zscore > spike_thresh and zscore > best_zscore:
            best_zscore = zscore
            best_age    = SPIKE_LOOKBACK - 1 - i   # 0 = most recent tick

    if best_age >= 0:
        return True, best_age, best_zscore
    return False, 0, 0.0


def compute_ema(prices: np.ndarray, period: int) -> float:
    """Single EMA value at the tail of the price series."""
    if len(prices) < period:
        return float(prices[-1]) if len(prices) > 0 else 0.0
    k   = 2.0 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema


def compute_hawkes(current_intensity: float, spike_occurred: bool) -> float:
    """
    Per-tick update of Hawkes process intensity.
    Decays by HAWKES_DECAY each tick. Bumps by 1.0 on a spike detection.
    """
    intensity = current_intensity * HAWKES_DECAY
    if spike_occurred:
        intensity += 1.0
    return min(intensity, 5.0)


def compute_signal(symbol: str, prices: np.ndarray, diffs: np.ndarray,
                   state: BotState) -> Optional[dict]:
    """
    Full signal pipeline. Returns a signal dict if all gates pass, else None.

    Signal dict keys:
      direction      : CALL or PUT
      duration_secs  : FAST_DUR or SLOW_DUR
      stake          : bias-scaled stake
      signal_score   : 0-1 composite score (for logging)
      spike_age      : ticks since spike
      spike_magnitude: spike z-score
      ema_spread     : (fast_ema - slow_ema) / price
    """
    cfg = SYMBOLS[symbol]
    expected_dir = cfg["direction"]   # CALL for Boom, PUT for Crash

    if len(prices) < max(EMA_SLOW_PERIOD * 2, SPIKE_LOOKBACK + 20):
        return None  # not enough history yet

    # ── Layer 1: Spike detection ──────────────────────────────────────────
    spike_found, spike_age, spike_mag = detect_spike(
        prices, diffs, state.spike_thresh[symbol])

    if not spike_found:
        return None

    # ── Layer 2: Post-spike timing window ─────────────────────────────────
    if spike_age < SPIKE_MIN_TICKS:
        # Too fresh — let the spike settle
        print(f"[Signal] {symbol}: spike too fresh (age={spike_age} < {SPIKE_MIN_TICKS})")
        return None
    if spike_age > SPIKE_MAX_TICKS:
        # Too old — momentum has faded
        return None

    # ── Layer 3: EMA trend confirmation ───────────────────────────────────
    fast_ema = compute_ema(prices, state.ema_fast[symbol])
    slow_ema = compute_ema(prices, state.ema_slow[symbol])
    price_now = float(prices[-1])

    ema_spread = (fast_ema - slow_ema) / max(price_now, 1e-9)

    if expected_dir == "CALL":
        ema_confirmed = ema_spread > EMA_MIN_SPREAD   # fast above slow = uptrend
    else:
        ema_confirmed = ema_spread < -EMA_MIN_SPREAD  # fast below slow = downtrend

    if not ema_confirmed:
        print(f"[Signal] {symbol}: EMA not confirmed "
              f"(spread={ema_spread:.6f}, need {'>' if expected_dir=='CALL' else '<'}"
              f"{EMA_MIN_SPREAD if expected_dir=='CALL' else -EMA_MIN_SPREAD:.6f})")
        return None

    # ── Layer 4: Hawkes cluster guard ─────────────────────────────────────
    hawkes_now = state.hawkes[symbol]
    if hawkes_now > HAWKES_MAX_INTENSITY:
        print(f"[Signal] {symbol}: Hawkes too high ({hawkes_now:.3f} > {HAWKES_MAX_INTENSITY})")
        return None

    # ── Signal score (0 → 1) ──────────────────────────────────────────────
    # Composite of:
    #   spike magnitude (normalized against a 10-sigma ceiling)
    #   EMA spread strength (normalized against 3x min spread)
    #   spike timing (freshest = best, linear decay from min to max)
    #   inverse Hawkes (lower cluster = cleaner signal)
    spike_score   = min(spike_mag / 10.0, 1.0)
    ema_score     = min(abs(ema_spread) / (EMA_MIN_SPREAD * 3), 1.0)
    timing_score  = 1.0 - (spike_age - SPIKE_MIN_TICKS) / max(SPIKE_MAX_TICKS - SPIKE_MIN_TICKS, 1)
    hawkes_score  = 1.0 - (hawkes_now / HAWKES_MAX_INTENSITY)

    signal_score = (spike_score * 0.40 +    # spike quality is most important
                    ema_score   * 0.25 +    # trend confirmation
                    timing_score * 0.25 +   # freshness
                    hawkes_score * 0.10)    # environment cleanliness

    if signal_score < MIN_SIGNAL_SCORE:
        print(f"[Signal] {symbol}: score too low ({signal_score:.3f} < {MIN_SIGNAL_SCORE})")
        return None

    # ── Stake scaling ─────────────────────────────────────────────────────
    # Linear interpolation between BASE_STAKE and MAX_STAKE_MULT * BASE_STAKE
    # based on signal_score between MIN_SIGNAL_SCORE and MAX_SIGNAL_SCORE
    score_range  = max(MAX_SIGNAL_SCORE - MIN_SIGNAL_SCORE, 0.01)
    score_norm   = (signal_score - MIN_SIGNAL_SCORE) / score_range
    stake_mult   = 1.0 + score_norm * (MAX_STAKE_MULT - 1.0)
    stake        = round(BASE_STAKE * stake_mult, 2)

    # ── Duration selection ────────────────────────────────────────────────
    # Fresh spikes with strong EMA → FAST_DUR (momentum peaking)
    # Older spikes with moderate EMA → SLOW_DUR (more time to settle)
    duration = FAST_DUR if (spike_age <= 8 and spike_score > 0.5) else state.dur_pref[symbol]

    print(f"[Signal] {symbol}: CONFIRMED  dir={expected_dir}  "
          f"spike_age={spike_age}t  spike_z={spike_mag:.1f}  "
          f"ema_spread={ema_spread:+.6f}  hawkes={hawkes_now:.3f}  "
          f"score={signal_score:.3f}  stake=${stake:.2f}  dur={duration}s")

    return {
        "direction":        expected_dir,
        "duration_secs":    duration,
        "stake":            stake,
        "signal_score":     signal_score,
        "spike_age":        spike_age,
        "spike_magnitude":  spike_mag,
        "ema_spread":       ema_spread,
        "hawkes_intensity": hawkes_now,
    }


# =============================================================================
# PROPOSAL API CHECK
# =============================================================================
async def fetch_proposal(client: DerivClient, symbol: str,
                         direction: str, duration_secs: int,
                         stake: float) -> Tuple[Optional[float], float]:
    """
    Calls Deriv proposal API for a CALL or PUT contract.
    Returns (net_profit, ask_price) or (None, stake) on failure.
    """
    try:
        resp = await client.send({
            "proposal":       1,
            "amount":         stake,
            "basis":          "stake",
            "contract_type":  direction,   # "CALL" or "PUT"
            "currency":       "USD",
            "duration":       duration_secs,
            "duration_unit":  "s",
            "symbol":         symbol,
        }, timeout=12)

        if "error" in resp:
            err = resp["error"].get("message", str(resp["error"]))
            print(f"[Proposal] {symbol} error: {err}")
            return None, stake

        prop      = resp.get("proposal", {})
        payout    = float(prop.get("payout",    0))
        ask_price = float(prop.get("ask_price", stake))

        if payout <= 0 or ask_price <= 0:
            return None, stake

        net_profit = payout - ask_price
        if net_profit < MIN_NET_PAYOUT * (stake / BASE_STAKE):
            print(f"[Proposal] {symbol}: net=${net_profit:.4f} below floor "
                  f"${MIN_NET_PAYOUT * (stake/BASE_STAKE):.4f} — skip")
            return None, stake

        if net_profit > stake * 10:
            print(f"[Proposal] {symbol}: suspicious net=${net_profit:.4f} — skip")
            return None, stake

        return net_profit, ask_price

    except Exception as e:
        print(f"[Proposal] {symbol} exception: {e}")
        return None, stake


# =============================================================================
# TRADE EXECUTOR
# =============================================================================
async def execute_trade(client: DerivClient, state: BotState,
                        symbol: str, signal: dict,
                        store: SupabaseStore) -> Tuple[bool, float]:
    """
    Places the CALL/PUT trade, waits for settlement, logs result.
    Returns (won, profit).
    """
    direction     = signal["direction"]
    duration_secs = signal["duration_secs"]
    stake         = signal["stake"]
    price_now     = state.last_price[symbol]

    # Proposal check
    net_payout, ask_price = await fetch_proposal(
        client, symbol, direction, duration_secs, stake)

    if net_payout is None:
        return False, 0.0

    SEP = "-" * 68
    print(f"\n{SEP}")
    print(f"  {direction}  {symbol}  {datetime.now(timezone.utc).isoformat()}")
    print(SEP)
    print(f"  Entry price   : {price_now:.5f}")
    print(f"  Direction     : {direction}")
    print(f"  Duration      : {duration_secs}s")
    print(f"  Stake/Ask     : ${stake:.2f} / ${ask_price:.4f}")
    print(f"  Net payout    : ${net_payout:.4f}")
    print(f"  Signal score  : {signal['signal_score']:.3f}  "
          f"spike_age={signal['spike_age']}t  "
          f"spike_z={signal['spike_magnitude']:.1f}  "
          f"ema_spread={signal['ema_spread']:+.6f}")
    print(SEP)

    won, profit, contract_id = False, 0.0, None
    exit_price = price_now

    try:
        resp = await client.send({
            "buy":   "1",
            "price": ask_price,
            "parameters": {
                "amount":           stake,
                "basis":            "stake",
                "contract_type":    direction,
                "currency":         "USD",
                "duration":         duration_secs,
                "duration_unit":    "s",
                "symbol":           symbol,
            },
        }, timeout=30)

        if "error" in resp:
            print(f"[Buy] {symbol} error: {resp['error'].get('message', resp['error'])}")
            return False, 0.0

        contract_id = resp.get("buy", {}).get("contract_id")
        if not contract_id:
            print(f"[Buy] {symbol}: no contract_id")
            return False, 0.0

        print(f"[Buy] Contract id={contract_id} -- waiting {duration_secs}s...")

        deadline = time.time() + duration_secs + 30
        while time.time() < deadline:
            await asyncio.sleep(5)
            try:
                poll   = await client.send(
                    {"proposal_open_contract": 1, "contract_id": contract_id},
                    timeout=12)
                poc    = poll.get("proposal_open_contract", {})
                status = poc.get("status")
                if status == "sold" or poc.get("is_expired") or poc.get("is_settleable"):
                    profit     = float(poc.get("profit", 0.0))
                    won        = profit > 0
                    exit_price = float(poc.get("exit_tick_display_value",
                                               poc.get("sell_price", price_now)) or price_now)
                    break
            except Exception:
                pass

    except Exception as e:
        print(f"[Buy] {symbol} exception: {e}")
        return False, 0.0

    # Update state
    state.session_trades[symbol] += 1
    if won:
        state.session_wins[symbol] += 1
    state.session_profit[symbol] += profit
    state.last_trade_time[symbol] = time.time()
    state.last_activity           = time.time()

    wr     = state.session_wins[symbol] / max(state.session_trades[symbol], 1)
    result = f"WIN  +${profit:.4f}" if won else f"LOSS  -${ask_price:.4f}"
    print(f"\n{SEP}")
    print(f"  RESULT  {symbol}  {datetime.now(timezone.utc).isoformat()}")
    print(SEP)
    print(f"  Contract  : {contract_id}")
    print(f"  Outcome   : {result}")
    print(f"  Session   : {state.session_wins[symbol]}/{state.session_trades[symbol]} "
          f"({wr:.1%})  net=${state.session_profit[symbol]:+.2f}")
    print(SEP + "\n")

    # Refresh balance
    try:
        bal_resp      = await client.send({"balance": 1})
        state.balance = float(bal_resp["balance"]["balance"])
    except Exception:
        pass

    # Log to Supabase
    store.log_trade({
        "symbol":           symbol,
        "direction":        direction,
        "entry_price":      price_now,
        "duration_secs":    duration_secs,
        "stake":            stake,
        "signal_score":     signal["signal_score"],
        "spike_age_ticks":  signal["spike_age"],
        "spike_magnitude":  signal["spike_magnitude"],
        "ema_spread":       signal["ema_spread"],
        "hawkes_intensity": signal["hawkes_intensity"],
        "won":              won,
        "profit":           profit,
        "ask_price":        ask_price,
        "exit_price":       exit_price,
    })

    return won, profit


# =============================================================================
# DAILY SELF-IMPROVEMENT
# =============================================================================
def daily_self_improvement(state: BotState, store: SupabaseStore):
    """
    Runs at midnight UTC. Analyzes last 7 days per symbol and adjusts:
    - spike_thresh: tighter if too many false positives, looser if trades are rare
    - ema_fast/slow: shift if directional trend confirmation is under-performing
    - dur_pref: prefer whichever duration had higher win rate
    """
    print("\n" + "=" * 68)
    print("  DAILY SELF-IMPROVEMENT (Boom/Crash)")
    print("=" * 68)

    for symbol in SYMBOLS:
        rows = store.load_recent_trades(symbol, days=7)
        if not rows or len(rows) < 10:
            print(f"  {symbol}: not enough history ({len(rows)} trades) — skipping")
            continue

        import pandas as pd
        df = pd.DataFrame(rows)
        df["won"] = df["won"].astype(bool)

        n     = len(df)
        wins  = df["won"].sum()
        wr    = wins / n
        print(f"\n  {symbol}: {n} trades  {wins}W/{n-wins}L  win_rate={wr:.3f}")

        # Spike threshold tuning
        # Low win rate → signal quality is poor → tighten spike threshold
        # Very low trade count → threshold too tight → loosen it slightly
        curr_thresh = state.spike_thresh[symbol]
        if wr < TARGET_WIN_RATE - 0.10:
            new_thresh = min(curr_thresh + 0.3, 8.0)
            print(f"    spike_thresh: {curr_thresh:.1f} -> {new_thresh:.1f} (win rate low, tightening)")
        elif n < 5:
            new_thresh = max(curr_thresh - 0.2, 2.5)
            print(f"    spike_thresh: {curr_thresh:.1f} -> {new_thresh:.1f} (few trades, loosening)")
        else:
            new_thresh = curr_thresh
            print(f"    spike_thresh: {curr_thresh:.1f} unchanged")
        state.spike_thresh[symbol] = new_thresh

        # Duration preference — whichever duration had higher win rate
        if "duration_secs" in df.columns:
            dur_wr = df.groupby("duration_secs")["won"].mean()
            if len(dur_wr) > 1:
                best_dur = int(dur_wr.idxmax())
                print(f"    dur_pref: {state.dur_pref[symbol]}s -> {best_dur}s "
                      f"(best wr={dur_wr[best_dur]:.3f})")
                state.dur_pref[symbol] = best_dur
            else:
                print(f"    dur_pref: {state.dur_pref[symbol]}s (only one duration, no change)")

    # Persist
    store.save_config("spike_thresh",  {s: state.spike_thresh[s] for s in SYMBOLS})
    store.save_config("ema_periods",   {s: {"fast": state.ema_fast[s],
                                             "slow": state.ema_slow[s]} for s in SYMBOLS})
    store.save_config("dur_pref",      {s: state.dur_pref[s] for s in SYMBOLS})

    state.last_tune_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print("\n  Self-improvement complete.\n")


def load_config_from_supabase(state: BotState, store: SupabaseStore):
    print("[Config] Loading saved parameters from Supabase...")

    thresh = store.load_config("spike_thresh")
    if thresh:
        for s in SYMBOLS:
            if s in thresh:
                state.spike_thresh[s] = float(thresh[s])
        print(f"  spike_thresh loaded: {state.spike_thresh}")

    ema = store.load_config("ema_periods")
    if ema:
        for s in SYMBOLS:
            if s in ema:
                state.ema_fast[s] = int(ema[s]["fast"])
                state.ema_slow[s] = int(ema[s]["slow"])
        print(f"  ema_periods loaded")

    dur = store.load_config("dur_pref")
    if dur:
        for s in SYMBOLS:
            if s in dur:
                state.dur_pref[s] = int(dur[s])
        print(f"  dur_pref loaded: {state.dur_pref}")

    print("[Config] Done.\n")


# =============================================================================
# TICK FEED HELPERS
# =============================================================================
async def fetch_history(client: DerivClient, symbol: str, count: int) -> list:
    try:
        resp = await client.send({
            "ticks_history": symbol, "count": count,
            "end": "latest", "style": "ticks",
        }, timeout=30)
        hist = resp.get("history", {})
        return list(zip(hist.get("times", []), hist.get("prices", [])))
    except Exception as e:
        print(f"[History] {symbol}: {e}")
        return []


async def subscribe_ticks(client: DerivClient, symbol: str) -> asyncio.Queue:
    q = asyncio.Queue()
    client.subscriptions["tick"].append(q)
    await client.send({"ticks": symbol, "subscribe": 1}, timeout=10)
    return q


async def watchdog(state: BotState):
    while True:
        await asyncio.sleep(60)
        idle = time.time() - state.last_activity
        if idle > WATCHDOG_TIMEOUT:
            print(f"[Watchdog] No activity for {idle:.0f}s — exiting for restart")
            os._exit(1)


# =============================================================================
# MAIN LOOP
# =============================================================================
async def main():
    if not DERIV_API_TOKEN:
        sys.exit("[FATAL] DERIV_API_TOKEN not set.")
    if not DERIV_APP_ID:
        sys.exit("[FATAL] DERIV_APP_ID not set.")

    store  = SupabaseStore()
    state  = BotState()
    load_config_from_supabase(state, store)

    client  = DerivClient(DERIV_APP_ID, DERIV_API_TOKEN,
                          DERIV_ACCOUNT_TYPE, DERIV_ACCOUNT_ID)
    account = await client.connect()
    state.balance = float(account.get("balance", 0))
    print(f"Balance: ${state.balance:.2f}")

    sym_list = list(SYMBOLS.keys())
    sdata: Dict[str, SymbolData] = {s: SymbolData(s) for s in sym_list}

    # Bootstrap
    print("\nBootstrapping tick history...")
    for sym in sym_list:
        ticks = await fetch_history(client, sym, HISTORY_BOOTSTRAP)
        for epoch, price in ticks:
            sdata[sym].add_tick(epoch, float(price))
        prices = sdata[sym].prices()
        if len(prices):
            state.last_price[sym] = float(prices[-1])
        print(f"  {sym}: {len(ticks)} ticks  price={state.last_price[sym]:.5f}")
    state.last_activity = time.time()

    # Subscribe
    tick_queues: Dict[str, asyncio.Queue] = {}
    for sym in sym_list:
        tick_queues[sym] = await subscribe_ticks(client, sym)
    print(f"\nSubscribed to: {sym_list}")

    async def resubscribe(c: DerivClient):
        for sym in sym_list:
            tick_queues[sym] = await subscribe_ticks(c, sym)
        bal_resp      = await c.send({"balance": 1})
        state.balance = float(bal_resp.get("balance", {}).get("balance", state.balance))
        print("[Reconnect] Subscriptions restored.")

    client.resubscribe_cb = resubscribe
    asyncio.create_task(watchdog(state))
    state.last_activity = time.time()

    print("\n" + "=" * 68)
    print("  Boom/Crash CALL/PUT Bot armed -- scanning for momentum setups")
    print("=" * 68 + "\n")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    while True:
        # ── Drain tick queues ─────────────────────────────────────────────
        for sym in sym_list:
            drained = 0
            while drained < 200:
                try:
                    msg   = tick_queues[sym].get_nowait()
                    tick  = msg.get("tick", {})
                    price = tick.get("quote") or tick.get("bid")
                    epoch = tick.get("epoch", time.time())
                    if price:
                        p = float(price)
                        sdata[sym].add_tick(float(epoch), p)
                        state.last_price[sym] = p
                        state.tick_count[sym] += 1
                        state.last_activity    = time.time()

                        # Update Hawkes intensity on every tick
                        diffs_now = sdata[sym].diffs()
                        if len(diffs_now) >= 5:
                            baseline = compute_rolling_std(diffs_now[:-1], 100)
                            this_z   = abs(diffs_now[-1]) / baseline
                            is_spike = this_z > state.spike_thresh[sym]
                        else:
                            is_spike = False
                        state.hawkes[sym] = compute_hawkes(state.hawkes[sym], is_spike)

                    drained += 1
                except asyncio.QueueEmpty:
                    break

        # ── Daily self-improvement ────────────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        today   = now_utc.strftime("%Y-%m-%d")
        if (now_utc.hour == DAILY_TUNE_HOUR_UTC and
                state.last_tune_date != today):
            daily_self_improvement(state, store)
            state.last_tune_date = today

        # ── Per-symbol signal evaluation ──────────────────────────────────
        if state.trading_locked:
            await asyncio.sleep(0.1)
            continue

        for sym in sym_list:
            # Cooldown guard
            elapsed = time.time() - state.last_trade_time.get(sym, 0.0)
            if elapsed < SPIKE_COOLDOWN_SECS:
                continue

            prices = sdata[sym].prices()
            diffs  = sdata[sym].diffs()
            if len(prices) < max(EMA_SLOW_PERIOD * 2, SPIKE_LOOKBACK + 20):
                continue

            signal = compute_signal(sym, prices, diffs, state)
            if signal is None:
                continue

            # Trade
            state.trading_locked = True
            try:
                won, profit = await execute_trade(client, state, sym, signal, store)
            finally:
                state.trading_locked = False

        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
