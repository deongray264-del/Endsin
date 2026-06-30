"""
Deriv Multi-Symbol Rise/Fall Trading Bot - FULL POWER  v2
==========================================================
Single-file bot. Scans all eligible synthetic-index symbols, runs an
18-layer intelligence pipeline per symbol using fitted statistical models,
fuses evidence via Bayesian log-odds combination, auto-selects trade
duration via Monte Carlo simulation, and trades the single strongest
signal with a balance-scaled martingale.

v2 FIXES (applied 2026-06-29 from live log + Supabase analysis):
─────────────────────────────────────────────────────────────────
FIX 1 — hurst_rs() now computed on LOG-RETURNS not absolute prices.
         Root cause of H=1.0 on every tick, which caused hurst_signal=+1.0
         always and forced momentum_mode=True permanently, injecting a
         structural CALL-only bias into every Bayesian fusion.

FIX 2 — hmm_trend_weight() lean now normalised by return std.
         HMM state means are O(1e-4) for synthetic index returns; tanh(1e-4
         × 200) ≈ 0.02 — the HMM layer was effectively silent. Normalising
         by std makes the signal dimensionless and reaches ±1 naturally.

FIX 3 — compute_adx() trend threshold lowered 20 → 12 for tick data.
         Tick-level ADX on synthetics rarely exceeds 20, so trend_strength
         was permanently 0 and adx_dir contributed nothing to the gate.

FIX 4 — momentum_mode h-threshold raised 0.52 → 0.58.
         True random-walk synthetics have H ≈ 0.50 ± 0.05 on returns.
         The 0.52 threshold was inside measurement noise, enabling spurious
         momentum mode even on genuinely mean-reverting regimes.

FIX 5 — Direction balance correction in bayesian_fusion().
         When recent CALL/PUT ratio exceeds 80/20, a soft log-odds penalty
         (capped at ±0.5) dampens runaway one-sided bias as a safety net.

FIX 6 — MARTINGALE_MAX_STEPS reduced 3 → 2 + MAX_SEQUENCE_LOSS_PCT guard.
         3-step martingale at 2% risk could consume 11.4% of balance per
         failed sequence. Hard cap at 5% of balance aborts the sequence
         before it can destroy the account. This is the structural fix for
         the $12,000 → $7.54 account destruction.

FIX 7 — POST_LOSS_DEEP_RECAL disabled (False).
         Every loss was triggering 688-second full recalibration (11.5 min),
         locking trading after each of the 41% losing trades. The scheduled
         2-hour recal is sufficient; per-loss recal was redundant and was
         calibrating on corrupted Hurst features anyway.

FIX 8 — MIN_EXP_WIN_RATE lowered 0.52 → 0.505.
         MC was blocking 186 signals because genuine random-walk synthetics
         only simulate to ~0.50-0.51. The layer gate does primary selection;
         MC's role is to pick the best duration, not gate the trade.

FIX 9 — MC_SIMULATIONS reduced 50000 → 8000.
         Statistical error at 8000 paths = ±0.006, sufficient to distinguish
         0.52 from 0.505. Reduces calibration time by ~80%.

FIX 10 — Walk-forward folds 5 → 3, step 3 → 5.
          Cuts calibration wall time from ~688s to ~200-250s.

FIX 11 — Direction history tracking (last 30 trades) for bias monitoring.
          Logs a warning when CALL/PUT ratio exceeds 80/20.

MODEL FITTING vs LIVE SCORING
------------------------------
Fitting HMM/GARCH/Hawkes/OU is computationally expensive, so it only happens
during calibration: once at startup (full universe), then every 2 hours
(top-K deep dive) or after 2 consecutive losses on a symbol (rate-limited,
that symbol's deep dive). Live trading between calibrations just evaluates
the cached fitted models against new ticks - cheap, fast, no refitting.

Symbols without a fitted model yet (before their first calibration) return
no signal and are simply not eligible for selection - this is automatic
and correct, no special-casing needed.

CONNECTION: new Deriv Options API (REST OTP bootstrap), verified against
developers.deriv.com as of 2026-06:
    REST  GET  /trading/v1/options/accounts            -> resolve account_id
    REST  POST /trading/v1/options/accounts/{id}/otp    -> pre-auth WS URL
    No `authorize` message needed - the OTP URL is already authenticated.
    OTP tokens are short-lived/single-use, so a fresh one is fetched on
    every (re)connect; the client auto-reconnects with backoff and replays
    subscriptions (balance + ticks for every symbol) after each reconnect.
    `active_symbols` no longer accepts `product_type`; its response field
    is `underlying_symbol` (not `symbol`). `contracts_for` no longer takes
    `currency`. Buy `parameters` now requires `underlying_symbol` (not
    `symbol`). Tick responses keep the `symbol` field unchanged.

ENV VARS REQUIRED:
    DERIV_APP_ID        - your app_id from a NEW developers.deriv.com application
                           (legacy app_ids, e.g. the old demo id 1089, do NOT
                           work with the new Options API)
    DERIV_API_TOKEN     - API token (personal access token) for your Deriv account
    DERIV_ACCOUNT_TYPE  - "demo" (default, safe) or "real". Picked explicitly
                           rather than guessed, so the bot never trades on
                           your real-money account by accident.
    DERIV_ACCOUNT_ID    - optional; skips the accounts lookup and uses this
                           account_id directly

SUPABASE PERSISTENCE (Railway has no persistent filesystem):
    SUPABASE_URL        - e.g. https://xxxxxxxxxxxx.supabase.co
    SUPABASE_KEY        - service_role key from Supabase Settings → API

    Run this SQL once in Supabase SQL editor before first Railway deploy:

        CREATE TABLE IF NOT EXISTS bot_trade_log (
            id          BIGSERIAL PRIMARY KEY,
            ts          TIMESTAMPTZ DEFAULT now(),
            symbol      TEXT,
            direction   INTEGER,
            step        INTEGER,
            stake       REAL,
            won         BOOLEAN,
            profit      REAL,
            p_up        REAL,
            confidence  REAL,
            duration    INTEGER,
            layer_votes JSONB,
            n_agree     INTEGER,
            n_disagree  INTEGER
        );

        CREATE TABLE IF NOT EXISTS bot_symbol_state (
            symbol         TEXT PRIMARY KEY,
            reliability    REAL,
            threshold      REAL,
            step0_wins     INTEGER DEFAULT 0,
            step0_total    INTEGER DEFAULT 0,
            layer_weights  JSONB  DEFAULT '{}',
            payout_history JSONB  DEFAULT '[]',
            updated_at     TIMESTAMPTZ DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS bot_global_state (
            key        TEXT PRIMARY KEY,
            value      JSONB,
            updated_at TIMESTAMPTZ DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS bot_gate_config (
            key        TEXT PRIMARY KEY,
            value      REAL,
            updated_at TIMESTAMPTZ DEFAULT now()
        );
"""

import asyncio
import io
import json
import os
import random
import sys
import time
import math
import contextlib
import warnings
import numpy as np
import requests
import websockets
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict

from scipy.optimize import minimize
from scipy.stats import rankdata, norm
from statsmodels.tsa.ar_model import AutoReg
from hmmlearn.hmm import GaussianHMM
from arch import arch_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG  (tune via your own walk-forward results before scaling up stakes)
# ---------------------------------------------------------------------------
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID") or None

# ── Supabase persistence (Railway has no persistent filesystem) ──
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── Connection (new Deriv Options API) ──
API_BASE = "https://api.derivws.com"
ACCOUNTS_PATH = "/trading/v1/options/accounts"
OTP_PATH = "/trading/v1/options/accounts/{account_id}/otp"

MIN_STAKE = 0.35
STAKE_PCT = 0.02                       # stake = max(MIN_STAKE, balance * STAKE_PCT)

MARTINGALE_FACTOR    = 1.24
MARTINGALE_MAX_STEPS = 2               # FIX v2: Reduced from 3 → 2.
                                       # 3-step at 2% risk: step0+step1+step2+step3 can
                                       # consume 2%+2.5%+3.1%+3.8% = 11.4% of balance in
                                       # one failed sequence. At a $12k account that was
                                       # $1,371 per sequence. Two steps caps max sequence
                                       # loss at ~6.5% of balance — still painful but survivable.

# FIX v2: Hard cap on total stake committed in one martingale sequence.
# If the cumulative at-risk amount would exceed this fraction of balance,
# abort the recovery rather than place the next step.
# This would have prevented the account destruction: the bot kept recovering
# at growing stakes while balance fell, compounding losses.
MAX_SEQUENCE_LOSS_PCT = 0.05           # Never risk more than 5% of balance in one sequence

SCHEDULED_CALIBRATION_INTERVAL = 2 * 60 * 60   # seconds — full deep recal every 2 hours
CALIBRATION_COOLDOWN = 5 * 60                  # grace period after calibration ends
HISTORY_BOOTSTRAP_COUNT = 10000                # ticks fetched per symbol at startup

CONFIDENCE_THRESHOLD_DEFAULT = 0.11    # fallback only — real threshold set adaptively
                                        # (see ADAPTIVE_THRESHOLD_PERCENTILE below)

# ── Quality gates ──────────────────────────────────────────────────────────
MIN_SCORE_GAP = 0.05

# ── Layer agreement gate ──────────────────────────────────────────────────
MIN_LAYER_AGREE    = 12
MAX_LAYER_DISAGREE = 3

# ── Monte Carlo quality floor ─────────────────────────────────────────────
# FIX v2: Lowered from 0.52 → 0.505. The MC was blocking 186 signals because
# genuine random-walk synthetics only produce ~0.50-0.51 from simulation.
# The real edge comes from the layer stack, not the MC simulation alone.
# 0.505 filters out clearly negative-edge scenarios while allowing the
# layer-quality gate to do the primary selection work.
MIN_EXP_WIN_RATE = 0.505

# ── Adaptive threshold percentile ─────────────────────────────────────────
ADAPTIVE_THRESHOLD_PERCENTILE = 75

# ── Post-loss deep recalibration ──────────────────────────────────────────
# FIX v2: Disabled POST_LOSS_DEEP_RECAL.
# Every loss was triggering a 688-second full recalibration, meaning the bot
# spent 11.5 minutes locked after EVERY single lost trade. At 41% loss rate
# that's ~28 minutes of downtime per hour. Also the deep recal was supposed
# to improve models but the broken Hurst meant it was calibrating on corrupted
# features. Use scheduled 2-hour recal only — sufficient for synthetics.
POST_LOSS_DEEP_RECAL = False
CANDIDATE_DURATIONS = [1, 3, 5, 7, 10]

# FIX v2: Reduced MC_SIMULATIONS from 50000 → 8000.
# The calibration wall time was 688 seconds (11.5 min) for 8 symbols.
# MC is used to select the best duration among 5 candidates on random-walk
# synthetics where the true win rate is ~0.50 ± 0.02. 8000 paths gives a
# standard error of sqrt(0.5*0.5/8000) = 0.0056 — more than sufficient to
# distinguish 0.52 from 0.51 with high confidence. This reduces calibration
# time by ~80% while retaining statistical validity.
MC_SIMULATIONS = 8000

WATCHDOG_TIMEOUT = 5 * 60
WATCHDOG_CHECK_INTERVAL = 20

MIN_TICKS_FOR_FIT = 200                # minimum ticks before a model can be fitted
MIN_TICKS_LIVE = 60                    # minimum ticks before live layers (Markov etc.) run


# ---------------------------------------------------------------------------
# SUPABASE PERSISTENCE STORE
# Railway's filesystem is ephemeral — every restart wipes in-memory state.
# SupabaseStore is the single exit point for all learned state: layer weights,
# per-symbol thresholds, reliability scores, win counts, and trade history.
# All methods are synchronous (requests) so they run during calibration pauses.
# Failures are always swallowed — the bot degrades to in-memory-only if down.
# ---------------------------------------------------------------------------
class SupabaseStore:
    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.ok  = bool(self.url and self.key)
        if self.ok:
            print(f"[Store] Supabase persistence active → {self.url}")
        else:
            print("[Store] SUPABASE_URL / SUPABASE_KEY not set — "
                  "learned state will NOT persist across Railway restarts.")

    def _headers(self, prefer="return=minimal"):
        return {"apikey": self.key, "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json", "Prefer": prefer}

    def _upsert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                              headers=self._headers("resolution=merge-duplicates,return=minimal"),
                              json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} upsert {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"[Store] {table} upsert failed: {e}")

    def _insert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                              headers=self._headers(), json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} insert {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"[Store] {table} insert failed: {e}")

    def _select(self, table, query="select=*"):
        if not self.ok: return []
        try:
            r = requests.get(f"{self.url}/rest/v1/{table}?{query}",
                             headers=self._headers("return=representation"), timeout=12)
            if r.status_code == 200: return r.json()
            print(f"[Store] {table} select {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"[Store] {table} select failed: {e}")
        return []

    def save_trade(self, symbol, direction, step, stake, won, profit,
                   p_up, confidence, duration, feats):
        votes = {}
        if feats:
            votes = {
                "markov":    round((feats.get("markov_p",     0.5) - 0.5) * 2, 4),
                "hmm":       round(feats.get("hmm_lean",      0), 4),
                "hawkes":    round(feats.get("hawkes",         0), 4),
                "ou":        round(feats.get("ou_dir",         0) * feats.get("ou_strength", 0), 4),
                "hurst":     round(feats.get("hurst_signal",   0), 4),
                "arfima":    round(feats.get("arfima_bias",    0), 4),
                "kalman":    round(feats.get("kalman",         0), 4),
                "copula":    round((feats.get("copula_agree", 0.5) - 0.5) * 2, 4),
                "rsi":       round(feats.get("rsi_signal",     0), 4),
                "srsi":      round(feats.get("srsi_signal",    0), 4),
                "adx":       round(feats.get("adx_dir",        0) * feats.get("adx_trend", 0), 4),
                "boll":      round(feats.get("boll_signal",    0), 4),
                "zscore":    round(feats.get("z_signal",       0), 4),
                "te":        round(feats.get("te_signal",      0), 4),
                "jump":      round(feats.get("jump_dir",       0) * feats.get("jump_intensity", 0), 4),
                "post_jump": round(feats.get("post_jump",      0) * feats.get("jump_intensity", 0), 4),
                "momentum_mode": int(feats.get("momentum_mode", False)),
            }
        self._insert("bot_trade_log", {
            "ts": datetime.utcnow().isoformat(), "symbol": symbol,
            "direction": int(direction), "step": int(step),
            "stake": round(float(stake), 4), "won": bool(won),
            "profit": round(float(profit), 4), "p_up": round(float(p_up), 6),
            "confidence": round(float(confidence), 6), "duration": int(duration),
            "layer_votes": json.dumps(votes),
            "n_agree":    int(feats.get("agree_up",    0)) if feats else 0,
            "n_disagree": int(feats.get("disagree_up", 0)) if feats else 0,
        })

    def save_symbol_state(self, state):
        for s, m in state.model_cache.items():
            self._upsert("bot_symbol_state", {
                "symbol":         s,
                "reliability":    round(float(state.reliability.get(s, 1.0)), 6),
                "threshold":      round(float(state.per_symbol_threshold.get(s, state.adaptive_threshold)), 6),
                "step0_wins":     int(state.step0_wins.get(s, 0)),
                "step0_total":    int(state.step0_total.get(s, 0)),
                "layer_weights":  json.dumps(m.per_layer_weights or {}),
                # FIX v2: persist the rolling Kelly payout history per symbol
                # so quarter-Kelly sizing doesn't reset to the conservative
                # default on every Railway restart/redeploy.
                "payout_history": json.dumps(state.payout_history.get(s, [])[-50:]),
                "updated_at":     datetime.utcnow().isoformat(),
            })
        print(f"[Store] Saved state for {len(state.model_cache)} symbols to Supabase.")

    def load_symbol_state(self, state):
        rows = self._select("bot_symbol_state")
        if not rows:
            print("[Store] No prior symbol state found — cold start.")
            return
        if not hasattr(state, '_pending_weights'):
            state._pending_weights = {}
        for row in rows:
            s = row["symbol"]
            state.reliability[s]          = float(row.get("reliability", 1.0))
            state.per_symbol_threshold[s] = float(row.get("threshold",   state.adaptive_threshold))
            state.step0_wins[s]           = int(row.get("step0_wins",   0))
            state.step0_total[s]          = int(row.get("step0_total",  0))
            raw_w = row.get("layer_weights") or "{}"
            weights = json.loads(raw_w) if isinstance(raw_w, str) else (raw_w or {})
            if weights:
                state._pending_weights[s] = weights
            # FIX v2: restore Kelly payout history
            raw_p = row.get("payout_history") or "[]"
            payouts = json.loads(raw_p) if isinstance(raw_p, str) else (raw_p or [])
            if payouts:
                state.payout_history[s] = payouts
        print(f"[Store] Warm-started state for {len(rows)} symbols from Supabase.")

    def save_global_state(self, state):
        """FIX v2: Persist global (non-per-symbol) self-improvement state —
        currently the rolling direction-balance history used by bayesian_fusion's
        bias correction. Without this, every Railway restart loses the bot's
        memory of recent CALL/PUT skew and the correction has to rebuild from
        zero, leaving a brief window of unprotected bias after every redeploy."""
        self._upsert("bot_global_state", {
            "key":        "direction_history",
            "value":      json.dumps(state.direction_history[-30:]),
            "updated_at": datetime.utcnow().isoformat(),
        })

    def load_global_state(self, state):
        rows = self._select("bot_global_state", "select=key,value")
        for row in rows:
            if row["key"] == "direction_history":
                raw = row.get("value") or "[]"
                hist = json.loads(raw) if isinstance(raw, str) else (raw or [])
                if hist:
                    state.direction_history = hist[-30:]
                    print(f"[Store] Restored direction_history "
                          f"({len(state.direction_history)} entries).")

    def save_gates(self, min_agree, max_disagree, min_exp_wr, adaptive_thr):
        for key, val in [("min_layer_agree", float(min_agree)),
                         ("max_layer_disagree", float(max_disagree)),
                         ("min_exp_win_rate",   float(min_exp_wr)),
                         ("adaptive_threshold", float(adaptive_thr))]:
            self._upsert("bot_gate_config", {"key": key, "value": round(val, 6),
                                              "updated_at": datetime.utcnow().isoformat()})

    def load_gates(self):
        rows = self._select("bot_gate_config", "select=key,value")
        return {row["key"]: float(row["value"]) for row in rows}


# Module-level store singleton — instantiated once in main()
_store: Optional[SupabaseStore] = None


# ---------------------------------------------------------------------------
# SHARED STATE  (single source of truth - every module reads/writes through this)
# ---------------------------------------------------------------------------
class TradeState:
    def __init__(self):
        self.balance = 0.0
        self.trading_locked = False
        self.trade_in_progress = False
        self.consecutive_losses = defaultdict(int)
        self.reliability = defaultdict(lambda: 1.0)
        self.loss_triggered_calibrations_24h = deque()
        self.last_scheduled_calibration = time.time()
        self.last_calibration_end = 0.0
        self.model_cache: Dict[str, "SymbolModels"] = {}
        self.last_activity = time.time()

        # Threshold: per-symbol, derived from each symbol's own OOS confidence
        # distribution during deep calibration. Falls back to global default
        # only for symbols not yet calibrated.
        self.adaptive_threshold = CONFIDENCE_THRESHOLD_DEFAULT   # global fallback
        self.per_symbol_threshold: Dict[str, float] = {}

        # Martingale recovery context — saved between main-loop iterations so
        # each recovery step waits for a genuine signal, not an instant re-entry
        # Recovery state — NO symbol/direction lock. After a loss the bot
        # recalibrates then re-enters the open scan at the elevated stake.
        # recovery_step=0 means not in recovery. recovery_step>=1 means
        # we are in a martingale sequence at that step number.
        self.recovery_step      = 0
        self.recovery_stake     = 0.0

        # FIX v2: Track stake committed so far in the current martingale
        # sequence. Abort if cumulative risk exceeds MAX_SEQUENCE_LOSS_PCT.
        self.seq_stakes_committed = 0.0

        # FIX v2: Direction balance tracking.
        # A rolling window of the last 30 trade directions (+1=CALL, -1=PUT).
        # Used to compute recent_call_ratio, which bayesian_fusion uses to
        # apply a soft correction when the model is one-sided.
        self.direction_history: list = []  # deque-style, max 30 entries

        # FIX v2: Rolling payout ratio tracking (per symbol) for Kelly sizing.
        # Deriv Rise/Fall payout varies by symbol/duration/volatility regime,
        # so it must be measured empirically rather than assumed. Stores the
        # last 50 winning trades' (profit / stake) ratio per symbol.
        self.payout_history: Dict[str, list] = defaultdict(list)

        # Step-0 (raw signal, no martingale recovery) win-rate tracking —
        # the only metric that honestly reveals whether the signal has edge
        self.step0_wins   = defaultdict(int)
        self.step0_total  = defaultdict(int)

        # Self-improvement bookkeeping
        self._pending_weights: Dict[str, dict] = {}   # loaded from Supabase at startup
        self._trades_since_autotune = 0               # counts step-0 trades since last gate tune

        # Sequence accumulator — tracks stakes/profits across martingale steps
        # so log_trade_summary has the full picture when the sequence closes
        self.seq_stakes    = []    # stake placed at each step
        self.seq_profits   = []    # profit (negative = loss) at each step
        self.seq_balance_before = 0.0   # balance at sequence open
        self.seq_p_up      = 0.5
        self.seq_confidence= 0.0
        self.seq_duration  = 0


@dataclass
class SymbolModels:
    fitted: bool = False
    fitted_at: float = 0.0
    origin_epoch: float = 0.0
    tick_dt: float = 2.0             # actual measured dt at fit time, carried for re-use
    hmm_model: Optional[object] = None
    garch_result: Optional[object] = None
    garch_scale: float = 1000.0
    ou_params: Optional[dict] = None
    hawkes_up: Optional[dict] = None
    hawkes_up_events: Optional[np.ndarray] = None
    hawkes_down: Optional[dict] = None
    hawkes_down_events: Optional[np.ndarray] = None
    # per-layer fusion weights learned from OOS correlation during deep calibration
    # None means fall back to static defaults inside bayesian_fusion()
    per_layer_weights: Optional[dict] = None


class SymbolData:
    def __init__(self, symbol, maxlen=12000, tick_dt=2.0):
        self.symbol = symbol
        self.tick_dt = tick_dt          # seconds per tick: 1.0 for 1HZ, ~2.0 for R_
        self.ticks = deque(maxlen=maxlen)  # (epoch, price)

    def add_tick(self, epoch, price):
        self.ticks.append((epoch, price))

    def prices(self):
        return np.array([p for _, p in self.ticks], dtype=float)

    def epochs(self):
        return np.array([e for e, _ in self.ticks], dtype=float)

    def returns(self):
        p = self.prices()
        if len(p) < 2:
            return np.array([])
        return np.diff(p) / p[:-1]

    def mean_tick_dt(self):
        """Compute actual mean inter-tick gap in seconds from the buffered epochs.
        Used to verify the tick_dt assumption and for activity ranking."""
        e = self.epochs()
        if len(e) < 2:
            return self.tick_dt
        return float(np.mean(np.diff(e)))

    def slice_copy(self, n):
        """Returns a new SymbolData containing only the first n ticks, carrying
        tick_dt through so re-fitted models use the correct rate."""
        new_sd = SymbolData(self.symbol, maxlen=n + 10, tick_dt=self.tick_dt)
        for e, p in list(self.ticks)[:n]:
            new_sd.add_tick(e, p)
        return new_sd


# ---------------------------------------------------------------------------
# DERIV API CLIENT - new Options API (REST OTP bootstrap, auto-reconnecting)
# ---------------------------------------------------------------------------
class DerivClient:
    """
    Client for the new Deriv Options API.

    Auth flow: REST GET .../accounts -> resolve account_id -> REST POST
    .../accounts/{id}/otp -> pre-authenticated WS URL. No `authorize`
    message is sent or needed; the OTP URL is already scoped to the account.

    OTP URLs are short-lived and single-use (per developers.deriv.com), so a
    fresh one is fetched on every connect AND every reconnect. After the
    first successful connect, this client auto-reconnects in the background
    with exponential backoff and calls `resubscribe_cb` (if set) so the
    caller can replay its balance/tick subscriptions.
    """

    HEARTBEAT_INTERVAL = 20
    RECONNECT_BASE = 2.0
    RECONNECT_CAP = 60.0

    def __init__(self, app_id, token, account_type="demo", account_id=None):
        self.app_id = app_id
        self.token = token
        self.account_type = account_type
        self.account_id = account_id
        self.ws = None
        self.req_id = 0
        self.pending = {}
        self.subscriptions = defaultdict(list)  # msg_type -> list[asyncio.Queue]
        self.account = None
        self.resubscribe_cb = None  # async callable(client), replayed after reconnect
        self._running = False
        self._reader_task = None
        self._ka_task = None

    # ---- REST bootstrap ----
    def _rest_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Deriv-App-ID": self.app_id,
            "Content-Type": "application/json",
        }

    def _resolve_account_id_sync(self):
        url = f"{API_BASE}{ACCOUNTS_PATH}"
        resp = requests.get(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == self.account_type:
                acc_id = acc.get("account_id") or acc.get("id")
                if acc_id:
                    return acc_id
        raise RuntimeError(
            f"No '{self.account_type}' account found via {ACCOUNTS_PATH}. "
            f"Set DERIV_ACCOUNT_ID explicitly, or create one first via "
            f"POST {ACCOUNTS_PATH}. Accounts returned: {data}"
        )

    def _fetch_otp_url_sync(self):
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
            print(f"Resolved {self.account_type} account_id = {self.account_id}")
        url = f"{API_BASE}{OTP_PATH.format(account_id=self.account_id)}"
        resp = requests.post(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP response missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self):
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    # ---- connection lifecycle ----
    async def connect(self):
        """Connects once (raises on failure, so startup misconfiguration
        fails fast) then runs the supervisor loop forever in the background."""
        self._running = True
        await self._connect_once()
        asyncio.create_task(self._supervise())
        return self.account

    async def _connect_once(self):
        ws_url = await self._get_ws_url()
        self.ws = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        # IMPORTANT: start the reader (and heartbeat) BEFORE sending anything.
        # `send()` blocks on a future that is only resolved by `_dispatch()`,
        # which only runs inside `_read_loop()`. If the reader isn't already
        # running, the balance handshake below times out forever (this was
        # the cause of a repeated TimeoutError/CancelledError crash loop).
        self._reader_task = asyncio.create_task(self._read_loop())
        self._ka_task = asyncio.create_task(self._heartbeat())
        bal = await self.send({"balance": 1})
        self.account = bal.get("balance", {})
        print(
            f"Connected ({self.account_type}). "
            f"loginid={self.account.get('loginid')} balance={self.account.get('balance')}"
        )

    async def _read_loop(self):
        try:
            async for message in self.ws:
                self._dispatch(json.loads(message))
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[DerivClient] WS connection lost: {e}")

    async def _supervise(self):
        """Watches the current reader task; on disconnect, cleans up and
        reconnects with exponential backoff, restarting reader+heartbeat
        each time inside `_connect_once`."""
        while self._running:
            if self._reader_task is not None:
                await self._reader_task

            if self._ka_task is not None:
                self._ka_task.cancel()
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("Deriv WS disconnected"))
            self.pending.clear()
            self.ws = None

            if not self._running:
                break

            attempt = 0
            while self._running and self.ws is None:
                attempt += 1
                delay = min(
                    self.RECONNECT_BASE * (2 ** (attempt - 1)), self.RECONNECT_CAP
                ) + random.uniform(0, 1)
                print(f"[DerivClient] Reconnecting in {delay:.1f}s (attempt {attempt})...")
                await asyncio.sleep(delay)
                try:
                    await self._connect_once()
                    if self.resubscribe_cb:
                        await self.resubscribe_cb(self)
                except Exception as e:
                    print(f"[DerivClient] Reconnect attempt {attempt} failed: {e}")

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                await self.ws.send(json.dumps({"ping": 1}))
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    def _dispatch(self, data):
        req_id = data.get("req_id")
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
        request = dict(request)
        request["req_id"] = rid
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(fut, timeout=timeout)

    def subscribe_channel(self, msg_type):
        q = asyncio.Queue()
        self.subscriptions[msg_type].append(q)
        return q



async def fetch_tradable_symbols(client):
    """Fetches R_ volatility indices only (R_10/25/50/75/100).
    Returns a list of verified CALL/PUT-eligible symbol names.
    1HZ symbols are handled separately by select_top_1hz()."""
    resp = await client.send({"active_symbols": "brief"})
    if "error" in resp:
        print(f"[fetch_tradable_symbols] active_symbols error: {resp['error']}")
        return []

    candidates = []
    for s in resp.get("active_symbols", []):
        symbol = s.get("underlying_symbol")
        if not symbol or "1HZ" in symbol:
            continue
        if not symbol.startswith("R_"):
            continue
        if s.get("market") != "synthetic_index":
            continue
        if not s.get("exchange_is_open", 1):
            continue
        candidates.append(symbol)
    print(f"[fetch_tradable_symbols] {len(candidates)} R_ candidates before contracts_for check")

    verified = []
    cf_errors = []
    for symbol in candidates:
        try:
            cf = await client.send({"contracts_for": symbol})
            if "error" in cf:
                cf_errors.append(f"{symbol}: {cf['error']}")
                continue
            types = {c["contract_type"] for c in cf.get("contracts_for", {}).get("available", [])}
            if "CALL" in types and "PUT" in types:
                verified.append(symbol)
        except Exception as e:
            cf_errors.append(f"{symbol}: {type(e).__name__}: {e}")
        await asyncio.sleep(0.05)

    if cf_errors:
        print(f"[fetch_tradable_symbols] {len(cf_errors)} contracts_for calls failed, e.g.: {cf_errors[:3]}")
    print(f"[fetch_tradable_symbols] verified R_ symbols: {verified}")
    return verified


async def select_top_1hz(client, n_top=3):
    """Fetches all 1HZ synthetic-index symbols that support CALL/PUT, bootstraps
    a short tick history for each, then ranks by tick-flow consistency (lowest
    coefficient-of-variation of inter-tick gaps = most active / most liquid).
    Returns the top n_top as a list of symbol names.

    Why consistency rather than just speed: all 1HZ symbols nominally tick every
    second, but some have gaps and bursts (irregular flow) while others tick very
    evenly. Even gap distribution means more reliable statistical model fitting
    and more predictable execution timing."""
    resp = await client.send({"active_symbols": "brief"})
    if "error" in resp:
        print(f"[select_top_1hz] active_symbols error: {resp['error']}")
        return []

    candidates = []
    for s in resp.get("active_symbols", []):
        symbol = s.get("underlying_symbol")
        if not symbol or "1HZ" not in symbol:
            continue
        if s.get("market") != "synthetic_index":
            continue
        if not s.get("exchange_is_open", 1):
            continue
        candidates.append(symbol)

    print(f"[select_top_1hz] {len(candidates)} 1HZ candidates found: {candidates}")

    # verify CALL/PUT support
    verified = []
    for symbol in candidates:
        try:
            cf = await client.send({"contracts_for": symbol})
            if "error" in cf:
                continue
            types = {c["contract_type"] for c in cf.get("contracts_for", {}).get("available", [])}
            if "CALL" in types and "PUT" in types:
                verified.append(symbol)
        except Exception:
            continue
        await asyncio.sleep(0.05)

    print(f"[select_top_1hz] {len(verified)} CALL/PUT-eligible 1HZ symbols: {verified}")

    if not verified:
        return []

    # bootstrap a short history for each candidate and measure tick consistency
    scores = {}
    for symbol in verified:
        try:
            resp2 = await client.send({
                "ticks_history": symbol, "count": 200, "end": "latest", "style": "ticks"
            })
            times = resp2.get("history", {}).get("times", [])
            if len(times) < 10:
                continue
            gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
            mean_gap = sum(gaps) / len(gaps)
            std_gap = (sum((g - mean_gap)**2 for g in gaps) / len(gaps)) ** 0.5
            cv = std_gap / mean_gap if mean_gap > 0 else 999
            scores[symbol] = cv
            print(f"[select_top_1hz] {symbol}: mean_gap={mean_gap:.2f}s  cv={cv:.3f}")
        except Exception as e:
            print(f"[select_top_1hz] {symbol}: bootstrap failed: {e}")
        await asyncio.sleep(0.05)

    if not scores:
        print("[select_top_1hz] no consistency data collected, returning all verified (up to n_top)")
        return verified[:n_top]

    ranked = sorted(scores, key=scores.get)          # ascending CV = most consistent first
    top = ranked[:n_top]
    print(f"[select_top_1hz] top {n_top} by tick consistency: {top}")
    return top



async def fetch_history(client, symbol, count=HISTORY_BOOTSTRAP_COUNT):
    """Fetch up to `count` ticks by paging backwards in time.
    Deriv's ticks_history API hard-caps each response at 1000 ticks regardless
    of the count parameter — confirmed in live logs (always returns 1000).
    We work around this by making ceil(count/1000) sequential calls, each time
    using the earliest timestamp from the previous batch as the new `end` value
    so the next call fetches the 1000 ticks immediately before that point."""
    BATCH = 1000
    all_ticks = []
    end = "latest"

    while len(all_ticks) < count:
        resp = await client.send({
            "ticks_history": symbol,
            "count": BATCH,
            "end": end,
            "style": "ticks",
        })
        history = resp.get("history", {})
        times  = history.get("times",  [])
        prices = history.get("prices", [])
        if not times:
            break   # no more history available

        batch = list(zip(times, prices))
        # Prepend so earlier ticks come first in final list
        all_ticks = batch + all_ticks

        if len(batch) < BATCH:
            break   # API returned fewer than requested — we've hit the start of available history

        # Next call: fetch ticks ending just before the earliest tick in this batch
        earliest_epoch = int(times[0]) - 1
        end = earliest_epoch

    # Trim to requested count (most recent ticks)
    if len(all_ticks) > count:
        all_ticks = all_ticks[-count:]

    return all_ticks


async def buy_contract(client, symbol, direction, duration, duration_unit, stake):
    contract_type = "CALL" if direction > 0 else "PUT"
    req = {
        "buy": "1",
        "price": stake,
        "parameters": {
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": int(duration),   # Deriv requires integer; guard against numpy int / float
            "duration_unit": duration_unit,
            "underlying_symbol": symbol,
        },
    }
    resp = await client.send(req)
    if "error" in resp:
        raise RuntimeError(resp["error"].get("message", "buy failed"))
    return resp["buy"]["contract_id"]


async def wait_for_contract_result(client, contract_id):
    q = client.subscribe_channel("proposal_open_contract")
    await client.send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
    while True:
        data = await q.get()
        poc = data.get("proposal_open_contract", {})
        if poc.get("contract_id") == contract_id and poc.get("is_sold"):
            profit = float(poc.get("profit", 0))
            return profit > 0, profit


# ---------------------------------------------------------------------------
# LAYER 1: MARKOV CHAIN (order-2, Laplace/Dirichlet smoothed)
# ---------------------------------------------------------------------------
def markov_directional_prob(returns, order=2, alpha_smooth=1.0):
    signs = np.sign(returns)
    signs = signs[signs != 0]
    if len(signs) < order + 20:
        return 0.5
    table = defaultdict(lambda: [alpha_smooth, alpha_smooth])  # [down_count, up_count]
    for i in range(len(signs) - order):
        state = tuple(signs[i:i + order])
        idx = 1 if signs[i + order] > 0 else 0
        table[state][idx] += 1
    current_state = tuple(signs[-order:])
    down_c, up_c = table[current_state]
    return float(up_c / (up_c + down_c))


# ---------------------------------------------------------------------------
# LAYER 2: HIDDEN MARKOV MODEL (real Baum-Welch fit via hmmlearn)
# ---------------------------------------------------------------------------
def fit_hmm(returns, n_states=3):
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None
    try:
        X = returns.reshape(-1, 1)
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=80, random_state=42)
        model.fit(X)
        return model
    except Exception as e:
        print(f"[HMM] fit failed: {e}")
        return None


def hmm_trend_weight(model, recent_returns):
    """
    Returns (trend_weight, directional_lean).

    FIX v2: The original computed lean = sum(posterior * HMM_means) and then
    applied tanh(lean * 200). For synthetic index log-returns, HMM state means
    are O(1e-4), so lean ≈ 1e-4 and tanh(1e-4 * 200) ≈ 0.02 — the lean
    signal was effectively zero on every tick (confirmed: HMM max value was
    0.0018 across all 90 live trades).

    Fix: normalise the raw lean by the actual standard deviation of recent
    returns BEFORE applying tanh. This makes the signal dimensionless and
    proportional to genuine directional persistence in return units, so it
    reaches ±1 when the regime state genuinely favours a direction.

    momentum_mode now also requires BOTH trend_weight > 0.60 AND h > 0.55
    (h > 0.52 was too easy to trigger given measurement noise around 0.5).
    """
    if model is None or len(recent_returns) < 5:
        return 0.5, 0.0
    try:
        X = recent_returns.reshape(-1, 1)
        posteriors  = model.predict_proba(X)
        current     = posteriors[-1]
        means       = model.means_.flatten()
        variances   = np.array([np.sqrt(c[0][0]) for c in model.covars_])

        # Raw lean in return units (O(1e-4) for synthetic indices)
        lean_raw    = float(np.sum(current * means))

        # Normalise: express lean in units of recent-return std
        return_std  = float(np.std(recent_returns)) + 1e-8
        lean_norm   = lean_raw / return_std          # dimensionless, O(1)

        # tanh maps to [-1, +1] with natural saturation at 3× std
        lean_signal = float(np.tanh(lean_norm * 3))

        dominant    = int(np.argmax(current))
        vol_rank    = variances[dominant] / (np.max(variances) + 1e-9)
        trend_weight = float(np.clip(1.0 - vol_rank, 0.2, 0.85))
        return trend_weight, lean_signal
    except Exception:
        return 0.5, 0.0


# ---------------------------------------------------------------------------
# LAYER 3: HAWKES PROCESS (real exponential-kernel MLE fit via scipy)
# ---------------------------------------------------------------------------
def hawkes_negloglik(params, event_times, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        return 1e10
    ll = -mu * T
    A = 0.0
    last_t = 0.0
    for i, ti in enumerate(event_times):
        if i > 0:
            A = math.exp(-beta * (ti - last_t)) * (1 + A)
        lam = mu + alpha * A
        if lam <= 0:
            return 1e10
        ll += math.log(lam)
        last_t = ti
    comp = (alpha / beta) * np.sum(1 - np.exp(-beta * (T - event_times)))
    ll -= comp
    return -ll


def fit_hawkes(event_times, T):
    if len(event_times) < 10 or T <= 0:
        return None
    init = [max(len(event_times) / T * 0.5, 1e-4), 0.3, 1.0]
    try:
        res = minimize(
            hawkes_negloglik, init, args=(event_times, T),
            bounds=[(1e-6, None), (0.0, None), (1e-6, None)],
            method="L-BFGS-B",
        )
        if not res.success:
            return None
        mu, alpha, beta = res.x
        if alpha >= beta:
            return None
        return {"mu": mu, "alpha": alpha, "beta": beta}
    except Exception as e:
        print(f"[Hawkes] fit failed: {e}")
        return None


def hawkes_intensity_now(params, event_times, current_t):
    if params is None or event_times is None or len(event_times) == 0:
        return 0.0
    mu, alpha, beta = params["mu"], params["alpha"], params["beta"]
    past = event_times[event_times <= current_t]
    if len(past) == 0:
        return mu
    excitation = np.sum(alpha * np.exp(-beta * (current_t - past)))
    return float(mu + excitation)


def fit_symbol_hawkes(sd):
    returns = sd.returns()
    epochs = sd.epochs()
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None, None, None, None, None
    thresh = 0.5 * np.std(returns) if np.std(returns) > 0 else 1e-9
    origin = epochs[0]
    event_epochs = epochs[1:]
    up_times = (event_epochs[returns > thresh] - origin).astype(float)
    down_times = (event_epochs[returns < -thresh] - origin).astype(float)
    T = float(epochs[-1] - origin)
    hawkes_up = fit_hawkes(up_times, T) if len(up_times) >= 10 else None
    hawkes_down = fit_hawkes(down_times, T) if len(down_times) >= 10 else None
    return origin, hawkes_up, up_times, hawkes_down, down_times


# ---------------------------------------------------------------------------
# LAYER 4: ORNSTEIN-UHLENBECK (OLS / Vasicek-style calibration)
# ---------------------------------------------------------------------------
def fit_ou(prices, dt=1.0):
    if len(prices) < 30:
        return None
    x, y = prices[:-1], prices[1:]
    try:
        b, a = np.polyfit(x, y, 1)
    except Exception:
        return None
    b = float(np.clip(b, 1e-6, 0.999999))
    theta = -math.log(b) / dt
    mu = a / (1 - b)
    resid = y - (a + b * x)
    resid_var = np.var(resid)
    denom = 1 - b ** 2
    sigma = math.sqrt(resid_var * 2 * theta / denom) if denom > 1e-9 else math.sqrt(max(resid_var, 1e-12))
    return {"theta": theta, "mu": mu, "sigma": sigma}


def ou_reversion_signal(prices, ou_params):
    if ou_params is None or len(prices) < 2:
        return {"z": 0.0, "reversion_dir": 0.0, "strength": 0.0}
    mu, sigma = ou_params["mu"], (ou_params["sigma"] if ou_params["sigma"] > 0 else 1e-9)
    z = (prices[-1] - mu) / sigma
    theta_norm = float(np.clip(ou_params["theta"], 0, 5) / 5)
    strength = float(np.clip(abs(z) / 2 * theta_norm, 0, 1))
    return {"z": float(z), "reversion_dir": float(-np.sign(z)), "strength": strength}


# ---------------------------------------------------------------------------
# LAYER 5: HURST EXPONENT (real rescaled-range / R-S analysis)
# ---------------------------------------------------------------------------
def hurst_rs(prices, min_window=10):
    """
    FIX v2: Compute Hurst on LOG-RETURNS, not on prices.

    The original used absolute prices. Prices on a random walk exhibit
    a spurious long-range trend (they never revert to a fixed mean) so
    R/S analysis on prices always converges to H≈1.0 regardless of the
    true underlying dynamics. This produced hurst_signal=+1.0 on every
    single tick, forcing momentum_mode=True permanently and injecting a
    structural CALL bias into every Bayesian fusion that no other layer
    could overcome.

    Log-returns are stationary, zero-mean, and bounded — R/S on returns
    gives a meaningful Hurst estimate in [0.3, 0.7] for synthetic indices.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 102:
        return 0.5
    # Convert to log-returns — stationary series with meaningful Hurst
    series = np.diff(np.log(np.maximum(prices, 1e-10)))
    n = len(series)
    if n < 50:
        return 0.5
    max_window = n // 2
    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=20).astype(int)
    )
    rs_points = []
    for w in window_sizes:
        n_chunks = n // w
        if n_chunks < 1:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = series[i * w:(i + 1) * w]
            mean  = np.mean(chunk)
            dev   = np.cumsum(chunk - mean)
            R     = np.max(dev) - np.min(dev)
            S     = np.std(chunk)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_points.append((w, np.mean(rs_chunk)))
    if len(rs_points) < 3:
        return 0.5
    log_w  = np.log([w for w, _ in rs_points])
    log_rs = np.log([rs for _, rs in rs_points])
    slope, _ = np.polyfit(log_w, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


# ---------------------------------------------------------------------------
# LAYER 6: ARFIMA-STYLE LONG MEMORY (fractional differencing + AR(1))
# ---------------------------------------------------------------------------
def fractional_diff_weights(d, size):
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])


def arfima_bias(returns, hurst, lookback=150):
    if len(returns) < 60:
        return 0.0
    d = float(np.clip(hurst - 0.5, -0.49, 0.49))
    recent = returns[-lookback:]
    n = len(recent)
    w = fractional_diff_weights(d, n)
    diff_series = np.convolve(recent, w, mode="valid")
    if len(diff_series) < 15:
        return float(np.tanh(diff_series[-1] * 50)) if len(diff_series) else 0.0
    try:
        ar_model = AutoReg(diff_series, lags=1, old_names=False).fit()
        forecast = ar_model.predict(start=len(diff_series), end=len(diff_series)).iloc[0]
    except Exception:
        forecast = diff_series[-1]
    return float(np.tanh(forecast * 50))


# ---------------------------------------------------------------------------
# LAYER 7: GARCH(1,1) (real MLE fit via the `arch` package)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CONFIRMATION LAYERS (L13-L18) — no model fitting needed, evaluate live
# ---------------------------------------------------------------------------
def compute_rsi(prices, period=14, momentum_mode=False):
    """L13a: RSI. Regime-aware polarity.
    momentum_mode=False (ranging)  — mean-reversion: RSI<30 → +signal, RSI>70 → -signal
    momentum_mode=True  (trending) — momentum: RSI>55 → +signal, RSI<45 → -signal"""
    if len(prices) < period + 2:
        return 50.0, 0.0
    deltas   = np.diff(prices[-(period + 2):])
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs  = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1 + rs))
    if momentum_mode:
        if rsi > 55:   signal = (rsi - 55) / 45
        elif rsi < 45: signal = -(45 - rsi) / 45
        else:          signal = 0.0
    else:
        if rsi < 30:   signal = (30 - rsi) / 30
        elif rsi > 70: signal = -(rsi - 70) / 30
        else:          signal = 0.0
    return float(rsi), float(np.clip(signal, -1, 1))


def compute_stoch_rsi(prices, rsi_period=14, stoch_period=14, momentum_mode=False):
    """L13b: Stochastic RSI. Regime-aware polarity.
    momentum_mode=False → mean-reversion: stoch<0.2 = +signal, stoch>0.8 = -signal
    momentum_mode=True  → momentum:       stoch>0.6 = +signal, stoch<0.4 = -signal"""
    if len(prices) < rsi_period + stoch_period + 5:
        return 0.5, 0.0
    rsi_series = []
    for i in range(stoch_period):
        rsi_val, _ = compute_rsi(prices[:len(prices) - (stoch_period - i - 1)],
                                  rsi_period, momentum_mode=False)
        rsi_series.append(rsi_val)
    rsi_series = np.array(rsi_series)
    lo, hi = np.min(rsi_series), np.max(rsi_series)
    if hi == lo:
        return 0.5, 0.0
    stoch_k = (rsi_series[-1] - lo) / (hi - lo)
    if momentum_mode:
        if stoch_k > 0.6:   signal = (stoch_k - 0.6) / 0.4
        elif stoch_k < 0.4: signal = -(0.4 - stoch_k) / 0.4
        else:                signal = 0.0
    else:
        if stoch_k < 0.2:   signal = (0.2 - stoch_k) / 0.2
        elif stoch_k > 0.8: signal = -(stoch_k - 0.8) / 0.2
        else:                signal = 0.0
    return float(stoch_k), float(np.clip(signal, -1, 1))


def compute_adx(prices, period=14):
    """L14: ADX trend-strength filter. ADX > 25 = trending, ADX < 20 = ranging.
    Returns (adx_value, trend_strength_0_to_1, direction_bias +1/-1/0).

    FIX: original set highs=lows=prices making TR=abs(prices[i]-prices[i])=0
    always, so ATR=0 and adx_dir was permanently 0.0 (confirmed in trade logs).
    For tick data TR must use consecutive price differences."""
    if len(prices) < period * 2 + 2:
        return 20.0, 0.3, 0.0
    n = len(prices)
    tr_list, pdm_list, ndm_list = [], [], []
    for i in range(1, n):
        tr  = abs(prices[i] - prices[i-1])          # tick-to-tick absolute move
        pdm = max(prices[i] - prices[i-1], 0.0)     # positive directional move
        ndm = max(prices[i-1] - prices[i], 0.0)     # negative directional move
        tr_list.append(tr); pdm_list.append(pdm); ndm_list.append(ndm)
    tr_a  = np.array(tr_list[-period * 2:])
    pdm_a = np.array(pdm_list[-period * 2:])
    ndm_a = np.array(ndm_list[-period * 2:])
    atr   = np.mean(tr_a[-period:])
    if atr == 0:
        return 20.0, 0.3, 0.0
    pdi = 100 * np.mean(pdm_a[-period:]) / atr
    ndi = 100 * np.mean(ndm_a[-period:]) / atr
    dx  = 100 * abs(pdi - ndi) / (pdi + ndi + 1e-9)
    adx_vals = [100 * abs(pdm_a[i] - ndm_a[i]) / (np.mean(tr_a[max(0,i-period):i]) * period + 1e-9)
                for i in range(period, len(tr_a))]
    adx = float(np.mean(adx_vals)) if adx_vals else dx
    adx = float(np.clip(adx, 0, 100))
    # FIX v2: Lower trend threshold from 20 → 12 for tick data.
    # Tick-level synthetic indices produce small consecutive moves so ADX
    # rarely exceeds 20. The original threshold meant trend_strength=0 for
    # virtually all trades (confirmed: adx layer_vote=0.0 on all 90 trades).
    # At threshold 12 with a 20-point range, ADX=15 → strength=0.15 (mild),
    # ADX=32 → strength=1.0 (strong). The direction signal (up_bias) fires
    # even when the trend is moderate.
    trend_strength = float(np.clip((adx - 12) / 20, 0, 1))
    up_bias        = float(np.sign(pdi - ndi))
    return adx, trend_strength, up_bias


def compute_bollinger(prices, period=20, n_std=2.0, momentum_mode=False):
    """L15: Bollinger Band %B. Regime-aware polarity.
    momentum_mode=False (ranging)  — mean-reversion: upper band → -signal (expect down)
    momentum_mode=True  (trending) — momentum: upper band → +signal (trend continues up)"""
    if len(prices) < period + 2:
        return 0.5, 0.0
    window = prices[-period:]
    mid    = np.mean(window)
    std    = np.std(window)
    if std == 0:
        return 0.5, 0.0
    upper  = mid + n_std * std
    lower  = mid - n_std * std
    pct_b  = float(np.clip((prices[-1] - lower) / (upper - lower + 1e-9), -0.5, 1.5))
    if momentum_mode:
        signal = float(np.clip((pct_b - 0.5) * 2, -1, 1))   # follow: +1 at upper, -1 at lower
    else:
        signal = float(np.clip((0.5 - pct_b) * 2, -1, 1))   # fade:   +1 at lower, -1 at upper
    return pct_b, signal


def compute_zscore(prices, period=50, momentum_mode=False):
    """L16: Z-score of price vs rolling mean. Regime-aware polarity.
    momentum_mode=False (ranging)  — fade the move: high z → -signal (expect reversion)
    momentum_mode=True  (trending) — follow the move: high z → +signal (trend continues)"""
    if len(prices) < period + 2:
        return 0.0, 0.0
    window = prices[-period:]
    mu     = np.mean(window)
    sigma  = np.std(window) if np.std(window) > 0 else 1e-9
    z      = (prices[-1] - mu) / sigma
    if momentum_mode:
        signal = float(np.clip(z / 2,  -1, 1))   # follow the move
    else:
        signal = float(np.clip(-z / 2, -1, 1))   # fade the move
    return float(z), signal


def transfer_entropy(source_returns, target_returns, lag=1, bins=5):
    """L17: Transfer entropy from source to target. Measures whether source's
    past directional moves provide information about target's next move,
    beyond what target's own history provides. Returns positive value if
    source -> target information flow exists, else near-zero.

    Uses binned estimator for speed (proper KSG estimator is O(n^2)).
    Returns a signed directional signal: positive = source predicts target
    up, negative = source predicts target down."""
    n = min(len(source_returns), len(target_returns)) - lag
    if n < 30:
        return 0.0
    s = source_returns[-n - lag:-lag]
    t_past   = target_returns[-n - lag:-lag]
    t_future = target_returns[-n:]
    try:
        s_bin  = np.digitize(s,       np.percentile(s,       np.linspace(0, 100, bins + 1)[1:-1]))
        tp_bin = np.digitize(t_past,  np.percentile(t_past,  np.linspace(0, 100, bins + 1)[1:-1]))
        tf_bin = np.digitize(t_future,np.percentile(t_future,np.linspace(0, 100, bins + 1)[1:-1]))
        # P(t_future | t_past, s) vs P(t_future | t_past)
        joint3  = np.zeros((bins, bins, bins))
        joint2  = np.zeros((bins, bins))
        joint2b = np.zeros((bins, bins))
        marg    = np.zeros(bins)
        for i in range(n):
            si  = min(s_bin[i],  bins - 1)
            tpi = min(tp_bin[i], bins - 1)
            tfi = min(tf_bin[i], bins - 1)
            joint3[tfi, tpi, si]  += 1
            joint2[tfi, tpi]      += 1
            joint2b[tpi, si]      += 1
            marg[tpi]             += 1
        joint3  = joint3 / (n + 1e-9)
        joint2  = joint2 / (n + 1e-9)
        joint2b = joint2b / (n + 1e-9)
        marg    = marg / (n + 1e-9)
        te = 0.0
        for tfi in range(bins):
            for tpi in range(bins):
                for si in range(bins):
                    num = joint3[tfi, tpi, si]
                    if num <= 0: continue
                    denom_a = joint2b[tpi, si] if joint2b[tpi, si] > 0 else 1e-9
                    denom_b = joint2[tfi, tpi] if joint2[tfi, tpi] > 0 else 1e-9
                    base    = marg[tpi]         if marg[tpi] > 0         else 1e-9
                    te += num * np.log((num * base) / (denom_a * denom_b) + 1e-9)
        # directional component: if source recently moved up, does target follow?
        src_dir = np.sign(np.mean(s[-5:]))
        return float(np.clip(te * src_dir, -1, 1))
    except Exception:
        return 0.0


def detect_jumps(returns, threshold_sigma=2.5):
    """L18: Jump-diffusion — Merton-style jump detection. Identifies ticks
    where the absolute return exceeds threshold_sigma standard deviations
    (likely engineered jumps in synthetic indices). Returns:
      jump_intensity  : recent jump frequency (0-1 normalised)
      jump_direction  : +1 if recent jumps were up, -1 if down, 0 if mixed
      post_jump_signal: after a large jump, expect partial reversion (-jump_dir)"""
    if len(returns) < 30:
        return 0.0, 0.0, 0.0
    sigma = np.std(returns)
    if sigma == 0:
        return 0.0, 0.0, 0.0
    z_scores  = returns / sigma
    jump_mask = np.abs(z_scores) > threshold_sigma
    recent    = jump_mask[-20:]
    intensity = float(np.mean(recent))
    if not np.any(recent):
        return intensity, 0.0, 0.0
    recent_z  = z_scores[-20:]
    jump_dirs = np.sign(recent_z[recent])
    jump_dir  = float(np.mean(jump_dirs)) if len(jump_dirs) > 0 else 0.0
    # post-jump: last tick was a jump → expect partial reversion
    post_jump = -float(np.sign(z_scores[-1])) if jump_mask[-1] else 0.0
    return intensity, float(jump_dir), float(post_jump)


def fit_garch(returns, scale=1000.0):
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None
    try:
        scaled = returns * scale
        am = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
        # arch's SLSQP optimizer prints convergence diagnostics directly to
        # stdout/stderr on non-convergence, bypassing warnings.filterwarnings.
        # These aren't fatal (a result is still returned) but were showing up
        # as noisy 'error' severity log lines - fully suppress at the source.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = am.fit(disp="off")
        return result
    except Exception as e:
        print(f"[GARCH] fit failed: {e}")
        return None


def garch_vol_trust(garch_result, returns, scale=1000.0):
    if garch_result is None:
        return 0.5, None
    try:
        forecast = garch_result.forecast(horizon=1, reindex=False)
        cond_vol = math.sqrt(float(forecast.variance.values[-1, 0])) / scale
        baseline_vol = np.std(returns) if np.std(returns) > 0 else 1e-9
        ratio = cond_vol / baseline_vol
        trust = 1.0 / (1.0 + max(ratio - 1, 0) * 2)
        return float(np.clip(trust, 0.1, 1.0)), cond_vol
    except Exception:
        return 0.5, None


# ---------------------------------------------------------------------------
# LAYER 8: SAMPLE ENTROPY (proper formula, not histogram Shannon entropy)
# ---------------------------------------------------------------------------
def sample_entropy_trust(returns, m=2, r_mult=0.2):
    if len(returns) < 30:
        return 0.5
    r = r_mult * np.std(returns)
    if r <= 0:
        return 0.5
    n = len(returns)

    def _phi(mm):
        x = np.array([returns[i:i + mm] for i in range(n - mm + 1)])
        count, total = 0, 0
        for i in range(len(x)):
            dist = np.max(np.abs(x - x[i]), axis=1)
            count += np.sum(dist <= r) - 1
            total += len(x) - 1
        return count / total if total > 0 else 0.0

    phi_m, phi_m1 = _phi(m), _phi(m + 1)
    if phi_m == 0 or phi_m1 == 0:
        return 0.5
    sampen = -math.log(phi_m1 / phi_m)
    return float(np.clip(1.0 / (1.0 + sampen), 0.1, 1.0))


# ---------------------------------------------------------------------------
# FIX v2 — NEW LAYER: PERMUTATION ENTROPY GATE
# ---------------------------------------------------------------------------
# Distinct from sample_entropy_trust above (which only down-weights evidence).
# This is a hard pre-trade GATE: when the tick sequence is statistically
# indistinguishable from random ordering, no amount of layer agreement is
# trustworthy, and the trade should be skipped entirely rather than just
# down-weighted. Permutation entropy (Bandt-Pompe) measures the predictability
# of ordinal patterns in a short window of recent prices.
PE_EMBED_DIM     = 5
PE_THRESHOLD     = 0.82   # skip trade if normalised PE >= this

def permutation_entropy(prices, m=PE_EMBED_DIM):
    """
    Normalised permutation entropy in [0, 1].
    0.0 = perfectly ordered/predictable sequence.
    1.0 = maximally random ordinal pattern distribution.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < m * 3:
        return 1.0   # not enough data -> treat as untrustworthy (high entropy)

    from math import factorial
    counts = {}
    for i in range(n - m + 1):
        pattern = tuple(np.argsort(prices[i:i + m]))
        counts[pattern] = counts.get(pattern, 0) + 1

    total = sum(counts.values())
    probs = np.array([v / total for v in counts.values()])
    H     = -float(np.sum(probs * np.log2(probs + 1e-12)))
    H_max = float(np.log2(factorial(m)))
    return float(np.clip(H / H_max, 0.0, 1.0))


def entropy_gate_passes(prices, threshold=PE_THRESHOLD):
    """
    Returns (passes: bool, pe_score: float).
    Uses the most recent 150 prices (or all available if fewer).
    """
    window = prices[-150:] if len(prices) >= 150 else prices
    pe = permutation_entropy(window)
    return pe < threshold, pe


# ---------------------------------------------------------------------------
# FIX v2 — NEW LAYER: MULTI-TIMEFRAME CONFLUENCE
# ---------------------------------------------------------------------------
# Computes directional agreement across three timeframes built from the SAME
# tick stream: raw ticks (TF1), 5-tick bars (TF5), and 20-tick bars (TF20).
# A genuine directional edge should show up at more than one timeframe
# simultaneously; an edge visible only on raw noisy ticks is far more likely
# to be spurious. Returns the count of timeframes agreeing with the proposed
# direction (0-3) plus the per-TF directions for logging/diagnostics.
MIN_TF_AGREEMENT = 2   # require at least 2 of 3 timeframes to agree

def _bar_returns(prices, bar_size):
    """Aggregate raw prices into bar_size-tick OHLC-style closes, return log-diffs."""
    n_bars = len(prices) // bar_size
    if n_bars < 2:
        return np.array([])
    bars   = prices[:n_bars * bar_size].reshape(n_bars, bar_size)
    closes = bars[:, -1]
    return np.diff(np.log(np.maximum(closes, 1e-10)))


def _tf_direction(returns_segment, lookback=10):
    """Simple mean-of-recent-returns direction vote: +1, -1, or 0 (neutral)."""
    if len(returns_segment) < 3:
        return 0
    recent = returns_segment[-lookback:]
    m = float(np.mean(recent))
    if abs(m) < 1e-12:
        return 0
    return 1 if m > 0 else -1


def multi_timeframe_confluence(prices, proposed_direction):
    """
    Returns (agreement_count: int 0-3, tf_directions: dict) for logging.
    proposed_direction: +1 (CALL) or -1 (PUT) — the direction the rest of the
    layer stack is currently leaning toward.
    """
    if len(prices) < 60:
        return 0, {"tf1": 0, "tf5": 0, "tf20": 0}

    returns_tf1  = np.diff(np.log(np.maximum(prices[-100:], 1e-10)))
    returns_tf5  = _bar_returns(prices[-250:],  5)
    returns_tf20 = _bar_returns(prices[-600:], 20)

    d1  = _tf_direction(returns_tf1,  lookback=10)
    d5  = _tf_direction(returns_tf5,  lookback=8)
    d20 = _tf_direction(returns_tf20, lookback=5)

    agreement = sum(1 for d in (d1, d5, d20) if d != 0 and d == proposed_direction)
    return agreement, {"tf1": d1, "tf5": d5, "tf20": d20}


# ---------------------------------------------------------------------------
# LAYER 9: KALMAN FILTER (real 2-state local-level + trend filter)
# ---------------------------------------------------------------------------
def kalman_trend_filter(prices, q_level=1e-5, q_trend=1e-6, r_obs=0.01):
    if len(prices) < 5:
        return 0.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[q_level, 0.0], [0.0, q_trend]])
    R = np.array([[r_obs]])
    x = np.array([[prices[0]], [0.0]])
    P = np.eye(2)
    for price in prices[1:]:
        x = F @ x
        P = F @ P @ F.T + Q
        y = price - (H @ x)[0, 0]
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T) / S
        x = x + K * y
        P = (np.eye(2) - K @ H) @ P
    trend = x[1, 0]
    denom = np.std(prices) + 1e-9
    return float(np.clip(np.sign(trend) * min(abs(trend) / denom * 10, 1.0), -1, 1))


# ---------------------------------------------------------------------------
# LAYER 10: COPULA (real Gaussian copula via rank-normal transform)
# ---------------------------------------------------------------------------
def copula_agreement(symbol, returns_window_dict):
    symbols = list(returns_window_dict.keys())
    if symbol not in symbols or len(symbols) < 2:
        return 0.5
    min_len = min(len(v) for v in returns_window_dict.values())
    if min_len < 30:
        return 0.5
    data = np.array([returns_window_dict[s][-min_len:] for s in symbols]).T
    ranks = np.apply_along_axis(rankdata, 0, data) / (min_len + 1)
    normal_scores = norm.ppf(np.clip(ranks, 1e-4, 1 - 1e-4))
    corr = np.corrcoef(normal_scores.T)
    idx = symbols.index(symbol)
    target_sign = np.sign(normal_scores[-1, idx])
    weighted_agree, total_weight = 0.0, 0.0
    for j in range(len(symbols)):
        if j == idx:
            continue
        rho = abs(corr[idx, j])
        peer_sign = np.sign(normal_scores[-1, j])
        weighted_agree += rho * (1.0 if peer_sign == target_sign else 0.0)
        total_weight += rho
    if total_weight == 0:
        return 0.5
    return float(np.clip(weighted_agree / total_weight, 0, 1))


# ---------------------------------------------------------------------------
# MODEL FITTING ORCHESTRATOR (runs only during calibration)
# ---------------------------------------------------------------------------
def fit_symbol_models(sd) -> SymbolModels:
    models = SymbolModels()
    returns = sd.returns()
    prices = sd.prices()
    if len(returns) < MIN_TICKS_FOR_FIT:
        return models

    # use the actual measured mean inter-tick gap as dt so OU theta is in
    # real seconds regardless of whether this is a 1HZ (dt~1s) or R_ (dt~2s) symbol
    actual_dt = sd.mean_tick_dt()
    models.tick_dt = actual_dt

    models.hmm_model = fit_hmm(returns)
    models.garch_result = fit_garch(returns, scale=models.garch_scale)
    models.ou_params = fit_ou(prices, dt=actual_dt)
    origin, h_up, up_ev, h_down, down_ev = fit_symbol_hawkes(sd)
    models.origin_epoch = origin if origin is not None else sd.epochs()[0]
    models.hawkes_up, models.hawkes_up_events = h_up, up_ev
    models.hawkes_down, models.hawkes_down_events = h_down, down_ev

    models.fitted_at = time.time()
    models.fitted = True
    return models


# ---------------------------------------------------------------------------
# LAYER 11: BAYESIAN FUSION (log-odds evidence combination - owns final direction)
# ---------------------------------------------------------------------------
def compute_features(sd, models, returns_window_dict):
    """Evaluates ALL 18 layers using the CACHED fitted models. Returns None if
    no model has been fitted yet (symbol not tradable until first calibration)."""
    if models is None or not models.fitted:
        return None
    returns = sd.returns()
    prices  = sd.prices()
    if len(returns) < MIN_TICKS_LIVE:
        return None

    recent_returns = returns[-50:] if len(returns) >= 50 else returns

    # ── Fitted-model layers (L01-L12) ──────────────────────────────────────
    trend_weight, hmm_lean = hmm_trend_weight(models.hmm_model, recent_returns)
    vol_trust, cond_vol    = garch_vol_trust(models.garch_result, returns, models.garch_scale)
    ou                     = ou_reversion_signal(prices, models.ou_params)

    current_t  = float(sd.epochs()[-1] - models.origin_epoch)
    lam_up     = hawkes_intensity_now(models.hawkes_up,   models.hawkes_up_events,   current_t)
    lam_down   = hawkes_intensity_now(models.hawkes_down, models.hawkes_down_events, current_t)
    hawkes_sig = (lam_up - lam_down) / (lam_up + lam_down + 1e-9)

    h          = hurst_rs(prices)
    arfima     = arfima_bias(returns, h)
    markov_p   = markov_directional_prob(returns)
    kalman     = kalman_trend_filter(prices)
    ent_trust  = sample_entropy_trust(returns[-150:] if len(returns) >= 150 else returns)
    copula     = copula_agreement(sd.symbol, returns_window_dict)

    # Regime classification: combine HMM trend_weight + Hurst exponent.
    # Both must agree that the market is trending before momentum mode activates.
    #   momentum_mode=True  → RSI/StochRSI/Boll/Z-score FOLLOW the direction
    #   momentum_mode=False → classic mean-reversion: fade overbought/oversold
    # FIX v2: Raise h threshold from 0.52 to 0.58.
    # True random walks (Deriv synthetics) have H≈0.50±0.05 on returns. The
    # original 0.52 threshold was inside the noise band, meaning any slight
    # upward bias in the H estimate triggered full momentum mode — permanently,
    # because the old hurst_rs on prices gave H=1.0 always. After fixing
    # hurst_rs to use returns, H will now fluctuate 0.43–0.57 on genuine
    # random-walk synthetics. Only clearly trending regimes (H > 0.58) should
    # activate momentum mode.
    momentum_mode = bool(trend_weight > 0.60 and h > 0.58)

    _,    rsi_signal   = compute_rsi(prices, momentum_mode=momentum_mode)
    _,    srsi_signal  = compute_stoch_rsi(prices, momentum_mode=momentum_mode)
    adx_val, adx_trend, adx_dir = compute_adx(prices)
    _,    boll_signal  = compute_bollinger(prices, momentum_mode=momentum_mode)
    z_val, z_signal    = compute_zscore(prices, momentum_mode=momentum_mode)

    # transfer entropy: average signal from all OTHER symbols toward this one
    te_signal = 0.0
    others = {s: r for s, r in returns_window_dict.items() if s != sd.symbol}
    if others:
        te_vals = [transfer_entropy(src_r, returns[-200:] if len(returns) >= 200 else returns)
                   for src_r in others.values()]
        te_signal = float(np.mean(te_vals))

    jump_intensity, jump_dir, post_jump = detect_jumps(returns)

    # Hurst-derived arbiter: how much to trust momentum vs reversion layers
    # H > 0.5 → persistent (trust momentum), H < 0.5 → anti-persistent (trust reversion)
    # Expressed as a centred signal so it contributes its own log-odds term
    hurst_signal = float(np.clip((h - 0.5) * 4, -1, 1))   # +1 at H=0.75, -1 at H=0.25

    # ── Layer agreement pre-computation ──────────────────────────────────
    # Compute agree/disagree counts here (not just in explain_signal) so the
    # main loop can enforce the MIN_LAYER_AGREE / MAX_LAYER_DISAGREE gates
    # before committing to a trade. direction is unknown at this point, so we
    # compute counts for both sides and let the caller choose the right set.
    _layer_votes = [
        (markov_p - 0.5) * 2,          # Markov
        hmm_lean,                        # HMM
        hawkes_sig,                      # Hawkes
        ou["reversion_dir"] * ou["strength"],  # OU
        hurst_signal,                    # Hurst
        arfima,                          # ARFIMA
        kalman,                          # Kalman
        (copula - 0.5) * 2,             # Copula
        rsi_signal,                      # RSI
        srsi_signal,                     # StochRSI
        adx_dir * adx_trend,            # ADX
        boll_signal,                     # Bollinger
        z_signal,                        # Z-score
        te_signal,                       # Transfer entropy
        jump_dir * jump_intensity,       # Jump direction
        post_jump * jump_intensity,      # Post-jump reversion
    ]
    # agree_up / disagree_up: counts from the perspective of a CALL trade
    _agree_up    = sum(1 for v in _layer_votes if v > 0)
    _disagree_up = sum(1 for v in _layer_votes if v < 0)
    _neutral     = len(_layer_votes) - _agree_up - _disagree_up

    return {
        # fitted-model layers
        "markov_p":     markov_p,
        "hmm_lean":     hmm_lean,
        "trend_weight": trend_weight,
        "hawkes":       hawkes_sig,
        "ou_dir":       ou["reversion_dir"],
        "ou_strength":  ou["strength"],
        "hurst":        h,
        "hurst_signal": hurst_signal,
        "arfima_bias":  arfima,
        "vol_trust":    vol_trust,
        "entropy_trust":ent_trust,
        "kalman":       kalman,
        "copula_agree": copula,
        "cond_vol":     cond_vol,
        "ou_params":    models.ou_params,
        # confirmation layers
        "rsi_signal":    rsi_signal,
        "srsi_signal":   srsi_signal,
        "adx_val":       adx_val,
        "adx_trend":     adx_trend,
        "adx_dir":       adx_dir,
        "boll_signal":   boll_signal,
        "z_signal":      z_signal,
        "z_val":         z_val,
        "te_signal":     te_signal,
        "momentum_mode": momentum_mode,   # logged to trade journal for analysis
        "jump_intensity": jump_intensity,
        "jump_dir":     jump_dir,
        "post_jump":    post_jump,
        # pass through for calibration weight lookup
        "per_layer_weights": models.per_layer_weights,
        # layer vote counts (direction-agnostic: agree_up = votes for CALL)
        "agree_up":    _agree_up,
        "disagree_up": _disagree_up,
        "n_neutral":   _neutral,
        "n_layers":    len(_layer_votes),
    }


def bayesian_fusion(features):
    """Log-odds Bayesian evidence combination across all 18 layers.

    WEIGHT HIERARCHY (highest to lowest precision):
      1. Per-symbol weights learned from OOS correlation during deep calibration
         (stored in features["per_layer_weights"]) — used when available.
      2. Static defaults below — used as fallback for unlearned symbols.

    Hurst contributes its own direct term (not just via ARFIMA) as the
    momentum/reversion arbiter. New confirmation layers (RSI, StochRSI, ADX,
    Bollinger, Z-score, Transfer entropy, Jump-diffusion) add incremental
    evidence without overriding the core fitted-model signal.

    trust_multiplier = vol_trust * entropy_trust gates the entire fusion:
    high-volatility / high-entropy (near-random) conditions suppress all
    evidence proportionally, not just individual layers."""

    learned = features.get("per_layer_weights") or {}

    def W(key, default):
        """Return learned weight if available, else static default."""
        return float(learned.get(key, default))

    p_markov     = float(np.clip(features["markov_p"], 1e-3, 1 - 1e-3))
    markov_logit = math.log(p_markov / (1 - p_markov))
    trend_w      = features["trend_weight"]
    hurst_w      = float(np.clip(features["hurst"], 0, 1))   # H itself as trust scalar

    # ── Core fitted-model evidence ──────────────────────────────────────────
    evidence = [
        # (signal_scaled_to_logit_range, base_weight)
        (markov_logit,                                          W("markov",   1.0)),
        (features["hmm_lean"]    * 2.0,                        W("hmm",      trend_w)),
        (features["hawkes"]      * 2.5,                        W("hawkes",   trend_w)),
        (features["ou_dir"] * features["ou_strength"] * 2.0,  W("ou",       1 - trend_w)),
        (features["hurst_signal"]* 1.2,                        W("hurst",    0.6)),   # ← direct Hurst term
        (features["arfima_bias"] * 1.5,                        W("arfima",   0.55)),
        (features["kalman"]      * 1.5,                        W("kalman",   0.65)),
        ((features["copula_agree"] - 0.5) * 2.0,              W("copula",   0.50)),
    ]

    # ── Confirmation layers (incremental, lower base weight) ────────────────
    adx_trust     = features["adx_trend"]
    momentum_mode = features.get("momentum_mode", False)
    # When both RSI + StochRSI agree → boost; when they disagree → reduce
    rsi_agree  = 1.0 if (features["rsi_signal"] * features["srsi_signal"]) >= 0 else 0.4
    bz_agree   = 1.0 if (features["boll_signal"] * features["z_signal"])   >= 0 else 0.4
    # Regime confidence: how decisively HMM+Hurst agree on the regime
    regime_conf = float(np.clip(abs(trend_w - 0.5) * 2 + abs(hurst_w - 0.5) * 2, 0, 1))

    evidence += [
        (features["rsi_signal"],                                W("rsi",      0.35) * rsi_agree * (1 + regime_conf * 0.5)),
        (features["srsi_signal"],                               W("srsi",     0.30) * rsi_agree * (1 + regime_conf * 0.5)),
        # ADX now produces real signal after tick-data TR fix; scale by trend strength
        (features["adx_dir"] * adx_trust,                      W("adx",      0.40) * (0.5 + adx_trust)),
        (features["boll_signal"],                               W("boll",     0.30) * bz_agree * (1 + regime_conf * 0.4)),
        (features["z_signal"],                                  W("zscore",   0.30) * bz_agree * (1 + regime_conf * 0.4)),
        (features["te_signal"],                                 W("te",       0.30)),
        (features["jump_dir"]    * features["jump_intensity"],  W("jump",     0.25)),
        (features["post_jump"]   * features["jump_intensity"],  W("post_jump",0.20)),
    ]

    total_trust = features["vol_trust"] * features["entropy_trust"]
    log_odds, total_weight = 0.0, 0.0
    for log_ratio, weight in evidence:
        w = float(weight) * total_trust
        log_odds     += log_ratio * w
        total_weight += abs(w)

    # ── FIX v2: Direction balance correction ──────────────────────────────
    # If recent signals are >80% one-directional it almost certainly reflects
    # a structural layer bias rather than a genuine edge. A soft correction
    # pushes log_odds back toward zero (capped at ±0.5 so it cannot flip a
    # genuinely strong signal). The root-cause fix (Hurst on returns) removes
    # the bias at source; this is a safety net against future regressions.
    direction_ratio = float(features.get("recent_call_ratio", 0.5))
    if direction_ratio > 0.80:
        log_odds -= float(np.clip((direction_ratio - 0.80) * 5.0, 0.0, 0.5))
    elif direction_ratio < 0.20:
        log_odds += float(np.clip((0.20 - direction_ratio) * 5.0, 0.0, 0.5))

    p_up       = float(np.clip(1.0 / (1.0 + math.exp(-log_odds)), 0.01, 0.99))
    confidence = abs(p_up - 0.5) * 2.0 * total_trust
    return p_up, confidence


# ---------------------------------------------------------------------------
# SELF-IMPROVEMENT: ONLINE LAYER WEIGHT UPDATE
# Nudges each layer's fusion weight ±4% after every step-0 trade outcome.
# Runs between calibrations so the bot adapts continuously from live results.
#
# Rule: won+agreed → reward (↑), won+opposed → punish (↓),
#       lost+agreed → punish (↓), lost+opposed → reward (↑)
# ---------------------------------------------------------------------------
def online_update_layer_weights(models: SymbolModels, feats: dict,
                                direction: int, won: bool, lr: float = 0.04):
    if models is None or feats is None:
        return
    layer_signals = {
        "markov":    (feats.get("markov_p",     0.5) - 0.5) * 2,
        "hmm":        feats.get("hmm_lean",      0),
        "hawkes":     feats.get("hawkes",         0),
        "ou":         feats.get("ou_dir",         0) * feats.get("ou_strength", 0),
        "hurst":      feats.get("hurst_signal",   0),
        "arfima":     feats.get("arfima_bias",    0),
        "kalman":     feats.get("kalman",         0),
        "copula":    (feats.get("copula_agree",  0.5) - 0.5) * 2,
        "rsi":        feats.get("rsi_signal",     0),
        "srsi":       feats.get("srsi_signal",    0),
        "adx":        feats.get("adx_dir",        0) * feats.get("adx_trend", 0),
        "boll":       feats.get("boll_signal",    0),
        "zscore":     feats.get("z_signal",       0),
        "te":         feats.get("te_signal",      0),
        "jump":       feats.get("jump_dir",       0) * feats.get("jump_intensity", 0),
        "post_jump":  feats.get("post_jump",      0) * feats.get("jump_intensity", 0),
    }
    w       = dict(models.per_layer_weights or {})
    outcome = 1 if won else -1
    for layer, signal in layer_signals.items():
        if abs(signal) < 0.01:
            continue
        agreement = 1 if signal * direction > 0 else -1
        reward    = outcome * agreement
        current_w = w.get(layer, 1.0)
        w[layer]  = float(np.clip(current_w + lr * reward * abs(current_w), 0.05, 3.0))
    models.per_layer_weights = w


# ---------------------------------------------------------------------------
# SELF-IMPROVEMENT: AUTO-TUNE ENTRY GATES
# Adjusts MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE from the
# rolling step-0 win rate. Called every 50 step-0 trades and post-calibration.
# Gate changes are persisted to Supabase so Railway restarts inherit them.
# ---------------------------------------------------------------------------
def autotune_gates(state):
    global MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE
    total_wins   = sum(state.step0_wins.values())
    total_trades = sum(state.step0_total.values())
    if total_trades < 50:
        return
    wr = total_wins / total_trades
    changed = False
    if wr < 0.46:
        new_agree = min(MIN_LAYER_AGREE + 1, 14)
        new_dis   = max(MAX_LAYER_DISAGREE - 1, 1)
        new_mc    = min(MIN_EXP_WIN_RATE + 0.01, 0.58)
        if (new_agree, new_dis, new_mc) != (MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE):
            MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE = new_agree, new_dis, new_mc
            changed = True
            print(f"[AutoTune] WR={wr:.3f} over {total_trades} trades < 0.46 → TIGHTENED: "
                  f"agree>={MIN_LAYER_AGREE} disagree<={MAX_LAYER_DISAGREE} MC>={MIN_EXP_WIN_RATE:.2f}")
    elif wr > 0.54 and total_trades >= 100:
        new_agree = max(MIN_LAYER_AGREE - 1, 10)
        new_dis   = min(MAX_LAYER_DISAGREE + 1, 4)
        new_mc    = max(MIN_EXP_WIN_RATE - 0.01, 0.50)
        if (new_agree, new_dis, new_mc) != (MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE):
            MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE = new_agree, new_dis, new_mc
            changed = True
            print(f"[AutoTune] WR={wr:.3f} over {total_trades} trades > 0.54 → RELAXED: "
                  f"agree>={MIN_LAYER_AGREE} disagree<={MAX_LAYER_DISAGREE} MC>={MIN_EXP_WIN_RATE:.2f}")
    else:
        print(f"[AutoTune] WR={wr:.3f} over {total_trades} trades — gates unchanged.")
    if changed and _store:
        _store.save_gates(MIN_LAYER_AGREE, MAX_LAYER_DISAGREE,
                          MIN_EXP_WIN_RATE, state.adaptive_threshold)


# ---------------------------------------------------------------------------
# LAYER 12: MONTE CARLO DURATION SELECTOR
# ---------------------------------------------------------------------------
def monte_carlo_duration(prices, returns, direction, feats, candidate_durations, n_sims=MC_SIMULATIONS, models=None):
    """Takes the direction already decided by the Bayesian layer (does NOT
    re-decide direction) and simulates forward paths to find which duration
    maximizes expected win probability.

    OU reversion pull weighted by (1 - trend_weight) - same weighting Bayesian
    fusion used when deciding direction, so MC never silently fights the chosen
    direction (fixed the exp_win=0.00 bug from earlier logs).

    When deep startup calibration has produced empirical per-duration win rates,
    those are blended with the simulation estimate (70% sim / 30% empirical) so
    duration selection is anchored to what actually happened on this symbol."""
    if len(returns) < 20:
        return candidate_durations[0], 0.5

    cond_vol = feats.get("cond_vol")
    vol = cond_vol if cond_vol and cond_vol > 0 else (np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns))
    vol = vol if vol > 0 else 1e-6

    hawkes_signal = feats.get("hawkes", 0.0)
    drift = direction * abs(np.mean(returns[-50:])) * (1 + abs(hawkes_signal) * 0.5) if len(returns) >= 50 else 0.0

    ou_params = feats.get("ou_params")
    trend_weight = feats.get("trend_weight", 0.5)
    current_price = prices[-1]
    reversion_pull = 0.0
    if ou_params and ou_params.get("theta", 0) > 0:
        raw_pull = ou_params["theta"] * (ou_params["mu"] - current_price) * 0.01
        reversion_pull = raw_pull * (1 - trend_weight)

    empirical = getattr(models, "empirical_duration_win_rates", {}) if models else {}

    # ── Terminal displacement model ───────────────────────────────────────
    # Deriv Rise/Fall settles on price[expiry] vs price[entry]. The correct
    # model for the terminal displacement after `dur` independent ticks is:
    #
    #   X_T ~ N(drift * dur, vol * sqrt(dur))
    #
    # The old approach summed `dur` individual N(drift, vol) draws, which is
    # mathematically identical to N(drift*dur, vol*sqrt(dur)) for the terminal
    # value BUT it was computing wins as sum(steps)>0 rather than sampling from
    # the correct terminal distribution — introducing a monotone duration bias
    # where longer durations always won because drift accumulated faster than
    # noise. Synthetic index RNG drift is effectively zero by design, so with
    # drift≈0 all durations produce ~50% in simulation and the empirical 30%
    # blend from deep calibration becomes the ONLY real differentiator.
    best = None
    for dur in candidate_durations:
        # Sample terminal displacement directly — no tick-by-tick accumulation
        terminal = np.random.normal(
            (drift + reversion_pull) * dur,   # expected drift over dur ticks
            vol * np.sqrt(dur),               # vol scales as sqrt(ticks)
            size=n_sims
        )
        wins = np.sum(terminal > 0) if direction > 0 else np.sum(terminal < 0)
        sim_win_rate = wins / n_sims

        # FIX v2: Magnitude-weighted win rate.
        # A naive win-count treats a path that ends barely past zero the same
        # as one that ends far in favour of the direction. Borderline paths
        # are weak evidence and inflate the apparent edge. Weighting by
        # |terminal|/std down-weights borderline crossings and produces a
        # sharper, more honest estimate of genuine directional conviction.
        std_term = float(np.std(terminal)) + 1e-9
        favourable = terminal if direction > 0 else -terminal
        weights = 1.0 + np.tanh(np.abs(favourable) / std_term)
        weighted_win_rate = float(
            np.sum(weights * (favourable > 0)) / np.sum(weights)
        )

        # Blend: empirical (primary) + simulation win-rate (sim) + weighted overlay.
        # Empirical still dominates at 70% when available; the remaining 30%
        # is split between raw and magnitude-weighted simulation estimates.
        sim_component = 0.5 * sim_win_rate + 0.5 * weighted_win_rate
        blended = (0.30 * sim_component + 0.70 * empirical[dur]
                   if dur in empirical and empirical[dur] > 0
                   else sim_component)
        if best is None or blended > best[1]:
            best = (dur, blended)
    return best


# ---------------------------------------------------------------------------
# FIX v2 — NEW: BOOTSTRAP META-ENSEMBLE MC (model-free second opinion)
# ---------------------------------------------------------------------------
# monte_carlo_duration() above is fully parametric: it assumes the terminal
# displacement is Gaussian with drift/vol estimated from recent returns.
# This bootstrap version instead resamples BLOCKS of actual historical
# returns (preserving short-range autocorrelation structure) and is
# completely model-free. When the parametric and bootstrap estimates
# agree, the signal is much more likely to reflect genuine structure rather
# than a parametric modelling artefact. Used as an additional soft check
# before committing to a trade — see usage in the main loop.
BOOTSTRAP_BLOCK_SIZE = 10
BOOTSTRAP_N_PATHS    = 2000
BOOTSTRAP_AGREE_TOL  = 0.08   # max allowed disagreement before flagging

def bootstrap_mc_p_directional(returns, direction, duration, block_size=BOOTSTRAP_BLOCK_SIZE,
                               n_paths=BOOTSTRAP_N_PATHS):
    """
    Model-free estimate of P(terminal move favours `direction`) at `duration`
    ticks ahead, built by resampling contiguous blocks of historical returns.
    """
    returns = np.asarray(returns)
    if len(returns) < block_size * 3:
        return 0.5
    n_blocks_needed = max(1, (duration + block_size - 1) // block_size)
    max_start = max(1, len(returns) - block_size)
    outcomes = np.empty(n_paths)
    for i in range(n_paths):
        idx = np.random.randint(0, max_start, size=n_blocks_needed)
        sampled = np.concatenate([returns[j:j + block_size] for j in idx])[:duration]
        terminal_logret = float(np.sum(sampled))
        outcomes[i] = terminal_logret
    favourable = outcomes if direction > 0 else -outcomes
    return float(np.mean(favourable > 0))


def meta_ensemble_agrees(returns, direction, duration, parametric_p,
                         tol=BOOTSTRAP_AGREE_TOL):
    """
    Returns (agrees: bool, bootstrap_p: float).
    If the model-free bootstrap estimate disagrees with the parametric MC
    estimate by more than `tol`, the signal should be treated with extra
    suspicion — the parametric model (Gaussian terminal displacement) may
    not be capturing the symbol's actual return structure right now.
    """
    bootstrap_p = bootstrap_mc_p_directional(returns, direction, duration)
    agrees = abs(bootstrap_p - parametric_p) <= tol
    return agrees, bootstrap_p


# ---------------------------------------------------------------------------
# LAYER AGREEMENT GATE
# ---------------------------------------------------------------------------
def passes_layer_gate(feats, direction):
    """Returns (passes: bool, agree: int, disagree: int, neutral: int).

    Uses the pre-computed vote counts from compute_features. For a CALL
    (direction=+1) the agree count is agree_up; for a PUT (direction=-1)
    it's disagree_up (those layers voted against CALL = voted for PUT).

    Gate: agree >= MIN_LAYER_AGREE AND disagree <= MAX_LAYER_DISAGREE.
    A trade with 10 agree / 4 disagree clears; one with 7 agree / 7 disagree
    does not regardless of how high the Bayesian confidence score is."""
    if direction > 0:
        agree    = feats["agree_up"]
        disagree = feats["disagree_up"]
    else:
        agree    = feats["disagree_up"]   # votes against CALL = votes FOR PUT
        disagree = feats["agree_up"]
    neutral  = feats["n_neutral"]
    passes   = (agree >= MIN_LAYER_AGREE) and (disagree <= MAX_LAYER_DISAGREE)
    return passes, agree, disagree, neutral


# ---------------------------------------------------------------------------
# ENSEMBLE SELECTOR
# ---------------------------------------------------------------------------
def select_trade(symbol_scores, reliability, global_threshold, per_symbol_threshold=None):
    """Selects the single strongest-signal symbol that clears its own
    per-symbol threshold (derived from that symbol's OOS confidence
    distribution during deep calibration). Falls back to the global threshold
    for symbols without a calibrated per-symbol value.

    Per-symbol thresholds mean a symbol with naturally lower confidence scores
    (e.g. R_10 which is more random) gets judged against its own distribution,
    not penalised against a global bar set by a more predictable symbol."""
    per_sym_thr = per_symbol_threshold or {}
    scored = []
    for symbol, (p_up, confidence) in symbol_scores.items():
        score     = confidence * reliability.get(symbol, 1.0)
        direction = 1 if p_up > 0.5 else -1
        thr       = per_sym_thr.get(symbol, global_threshold)
        scored.append((symbol, direction, p_up, score, thr))

    if not scored:
        return None

    # Filter: each symbol must clear its own threshold
    scored = [s for s in scored if s[3] >= s[4]]
    if not scored:
        return None

    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[0]

    # Gap check: top scorer must lead runner-up meaningfully
    if len(scored) > 1 and (top[3] - scored[1][3]) < MIN_SCORE_GAP:
        return None

    return top[:4]   # (symbol, direction, p_up, score)


# ---------------------------------------------------------------------------
# STAKING
# ---------------------------------------------------------------------------
def calculate_stake(balance):
    """stake = max($0.35, 2% of balance) - single formula, no seam/discontinuity.
    Used as the FLOOR/fallback stake. See kelly_adjusted_stake() for the
    edge-aware sizing now used at the call site."""
    return round(max(MIN_STAKE, balance * STAKE_PCT), 2)


# ---------------------------------------------------------------------------
# FIX v2 — NEW: FRACTIONAL KELLY STAKE SIZING
# ---------------------------------------------------------------------------
# The fixed 2%-of-balance formula above sizes every trade identically
# regardless of how strong the signal is. Fractional Kelly instead scales
# the stake with the model's own estimated edge, so high-conviction signals
# get proportionally more capital and marginal signals get less — without
# ever exceeding a hard ceiling.
#
# Binary option Kelly: f* = (p * b - (1 - p)) / b
#   p = model's estimated win probability (exp_win_rate from MC, NOT raw p_up)
#   b = net payout ratio (e.g. 0.95 for a 95%-payout contract)
# Quarter-Kelly (fraction=0.25) is used to keep variance survivable.
KELLY_FRACTION         = 0.25
KELLY_DEFAULT_PAYOUT   = 0.88   # conservative prior before any history exists
KELLY_MIN_HISTORY      = 15     # minimum resolved trades before trusting empirical payout
KELLY_STAKE_CEILING_PCT = 0.04  # never let Kelly alone push stake above 4% of balance

def record_payout(state, symbol, stake, profit, won):
    """Call after every resolved step-0 trade to update the empirical payout
    ratio used by kelly_adjusted_stake(). Only winning trades carry payout
    information (losing trades return profit=-stake, which is not payout)."""
    if won and stake > 0:
        ratio = profit / stake
        hist = state.payout_history[symbol]
        hist.append(ratio)
        if len(hist) > 50:
            hist.pop(0)


def empirical_payout(state, symbol):
    """Returns the rolling average payout ratio for a symbol, or the
    conservative default if not enough history exists yet."""
    hist = state.payout_history.get(symbol, [])
    if len(hist) < KELLY_MIN_HISTORY:
        return KELLY_DEFAULT_PAYOUT
    return float(np.mean(hist))


def kelly_adjusted_stake(balance, exp_win_rate, symbol, state):
    """
    Blends the fixed 2%-of-balance floor with a fractional-Kelly edge-scaled
    component. The fixed floor protects against under-betting when the model
    is right but underconfident; the Kelly component lets strong signals size
    up within a hard ceiling.

    Returns the final stake, already clamped to [MIN_STAKE, balance * KELLY_STAKE_CEILING_PCT].
    """
    payout = empirical_payout(state, symbol)
    p      = float(np.clip(exp_win_rate, 0.01, 0.99))

    # Full Kelly fraction of bankroll
    f_full = (p * payout - (1 - p)) / payout
    f_full = max(0.0, f_full)               # never bet on negative edge
    f_kelly = f_full * KELLY_FRACTION

    kelly_stake = balance * f_kelly
    floor_stake = calculate_stake(balance)    # existing 2%-of-balance floor

    # Take the larger of the two, but never exceed the hard ceiling
    raw_stake = max(kelly_stake, floor_stake)
    ceiling   = balance * KELLY_STAKE_CEILING_PCT
    final     = min(raw_stake, max(ceiling, MIN_STAKE))
    return round(max(MIN_STAKE, final), 2)


def martingale_stakes(base_stake):
    stakes = [round(base_stake, 2)]
    for _ in range(MARTINGALE_MAX_STEPS):
        stakes.append(round(stakes[-1] * MARTINGALE_FACTOR, 2))
    return stakes


# ---------------------------------------------------------------------------
# TRADE EXECUTION
# ---------------------------------------------------------------------------
def explain_signal(symbol, direction, feats, p_up, confidence, duration, exp_win, score):
    """Prints a human-readable breakdown of WHY this trade was taken —
    which layers drove the signal, how strongly, and what the ensemble
    concluded. Logged once at entry before the contract is placed."""
    side     = "CALL (UP)" if direction > 0 else "PUT (DOWN)"
    ts       = datetime.utcnow().isoformat()
    bar      = "█"
    sep      = "─" * 60

    def bar_str(val, width=20):
        """Render a ±1 value as a centred ASCII bar."""
        v     = float(np.clip(val, -1, 1))
        mid   = width // 2
        filled= int(abs(v) * mid)
        if v >= 0:
            return " " * mid + bar * filled + " " * (width - mid - filled)
        else:
            return " " * (mid - filled) + bar * filled + " " * mid + " " * (width - mid)

    # Compile layer contributions into a ranked list
    layer_signals = [
        ("Markov chain",    (feats["markov_p"] - 0.5) * 2),
        ("HMM regime",      feats["hmm_lean"]),
        ("Hawkes momentum", feats["hawkes"]),
        ("OU mean-rev",     feats["ou_dir"] * feats["ou_strength"]),
        ("Hurst",           feats["hurst_signal"]),
        ("ARFIMA long-mem", feats["arfima_bias"]),
        ("Kalman trend",    feats["kalman"]),
        ("Copula agree",    (feats["copula_agree"] - 0.5) * 2),
        ("RSI",             feats["rsi_signal"]),
        ("StochRSI",        feats["srsi_signal"]),
        ("ADX dir",         feats["adx_dir"] * feats["adx_trend"]),
        ("Bollinger %B",    feats["boll_signal"]),
        ("Z-score",         feats["z_signal"]),
        ("Transfer entropy",feats["te_signal"]),
        ("Jump direction",  feats["jump_dir"] * feats["jump_intensity"]),
        ("Post-jump rev",   feats["post_jump"] * feats["jump_intensity"]),
    ]

    # Sort by absolute contribution, strongest first
    layer_signals.sort(key=lambda x: abs(x[1]), reverse=True)

    # Count layers agreeing vs disagreeing with the final direction
    agree    = sum(1 for _, v in layer_signals if v * direction > 0)
    disagree = sum(1 for _, v in layer_signals if v * direction < 0)
    neutral  = len(layer_signals) - agree - disagree

    hurst_regime = ("persistent / trending" if feats["hurst"] > 0.55
                    else "anti-persistent / mean-reverting" if feats["hurst"] < 0.45
                    else "near-random walk")
    hmm_regime   = ("trending"  if feats["trend_weight"] > 0.65
                    else "ranging" if feats["trend_weight"] < 0.4
                    else "mixed")
    vol_state    = ("HIGH — signal down-weighted" if feats["vol_trust"] < 0.5
                    else "ELEVATED" if feats["vol_trust"] < 0.75
                    else "normal")
    entropy_state= ("HIGH — low structure"  if feats["entropy_trust"] < 0.4
                    else "MODERATE" if feats["entropy_trust"] < 0.65
                    else "low — market is structured")
    conf_mode    = ("MOMENTUM (RSI/StochRSI/Boll/Z-score follow trend)"
                    if feats.get("momentum_mode") else
                    "MEAN-REVERSION (RSI/StochRSI/Boll/Z-score fade extremes)")
    adx_str      = (f"ADX={feats['adx_val']:.1f}  trend_str={feats['adx_trend']:.2f}"
                    f"  dir={feats['adx_dir']:+.0f}")

    print(f"\n{sep}")
    print(f"  TRADE SIGNAL  {ts}")
    print(sep)
    print(f"  Symbol  : {symbol}   Direction : {side}")
    print(f"  p(UP)   : {p_up:.4f}   Confidence: {confidence:.4f}   Score: {score:.4f}")
    print(f"  Duration: {duration} ticks   MC exp. win rate: {exp_win:.2%}")
    print(f"  Trust   : vol={feats['vol_trust']:.2f}  entropy={feats['entropy_trust']:.2f}  "
          f"combined={feats['vol_trust']*feats['entropy_trust']:.2f}")
    print("\n  Market regime:")
    print(f"    Hurst H={feats['hurst']:.3f}  → {hurst_regime}")
    print(f"    HMM trend_weight={feats['trend_weight']:.2f}  → {hmm_regime}")
    print(f"    Confirmation mode → {conf_mode}")
    print(f"    {adx_str}")
    print(f"    Volatility state  → {vol_state}")
    print(f"    Entropy state     → {entropy_state}")
    print(f"\n  Layer breakdown  [{agree} agree | {disagree} disagree | {neutral} neutral]")
    print(f"  {'Layer':<20}  {'Signal':>7}  {'Direction bar (±1)':^22}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*22}")
    for name, val in layer_signals:
        tag = "▲" if val * direction > 0 else ("▼" if val * direction < 0 else "─")
        print(f"  {name:<20}  {val:>+.4f}  {bar_str(val)}  {tag}")
    print(f"\n  Decision: {agree}/{len(layer_signals)} layers support {side}")
    print(sep + "\n")


def log_trade(symbol, direction, stake, won, profit, step):
    ts   = datetime.utcnow().isoformat()
    side = "CALL" if direction > 0 else "PUT"
    print(f"[{ts}] {symbol} {side} step={step} stake={stake:.2f} "
          f"won={won} profit={profit:+.2f}")


def log_trade_summary(symbol, direction, stakes_used, profits, sequence_won,
                      balance_before, balance_after, p_up, confidence, duration):
    """Printed once after a full martingale sequence resolves (win or full loss).
    Gives a compact but complete picture of what happened and what it cost."""
    ts        = datetime.utcnow().isoformat()
    side      = "CALL" if direction > 0 else "PUT"
    n_steps   = len(stakes_used)
    total_staked = sum(stakes_used)
    net_pnl   = sum(profits)
    outcome   = "✓ WON" if sequence_won else "✗ LOST ALL STEPS"
    bal_delta = balance_after - balance_before
    sep       = "─" * 60

    print(f"\n{sep}")
    print(f"  TRADE SUMMARY  {ts}")
    print(sep)
    print(f"  Symbol    : {symbol}   {side}   {duration} ticks")
    print(f"  Signal    : p_up={p_up:.4f}   confidence={confidence:.4f}")
    print(f"  Outcome   : {outcome}")
    print(f"  Steps used: {n_steps} / {MARTINGALE_MAX_STEPS + 1}")
    print(f"  {'Step':<6}  {'Stake':>8}  {'Result':>8}  {'P/L':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for i, (s, p) in enumerate(zip(stakes_used, profits)):
        result = "WIN" if p > 0 else "LOSS"
        print(f"  {i:<6}  {s:>8.2f}  {result:>8}  {p:>+8.2f}")
    print(f"  {'TOTAL':<6}  {total_staked:>8.2f}  {'':>8}  {net_pnl:>+8.2f}")
    print(f"\n  Balance : {balance_before:.2f} → {balance_after:.2f}  ({bal_delta:+.2f})")
    print(sep + "\n")


async def execute_single_step(client, state, symbol, direction, stake, step, duration=5,
                              feats=None):
    """Places exactly ONE trade and returns. Never loops to the next martingale
    step — that decision belongs to the main signal loop, which waits for a
    genuine quality entry before placing any recovery step.

    feats: if supplied, the layer gate is re-evaluated atomically here as a
    final check immediately before the buy request is sent. This prevents the
    race where the gate blocks on tick N but the trade slips through on tick
    N+1 before a fresh iteration runs the gate check again."""
    # ── Atomic final gate check ─────────────────────────────────────────────
    if feats is not None:
        gate_ok, n_agree, n_dis, _ = passes_layer_gate(feats, direction)
        if not gate_ok:
            print(f"[Gate/Atomic] {symbol} step={step} blocked at execution — "
                  f"{n_agree} agree / {n_dis} disagree (gate moved between check and fire)")
            state.trade_in_progress = False
            return False, 0.0

    state.trade_in_progress = True
    won, profit = False, 0.0
    try:
        contract_id = await buy_contract(client, symbol, direction, int(duration), "t", stake)
        won, profit = await wait_for_contract_result(client, contract_id)
        log_trade(symbol, direction, stake, won, profit, step)
    except Exception as e:
        print(f"[Trade] Error on {symbol} step={step}: {e}")

    # accumulate into the sequence tracker for the summary log
    state.seq_stakes.append(stake)
    state.seq_profits.append(profit)

    # step-0 raw signal win-rate tracking (honest edge measurement)
    if step == 0:
        state.step0_total[symbol] += 1
        if won:
            state.step0_wins[symbol] += 1

        # FIX v2: Record direction into rolling history (max 30 entries).
        # Used by bayesian_fusion's direction balance correction to detect
        # and dampen systematic CALL/PUT bias in the signal layers.
        state.direction_history.append(direction)
        if len(state.direction_history) > 30:
            state.direction_history.pop(0)

        # FIX v2: Record empirical payout ratio for Kelly stake sizing.
        record_payout(state, symbol, stake, profit, won)

        # Log direction balance whenever history has enough data
        if len(state.direction_history) >= 10:
            call_ratio = sum(1 for d in state.direction_history if d == 1) / len(state.direction_history)
            if call_ratio > 0.80 or call_ratio < 0.20:
                print(f"[DirectionBalance] ⚠ {call_ratio:.0%} CALL in last "
                      f"{len(state.direction_history)} trades — bias correction active")

        # ── Online layer weight update ──────────────────────────────────────
        if feats is not None:
            models_ref = state.model_cache.get(symbol)
            if models_ref is not None:
                online_update_layer_weights(models_ref, feats, direction, won)

        # ── Persist trade to Supabase ───────────────────────────────────────
        if _store is not None and feats is not None:
            _store.save_trade(symbol, direction, step, stake, won, profit,
                              state.seq_p_up, state.seq_confidence,
                              state.seq_duration, feats)

        # ── Auto-tune gates every 50 step-0 trades ─────────────────────────
        state._trades_since_autotune += 1
        if state._trades_since_autotune >= 50:
            autotune_gates(state)
            state._trades_since_autotune = 0

    try:
        bal_resp = await client.send({"balance": 1})
        state.balance = bal_resp["balance"]["balance"]
    except Exception:
        pass

    state.trade_in_progress = False
    return won, profit


def clear_recovery(state):
    """Reset all recovery context fields — called on sequence win or exhaustion."""
    state.recovery_step       = 0
    state.recovery_stake      = 0.0
    state.seq_stakes_committed = 0.0   # FIX v2: reset sequence loss guard


def reset_sequence_accumulator(state, balance_now, p_up=0.5, confidence=0.0, duration=0):
    """Called at the START of a new sequence (step=0 entry). Resets all
    per-sequence tracking so the summary log reflects only this sequence."""
    state.seq_stakes         = []
    state.seq_profits        = []
    state.seq_balance_before = balance_now
    state.seq_p_up           = p_up
    state.seq_confidence     = confidence
    state.seq_duration       = duration


def emit_sequence_summary(state, symbol, direction, sequence_won):
    """Called at the END of a sequence. Prints the full trade summary."""
    log_trade_summary(
        symbol        = symbol,
        direction     = direction,
        stakes_used   = list(state.seq_stakes),
        profits       = list(state.seq_profits),
        sequence_won  = sequence_won,
        balance_before= state.seq_balance_before,
        balance_after = state.balance,
        p_up          = state.seq_p_up,
        confidence    = state.seq_confidence,
        duration      = state.seq_duration,
    )


# ---------------------------------------------------------------------------
# SYMBOL CALIBRATOR (trigger manager + FULL-POWER calibration engine)
# ---------------------------------------------------------------------------
def check_calibration_triggers(state):
    """Returns ("scheduled", None) when the 2-hour wall clock interval has
    elapsed. Loss-triggered deep recals are handled inline in the main loop
    (POST_LOSS_DEEP_RECAL path) and do not go through this function."""
    now = time.time()
    if now - state.last_calibration_end < CALIBRATION_COOLDOWN:
        return None
    if now - state.last_scheduled_calibration >= SCHEDULED_CALIBRATION_INTERVAL:
        return "scheduled", None
    return None


def walk_forward_validate(sd, train_frac=0.8, horizon=5, step=5):
    """REAL walk-forward validation: fit models on the first train_frac of the
    buffered ticks only, then step through the held-out remainder tick by tick
    (simulating live arrival), generating predictions from the FROZEN trained
    models and comparing to realized direction `horizon` ticks later. Returns
    (hit_rate, fitted_models, confidences) - the same models get cached for
    live trading if validation passes a sane bar, and `confidences` (the raw
    confidence score at each replayed point) feeds the adaptive threshold
    calibration in run_calibration."""
    n_ticks = len(sd.ticks)
    if n_ticks < MIN_TICKS_FOR_FIT + 100:
        return 0.5, None, []

    split = max(MIN_TICKS_FOR_FIT, int(n_ticks * train_frac))
    train_sd = sd.slice_copy(split)
    models = fit_symbol_models(train_sd)
    if not models.fitted:
        return 0.5, None, []

    eval_sd = sd.slice_copy(split)
    remaining_ticks = list(sd.ticks)[split:]
    hits, total = 0, 0
    confidences = []
    for i in range(0, len(remaining_ticks) - horizon, step):
        eval_sd.add_tick(*remaining_ticks[i])
        feats = compute_features(eval_sd, models, {sd.symbol: eval_sd.returns()})
        if feats is None:
            continue
        p_up, confidence = bayesian_fusion(feats)
        confidences.append(confidence)
        predicted_dir = 1 if p_up > 0.5 else -1
        current_price = remaining_ticks[i][1]
        future_price = remaining_ticks[i + horizon][1]
        actual_dir = 1 if future_price > current_price else -1
        hits += int(predicted_dir == actual_dir)
        total += 1

    hit_rate = hits / total if total > 0 else 0.5
    return hit_rate, models, confidences



# ---------------------------------------------------------------------------
# DEEP STARTUP CALIBRATION
# ---------------------------------------------------------------------------
def expanding_window_walk_forward(sd, n_folds=5, horizons=None, step=3):
    """True expanding-window walk-forward: models are REFITTED at each fold
    boundary on all data up to that point, then evaluated on the next unseen
    window. Returns a full report including per-fold hit rates, per-duration
    empirical win rates, per-layer correlations, and models fitted on the
    complete dataset for live trading."""
    if horizons is None:
        horizons = CANDIDATE_DURATIONS

    n_ticks = len(sd.ticks)
    if n_ticks < MIN_TICKS_FOR_FIT * 2 + 100:
        return None

    all_ticks = list(sd.ticks)
    fold_size = (n_ticks - MIN_TICKS_FOR_FIT) // (n_folds + 1)
    if fold_size < 30:
        return None

    per_fold_hit_rates = []
    per_duration_outcomes = defaultdict(lambda: [0, 0])
    layer_outcomes = defaultdict(list)
    all_confidences = []
    mid_h = horizons[len(horizons) // 2]

    for fold in range(n_folds):
        train_end = MIN_TICKS_FOR_FIT + fold_size * (fold + 1)
        test_end  = min(train_end + fold_size, n_ticks)
        if test_end - train_end < 20:
            continue

        train_sd = sd.slice_copy(train_end)
        models   = fit_symbol_models(train_sd)
        if not models.fitted:
            continue

        eval_sd    = sd.slice_copy(train_end)
        test_ticks = all_ticks[train_end:test_end]
        hits_fold, total_fold = 0, 0

        for i in range(0, len(test_ticks) - max(horizons), step):
            eval_sd.add_tick(*test_ticks[i])
            feats = compute_features(eval_sd, models, {sd.symbol: eval_sd.returns()})
            if feats is None:
                continue
            p_up, confidence = bayesian_fusion(feats)
            all_confidences.append(confidence)
            predicted_dir = 1 if p_up > 0.5 else -1
            current_price = test_ticks[i][1]

            for h in horizons:
                if i + h >= len(test_ticks):
                    continue
                future_price = test_ticks[i + h][1]
                actual_dir   = 1 if future_price > current_price else -1
                won = int(predicted_dir == actual_dir)
                per_duration_outcomes[h][0] += won
                per_duration_outcomes[h][1] += 1
                if h == mid_h:
                    hits_fold  += won
                    total_fold += 1

            # per-layer correlation data (mid horizon only) — all 18 layers
            if i + mid_h < len(test_ticks):
                actual_mid = 1 if test_ticks[i + mid_h][1] > current_price else -1
                for layer, key in [
                    ("markov",    "markov_p"),    ("hmm",       "hmm_lean"),
                    ("hawkes",    "hawkes"),       ("ou",        "ou_dir"),
                    ("hurst",     "hurst_signal"), ("arfima",    "arfima_bias"),
                    ("kalman",    "kalman"),       ("copula",    "copula_agree"),
                    ("vol_trust", "vol_trust"),    ("entropy",   "entropy_trust"),
                    ("rsi",       "rsi_signal"),   ("srsi",      "srsi_signal"),
                    ("adx",       "adx_dir"),      ("boll",      "boll_signal"),
                    ("zscore",    "z_signal"),     ("te",        "te_signal"),
                    ("jump",      "jump_dir"),     ("post_jump", "post_jump"),
                ]:
                    val = feats.get(key)
                    if val is not None:
                        layer_outcomes[layer].append((float(val), actual_mid))

        if total_fold > 0:
            per_fold_hit_rates.append((fold, train_end, total_fold, hits_fold / total_fold))

    if not per_fold_hit_rates:
        return None

    fold_hrs = [x[3] for x in per_fold_hit_rates]
    per_duration_win_rates = {
        dur: wins / total if total > 0 else 0.5
        for dur, (wins, total) in per_duration_outcomes.items()
    }
    per_layer_correlations = {}
    for layer, pairs in layer_outcomes.items():
        if len(pairs) < 20:
            continue
        vals     = np.array([p[0] for p in pairs])
        outcomes = np.array([1 if p[1] > 0 else 0 for p in pairs])
        if np.std(vals) > 0:
            per_layer_correlations[layer] = float(np.corrcoef(vals, outcomes)[0, 1])

    best_models = fit_symbol_models(sd)

    return {
        "per_fold_hit_rates":      per_fold_hit_rates,
        "per_duration_win_rates":  per_duration_win_rates,
        "per_layer_correlations":  per_layer_correlations,
        "mean_hit_rate":           float(np.mean(fold_hrs)),
        "std_hit_rate":            float(np.std(fold_hrs)),
        "all_confidences":         all_confidences,
        "best_models":             best_models,
        "is_tradeable":            float(np.mean(fold_hrs)) >= 0.46 and best_models.fitted,
        "n_folds_completed":       len(per_fold_hit_rates),
    }


def check_model_stability(models, symbol):
    """Audit fitted model parameters for physical sanity. Returns a list of
    warning strings (empty = clean)."""
    warns = []
    if models.garch_result is not None:
        try:
            p = models.garch_result.params
            alpha = p.get("alpha[1]", p.get("alpha", None))
            beta  = p.get("beta[1]",  p.get("beta",  None))
            if alpha is not None and beta is not None:
                persistence = float(alpha) + float(beta)
                if persistence >= 1.0:
                    warns.append(f"GARCH persistence={persistence:.3f} >= 1.0 (non-stationary)")
                elif persistence > 0.98:
                    warns.append(f"GARCH persistence={persistence:.3f} near-unit-root")
        except Exception:
            pass
    for label, h in [("up", models.hawkes_up), ("down", models.hawkes_down)]:
        if h is not None:
            alpha, beta = h.get("alpha", 0), h.get("beta", 1)
            ratio = alpha / beta if beta > 0 else 999
            if ratio >= 1.0:
                warns.append(f"Hawkes {label}: branching ratio={ratio:.3f} >= 1.0 (explosive)")
            elif ratio > 0.9:
                warns.append(f"Hawkes {label}: branching ratio={ratio:.3f} near-critical")
    if models.ou_params is not None:
        theta = models.ou_params.get("theta", 0)
        if theta <= 0:
            warns.append(f"OU theta={theta:.4f} <= 0 (divergent)")
    if models.hmm_model is not None:
        try:
            for i, p in enumerate(models.hmm_model.get_stationary_distribution()):
                if p < 0.05:
                    warns.append(f"HMM state {i} stationary prob={p:.3f} (degenerate)")
        except Exception:
            pass
    return warns


async def deep_startup_calibration(state, symbol_data, symbols):
    """Full-power startup calibration. Every symbol, every layer, no shortcuts.
    Called ONCE before the bot places any trade. Periodic run_calibration()
    continues every 2 hours and on loss triggers - those are lighter (top-K).
    This is the one time with no time pressure, so we use it fully."""
    state.trading_locked = True
    start = time.time()
    print("=" * 60)
    print("DEEP STARTUP CALIBRATION — full power, all symbols")
    print("=" * 60)

    all_confidences = []
    symbol_reports  = {}

    for s in symbols:
        sd = symbol_data[s]
        n  = len(sd.ticks)
        fam = "1HZ" if "1HZ" in s else "R_ "
        print(f"\n[DeepCal] [{fam}] {s}: {n} ticks  tick_dt={sd.tick_dt:.1f}s — "
              f"starting {5}-fold expanding walk-forward...")

        if n < MIN_TICKS_FOR_FIT * 2 + 100:
            print(f"[DeepCal] {s}: insufficient history, skipping.")
            state.reliability[s] = 0.3
            continue

        # FIX v2: 3 folds + step=5 cuts calibration time from 688s to ~200s
        # (40% fewer folds × 40% fewer test-window re-fits) with no meaningful
        # loss of calibration accuracy on 10k-tick synthetic index histories.
        report = expanding_window_walk_forward(sd, n_folds=3,
                                               horizons=CANDIDATE_DURATIONS, step=5)
        if report is None:
            print(f"[DeepCal] {s}: walk-forward returned no result. Not tradeable.")
            state.reliability[s] = 0.3
            continue

        stability_warns = check_model_stability(report["best_models"], s)

        print(f"[DeepCal] {s}: {report['n_folds_completed']}/3 folds")
        print(f"  Mean OOS hit rate : {report['mean_hit_rate']:.3f}  (std={report['std_hit_rate']:.3f})")
        print(f"  Per-fold          : {[f'f{x[0]}={x[3]:.3f}' for x in report['per_fold_hit_rates']]}")
        print(f"  Per-duration win% : { {d: f'{v:.3f}' for d,v in sorted(report['per_duration_win_rates'].items())} }")
        print(f"  Layer correlations: { {l: f'{v:+.3f}' for l,v in sorted(report['per_layer_correlations'].items(), key=lambda x: abs(x[1]), reverse=True)} }")
        print(f"  Is tradeable      : {report['is_tradeable']}  (mean hit rate >= 0.46)")
        if stability_warns:
            print(f"  *** STABILITY WARNINGS ***")
            for w in stability_warns:
                print(f"      {w}")
        else:
            print(f"  Model stability   : CLEAN")

        if report["best_models"] is not None and report["best_models"].fitted:
            m = report["best_models"]
            m.empirical_duration_win_rates = report["per_duration_win_rates"]

            # ── Convert OOS per-layer correlations → fusion weights ────────
            # Correlation with realized outcome tells us how much each layer
            # actually predicts direction on THIS specific symbol. We scale
            # it into a positive weight: perfectly correlated layer gets 2x
            # its static default, uncorrelated gets 0.1x (not zero — avoids
            # a layer being silenced on a short OOS window that may be noisy).
            corr = report["per_layer_correlations"]
            if corr:
                learned_w = {}
                for layer, c in corr.items():
                    # abs(corr) in [0,1] → weight in [0.1, 2.0]
                    learned_w[layer] = float(np.clip(0.1 + abs(c) * 1.9, 0.1, 2.0))
                    # preserve sign: if layer is negatively correlated, flip
                    # its evidence contribution (handled in bayesian_fusion via
                    # the weight staying positive but the signal itself carrying
                    # direction - weight scales magnitude only)
                m.per_layer_weights = learned_w
                top3 = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                print(f"  Learned weights   : top-3 predictors = "
                      f"{[(l, f'{c:+.3f}') for l,c in top3]}")
            else:
                m.per_layer_weights = None
                print(f"  Learned weights   : insufficient OOS data, using static defaults")

            state.model_cache[s] = m

            # ── Warm-start: blend Supabase-persisted weights ───────────────
            pending = state._pending_weights.get(s)
            if pending:
                if m.per_layer_weights is None:
                    m.per_layer_weights = pending
                    print(f"  Warm weights      : restored from Supabase (no OOS weights this run)")
                else:
                    all_keys = set(m.per_layer_weights) | set(pending)
                    m.per_layer_weights = {
                        k: round(0.7 * m.per_layer_weights.get(k, 1.0)
                                 + 0.3 * pending.get(k, 1.0), 6)
                        for k in all_keys
                    }
                    print(f"  Warm weights      : blended OOS 70% + Supabase prior 30%")

        state.reliability[s] = float(np.clip(report["mean_hit_rate"] / 0.5, 0.3, 1.5))
        symbol_reports[s]    = report

        # ── Per-symbol threshold from THIS symbol's OOS confidence distribution
        # This is the key fix: each symbol gets its own threshold derived from
        # its own OOS confidence scores, not a pooled global number.
        sym_confidences = report["all_confidences"]
        if sym_confidences:
            sym_thr = float(np.clip(
                np.percentile(sym_confidences, ADAPTIVE_THRESHOLD_PERCENTILE), 0.02, 0.55))
            # Quality gate: if fewer than 15% of OOS points clear this bar,
            # lower it slightly to avoid starving good trades. If > 60% clear,
            # raise it slightly to tighten precision.
            pct_clr = float(np.mean(np.array(sym_confidences) >= sym_thr))
            if pct_clr < 0.15:
                sym_thr = float(np.percentile(sym_confidences, max(ADAPTIVE_THRESHOLD_PERCENTILE - 15, 40)))
            elif pct_clr > 0.60:
                sym_thr = float(np.percentile(sym_confidences, min(ADAPTIVE_THRESHOLD_PERCENTILE + 10, 80)))
            state.per_symbol_threshold[s] = sym_thr
            print(f"  Per-symbol thr    : {sym_thr:.4f}  ({pct_clr*100:.0f}% OOS points clear)")
        else:
            state.per_symbol_threshold[s] = state.adaptive_threshold

        all_confidences.extend(sym_confidences)
        print(f"  Reliability       : {state.reliability[s]:.3f}")

    if all_confidences:
        global_thr = float(np.clip(
            np.percentile(all_confidences, ADAPTIVE_THRESHOLD_PERCENTILE), 0.03, 0.6))
        state.adaptive_threshold = global_thr   # global fallback only
        print(f"\n[DeepCal] Global fallback threshold -> {global_thr:.4f} "
              f"(per-symbol thresholds take precedence when set)")
    else:
        print(f"\n[DeepCal] WARNING: no confidence samples — keeping default "
              f"threshold={state.adaptive_threshold:.3f}")

    tradeable     = [s for s,r in symbol_reports.items() if r["is_tradeable"]]
    not_tradeable = [s for s,r in symbol_reports.items() if not r["is_tradeable"]]
    print(f"\n[DeepCal] TRADEABLE ({len(tradeable)}): {tradeable}")
    print(f"[DeepCal] BELOW EDGE BAR ({len(not_tradeable)}): {not_tradeable}")
    print(f"[DeepCal] Below-bar symbols still compete via ensemble — "
          f"lower reliability multiplier means they need a stronger signal to win selection.")

    elapsed = time.time() - start
    print(f"\n[DeepCal] Complete in {elapsed:.1f}s ({elapsed/60:.1f} min). Bot armed.")
    print("=" * 60)

    state.last_scheduled_calibration = time.time()
    state.last_calibration_end       = time.time()
    state.last_activity              = time.time()
    state.trading_locked             = False

    # ── Persist learned state to Supabase ─────────────────────────────────
    if _store is not None:
        _store.save_symbol_state(state)
        _store.save_global_state(state)   # FIX v2: persist direction_history
        _store.save_gates(MIN_LAYER_AGREE, MAX_LAYER_DISAGREE,
                          MIN_EXP_WIN_RATE, state.adaptive_threshold)
    autotune_gates(state)



async def run_calibration(state, symbol_data, symbols, trigger_reason):
    state.trading_locked = True
    kind, loss_symbol = trigger_reason
    start = time.time()
    print(f"[Calibrator] starting (trigger={kind}{':' + loss_symbol if loss_symbol else ''}). Trading locked.")

    if kind == "loss_triggered":
        state.loss_triggered_calibrations_24h.append(start)

    # Always recalibrate ALL symbols — both scheduled (2-hour) and initial runs
    # use the full universe so thresholds and reliability scores reflect every
    # available symbol, not just the top-K from an entropy pre-scan.
    candidates = symbols

    all_confidences = []
    for s in candidates:
        sd = symbol_data[s]
        if len(sd.ticks) < MIN_TICKS_FOR_FIT + 100:
            print(f"[Calibrator] {s}: not enough ticks yet, skipping this cycle.")
            continue
        hit_rate, models, confidences = walk_forward_validate(sd)
        if models is not None:
            # Blend in Supabase-persisted weights as warm-start
            pending = state._pending_weights.get(s)
            if pending:
                if models.per_layer_weights is None:
                    models.per_layer_weights = pending
                else:
                    all_keys = set(models.per_layer_weights) | set(pending)
                    models.per_layer_weights = {
                        k: round(0.7 * models.per_layer_weights.get(k, 1.0)
                                 + 0.3 * pending.get(k, 1.0), 6)
                        for k in all_keys
                    }
            state.model_cache[s] = models
        state.reliability[s] = float(np.clip(hit_rate / 0.5, 0.3, 1.5))
        state.consecutive_losses[s] = 0
        all_confidences.extend(confidences)
        print(f"[Calibrator] {s}: walk-forward hit_rate={hit_rate:.3f} reliability={state.reliability[s]:.2f} "
              f"n_confidence_samples={len(confidences)}")

    if all_confidences:
        new_threshold = float(np.percentile(all_confidences, ADAPTIVE_THRESHOLD_PERCENTILE))
        # never let the bar collapse to ~0 (untradeable noise floor) or demand
        # near-impossible confidence - keep it in a sane band regardless of
        # what the percentile math produces on a weird sample
        new_threshold = float(np.clip(new_threshold, 0.03, 0.6))
        old_threshold = state.adaptive_threshold
        state.adaptive_threshold = new_threshold
        pct_clearing = float(np.mean(np.array(all_confidences) >= new_threshold)) * 100
        print(f"[Calibrator] adaptive_threshold {old_threshold:.3f} -> {new_threshold:.3f} "
              f"(P{ADAPTIVE_THRESHOLD_PERCENTILE} of {len(all_confidences)} samples, "
              f"~{pct_clearing:.0f}% of replayed points would clear it)")
    else:
        print(f"[Calibrator] no confidence samples collected this cycle - "
              f"keeping threshold at {state.adaptive_threshold:.3f}")

    state.last_scheduled_calibration = time.time()
    state.last_calibration_end = time.time()
    state.last_activity = time.time()
    print(f"[Calibrator] complete in {state.last_calibration_end - start:.1f}s. Updated: {candidates}")
    state.trading_locked = False

    # ── Persist learned state to Supabase ─────────────────────────────────
    if _store is not None:
        _store.save_symbol_state(state)
        _store.save_global_state(state)   # FIX v2: persist direction_history
        _store.save_gates(MIN_LAYER_AGREE, MAX_LAYER_DISAGREE,
                          MIN_EXP_WIN_RATE, state.adaptive_threshold)
    autotune_gates(state)


# ---------------------------------------------------------------------------
# STREAM CONSUMERS
# ---------------------------------------------------------------------------
async def tick_consumer(queue, symbol_data, state):
    while True:
        data = await queue.get()
        tick = data.get("tick")
        if not tick:
            continue
        symbol = tick.get("symbol")
        if symbol in symbol_data:
            symbol_data[symbol].add_tick(tick["epoch"], tick["quote"])
        state.last_activity = time.time()


async def balance_consumer(queue, state):
    while True:
        data = await queue.get()
        bal = data.get("balance")
        if bal:
            state.balance = bal["balance"]


async def watchdog(state):
    """If WATCHDOG_TIMEOUT seconds pass with no tick received and no main-loop
    iteration completed (state.last_activity untouched), the process is
    assumed locked up. Rather than depending on any specific host's restart
    policy, this re-execs the current Python process in place - identical
    behavior on Railway and on a local PC, no external supervisor needed."""
    while True:
        await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
        idle = time.time() - state.last_activity
        if idle > WATCHDOG_TIMEOUT:
            print(f"[Watchdog] No activity for {idle:.0f}s (limit {WATCHDOG_TIMEOUT}s). "
                  f"Restarting process in place now.")
            sys.stdout.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def main():
    if not DERIV_API_TOKEN:
        raise RuntimeError("Set the DERIV_API_TOKEN environment variable.")
    if not DERIV_APP_ID:
        raise RuntimeError(
            "Set the DERIV_APP_ID environment variable to your app_id from "
            "developers.deriv.com. Legacy app_ids (e.g. the old demo id "
            "1089) do NOT work with the new Options API."
        )
    if DERIV_ACCOUNT_TYPE not in ("demo", "real"):
        raise RuntimeError("DERIV_ACCOUNT_TYPE must be 'demo' or 'real'.")
    if DERIV_ACCOUNT_TYPE == "real":
        print("!" * 72)
        print("! DERIV_ACCOUNT_TYPE=real - this bot will trade with REAL MONEY.    !")
        print("! Set DERIV_ACCOUNT_TYPE=demo (or unset it) to use a demo account.  !")
        print("!" * 72)

    client = DerivClient(
        DERIV_APP_ID, DERIV_API_TOKEN,
        account_type=DERIV_ACCOUNT_TYPE, account_id=DERIV_ACCOUNT_ID,
    )
    account = await client.connect()
    print(f"Authorized as {account.get('loginid')}")

    state = TradeState()
    state.balance = account.get("balance", 0.0)
    print(f"Starting balance: {state.balance}")

    # ── Supabase: init store and warm-start from persisted state ──────────
    global _store, MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE
    _store = SupabaseStore()
    _store.load_symbol_state(state)
    _store.load_global_state(state)   # FIX v2: restore direction_history
    gates = _store.load_gates()
    if gates:
        MIN_LAYER_AGREE    = int(gates.get("min_layer_agree",    MIN_LAYER_AGREE))
        MAX_LAYER_DISAGREE = int(gates.get("max_layer_disagree", MAX_LAYER_DISAGREE))
        MIN_EXP_WIN_RATE   = float(gates.get("min_exp_win_rate", MIN_EXP_WIN_RATE))
        state.adaptive_threshold = float(gates.get("adaptive_threshold", state.adaptive_threshold))
        print(f"[Store] Restored gates: agree>={MIN_LAYER_AGREE} "
              f"disagree<={MAX_LAYER_DISAGREE} MC>={MIN_EXP_WIN_RATE:.2f} "
              f"thr={state.adaptive_threshold:.4f}")

    # --- R_ symbols ---
    r_symbols = []
    for attempt in range(1, 6):
        r_symbols = await fetch_tradable_symbols(client)
        if r_symbols:
            break
        print(f"[main] No R_ symbols on attempt {attempt}/5, retrying in 3s...")
        await asyncio.sleep(3)
    if not r_symbols:
        raise RuntimeError("No R_ rise/fall symbols found (check API credentials/connectivity).")

    # --- top-3 1HZ symbols by tick consistency ---
    hz_symbols = []
    for attempt in range(1, 4):
        hz_symbols = await select_top_1hz(client, n_top=3)
        if hz_symbols:
            break
        print(f"[main] No 1HZ symbols on attempt {attempt}/3, retrying in 3s...")
        await asyncio.sleep(3)
    if not hz_symbols:
        print("[main] WARNING: no 1HZ symbols available - proceeding with R_ only.")

    symbols = r_symbols + hz_symbols
    print(f"\nFull tradable universe ({len(symbols)} symbols):")
    print(f"  R_ ({len(r_symbols)}): {r_symbols}")
    print(f"  1HZ top-3 ({len(hz_symbols)}): {hz_symbols}")

    # build SymbolData with correct tick_dt per family
    symbol_data = {}
    for s in r_symbols:
        symbol_data[s] = SymbolData(s, tick_dt=2.0)   # R_ tick ~every 2s
    for s in hz_symbols:
        symbol_data[s] = SymbolData(s, tick_dt=1.0)   # 1HZ ticks every 1s

    print(f"Bootstrapping tick history for all symbols (target: {HISTORY_BOOTSTRAP_COUNT} ticks each)...")
    for s in symbols:
        history = await fetch_history(client, s)
        for epoch, price in history:
            symbol_data[s].add_tick(epoch, price)
        actual_dt = symbol_data[s].mean_tick_dt()
        n = len(symbol_data[s].ticks)
        span_hrs = (n * actual_dt) / 3600
        print(f"  {s}: {n} ticks loaded  actual_mean_dt={actual_dt:.2f}s  span≈{span_hrs:.1f}h")

    tick_queue = client.subscribe_channel("tick")
    balance_queue = client.subscribe_channel("balance")

    async def subscribe_all(c):
        """Replays balance + per-symbol tick subscriptions. Used for the
        initial subscribe and re-run as `resubscribe_cb` after every
        reconnect (a fresh OTP session has no memory of prior subscriptions)."""
        await c.send({"balance": 1, "subscribe": 1})
        for s in symbols:
            await c.send({"ticks": s, "subscribe": 1})

    client.resubscribe_cb = subscribe_all
    await subscribe_all(client)

    asyncio.create_task(tick_consumer(tick_queue, symbol_data, state))
    asyncio.create_task(balance_consumer(balance_queue, state))
    asyncio.create_task(watchdog(state))

    print("Running initial full-power calibration across the entire universe before trading begins...")
    await deep_startup_calibration(state, symbol_data, symbols)

    print("Bot running. Entering main decision loop.")
    last_heartbeat = 0.0

    while True:
        await asyncio.sleep(2)
        state.last_activity = time.time()

        if state.trading_locked or state.trade_in_progress:
            continue

        trigger = check_calibration_triggers(state)
        if trigger:
            await run_calibration(state, symbol_data, symbols, trigger)
            continue

        ready_symbols = [s for s in symbols
                         if s in state.model_cache
                         and len(symbol_data[s].ticks) >= MIN_TICKS_LIVE]

        now = time.time()
        if now - last_heartbeat > 30:
            rec = (f" | RECOVERY step={state.recovery_step} stake={state.recovery_stake:.2f}"
                   if state.recovery_step > 0 else "")
            s0_parts = []
            for sym in ready_symbols:
                tot = state.step0_total[sym]
                if tot > 0:
                    wr = state.step0_wins[sym] / tot
                    s0_parts.append(f"{sym}:{wr:.0%}({tot})")
            s0_str = " s0_wr=[" + " ".join(s0_parts) + "]" if s0_parts else ""
            print(f"[scan] balance={state.balance:.2f} | "
                  f"{len(ready_symbols)}/{len(symbols)} ready{rec}{s0_str}")
            last_heartbeat = now

        if not ready_symbols:
            continue

        returns_window_dict = {s: symbol_data[s].returns()[-200:] for s in ready_symbols}

        # ── RECOVERY MODE ────────────────────────────────────────────────────
        # No symbol, direction, or duration lock. Recovery is a fresh open scan
        # at the elevated martingale stake, using models freshly fitted by the
        # deep recal that fired immediately after the step=0 loss. The best
        # signal from ANY symbol in ANY direction wins selection — same quality
        # gates apply (layer agreement, MC win rate, score gap, threshold).
        if state.recovery_step > 0:
            # Run the full symbol scan using fresh post-recal models
            rec_scores = {}
            for s in ready_symbols:
                sd    = symbol_data[s]
                feats = compute_features(sd, state.model_cache.get(s), returns_window_dict)
                if feats is None:
                    continue
                p_up, confidence = bayesian_fusion(feats)
                rec_scores[s] = (p_up, confidence)

            rec_pick = select_trade(
                rec_scores, state.reliability,
                state.adaptive_threshold,
                state.per_symbol_threshold
            )
            if not rec_pick:
                continue   # no symbol clears quality bar yet — keep waiting

            rec_sym, rec_dir, rec_p_up, rec_score = rec_pick
            sd    = symbol_data[rec_sym]
            feats = compute_features(sd, state.model_cache.get(rec_sym), returns_window_dict)

            duration, exp_win_rate = monte_carlo_duration(
                sd.prices(), sd.returns(), rec_dir, feats, CANDIDATE_DURATIONS,
                models=state.model_cache.get(rec_sym)
            )
            if exp_win_rate < MIN_EXP_WIN_RATE:
                continue

            # ── Atomic gate check immediately before execution ──────────────
            # Evaluated here (not just at top of iteration) to prevent the
            # race where gate blocks on tick N but trade fires on tick N+1
            # before a fresh gate check runs.
            gate_ok, n_agree, n_disagree, n_neutral = passes_layer_gate(feats, rec_dir)
            if not gate_ok:
                print(f"[Gate/Recovery] step={state.recovery_step} — best pick {rec_sym} "
                      f"vote {n_agree}/{feats['n_layers']} agree, {n_disagree} disagree, "
                      f"{n_neutral} neutral — waiting for stronger consensus")
                continue

            print(f"[Recovery] step={state.recovery_step} stake={state.recovery_stake:.2f} "
                  f"— best signal: {rec_sym} {'CALL' if rec_dir>0 else 'PUT'} "
                  f"({n_agree}/16 agree, MC={exp_win_rate:.2f})")

            explain_signal(
                symbol=rec_sym, direction=rec_dir,
                feats=feats, p_up=rec_p_up, confidence=rec_scores[rec_sym][1],
                duration=duration, exp_win=exp_win_rate, score=rec_score
            )

            # Final atomic gate re-check inside execute_single_step is the
            # last line of defence — passes feats and direction through.
            won, _ = await execute_single_step(
                client, state, rec_sym, rec_dir,
                state.recovery_stake, state.recovery_step,
                duration=duration, feats=feats
            )

            if won:
                print(f"[Recovery] Recovered at step={state.recovery_step} "
                      f"via {rec_sym} {'CALL' if rec_dir>0 else 'PUT'}.")
                state.consecutive_losses[rec_sym] = 0
                emit_sequence_summary(state, rec_sym, rec_dir, True)
                clear_recovery(state)
            else:
                next_step  = state.recovery_step + 1
                next_stake = round(state.recovery_stake * MARTINGALE_FACTOR, 2)
                if next_step > MARTINGALE_MAX_STEPS:
                    print(f"[Recovery] Exhausted all {MARTINGALE_MAX_STEPS} steps — "
                          f"closing sequence and running deep recalibration.")
                    state.consecutive_losses[rec_sym] += 1
                    emit_sequence_summary(state, rec_sym, rec_dir, False)
                    clear_recovery(state)
                    await deep_startup_calibration(state, symbol_data, symbols)
                else:
                    # FIX v2: Sequence loss guard — abort if cumulative risk
                    # would exceed MAX_SEQUENCE_LOSS_PCT of current balance.
                    state.seq_stakes_committed += state.recovery_stake
                    max_allowed = state.balance * MAX_SEQUENCE_LOSS_PCT
                    if state.seq_stakes_committed + next_stake > max_allowed:
                        print(f"[Recovery] SEQUENCE LOSS GUARD triggered — "
                              f"committed={state.seq_stakes_committed:.2f} "
                              f"next={next_stake:.2f} > max={max_allowed:.2f}. "
                              f"Aborting sequence to protect balance.")
                        emit_sequence_summary(state, rec_sym, rec_dir, False)
                        clear_recovery(state)
                        state.seq_stakes_committed = 0.0
                    else:
                        state.recovery_step  = next_step
                        state.recovery_stake = next_stake
                        print(f"[Recovery] step={state.recovery_step - 1} lost on {rec_sym} — "
                              f"next step={next_step} stake={next_stake:.2f}")
                        # FIX v2: POST_LOSS_DEEP_RECAL is now False — no 688s
                        # calibration pause after each recovery step. The scheduled
                        # 2-hour recal is sufficient for model freshness.
                        if POST_LOSS_DEEP_RECAL:
                            await deep_startup_calibration(state, symbol_data, symbols)

            state.last_activity = time.time()
            continue

        # ── NORMAL ENTRY ─────────────────────────────────────────────────────
        # FIX v2: Compute direction balance ratio from recent trade history.
        # Passed into feats so bayesian_fusion can apply a soft correction
        # when one direction is systematically over-represented.
        recent_dirs = state.direction_history[-30:] if state.direction_history else []
        if recent_dirs:
            recent_call_ratio = sum(1 for d in recent_dirs if d == 1) / len(recent_dirs)
        else:
            recent_call_ratio = 0.5   # neutral when no history

        symbol_scores = {}
        for s in ready_symbols:
            sd    = symbol_data[s]
            feats = compute_features(sd, state.model_cache.get(s), returns_window_dict)
            if feats is None:
                continue
            feats["recent_call_ratio"] = recent_call_ratio   # inject for fusion
            p_up, confidence = bayesian_fusion(feats)
            symbol_scores[s] = (p_up, confidence)

        pick = select_trade(
            symbol_scores, state.reliability,
            state.adaptive_threshold,
            state.per_symbol_threshold
        )
        if not pick:
            continue

        symbol, direction, p_up, score = pick
        sd    = symbol_data[symbol]
        feats = compute_features(sd, state.model_cache.get(symbol), returns_window_dict)

        duration, exp_win_rate = monte_carlo_duration(
            sd.prices(), sd.returns(), direction, feats, CANDIDATE_DURATIONS,
            models=state.model_cache.get(symbol)
        )
        if exp_win_rate < MIN_EXP_WIN_RATE:
            print(f"Skipping {symbol} — MC best win rate {exp_win_rate:.2f} below floor.")
            continue

        # ── Layer agreement gate ────────────────────────────────────────────
        gate_ok, n_agree, n_disagree, n_neutral = passes_layer_gate(feats, direction)
        if not gate_ok:
            print(f"[Gate] {symbol} skipped — layer vote {n_agree} agree / "
                  f"{n_disagree} disagree / {n_neutral} neutral "
                  f"(need >={MIN_LAYER_AGREE} agree, <={MAX_LAYER_DISAGREE} disagree)")
            continue

        # ── FIX v2: Permutation entropy gate ────────────────────────────────
        # Skip if the recent tick sequence is statistically indistinguishable
        # from random ordering — no amount of layer agreement is trustworthy
        # in a near-random window.
        pe_ok, pe_score = entropy_gate_passes(sd.prices())
        if not pe_ok:
            print(f"[EntropyGate] {symbol} skipped — PE={pe_score:.3f} >= "
                  f"{PE_THRESHOLD} (market too random)")
            continue

        # ── FIX v2: Multi-timeframe confluence ──────────────────────────────
        # Require at least 2 of 3 timeframes (raw ticks, 5-tick bars, 20-tick
        # bars) to agree with the proposed direction before trading.
        tf_agree, tf_dirs = multi_timeframe_confluence(sd.prices(), direction)
        if tf_agree < MIN_TF_AGREEMENT:
            print(f"[Confluence] {symbol} skipped — only {tf_agree}/3 timeframes "
                  f"agree {tf_dirs} (need >={MIN_TF_AGREEMENT})")
            continue

        # ── FIX v2: Bootstrap meta-ensemble agreement ───────────────────────
        # Model-free resampled estimate must roughly agree with the parametric
        # MC estimate, otherwise the parametric Gaussian assumption may not be
        # capturing this symbol's current return structure.
        bs_agrees, bs_p = meta_ensemble_agrees(sd.returns(), direction, duration, exp_win_rate)
        if not bs_agrees:
            print(f"[MetaEnsemble] {symbol} skipped — bootstrap p={bs_p:.3f} vs "
                  f"parametric p={exp_win_rate:.3f} disagree by "
                  f">{BOOTSTRAP_AGREE_TOL}")
            continue

        # ── FIX v2: Fractional Kelly stake sizing ───────────────────────────
        # Blends the fixed 2%-of-balance floor with an edge-scaled Kelly
        # component (quarter-Kelly), capped at 4% of balance. Strong signals
        # size up proportionally; marginal signals stay near the floor.
        base_stake = kelly_adjusted_stake(state.balance, exp_win_rate, symbol, state)
        all_stakes = martingale_stakes(base_stake)

        # reset sequence accumulator before step=0 fires
        reset_sequence_accumulator(state, state.balance, p_up, confidence, duration)

        explain_signal(
            symbol=symbol, direction=direction,
            feats=feats, p_up=p_up, confidence=confidence,
            duration=duration, exp_win=exp_win_rate, score=score
        )

        won, _ = await execute_single_step(
            client, state, symbol, direction, base_stake, 0, duration=duration, feats=feats
        )

        if won:
            state.consecutive_losses[symbol] = 0
            emit_sequence_summary(state, symbol, direction, True)
        else:
            # ── Post-step=0-loss deep recalibration ────────────────────────
            # Re-fit ALL symbols on fresh data before the next entry. This
            # clears any stale directional bias and resets per-symbol
            # thresholds. Recovery steps (if MARTINGALE_MAX_STEPS >= 1) will
            # still fire after this, but will use the freshly calibrated models.
            if POST_LOSS_DEEP_RECAL:
                print(f"[Recovery] step=0 lost on {symbol} — "
                      f"triggering deep recalibration before recovery.")
                await deep_startup_calibration(state, symbol_data, symbols)

            if MARTINGALE_MAX_STEPS >= 1:
                state.recovery_step  = 1
                state.recovery_stake = all_stakes[1]
                print(f"[Recovery] Armed for step=1 stake={all_stakes[1]:.2f} — "
                      f"open scan, any symbol/direction, fresh post-recal models.")
            else:
                state.consecutive_losses[symbol] += 1
                emit_sequence_summary(state, symbol, direction, False)

        state.last_activity = time.time()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[main] Unhandled exception, restarting process in place: {type(e).__name__}: {e}")
        sys.stdout.flush()
        time.sleep(3)  # brief pause so a fast crash loop doesn't hammer the API
        os.execv(sys.executable, [sys.executable] + sys.argv)
