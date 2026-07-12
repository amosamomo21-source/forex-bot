"""
PDHL Cooldown Audit — NATGAS_USD H1, 10 years
Mirrors live bot mechanics: SL=prev_mid, TP=1.5x range, 2xATR trailing,
break-even lock at +$1,500, auto-close at +$6,000.

Modes compared:
  baseline  — no cooldown (current live bot behaviour)
  option_a  — 6h cooldown after profit_target ($6k) close only
  option_b  — 1 entry per direction per day (no same-day re-entry)
  option_c  — pullback required: price must return inside range before re-entry
"""
from dotenv import load_dotenv; load_dotenv()
import numpy as np, pandas as pd
from data import load_oanda_data

INITIAL           = 100_000
SLEEVE_FRAC       = 1 / 20      # $5,000 per sleeve (20-sleeve portfolio)
RISK_FRAC         = 1.5         # H1_RISK_PCT in demo mode
TP_MULT           = 1.5
TRAIL_ATR         = 2.0         # trailing stop multiplier
BE_LOCK_USD       = 1_500       # break-even lock threshold
PROFIT_TARGET_USD = 6_000       # auto-close profit target
COOLDOWN_H        = 6           # Option A cooldown hours after profit_target


def _atr14(h, l, c):
    h = pd.Series(h, dtype=float)
    l = pd.Series(l, dtype=float)
    c = pd.Series(c, dtype=float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=14, adjust=False).mean()


def run_simulation(h1, d1, mode="baseline"):
    d1 = d1.copy()
    d1["ph"] = d1["High"].shift(1)
    d1["pl"] = d1["Low"].shift(1)
    d1["pm"] = (d1["ph"] + d1["pl"]) / 2
    d1["pr"] = d1["ph"] - d1["pl"]
    d1_by_date = {ts.date(): row for ts, row in d1.iterrows()}

    atr_s = _atr14(h1["High"].values, h1["Low"].values, h1["Close"].values)

    equity   = INITIAL
    pos      = 0          # 1 long, -1 short, 0 flat
    ep       = sl = tp = 0.0
    units    = 0
    trades   = []

    # Option A
    cooldown_until = None

    # Option B
    day_long = day_short = None

    # Option C
    saw_inside = True

    for i in range(50, len(h1)):
        ts    = h1.index[i]
        price = float(h1["Close"].iloc[i])
        date  = ts.date()

        row = d1_by_date.get(date)
        if row is None or pd.isna(row["ph"]) or row["pr"] <= 0:
            continue
        ph = float(row["ph"])
        pl = float(row["pl"])
        pm = float(row["pm"])
        pr = float(row["pr"])
        av = float(atr_s.iloc[i])
        if np.isnan(av) or av <= 0:
            continue

        # ── Manage open position ──────────────────────────────────────────
        if pos != 0:
            unreal = pos * (price - ep) * units

            sl_hit = (pos == 1 and price <= sl) or (pos == -1 and price >= sl)
            tp_hit = (pos == 1 and price >= tp) or (pos == -1 and price <= tp)
            pt_hit = unreal >= PROFIT_TARGET_USD

            if pt_hit or tp_hit or sl_hit:
                if pt_hit:
                    exit_p, reason = price, "profit_target"
                elif tp_hit:
                    exit_p, reason = tp, "tp"
                else:
                    exit_p, reason = sl, "sl"

                pnl = pos * (exit_p - ep) * units
                equity += pnl
                trades.append({"ts": ts, "pnl": pnl, "reason": reason, "dir": pos})

                if mode == "option_a" and reason == "profit_target":
                    cooldown_until = ts + pd.Timedelta(hours=COOLDOWN_H)
                if mode == "option_c":
                    saw_inside = False

                pos = 0
                continue

            # Trailing stop
            if pos == 1:
                trail = price - TRAIL_ATR * av
                if unreal >= BE_LOCK_USD and sl < ep:
                    trail = max(trail, ep)
                sl = max(sl, trail)
            else:
                trail = price + TRAIL_ATR * av
                if unreal >= BE_LOCK_USD and sl > ep:
                    trail = min(trail, ep)
                sl = min(sl, trail)
            continue

        # ── Flat: Option C pullback tracking ─────────────────────────────
        if mode == "option_c" and not saw_inside:
            if pl <= price <= ph:
                saw_inside = True

        # ── Entry gates ───────────────────────────────────────────────────
        can_long  = price > ph
        can_short = price < pl

        if mode == "option_b":
            if can_long  and day_long  == date: can_long  = False
            if can_short and day_short == date: can_short = False

        if mode == "option_a" and cooldown_until and ts < cooldown_until:
            can_long = can_short = False

        if mode == "option_c" and not saw_inside:
            can_long = can_short = False

        # ── Entry ─────────────────────────────────────────────────────────
        sleeve_eq   = equity * SLEEVE_FRAC
        risk_amount = sleeve_eq * RISK_FRAC

        if can_long:
            sd = abs(price - pm)
            if sd > 0:
                u = int(risk_amount / sd)
                if u > 0:
                    pos = 1; ep = price; sl = pm; tp = price + TP_MULT * pr; units = u
                    if mode == "option_b": day_long = date

        elif can_short:
            sd = abs(price - pm)
            if sd > 0:
                u = int(risk_amount / sd)
                if u > 0:
                    pos = -1; ep = price; sl = pm; tp = price - TP_MULT * pr; units = u
                    if mode == "option_b": day_short = date

    return pd.DataFrame(trades)


def summary(df, yrs=10):
    if df.empty or len(df) < 5:
        return dict(n=0, sh=0, wr=0, ann=0, dd=0, pt=0, total=0)
    t   = df["pnl"]
    cum = t.cumsum() + INITIAL
    dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    ann = (cum.iloc[-1] / INITIAL) ** (1 / yrs) - 1
    sh  = t.mean() / t.std() * np.sqrt(252 * 24) if t.std() > 0 else 0   # H1 bars
    return dict(
        n=len(t), sh=sh, wr=(t > 0).mean(), ann=ann, dd=dd,
        pt=(df["reason"] == "profit_target").mean(),
        total=t.sum(),
    )


# ══════════════════════════════════════════════════════════════════════════════
print("Loading NATGAS_USD H1 + D1 (10y)…")
h1 = load_oanda_data("NATGAS_USD", period="10y", interval="1h")
d1 = load_oanda_data("NATGAS_USD", period="10y", interval="1d")
years = sorted(h1.index.year.unique())

MODES = [
    ("baseline", "Baseline — no cooldown (live bot)"),
    ("option_a", "Option A — 6h after $6k close    "),
    ("option_b", "Option B — 1/direction/day        "),
    ("option_c", "Option C — pullback required      "),
]

results = {}
for key, label in MODES:
    print(f"  Simulating {label.strip()}…")
    results[key] = (run_simulation(h1, d1, mode=key), label)

sep = "─" * 74

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'═'*74}")
print("PDHL COOLDOWN AUDIT — NATGAS_USD H1 10y")
print(f"{'═'*74}")
print(f"\n{'Mode':<36} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PT%':>5}  Total$")
print(sep)
for key, (df, label) in results.items():
    s = summary(df)
    print(f"{label:<36} {s['sh']:>+6.2f} {s['ann']*100:>+6.1f}%"
          f" {s['dd']*100:>5.1f}% {s['wr']*100:>5.1f}%"
          f" {s['n']:>5} {s['pt']*100:>4.1f}%  ${s['total']:>+10,.0f}")

# ── Year-by-year ──────────────────────────────────────────────────────────────
print(f"\n{'Year':<6} {'Baseline':>12} {'Option A':>12} {'Option B':>12} {'Option C':>12}")
print(sep)
for y in years:
    row = f"{y:<6}"
    for key, (df, _) in results.items():
        yt = df[df["ts"].dt.year == y]["pnl"].sum() if not df.empty else 0
        row += f"  {yt:>+10,.0f}"
    print(row)

# ── Profitable years per mode ─────────────────────────────────────────────────
print(f"\n{'Mode':<36} {'Profitable years':>16}")
print(sep)
for key, (df, label) in results.items():
    if df.empty:
        print(f"{label:<36}  no trades"); continue
    prof = sum(
        1 for y in years
        if df[df["ts"].dt.year == y]["pnl"].sum() > 0
    )
    print(f"{label:<36}  {prof}/{len(years)}")

# ── Close reason breakdown ────────────────────────────────────────────────────
print(f"\nClose reason breakdown (% of trades)")
print(sep)
for key, (df, label) in results.items():
    if df.empty: continue
    r = df["reason"].value_counts(normalize=True) * 100
    parts = "  ".join(f"{k}: {v:.1f}%" for k, v in r.items())
    print(f"{label.strip()}: {parts}")

# ── Re-entry analysis ─────────────────────────────────────────────────────────
print(f"\nRe-entry trades blocked per mode (vs baseline)")
baseline_n = summary(results["baseline"][0])["n"]
for key, (df, label) in results.items():
    if key == "baseline": continue
    blocked = baseline_n - summary(df)["n"]
    print(f"  {label.strip()}: {blocked:+d} trades vs baseline")
