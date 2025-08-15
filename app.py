# stick2discipline — Futures Journal (USDⓈ‑M & COIN‑M) with Discipline System
# Run: streamlit run app.py

import os, json, uuid, datetime as dt
import pandas as pd, numpy as np
import streamlit as st, altair as alt

# ================= Config =================
APP_TITLE = "🧠 stick2discipline — USDⓈ‑M & COIN‑M"
DATA_DIR = "data"
LOG_PATH = os.path.join(DATA_DIR, "trade_log.csv")
WITHDRAW_LOG = os.path.join(DATA_DIR, "withdrawals.csv")

DEFAULTS = {
    "symbol": "DOTUSDT",
    "base_usdt": 1000.0,     # neutral default for public repo
    "base_coin": 0.0,        # start empty for COIN‑M
    "ref_price": 4.00,       # USD per coin (for COIN view)
    "leverage": 12,
    "risk_pct": 5.0,
    "rr": 2.0,
    "daily_target_usd": 50.0,
    "checkpoints_usd": [500.0, 1000.0, 2000.0],
    "checkpoint_withdrawals_usd": [50.0, 100.0, 200.0],
    "face_value_usd": 10.0,  # DOT COIN‑M: 1 contract = 10 USD
}

REASON_TAGS = [
    "FVG","OB_Wall","Liquidity_Imbalance","Tape_Burst",
    "Trend_15m","Trend_4h","BTC.D","News_Pos","News_Neg","Confluence_3+"
]
FEELINGS = [
    "Calm/Focused","Confident","Flow State",
    "FOMO","Revenge","Over‑confident","Anxious","Hesitant/Fearful",
    "Bored","Tired","Distracted","Impatient"
]

# ================= Utils =================
def ensure_dirs(): os.makedirs(DATA_DIR, exist_ok=True)

def load_df(path, columns):
    ensure_dirs()
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            for c in columns:
                if c not in df.columns: df[c] = np.nan
            return df[columns]
        except Exception:
            pass
    return pd.DataFrame(columns=columns)

def save_df(df, path): ensure_dirs(); df.to_csv(path, index=False)

def today_iso(): return dt.datetime.now().strftime("%Y-%m-%d")

def to_day(ts_like):
    """Return datetime64[ns] normalized to 00:00; safe for strings/NaT."""
    s = pd.to_datetime(ts_like, errors="coerce").dt.tz_localize(None)
    return s.dt.normalize()

# ===== USDⓈ‑M sizing (linear) =====
def suggest_qty_linear(entry, sl, risk_usd):
    if entry<=0 or sl<=0: return 0.0
    dP = abs(entry - sl)
    return 0.0 if dP==0 else round(risk_usd/dP, 3)

# ===== COIN‑M (inverse) math =====
def coinm_contracts_for_risk(entry, sl, risk_usd, face_usd):
    if entry<=0 or sl<=0 or risk_usd<=0 or face_usd<=0: return 0.0
    risk_coin = risk_usd / entry
    denom = abs(face_usd*(1.0/entry - 1.0/sl))
    return 0.0 if denom==0 else max(0.0, risk_coin/denom)

def coinm_dot_notional(entry, contracts, face_usd):
    if entry<=0 or face_usd<=0: return 0.0
    return (face_usd/entry)*contracts

def coinm_loss_coin_at_sl(entry, sl, contracts, face_usd):
    if entry<=0 or sl<=0 or contracts<=0 or face_usd<=0: return 0.0
    return abs(face_usd*contracts*(1.0/entry - 1.0/sl))

def coinm_pnl_coin(entry, close, contracts, face_usd, side):
    if entry<=0 or close<=0 or contracts<=0 or face_usd<=0: return 0.0
    if side.upper()=="LONG":
        return face_usd*contracts*((1.0/entry) - (1.0/close))
    else:
        return face_usd*contracts*((1.0/close) - (1.0/entry))

# ===== TP calc =====
def auto_tp(entry, sl, side, rr):
    if entry==0 or sl==0: return 0.0
    risk_per_unit = abs(entry - sl)
    return round(entry + (rr*risk_per_unit if side=="LONG" else -rr*risk_per_unit), 6)

# ===== Equity curve =====
def build_equity_curve_usd(trades_df, withdraw_df, base_usd):
    t = trades_df.copy(); w = withdraw_df.copy()
    if not t.empty:
        t["date"] = to_day(t["date"]); t = t.dropna(subset=["date"])
        pnl = t.groupby("date", as_index=True)["pnl_usd"].sum()
    else:
        pnl = pd.Series(dtype="float64")
    if not w.empty:
        w["date"] = to_day(w["date"]); w = w.dropna(subset=["date"])
        wd  = w.groupby("date", as_index=True)["amount_usd"].sum()
    else:
        wd = pd.Series(dtype="float64")

    all_dates = sorted(set(pnl.index.tolist()) | set(wd.index.tolist()))
    if not all_dates:
        return pd.DataFrame({"date":[pd.Timestamp.today().normalize()], "equity_usd":[base_usd]})

    pnl_aligned = pd.Series(index=all_dates, data=0.0).add(pnl, fill_value=0.0)
    wd_aligned  = pd.Series(index=all_dates, data=0.0).add(wd,  fill_value=0.0)

    df = pd.DataFrame({"date": all_dates, "pnl": pnl_aligned.values, "withdraw": wd_aligned.values})
    df["cum_pnl"] = df["pnl"].cumsum()
    df["cum_wd"]  = df["withdraw"].cumsum()
    df["equity_usd"] = base_usd + df["cum_pnl"] - df["cum_wd"]
    return df[["date","equity_usd"]]

def build_tag_winrate(trades_df):
    if trades_df.empty or "reason_tags" not in trades_df.columns:
        return pd.DataFrame(columns=["tag","trades","wins","winrate"])
    df = trades_df.copy()
    df["reason_tags"] = df["reason_tags"].fillna("").astype(str)
    rows=[]
    for _,r in df.iterrows():
        tags=[t.strip() for t in r["reason_tags"].split(",") if t.strip()]
        for t in set(tags): rows.append({"tag":t,"result":r["result"]})
    if not rows: return pd.DataFrame(columns=["tag","trades","wins","winrate"])
    x=pd.DataFrame(rows)
    g=x.groupby("tag")["result"].value_counts().unstack(fill_value=0).reset_index()
    g["trades"]=g.get("WIN",0)+g.get("LOSS",0)
    g["wins"]=g.get("WIN",0)
    g["winrate"]=(g["wins"]/g["trades"].replace(0,np.nan)*100).fillna(0).round(1)
    return g.sort_values(["winrate","trades"], ascending=[False,False])

# ================= App =================
st.set_page_config(page_title="stick2discipline", page_icon="🧠", layout="wide")
st.title(APP_TITLE)
st.caption("Discipline‑first flow: Checklist → Sizing & TP → Log (exact COIN‑M math) → Review. Data stays local.")

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("⚙️ Settings")
    symbol = st.text_input("Symbol", value=DEFAULTS["symbol"])
    contract_mode = st.radio("Contract Type", ["USDⓈ‑M (linear)", "COIN‑M (inverse)"], index=1)
    leverage = st.number_input("Leverage (x)", value=int(DEFAULTS["leverage"]), step=1, min_value=1)
    risk_pct = st.slider("Risk per trade (%)", 1.0, 10.0, float(DEFAULTS["risk_pct"]), 0.5)
    rr = st.slider("Reward:Risk", 1.0, 3.0, float(DEFAULTS["rr"]), 0.1)
    strict_gating = st.toggle("🔒 Strict: Hide Sizing until Checklist is passed", value=True)
    st.divider()

    if contract_mode == "USDⓈ‑M (linear)":
        base_usd = st.number_input("Base Equity (USD)", value=float(DEFAULTS["base_usdt"]), step=10.0)
        display_currency = "USD"; face_value_usd=None; base_coin=None; ref_price=None
    else:
        base_coin = st.number_input("Base Equity (COIN units)", value=float(DEFAULTS["base_coin"]), step=0.1)
        ref_price = st.number_input("Reference Price (USD/COIN)", value=float(DEFAULTS["ref_price"]), step=0.01)
        base_usd = base_coin * ref_price
        display_currency = st.radio("Display Equity As", ["USD","COIN"], index=0)
        face_value_usd = st.number_input("COIN‑M Face (USD/contract)", value=float(DEFAULTS["face_value_usd"]), step=1.0)

    st.divider()
    daily_target_usd = st.number_input("Daily Target (USD)", value=float(DEFAULTS["daily_target_usd"]), step=5.0)
    st.write("🏁 **Checkpoints (USD)**")
    c1,c2,c3 = st.columns(3)
    cp1=c1.number_input("CP1", value=float(DEFAULTS["checkpoints_usd"][0]), step=10.0)
    cp2=c2.number_input("CP2", value=float(DEFAULTS["checkpoints_usd"][1]), step=10.0)
    cp3=c3.number_input("CP3", value=float(DEFAULTS["checkpoints_usd"][2]), step=10.0)
    w1,w2,w3 = st.columns(3)
    wd1=w1.number_input("WD1", value=float(DEFAULTS["checkpoint_withdrawals_usd"][0]), step=10.0)
    wd2=w2.number_input("WD2", value=float(DEFAULTS["checkpoint_withdrawals_usd"][1]), step=10.0)
    wd3=w3.number_input("WD3", value=float(DEFAULTS["checkpoint_withdrawals_usd"][2]), step=10.0)
    checkpoints=[cp1,cp2,cp3]; checkpoint_withdrawals=[wd1,wd2,wd3]

# ---------- Data ----------
trade_cols = [
    "id","timestamp","date","symbol","mode","side","entry","sl","tp","close",
    "qty_or_contracts","risk_usd","rr","result","pnl_coin","pnl_usd",
    "reason_tags","notes","day_tag","peak_equity_usd","discipline_score","feelings","checklist_json"
]
withdraw_cols = ["timestamp","date","reason","amount_usd","equity_after_usd"]

trades = load_df(LOG_PATH, trade_cols)
withdraws = load_df(WITHDRAW_LOG, withdraw_cols)

# normalize any legacy dates
if not trades.empty:
    trades["date"] = to_day(trades["date"]); trades["day_tag"] = trades["date"].dt.strftime("%Y-%m-%d")
if not withdraws.empty:
    withdraws["date"] = to_day(withdraws["date"])

# ---------- Metrics ----------
equity_usd = (base_usd + (trades["pnl_usd"].fillna(0).sum() if not trades.empty else 0.0)
              - (withdraws["amount_usd"].fillna(0).sum() if not withdraws.empty else 0.0))
equity_usd = float(round(equity_usd,2))
equity_coin = (equity_usd / ref_price) if (contract_mode.startswith("COIN") and ref_price and ref_price>0) else None
wins = int((trades["result"]=="WIN").sum()) if not trades.empty else 0
losses = int((trades["result"]=="LOSS").sum()) if not trades.empty else 0
winrate = (wins/max(1,wins+losses))*100
today = today_iso()
today_pnl_usd = float(trades.loc[trades["day_tag"]==today, "pnl_usd"].sum()) if not trades.empty else 0.0
prog = min(1.0, max(0.0, (today_pnl_usd / daily_target_usd) if daily_target_usd>0 else 0.0))

# ---------- Header ----------
k1,k2,k3,k4,k5 = st.columns(5)
if contract_mode=="USDⓈ‑M (linear) or display_currency=='USD'":
    pass  # placeholder (won't run); below we handle both paths explicitly
if contract_mode=="USDⓈ‑M (linear)" or display_currency=="USD":
    k1.metric("💼 Current Equity", f"${equity_usd:,.2f}")
else:
    k1.metric("💼 Current Equity", f"{equity_coin:,.4f} COIN")
k2.metric("✅ Winrate", f"{winrate:.1f}%")
k3.metric("💰 Total PnL (USD)", f"${(trades['pnl_usd'].sum() if not trades.empty else 0):.2f}")
k4.metric("📅 Today PnL (USD)", f"${today_pnl_usd:.2f}")
k5.metric("🎯 Daily Progress", f"{int(prog*100)}%")
st.progress(prog, text=f"Target ${daily_target_usd:.0f} — {int(prog*100)}% reached")

# ===== Tabs =====
tab_check, tab_size, tab_log, tab_review = st.tabs(["🧭 Pre‑Trade Checklist", "📐 Sizing & TP", "📝 Log Trade", "📊 Review"])

# ---------- Checklist ----------
with tab_check:
    st.subheader("🧭 Pre‑Trade Discipline")
    st.caption("Complete before sizing (if Strict ON).")
    # Setup
    st.markdown("### ✅ Setup Confirmation")
    sc1=st.checkbox("HTF bias aligns with trade")
    sc2=st.checkbox("Clear trigger (FVG/OB/Break‑retest/Liquidity)")
    sc3=st.checkbox("Volume/price action confirm")
    sc4=st.checkbox("No high‑impact news in 15–30m")
    # Risk/Exec
    st.markdown("### 💰 Risk & Execution")
    re1=st.checkbox("Risk = dashboard % (no oversize)")
    re2=st.checkbox("Size matches Binance (qty/contracts & leverage)")
    re3=st.checkbox("SL at invalidation (not random)")
    re4=st.checkbox("RR ≥ minimum (e.g., 1.8–2.0)")
    # Mindset
    st.markdown("### 🧠 Mindset & Emotions")
    me1=st.checkbox("Okay to lose this trade")
    me2=st.checkbox("No FOMO / Revenge / Over‑confidence")
    me3=st.checkbox("Will stop after 2 losses today")
    feel = st.multiselect("How do you feel? (pick 1–3)", FEELINGS, default=["Calm/Focused"])

    answers=[sc1,sc2,sc3,sc4,re1,re2,re3,re4,me1,me2,me3]
    score=int(100*(sum(1 for a in answers if a)/len(answers)))
    st.metric("🧮 Discipline Score", f"{score}%")
    if st.button("✅ Save Checklist for Next Trade", use_container_width=True):
        st.session_state["pretrade_checklist"]={
            "score":score,"feelings":feel,
            "answers":{"setup":[sc1,sc2,sc3,sc4],"risk_exec":[re1,re2,re3,re4],"mindset":[me1,me2,me3]},
            "timestamp":dt.datetime.now().isoformat()
        }
        st.success("Checklist saved — go to Sizing & TP.")

# ---------- Sizing & TP ----------
with tab_size:
    st.subheader("📐 Sizing & TP")
    if strict_gating:
        msg=None; chk=st.session_state.get("pretrade_checklist")
        if not chk: msg="Complete the Pre‑Trade Checklist first."
        elif chk["score"]<70: msg="Discipline score too low (<70%)."
        else:
            bad={"FOMO","Revenge","Over‑confident","Anxious","Impatient","Bored","Tired","Distracted","Hesitant/Fearful"}
            if any(f in bad for f in chk.get("feelings",[])): msg="Risky feelings detected. Re‑center first."
        if msg: st.warning("🔒 "+msg); st.stop()

    c = st.columns([1,1,1,1,1])
    side = c[0].selectbox("Side", ["LONG","SHORT"])
    entry = c[1].number_input("Entry (USD)", value=0.0, step=0.0001, format="%.6f")
    sl    = c[2].number_input("Stop Loss (USD)", value=0.0, step=0.0001, format="%.6f")
    tp_in = c[3].number_input("Take Profit (USD) — optional", value=0.0, step=0.0001, format="%.6f")
    manual= c[4].number_input(("Contracts" if contract_mode.startswith("COIN") else "Qty"), value=0.0, step=0.01)

    risk_usd = round(equity_usd*(risk_pct/100.0),2)
    st.caption(f"Risk per trade: **${risk_usd}** | RR target: **{rr}** | Mode: **{contract_mode}**")

    tp_auto = auto_tp(entry, sl, side, rr) if (tp_in==0 and entry>0 and sl>0) else 0.0
    if tp_auto>0: st.info(f"Suggested TP for RR={rr}: **{tp_auto}**")

    if contract_mode=="USDⓈ‑M (linear)":
        if manual==0 and entry>0 and sl>0:
            qty = suggest_qty_linear(entry, sl, risk_usd)
            notional = qty*entry; margin = notional/max(1,leverage)
            st.info(f"Suggest **Qty**: **{qty}** | Notional: **${notional:,.2f}** | Init margin (@{leverage}×): **${margin:,.2f}**")
    else:
        if manual==0 and entry>0 and sl>0 and face_value_usd:
            raw = coinm_contracts_for_risk(entry, sl, risk_usd, face_value_usd)
            N = int(round(raw))
            notional_coin = coinm_dot_notional(entry, raw, face_value_usd)
            margin_coin   = notional_coin/max(1,leverage)
            risk_after = coinm_loss_coin_at_sl(entry, sl, N, face_value_usd)*entry
            st.info(
                f"Suggest **Contracts**: **{raw:.2f}** → **{N}** (rounded)\n"
                f"• Notional: ~**{notional_coin:.4f} COIN**  • Init margin (@{leverage}×): **{margin_coin:.4f} COIN**\n"
                f"• Est. risk after rounding: **${risk_after:.2f}**"
            )
        if manual>0 and entry>0 and sl>0 and face_value_usd:
            loss_coin = coinm_loss_coin_at_sl(entry, sl, manual, face_value_usd)
            st.warning(f"Manual **{manual:.0f} contracts** → Est. loss at SL ≈ **{loss_coin:.6f} COIN** (~${loss_coin*entry:.2f})")

    st.success("Copy **Qty/Contracts + TP/SL** to Binance. Then log in the next tab after exit.")

# ---------- Log Trade ----------
with tab_log:
    st.subheader("📝 Log Trade (after exit)")
    r1 = st.columns([1,1,1,1,1,1])
    l_side = r1[0].selectbox("Side", ["LONG","SHORT"], key="l_side")
    l_entry= r1[1].number_input("Entry", value=0.0, step=0.0001, format="%.6f", key="l_entry")
    l_sl   = r1[2].number_input("SL", value=0.0, step=0.0001, format="%.6f", key="l_sl")
    l_tp   = r1[3].number_input("TP", value=0.0, step=0.0001, format="%.6f", key="l_tp")
    l_qty  = r1[4].number_input(("Contracts" if contract_mode.startswith("COIN") else "Qty"), value=0.0, step=0.001, format="%.3f", key="l_qty")
    l_rr   = r1[5].number_input("RR (target)", value=float(rr), step=0.1, key="l_rr")

    r2 = st.columns([1,1,1,2])
    l_close = r2[0].number_input("Close Price", value=0.0, step=0.0001, format="%.6f")
    l_result= r2[1].selectbox("Result", ["WIN","LOSS"], key="l_result")
    l_tags  = r2[2].multiselect("Reason Tags", REASON_TAGS, default=["FVG","Confluence_3+"])
    l_notes = r2[3].text_input("Notes (short reason)")

    def pnl_usd_linear(entry, close, qty, side):
        if entry<=0 or close<=0 or qty<=0: return 0.0
        move = (close-entry) if side=="LONG" else (entry-close)
        return qty*move

    if st.button("✅ Add Trade", use_container_width=True):
        if l_entry<=0 or l_close<=0 or l_qty<=0:
            st.error("Entry/Close and Qty/Contracts are required.")
        else:
            risk_usd = round(equity_usd*(risk_pct/100.0),2)
            if contract_mode=="USDⓈ‑M (linear)":
                pnl_usd = pnl_usd_linear(l_entry, l_close, l_qty, l_side); pnl_coin=0.0; qty_save=l_qty
            else:
                N=int(round(l_qty)); qty_save=N
                pnl_coin = coinm_pnl_coin(l_entry, l_close, N, face_value_usd, l_side)
                pnl_usd  = pnl_coin * l_close

            new = {
                "id": str(uuid.uuid4())[:8],
                "timestamp": dt.datetime.now().isoformat(),
                "date": today_iso(),
                "symbol": symbol,
                "mode": "COIN-M" if contract_mode.startswith("COIN") else "USD-M",
                "side": l_side, "entry": l_entry, "sl": l_sl, "tp": l_tp, "close": l_close,
                "qty_or_contracts": qty_save, "risk_usd": risk_usd, "rr": l_rr, "result": l_result,
                "pnl_coin": round(pnl_coin,6), "pnl_usd": round(float(pnl_usd),2),
                "reason_tags": ",".join(l_tags), "notes": l_notes,
                "day_tag": today_iso(), "peak_equity_usd": equity_usd,
                "discipline_score": st.session_state.get("pretrade_checklist",{}).get("score"),
                "feelings": ",".join(st.session_state.get("pretrade_checklist",{}).get("feelings",[])),
                "checklist_json": json.dumps(st.session_state.get("pretrade_checklist",{}).get("answers",{}))
            }
            trades = pd.concat([trades, pd.DataFrame([new])], ignore_index=True)
            save_df(trades, LOG_PATH)

            new_eq = base_usd + trades["pnl_usd"].fillna(0).sum() - (withdraws["amount_usd"].fillna(0).sum() if not withdraws.empty else 0.0)
            suggests=[wd for cp,wd in zip([cp1,cp2,cp3], checkpoint_withdrawals) if new_eq>=cp]
            if suggests:
                st.warning("🎉 Checkpoint hit! Suggested USD withdrawals: " + ", ".join([f"${x:.0f}" for x in suggests]))
            st.success(f"Trade logged. PnL: {('+' if pnl_usd>=0 else '')}${pnl_usd:.2f}  ({pnl_coin:+.6f} COIN)")

# ---------- Review ----------
with tab_review:
    st.subheader("📊 Review & Insights")
    left,right = st.columns([2,1])

    with left:
        st.markdown("#### 📈 Equity Over Time")
        eq_df = build_equity_curve_usd(trades, withdraws, base_usd)
        if contract_mode.startswith("COIN") and display_currency=="COIN" and (ref_price and ref_price>0):
            cdf = eq_df.copy(); cdf["equity_coin"] = cdf["equity_usd"]/ref_price
            y = alt.Y("equity_coin:Q", title="Equity (COIN)")
            tip = ["date:T", alt.Tooltip("equity_coin:Q", title="Equity (COIN)", format=".4f")]
        else:
            cdf = eq_df.copy(); y = alt.Y("equity_usd:Q", title="Equity (USD)")
            tip = ["date:T", alt.Tooltip("equity_usd:Q", title="Equity (USD)", format=",.2f")]
        chart=(alt.Chart(cdf).mark_line(point=True).encode(x=alt.X("date:T", title="Date"), y=y, tooltip=tip)
               .properties(height=280, width="container"))
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### 🧩 Per‑Tag Winrate (Heatmap)")
        tag_df = build_tag_winrate(trades)
        if tag_df.empty or tag_df["trades"].sum()==0:
            st.info("Tag some trades to populate the heatmap.")
        else:
            heat=(alt.Chart(tag_df.assign(Winrate="Winrate")).mark_rect().encode(
                y=alt.Y("tag:N", sort="-x", title="Reason Tag"),
                x=alt.X("Winrate:N", title=None),
                color=alt.Color("winrate:Q", title="Winrate %"),
                tooltip=[alt.Tooltip("tag:N"),alt.Tooltip("trades:Q"),alt.Tooltip("wins:Q"),alt.Tooltip("winrate:Q")])
                .properties(height=max(220,22*len(tag_df)), width=180))
            st.altair_chart(heat, use_container_width=True)
            st.dataframe(tag_df[["tag","trades","wins","winrate"]]
                         .rename(columns={"winrate":"winrate_%"})
                         .sort_values(["winrate_%","trades"], ascending=[False,False]),
                         use_container_width=True, height=240)

    with right:
        st.markdown("#### 🧠 Discipline vs Results")
        if trades.empty or "discipline_score" not in trades.columns:
            st.info("No trades with discipline data yet.")
        else:
            tmp=trades.dropna(subset=["discipline_score"]).copy()
            if tmp.empty: st.info("No discipline data yet.")
            else:
                tmp["discipline_bucket"]=pd.cut(tmp["discipline_score"], bins=[-1,59,69,79,89,100],
                                                labels=["<60","60‑69","70‑79","80‑89","90‑100"])
                agg=tmp.groupby("discipline_bucket").agg(
                    trades=("id","count"),
                    winrate=("result", lambda s:(s=="WIN").mean()*100),
                    avg_pnl=("pnl_usd","mean")).reset_index().sort_values("discipline_bucket")
                st.dataframe(agg, use_container_width=True, height=260)
                st.caption("Goal: keep most trades in **80‑100** bucket.")

# ---------- Withdrawals ----------
st.divider()
st.subheader("🏦 Log Withdrawal (USD)")
wc1,wc2=st.columns([1,2])
w_amount_usd=wc1.number_input("Amount (USD)", value=0.0, step=10.0)
w_reason=wc2.selectbox("Reason", ["Checkpoint","Safety","Emergency","Other"])
if st.button("💸 Add Withdrawal", use_container_width=True):
    if w_amount_usd<=0: st.error("Enter an amount.")
    else:
        row={"timestamp":dt.datetime.now().isoformat(),"date":today_iso(),
             "reason":w_reason,"amount_usd":round(float(w_amount_usd),2),
             "equity_after_usd": float(equity_usd - round(float(w_amount_usd),2))}
        withdraws=pd.concat([withdraws,pd.DataFrame([row])], ignore_index=True)
        save_df(withdraws, WITHDRAW_LOG)
        st.success(f"Withdrawal logged: ${float(w_amount_usd):.2f}")

# ---------- Nudges ----------
if today_pnl_usd <= -(equity_usd*0.10): st.error("🛑 Down >10% today. Stop & review.")
elif prog >= 1.0: st.success("🎯 Daily target reached. Protect gains: stop or switch to sim.")
else: st.info("🚦 Only A+ setups. No chase. No FOMO.")
