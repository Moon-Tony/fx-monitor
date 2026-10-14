"""과거 데이터 백테스트: 외환 3종 점수 신호로 코스피 ETF(예: KODEX 레버리지)를 매매했다면?

조건은 backtest_settings.toml 에서만 바꾸면 됩니다 (이 파일은 고치지 않아도 됩니다).
결과는 backtest_results/ 폴더에 (보고서 report.md, 거래기록 trades.csv, 그래프 equity.png) 저장되고,
실행할 때마다 backtest_results/summary.csv 에 한 줄씩 쌓여서 조건별 비교가 쉽습니다.
"""
import csv
import os
import re
import sys
import tomllib
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

SETTINGS_FILE = "backtest_settings.toml"
RESULT_DIR = "backtest_results"
KST = ZoneInfo("Asia/Seoul")
NAME = {1: "상승", 0: "중립", -1: "하락"}


# ---------------------------------------------------------------- 데이터
def clean_index(df):
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def download(ticker, start, end):
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise ValueError(f"{ticker} 데이터를 받지 못했습니다")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return clean_index(df[["Open", "High", "Low", "Close"]].dropna())


# ---------------------------------------------------------------- 신호
def fx_score(close, s, m, l):
    ma_s, ma_m, ma_l = (close.rolling(n).mean() for n in (s, m, l))
    score = pd.Series(0.0, index=close.index)
    score[(close > ma_m) & (ma_s > ma_l)] = 1
    score[(close < ma_m) & (ma_s < ma_l)] = -1
    return score.where(ma_s.notna() & ma_m.notna() & ma_l.notna())


def confirm(raw, n):
    """같은 신호가 n일 연속 이어져야 포지션 변경으로 인정 (n=1이면 즉시 인정)"""
    vals, out = raw.tolist(), []
    cur = vals[0]
    for i, v in enumerate(vals):
        if i >= n - 1 and all(x == v for x in vals[i - n + 1:i + 1]):
            cur = v
        out.append(cur)
    return pd.Series(out, index=raw.index)


def build_signal(fx, sg):
    scores = [fx_score(df["Close"], sg["ma_short"], sg["ma_mid"], sg["ma_long"]).rename(k)
              for k, df in fx.items()]
    total = pd.concat(scores, axis=1).dropna().sum(axis=1)
    raw = pd.Series(0, index=total.index)
    raw[total >= sg["buy_threshold"]] = 1
    raw[total <= sg["sell_threshold"]] = -1
    return confirm(raw, int(sg["confirm_days"]))


def align(sig, trade_index):
    """외환 일봉 D의 신호는 D 다음 날부터(코스피 ETF 기준) 사용 -> 미래 데이터 엿보기 방지"""
    s = sig.copy()
    s.index = s.index + pd.Timedelta(days=1)
    return s.reindex(trade_index, method="ffill")


# ---------------------------------------------------------------- 시뮬레이션
def simulate(px, sig, cfg):
    en, ex, co = cfg["entry"], cfg["exit"], cfg["cost"]
    at_open = en["execution"] == "next_open"
    reentry = bool(en["reentry_after_take_profit"])
    tp, sl = ex["take_profit_pct"] / 100, ex["stop_loss_pct"] / 100
    on_neutral, on_down = bool(ex["exit_on_neutral"]), bool(ex["exit_on_down"])
    fee, slip = co["fee_pct_per_side"] / 100, co["slippage_pct_per_side"] / 100

    st = dict(cash=1.0, shares=0.0, in_pos=False, entry=0.0, dt=None, base=1.0, rearm=False)
    trades, eq = [], []

    def buy(dt, price):
        fill = price * (1 + slip)
        st.update(base=st["cash"], shares=st["cash"] * (1 - fee) / fill, cash=0.0,
                  in_pos=True, entry=fill, dt=dt, rearm=False)

    def sell(dt, price, reason):
        fill = price * (1 - slip)
        cash = st["shares"] * fill * (1 - fee)
        trades.append(dict(entry_date=st["dt"].date(), entry_price=round(st["entry"], 2),
                           exit_date=dt.date(), exit_price=round(fill, 2), reason=reason,
                           return_pct=round((cash / st["base"] - 1) * 100, 2),
                           hold_days=(dt - st["dt"]).days))
        st.update(cash=cash, shares=0.0, in_pos=False, rearm=(reentry and reason == "익절"))

    def intraday(dt, o, h, l, gap_ok):
        """장중 손절/익절 확인 (같은 날 둘 다 가능하면 손절을 먼저 적용 = 보수적)"""
        if sl > 0:
            p = st["entry"] * (1 - sl)
            if gap_ok and o <= p:
                return sell(dt, o, "손절")
            if l <= p:
                return sell(dt, p, "손절")
        if tp > 0:
            p = st["entry"] * (1 + tp)
            if gap_ok and o >= p:
                return sell(dt, o, "익절")
            if h >= p:
                return sell(dt, p, "익절")

    prev = None
    for dt, r in px.iterrows():
        o, h, l, c = r["Open"], r["High"], r["Low"], r["Close"]
        s = int(sig.loc[dt])
        if s != 1:
            st["rearm"] = False
        sig_exit = (s == 0 and on_neutral) or (s == -1 and on_down)
        exited = False
        if st["in_pos"]:
            if at_open:
                if sig_exit:
                    sell(dt, o, "신호")
                else:
                    intraday(dt, o, h, l, True)
            else:
                intraday(dt, o, h, l, True)
                if st["in_pos"] and sig_exit:
                    sell(dt, c, "신호")
            exited = not st["in_pos"]
        if not st["in_pos"] and not exited:
            changed_up = prev is not None and s == 1 and prev != 1
            if s == 1 and (changed_up or st["rearm"]):
                if at_open:
                    buy(dt, o)
                    intraday(dt, o, h, l, False)
                else:
                    buy(dt, c)
        eq.append(st["cash"] + st["shares"] * c)
        prev = s
    holding = None
    if st["in_pos"]:
        holding = dict(entry_date=st["dt"].date(), entry_price=round(st["entry"], 2),
                       now_price=round(px["Close"].iloc[-1], 2),
                       return_pct=round((st["shares"] * px["Close"].iloc[-1] / st["base"] - 1) * 100, 2))
    return pd.Series(eq, index=px.index), pd.DataFrame(trades), holding


# ---------------------------------------------------------------- 성과 계산
def stats(eq):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else float("nan")
    mdd = (eq / eq.cummax() - 1).min()
    return dict(total=total * 100, cagr=cagr * 100, mdd=mdd * 100)


def yearly(eq):
    ye = eq.groupby(eq.index.year).last()
    base = ye.shift(1)
    base.iloc[0] = eq.iloc[0]
    return (ye / base - 1) * 100


def trade_stats(trades, eq, sig):
    n = len(trades)
    if n == 0:
        return dict(n=0, win=float("nan"), avg=float("nan"), hold=float("nan"))
    return dict(n=n, win=(trades["return_pct"] > 0).mean() * 100,
                avg=trades["return_pct"].mean(), hold=trades["hold_days"].mean())


# ---------------------------------------------------------------- 보고서
def fmt(x, d=1):
    return "-" if pd.isna(x) else f"{x:,.{d}f}"


def save_outputs(cfg, raw_text, eq, bench, trades, holding, sig, label):
    stamp = datetime.now(KST).strftime("%Y%m%d-%H%M")
    safe = re.sub(r"[^\w.-]", "_", label) or "run"
    folder = os.path.join(RESULT_DIR, f"{safe}_{stamp}")
    os.makedirs(folder, exist_ok=True)

    ss, bs = stats(eq), stats(bench)
    ts = trade_stats(trades, eq, sig)
    exposure = (eq.pct_change().fillna(0) != 0).mean() * 100
    ys, yb = yearly(eq), yearly(bench)

    chart = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(eq.index, eq / eq.iloc[0], label="Strategy")
        ax.plot(bench.index, bench / bench.iloc[0], label="Buy & Hold", alpha=0.7)
        ax.set_yscale("log")
        ax.set_title("Equity curve (start = 1.0, log scale)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "equity.png"), dpi=110)
        plt.close(fig)
        chart = True
    except Exception as e:  # 그래프는 없어도 보고서는 만든다
        print("그래프 생성 건너뜀:", e)

    trades.to_csv(os.path.join(folder, "trades.csv"), index=False, encoding="utf-8-sig")

    L = [f"# 백테스트 결과: {label}", "",
         f"- 실행 시각: {datetime.now(KST):%Y-%m-%d %H:%M} (KST)",
         f"- 기간: {eq.index[0].date()} ~ {eq.index[-1].date()}",
         f"- 매매 종목: {cfg['market']['trade_ticker']}", "",
         "## 핵심 성과", "",
         "| 항목 | 이 전략 | 그냥 보유(Buy & Hold) |", "|---|---|---|",
         f"| 누적 수익률(%) | {fmt(ss['total'])} | {fmt(bs['total'])} |",
         f"| 연평균 수익률 CAGR(%) | {fmt(ss['cagr'])} | {fmt(bs['cagr'])} |",
         f"| 최대 낙폭 MDD(%) | {fmt(ss['mdd'])} | {fmt(bs['mdd'])} |",
         f"| 거래 횟수 | {ts['n']} | - |",
         f"| 승률(%) | {fmt(ts['win'])} | - |",
         f"| 거래당 평균 수익률(%) | {fmt(ts['avg'], 2)} | - |",
         f"| 평균 보유일 | {fmt(ts['hold'])} | - |",
         f"| 보유 중이었던 날 비율(%) | {fmt(exposure)} | 100 |", ""]
    if holding:
        L += [f"- 백테스트 종료 시점에 보유 중: {holding['entry_date']} 진입, 현재 수익률 {holding['return_pct']}%", ""]
    if chart:
        L += ["![equity](equity.png)", ""]
    L += ["## 연도별 수익률(%)", "", "| 연도 | 이 전략 | 그냥 보유 |", "|---|---|---|"]
    for y in ys.index:
        L.append(f"| {y} | {fmt(ys[y])} | {fmt(yb.get(y, float('nan')))} |")
    L += ["", "## 최근 거래 30건", ""]
    if len(trades):
        L += ["| 진입일 | 진입가 | 청산일 | 청산가 | 사유 | 수익률(%) | 보유일 |", "|---|---|---|---|---|---|---|"]
        for _, t in trades.tail(30).iterrows():
            L.append(f"| {t.entry_date} | {t.entry_price} | {t.exit_date} | {t.exit_price} | "
                     f"{t.reason} | {t.return_pct} | {t.hold_days} |")
    else:
        L.append("거래가 없었습니다.")
    L += ["", "## 가정과 한계 (꼭 읽어주세요)", "",
          "- 신호는 외환 **일봉 종가** 기준이고, 매매는 **다음 코스피 거래일**에 체결된다고 가정했습니다(미래 데이터 엿보기 방지).",
          "- 실제 알림은 15분마다 장중 가격으로 판단하므로 결과가 이 백테스트와 다를 수 있습니다.",
          "- 전액 매수이며 소수점 수량도 허용합니다. 호가 단위, 미체결, 시초가 급변은 반영하지 않았습니다.",
          "- 같은 날 손절과 익절이 모두 가능한 경우 손절이 먼저라고 보수적으로 가정했습니다.",
          "- 과거 결과가 미래를 보장하지 않습니다. 조건을 여러 번 바꿔가며 가장 좋은 값만 고르면 과거에만 잘 맞는 설정(과최적화)이 되기 쉽습니다. 연도별 결과가 고르게 좋은지 함께 보세요.",
          "", "## 이번에 사용한 설정", "", "```toml", raw_text.strip(), "```", ""]
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "summary.csv")
    new = not os.path.exists(path)
    sg, en, ex, co = cfg["signal"], cfg["entry"], cfg["exit"], cfg["cost"]
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["실행시각", "이름", "기간시작", "기간끝", "이평(단/중/장)", "매수/매도 점수",
                        "신호확인일수", "체결", "익절%", "손절%", "중립청산", "하락청산", "익절후재진입",
                        "수수료%", "슬리피지%", "전략누적%", "전략CAGR%", "전략MDD%", "거래수", "승률%",
                        "보유누적%", "보유CAGR%", "보유MDD%", "결과폴더"])
        w.writerow([datetime.now(KST).strftime("%Y-%m-%d %H:%M"), label, eq.index[0].date(), eq.index[-1].date(),
                    f"{sg['ma_short']}/{sg['ma_mid']}/{sg['ma_long']}", f"{sg['buy_threshold']}/{sg['sell_threshold']}",
                    sg["confirm_days"], en["execution"], ex["take_profit_pct"], ex["stop_loss_pct"],
                    ex["exit_on_neutral"], ex["exit_on_down"], en["reentry_after_take_profit"],
                    co["fee_pct_per_side"], co["slippage_pct_per_side"],
                    round(ss["total"], 1), round(ss["cagr"], 1), round(ss["mdd"], 1), ts["n"], round(ts["win"], 1) if ts["n"] else "",
                    round(bs["total"], 1), round(bs["cagr"], 1), round(bs["mdd"], 1), folder])
    return folder, ss, bs, ts


# ---------------------------------------------------------------- 실행
DEFAULTS = {
    "start_date": "2012-01-01", "end_date": "", "label": "기본",
    "market": {"trade_ticker": "122630.KS",
               "fx_pairs": ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"]},
    "signal": {"ma_short": 5, "ma_mid": 10, "ma_long": 20,
               "buy_threshold": 2, "sell_threshold": -2, "confirm_days": 1},
    "entry": {"execution": "next_open", "reentry_after_take_profit": False},
    "exit": {"take_profit_pct": 5.0, "stop_loss_pct": 0.0,
             "exit_on_neutral": True, "exit_on_down": True},
    "cost": {"fee_pct_per_side": 0.015, "slippage_pct_per_side": 0.05},
}


def merged(user):
    cfg = {}
    for k, v in DEFAULTS.items():
        cfg[k] = {**v, **user.get(k, {})} if isinstance(v, dict) else user.get(k, v)
    return cfg


def run(cfg, raw_text="", loader=download):
    if cfg["entry"]["execution"] not in ("next_open", "next_close"):
        raise ValueError('execution 은 "next_open" 또는 "next_close" 만 가능합니다')
    if int(cfg["signal"]["confirm_days"]) < 1:
        raise ValueError("confirm_days 는 1 이상이어야 합니다")
    start = cfg["start_date"]
    end = cfg["end_date"] or None
    warm = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    fx = {p: loader(p, warm, end) for p in cfg["market"]["fx_pairs"]}
    px = loader(cfg["market"]["trade_ticker"], start, end)

    sig = align(build_signal(fx, cfg["signal"]), px.index).dropna()
    px = px.loc[sig.index]
    px = px[px.index >= pd.Timestamp(start)]
    sig = sig.loc[px.index]
    if len(px) < 30:
        raise ValueError("백테스트에 쓸 데이터가 너무 적습니다 (기간/종목 코드를 확인하세요)")

    eq, trades, holding = simulate(px, sig, cfg)
    bench = px["Close"]
    return save_outputs(cfg, raw_text, eq, bench, trades, holding, sig, cfg["label"])


def main():
    with open(SETTINGS_FILE, "rb") as f:
        raw = f.read()
    cfg = merged(tomllib.loads(raw.decode("utf-8")))
    folder, ss, bs, ts = run(cfg, raw.decode("utf-8"))
    print(f"결과 폴더: {folder}")
    print(f"전략  누적 {ss['total']:.1f}%  CAGR {ss['cagr']:.1f}%  MDD {ss['mdd']:.1f}%  거래 {ts['n']}회")
    print(f"보유  누적 {bs['total']:.1f}%  CAGR {bs['cagr']:.1f}%  MDD {bs['mdd']:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("오류:", e, file=sys.stderr)
        sys.exit(1)
