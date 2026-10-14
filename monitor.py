"""외환 3종 점수 기반 포지션 변화 알림 (텔레그램)

규칙 (통화별, 일봉 기준 단순이평선):
  현재가 > 10일선 AND 5일선 > 20일선  -> +1
  현재가 < 10일선 AND 5일선 < 20일선  -> -1
  그 외                               ->  0
합산 >= +2 : 상승 / 합산 <= -2 : 하락 / 그 외 : 중립
포지션이 바뀔 때만 텔레그램으로 알림을 보냅니다.
매 실행 결과는 status.md(최신 1건)와 history.csv(누적 기록)에 남깁니다.
"""
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

# ---------------- 설정 (필요하면 여기만 고치세요) ----------------
PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
}
BUY_THRESHOLD = 2     # 합산 점수가 이 값 이상이면 상승
SELL_THRESHOLD = -2   # 합산 점수가 이 값 이하이면 하락
STATE_FILE = "state.json"      # 직전 포지션 기억용
STATUS_FILE = "status.md"      # 가장 최근 확인 결과 (보기 편한 표)
HISTORY_FILE = "history.csv"   # 확인할 때마다 한 줄씩 쌓이는 기록
KST = ZoneInfo("Asia/Seoul")
# ---------------------------------------------------------------


def in_window(now):
    """월요일 08:00 ~ 토요일 06:00 (한국시간) 이면 True"""
    wd = now.weekday()  # 월=0 ... 일=6
    minutes = now.hour * 60 + now.minute
    if wd == 0:
        return minutes >= 8 * 60
    if 1 <= wd <= 4:
        return True
    if wd == 5:
        return minutes <= 6 * 60
    return False


def score_from_prices(close):
    """종가 리스트(최신값 마지막)로 점수와 지표를 계산"""
    if len(close) < 20:
        raise ValueError("데이터가 20개 미만입니다")
    price = close[-1]
    ma5 = sum(close[-5:]) / 5
    ma10 = sum(close[-10:]) / 10
    ma20 = sum(close[-20:]) / 20
    if price > ma10 and ma5 > ma20:
        score = 1
    elif price < ma10 and ma5 < ma20:
        score = -1
    else:
        score = 0
    return score, price, ma5, ma10, ma20


def fetch_close(ticker):
    """(종가 리스트, 마지막 봉의 날짜) 를 돌려줍니다"""
    import yfinance as yf

    df = yf.download(ticker, period="3mo", interval="1d",
                     progress=False, auto_adjust=True)
    close = df["Close"]
    if hasattr(close, "columns"):  # 열이 여러 개로 오는 경우 대비
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) == 0:
        raise ValueError(f"{ticker} 데이터가 비어 있습니다")
    last = close.index[-1].to_pydatetime()
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - last > timedelta(days=5):
        raise ValueError(f"{ticker} 데이터가 너무 오래되었습니다 ({last})")
    return [float(x) for x in close.tolist()], last.strftime("%Y-%m-%d")


def position_from_total(total):
    if total >= BUY_THRESHOLD:
        return "상승"
    if total <= SELL_THRESHOLD:
        return "하락"
    return "중립"


def vs(a, b):
    if a > b:
        return "위"
    if a < b:
        return "아래"
    return "같음"


def send_telegram(text):
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text},
        timeout=30,
    )
    r.raise_for_status()


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            pos = json.load(f).get("position")
    except FileNotFoundError:
        return None
    if not pos:
        return None
    # 예전 이름("상승(매수)" 등)이 저장돼 있어도 "상승"으로 간주 (불필요한 알림 방지)
    return pos.split("(")[0].strip()


def save_state(position):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"position": position}, f, ensure_ascii=False)


def write_records(now, rows, total, position):
    """history.csv 에 한 줄씩 누적하고, status.md 는 최신 결과로 덮어씁니다"""
    stamp = now.strftime("%Y-%m-%d %H:%M")

    new_file = not os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["확인시각(KST)", "통화쌍", "점수", "현재가",
                        "5일선", "10일선", "20일선", "데이터날짜",
                        "합산점수", "포지션"])
        for r in rows:
            w.writerow([stamp, r["name"], r["score"],
                        f"{r['price']:.5f}", f"{r['ma5']:.5f}",
                        f"{r['ma10']:.5f}", f"{r['ma20']:.5f}",
                        r["bar"], total, position])

    md = [
        "# 최근 확인 결과",
        "",
        f"- 확인 시각: **{stamp} (KST)**",
        f"- 현재 포지션: **{position}** (합산 점수 {total:+d})",
        "",
        "| 통화쌍 | 점수 | 현재가 | 5일선 | 10일선 | 20일선 "
        "| 현재가 vs 10일선 | 5일선 vs 20일선 | 데이터 날짜 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['name']} | {r['score']:+d} | {r['price']:.5f} "
            f"| {r['ma5']:.5f} | {r['ma10']:.5f} | {r['ma20']:.5f} "
            f"| {vs(r['price'], r['ma10'])} | {vs(r['ma5'], r['ma20'])} "
            f"| {r['bar']} |"
        )
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main():
    now = datetime.now(KST)
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    if not manual and not in_window(now):
        print("확인 시간대가 아니라서 종료합니다:", now)
        return

    rows, total = [], 0
    for name, ticker in PAIRS.items():
        closes, bar = fetch_close(ticker)
        score, price, ma5, ma10, ma20 = score_from_prices(closes)
        total += score
        rows.append(dict(name=name, score=score, price=price,
                         ma5=ma5, ma10=ma10, ma20=ma20, bar=bar))

    position = position_from_total(total)
    previous = load_state()
    print(now, [(r["name"], r["score"]) for r in rows],
          "합산", total, "포지션", position, "이전", previous)

    write_records(now, rows, total, position)

    lines = [f"{r['name']}: {r['score']:+d}  (현재가 {r['price']:.5f})"
             for r in rows]
    detail = "\n".join(lines) + f"\n합산 점수: {total:+d}\n{now:%m/%d %H:%M} (KST)"
    if previous is None:
        send_telegram(f"✅ 모니터링 시작\n현재 포지션: {position}\n{detail}")
    elif position != previous:
        send_telegram(f"🔔 포지션 변화: {previous} → {position}\n{detail}")
    elif manual:
        send_telegram(f"🧪 수동 테스트 (변화 없음)\n현재 포지션: {position}\n{detail}")
    save_state(position)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("오류:", e, file=sys.stderr)
        sys.exit(1)  # 실패하면 GitHub가 이메일로 알려줍니다
