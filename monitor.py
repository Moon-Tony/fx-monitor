"""외환 3종 점수 기반 포지션 변화 알림 (텔레그램)

규칙 (통화별, 일봉 기준 단순이평선):
  현재가 > 10일선 AND 5일선 > 20일선  -> +1
  현재가 < 10일선 AND 5일선 < 20일선  -> -1
  그 외                               ->  0
합산 >= +2 : 상승(매수) / 합산 <= -2 : 하락(매도) / 그 외 : 중립(현금)
포지션이 바뀔 때만 텔레그램으로 알림을 보냅니다.
"""
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
BUY_THRESHOLD = 2     # 합산 점수가 이 값 이상이면 상승(매수)
SELL_THRESHOLD = -2   # 합산 점수가 이 값 이하이면 하락(매도)
STATE_FILE = "state.json"
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
    return [float(x) for x in close.tolist()]


def position_from_total(total):
    if total >= BUY_THRESHOLD:
        return "상승(매수)"
    if total <= SELL_THRESHOLD:
        return "하락(매도)"
    return "중립(현금)"


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
            return json.load(f).get("position")
    except FileNotFoundError:
        return None


def save_state(position):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"position": position}, f, ensure_ascii=False)


def main():
    now = datetime.now(KST)
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    if not manual and not in_window(now):
        print("확인 시간대가 아니라서 종료합니다:", now)
        return

    lines, total = [], 0
    for name, ticker in PAIRS.items():
        score, price, ma5, ma10, ma20 = score_from_prices(fetch_close(ticker))
        total += score
        lines.append(f"{name}: {score:+d}  (현재가 {price:.4f})")

    position = position_from_total(total)
    previous = load_state()
    print(now, lines, "합산", total, "포지션", position, "이전", previous)

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
