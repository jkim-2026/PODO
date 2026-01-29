import httpx
from typing import List, Dict, Any
from config.settings import SLACK_WEBHOOK_URL, SLACK_ALERT_ENABLED


async def send_slack_alert(status: str, alerts: List[Dict[str, Any]], session_info: Dict[str, Any]):
    """
    Slack 웹훅으로 알림 전송

    Args:
        status: 시스템 상태 (healthy, warning, critical)
        alerts: 알림 목록
        session_info: 세션 정보
    """
    if not SLACK_ALERT_ENABLED or not SLACK_WEBHOOK_URL:
        return

    # Critical 알림만 전송
    critical_alerts = [a for a in alerts if a["level"] == "critical"]
    if not critical_alerts:
        return

    # Slack 메시지 포맷
    alert_text = "\n".join([
        f"• *{a['message']}* (현재값: {a['value']}, 임계값: {a['threshold']})"
        for a in critical_alerts
    ])

    # 세션 정보 포맷
    session_id = session_info.get("id", "N/A")
    if session_id == "N/A":
        session_display = "전체 데이터"
    elif session_id is None:
        session_display = "세션 없음"
    else:
        session_display = f"세션 {session_id}"

    message = {
        "text": ":rotating_light: *PCB 시스템 Critical 알림*",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 PCB 검사 시스템 Critical 알림"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*상태:*\n{status}"},
                    {"type": "mrkdwn", "text": f"*세션:*\n{session_display}"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*알림 내용:*\n{alert_text}"
                }
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(SLACK_WEBHOOK_URL, json=message, timeout=5.0)
            response.raise_for_status()
        except Exception as e:
            print(f"Slack 알림 전송 실패: {e}")
