"""Caregiver notifications through ntfy.sh: a plain HTTP POST, no account; the topic is the key."""

from __future__ import annotations

import urllib.request


class Notifier:
    """Posts to https://ntfy.sh/<topic>. A no-op when topic is empty; never raises."""

    def __init__(self, topic: str = "", server: str = "https://ntfy.sh", timeout_s: float = 3.0):
        self.topic = (topic or "").strip()
        self.server = server.rstrip("/")
        self.timeout_s = timeout_s

    def help(self, reason: str, url: str = "") -> bool:
        """Urgent: the robot is waiting for the caregiver. url opens on tap (the dashboard)."""
        return self._post(reason, "Byte to Bite: help requested", "high", "warning,robot", url)

    def info(self, msg: str) -> bool:
        return self._post(msg, "Byte to Bite", "default", "robot")

    def _post(self, body: str, title: str, priority: str, tags: str, click: str = "") -> bool:
        if not self.topic:
            return False
        # urllib sends headers as latin-1, so keep them ASCII; the body may be UTF-8.
        headers = {"Title": title, "Priority": priority, "Tags": tags}
        if click:
            headers["Click"] = click.encode("ascii", "ignore").decode()
        req = urllib.request.Request(
            f"{self.server}/{self.topic}", data=body.encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                r.read()
            return True
        except Exception as e:  # a notification must never break the run
            print(f"ntfy: could not notify {self.topic!r}: {e}")
            return False
