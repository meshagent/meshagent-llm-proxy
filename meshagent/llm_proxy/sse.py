import logging
from typing import Optional


logger = logging.getLogger("meshagent.llm_proxy.sse")


class SSEEvent:
    """The object created as the result of received events."""

    data: str
    event: str
    id: Optional[str]
    retry: Optional[bool]

    def __init__(
        self,
        data: str = "",
        event: str = "message",
        id: Optional[str] = None,
        retry: Optional[bool] = None,
    ):
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry

    def dump(self) -> str:
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")

        if self.event != "message":
            lines.append(f"event: {self.event}")

        if self.retry:
            lines.append(f"retry: {self.retry}")

        lines.extend(f"data: {d}" for d in self.data.split("\n"))
        return "\n".join(lines) + "\n\n"

    def encode(self) -> bytes:
        return self.dump().encode("utf-8")

    @classmethod
    def parse(cls, raw: str) -> "SSEEvent":
        msg = cls()
        for line in raw.splitlines():
            parts = line.split(":", 1)
            if len(parts) != 2:
                logger.warning("Invalid SSE line: %s", line)
                continue

            name, value = parts
            if value.startswith(" "):
                value = value[1:]

            if name == "data":
                if msg.data:
                    msg.data = f"{msg.data}\n{value}"
                else:
                    msg.data = value
            elif name == "event":
                msg.event = value
            elif name == "id":
                msg.id = value
            elif name == "retry":
                msg.retry = bool(value)

        return msg

    def __str__(self) -> str:
        return self.data
