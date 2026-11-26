"""Limit POST bodies while FastAPI reads them, including streamed requests."""

from fastapi import HTTPException


MAX_POST_BODY_BYTES = 64 * 1024
BODY_TOO_LARGE = "요청 본문은 최대 64 KiB까지 허용됩니다."


class PostBodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        received = 0

        async def receive_limited():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_POST_BODY_BYTES:
                    raise HTTPException(status_code=413, detail=BODY_TOO_LARGE)
            return message

        await self.app(scope, receive_limited, send)
