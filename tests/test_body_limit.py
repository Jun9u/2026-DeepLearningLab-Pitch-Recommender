import json
import unittest
from unittest.mock import Mock

from web.backend.app import app
from web.backend.body_limit import BODY_TOO_LARGE, MAX_POST_BODY_BYTES


class PostBodyLimitTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, method, path, chunks, *, content_length=None):
        service = Mock()
        service.recommend.return_value = {"result": "recommend"}
        service.compare.return_value = {"result": "compare"}
        messages = iter(chunks)
        replies = []

        async def receive():
            return next(messages, {"type": "http.disconnect"})

        async def send(message):
            replies.append(message)

        headers = [(b"content-type", b"application/json")]
        if content_length is not None:
            headers.append((b"content-length", str(content_length).encode()))
        await app(
            {"type": "http", "method": method, "path": path, "root_path": "", "headers": headers,
             "query_string": b"", "state": {"service": service}},
            receive, send,
        )
        status = replies[0]["status"]
        body = json.loads(b"".join(message.get("body", b"") for message in replies[1:]))
        return status, body, service

    def payload(self, *, compare=False):
        data = {"pitcher": 123, "balls": 0, "strikes": 0, "outs_when_up": 1, "inning": 1}
        if compare:
            data.update(action_pitch_type="FF", action_zone="Z5")
        return json.dumps(data).encode()

    async def test_accepts_a_valid_request_at_the_byte_limit(self):
        body = self.payload().ljust(MAX_POST_BODY_BYTES, b" ")
        status, response, service = await self.call(
            "POST", "/recommend", [{"type": "http.request", "body": body}], content_length=len(body),
        )
        self.assertEqual((status, response), (200, {"result": "recommend"}))
        service.recommend.assert_called_once()

    async def test_rejects_an_oversize_body_before_inference(self):
        body = self.payload().ljust(MAX_POST_BODY_BYTES + 1, b" ")
        status, response, service = await self.call(
            "POST", "/recommend", [{"type": "http.request", "body": body}], content_length=len(body),
        )
        self.assertEqual((status, response), (413, {"detail": BODY_TOO_LARGE}))
        service.recommend.assert_not_called()

    async def test_counts_bytes_across_chunks_without_content_length(self):
        body = self.payload(compare=True).ljust(MAX_POST_BODY_BYTES + 1, b" ")
        status, response, service = await self.call("POST", "/compare", [
            {"type": "http.request", "body": body[:MAX_POST_BODY_BYTES], "more_body": True},
            {"type": "http.request", "body": body[MAX_POST_BODY_BYTES:]},
        ])
        self.assertEqual((status, response), (413, {"detail": BODY_TOO_LARGE}))
        service.compare.assert_not_called()

    async def test_get_liveness_still_works_without_a_body(self):
        status, response, _ = await self.call("GET", "/live", [])
        self.assertEqual((status, response), (200, {"status": "ok"}))
