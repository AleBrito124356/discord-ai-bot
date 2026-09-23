"""NimClient against a mock HTTP transport: retries, errors, payloads.

openai 1.x is built on ``httpx`` and openai 3.x on ``httpx2``; the test uses
whichever module the installed SDK imports, so no real request is ever made.
"""
from __future__ import annotations

import base64
import io
import json

import openai._base_client as _openai_base
import pytest
from PIL import Image

from bot.nim_client import NimClient, NimError, make_nim_client
from bot.offline import OfflineNimClient

HTTPX = getattr(_openai_base, "httpx2", None) or getattr(_openai_base, "httpx")


def _chat_body(content="hello"):
    return {
        "id": "cmpl-1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


class Server:
    """Scripted responses: each item is (status, json_body)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        status, body = self.responses.pop(0) if self.responses else (200, _chat_body())
        return HTTPX.Response(status, json=body)

    def body(self, i=-1):
        return json.loads(self.requests[i].content)


@pytest.fixture
def sleeps(monkeypatch):
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("bot.nim_client.asyncio.sleep", fake_sleep)
    return delays


@pytest.fixture
async def client_for(settings):
    clients = []
    settings.nvidia_api_key = "nvapi-test"
    settings.nim_base_url = "http://nim.test/v1"

    def factory(server):
        http = HTTPX.AsyncClient(transport=HTTPX.MockTransport(server))
        client = NimClient(settings, http_client=http)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        await client.close()


async def test_chat_sends_settings_and_strips_the_reply(client_for, settings):
    server = Server((200, _chat_body("  hi there  ")))
    reply = await client_for(server).chat([{"role": "user", "content": "yo"}])
    assert reply == "hi there"
    body = server.body()
    assert body["model"] == settings.chat_model
    assert body["max_tokens"] == settings.max_tokens and body["temperature"] == settings.temperature
    assert server.requests[0].headers["authorization"] == "Bearer nvapi-test"
    assert str(server.requests[0].url) == "http://nim.test/v1/chat/completions"


async def test_server_errors_are_retried_with_backoff(client_for, sleeps):
    server = Server((503, {"error": "busy"}), (502, {"error": "bad gw"}), (200, _chat_body("ok")))
    assert await client_for(server).chat([{"role": "user", "content": "x"}]) == "ok"
    assert len(server.requests) == 3 and sleeps == [1.0, 2.0]


async def test_rate_limits_are_retried(client_for, sleeps):
    server = Server((429, {"error": "slow down"}), (200, _chat_body("ok")))
    assert await client_for(server).chat([{"role": "user", "content": "x"}]) == "ok"
    assert sleeps == [1.0]


async def test_persistent_outage_gives_a_friendly_error(client_for, sleeps):
    server = Server(*[(500, {"error": "down"})] * 3)
    with pytest.raises(NimError, match="unavailable right now"):
        await client_for(server).chat([{"role": "user", "content": "x"}])
    assert len(server.requests) == 3


@pytest.mark.parametrize(
    "status,message",
    [(401, "rejected the API key"), (404, "model name is probably wrong"), (400, "request failed \\(400\\)")],
)
async def test_client_errors_fail_fast_with_friendly_messages(client_for, sleeps, status, message):
    server = Server((status, {"error": {"message": "nope"}}))
    with pytest.raises(NimError, match=message):
        await client_for(server).chat([{"role": "user", "content": "x"}])
    assert len(server.requests) == 1 and sleeps == []


async def test_embeddings_are_reordered_by_index(client_for, settings):
    server = Server((200, {
        "object": "list", "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "data": [
            {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
            {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
        ],
    }))
    client = client_for(server)
    vectors = await client.embed(["first", "second"], input_type="query")
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    body = server.body()
    assert body["input"] == ["first", "second"] and body["model"] == settings.embed_model
    assert body["input_type"] == "query" and body["truncate"] == "END"
    assert client.embed_model_name == settings.embed_model
    assert await client.embed([]) == [] and len(server.requests) == 1


async def test_vision_sends_a_normalised_data_uri(client_for, settings):
    server = Server((200, _chat_body("a blue image")))
    buf = io.BytesIO()
    Image.new("RGB", (3000, 2000), (0, 0, 255)).save(buf, format="WEBP")
    reply = await client_for(server).vision("what?", buf.getvalue(), "image/webp")
    assert reply == "a blue image"
    content = server.body()["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "what?"}
    uri = content[1]["image_url"]["url"]
    assert uri.startswith("data:image/jpeg;base64,")
    b64 = uri.split(",", 1)[1]
    assert len(b64) <= settings.image_max_b64
    assert Image.open(io.BytesIO(base64.b64decode(b64))).size == (1568, 1045)
    assert server.body()["model"] == settings.vision_model


async def test_vision_rejects_non_images_without_a_request(client_for):
    server = Server()
    with pytest.raises(NimError, match="not an image format"):
        await client_for(server).vision("what?", b"not an image", "image/png")
    assert server.requests == []


async def test_make_nim_client_picks_the_backend(settings):
    settings.offline = True
    assert isinstance(make_nim_client(settings), OfflineNimClient)
    settings.offline = False
    live = make_nim_client(settings)
    try:
        assert isinstance(live, NimClient) and not live.offline
    finally:
        await live.close()


def test_offline_flag_from_environment(monkeypatch):
    from bot.config import load_settings

    monkeypatch.setenv("BOT_OFFLINE", "1")
    assert load_settings().offline
    monkeypatch.delenv("BOT_OFFLINE")
    monkeypatch.setenv("NIM_BASE_URL", "offline")
    assert load_settings().offline
    monkeypatch.setenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    assert not load_settings().offline
