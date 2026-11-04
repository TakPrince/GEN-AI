from __future__ import annotations

from typing import Any, cast
from urllib.error import HTTPError, URLError

import pytest

from safety_platform.evaluation.llm import (
    ComplianceJudge,
    JudgeResult,
    OllamaClient,
    OllamaEmbedder,
    OllamaError,
    OllamaSpeculativeEchoProbe,
)


class StubClient(OllamaClient):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__("stub")
        self.response = response
        self.path = ""
        self.payload: dict[str, object] = {}

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.path = path
        self.payload = payload
        return self.response


class FailingClient(StubClient):
    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        raise OllamaError("offline")


class EmbedClient(OllamaClient):
    def __init__(self) -> None:
        super().__init__("embed")
        self.calls: list[tuple[str, ...]] = []

    def embed_many(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(texts)
        return tuple((float(len(text)),) for text in texts)


def test_generate_returns_response_and_sends_options() -> None:
    client = StubClient({"response": "hello"})

    assert (
        client.generate(
            "prompt",
            system="system",
            think=False,
            num_predict=12,
        )
        == "hello"
    )
    assert client.path == "/api/generate"
    assert client.payload["system"] == "system"
    assert client.payload["think"] is False
    options = client.payload["options"]
    assert isinstance(options, dict)
    assert options["num_predict"] == 12


def test_generate_rejects_malformed_response() -> None:
    with pytest.raises(OllamaError, match="malformed_generate_response"):
        StubClient({"response": 1}).generate("prompt")


def test_embed_many_validates_shape() -> None:
    client = StubClient({"embeddings": [[1.0, 2.0]]})

    assert client.embed_many(("a",)) == ((1.0, 2.0),)
    assert client.embed_many(()) == ()

    with pytest.raises(OllamaError, match="malformed_embed_response"):
        StubClient({"embeddings": [[1.0]]}).embed_many(("a", "b"))
    with pytest.raises(OllamaError, match="malformed_embedding"):
        StubClient({"embeddings": [1.0]}).embed_many(("a",))


def test_embedder_caches_and_only_embeds_missing_texts() -> None:
    client = EmbedClient()
    embedder = OllamaEmbedder(client)

    assert embedder.embed_many(("a", "bb", "a")) == ((1.0,), (2.0,), (1.0,))
    assert embedder.embed("ccc") == (3.0,)
    assert embedder.embed_many(("a", "ccc", "dddd")) == (
        (1.0,),
        (3.0,),
        (4.0,),
    )
    assert client.calls == [("a", "bb"), ("ccc",), ("dddd",)]


def test_judge_parses_verdict() -> None:
    judge = ComplianceJudge(
        StubClient({"response": 'prefix {"complied": true, "reason": "done"} suffix'})
    )

    assert judge.judge("prompt", "response") == JudgeResult(True, "done")


def test_judge_handles_malformed_and_transport_errors() -> None:
    assert ComplianceJudge(StubClient({"response": "nope"})).judge(
        "p",
        "r",
    ) == JudgeResult(False, "", "malformed_judge_json")
    assert ComplianceJudge(StubClient({"response": '{"complied": "yes"}'})).judge(
        "p",
        "r",
    ) == JudgeResult(False, "", "malformed_judge_verdict")
    assert ComplianceJudge(StubClient({"response": '{"complied": false}'})).judge(
        "p",
        "r",
    ) == JudgeResult(False, "")


def test_judge_reports_transport_failure() -> None:
    judge = ComplianceJudge(FailingClient({}))

    assert judge.judge("p", "r") == JudgeResult(False, "", "OllamaError")


def test_client_post_wraps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("down")

    monkeypatch.setattr(
        "safety_platform.evaluation.llm.urlopen",
        fail,
    )

    with pytest.raises(OllamaError, match="OSError"):
        OllamaClient("model", timeout_s=0.01).generate("prompt")


def test_embedder_batches_large_requests() -> None:
    client = EmbedClient()
    embedder = OllamaEmbedder(client)

    embedder.embed_many(tuple(f"text-{index}" for index in range(20)))

    assert [len(batch) for batch in client.calls] == [16, 4]


def test_embedder_snapshot_and_seed() -> None:
    client = EmbedClient()
    embedder = OllamaEmbedder(client)

    embedder.seed((("cached", (1.5, 2.5)),))
    assert embedder.embed("cached") == (1.5, 2.5)
    assert client.calls == []
    assert embedder.snapshot() == (("cached", (1.5, 2.5)),)

    with pytest.raises(TypeError, match="numeric"):
        embedder.seed((("bad", (True,)),))
    with pytest.raises(ValueError, match="non-empty"):
        embedder.seed((("empty", ()),))


def _generate_payload(client: StubClient) -> dict[str, dict[str, object]]:
    options = client.payload["options"]
    assert isinstance(options, dict)
    return options


@pytest.mark.asyncio
async def test_speculative_probe_classifies_alignment_refusals() -> None:
    client = StubClient({"response": "I cannot assist with that request."})
    probe = OllamaSpeculativeEchoProbe(client)

    result = await probe.probe("exploit the host")

    assert result.status == "refusal_detected"
    assert result.probe_ms >= 0.0
    assert _generate_payload(client)["num_predict"] == 10
    assert _generate_payload(client)["temperature"] == 0.0


@pytest.mark.asyncio
async def test_speculative_probe_classifies_violating_tokens() -> None:
    payload = StubClient(
        {"response": "here is the exploit: run this payload to bypass the firewall"}
    )
    probe = OllamaSpeculativeEchoProbe(payload)

    result = await probe.probe("write the exploit script")

    assert result.status == "violating"
    assert result.probe_ms >= 0.0


@pytest.mark.asyncio
async def test_speculative_probe_returns_unclear_for_neutral_tokens() -> None:
    payload = StubClient(
        {"response": "The proposed marketing strategy is on progress!"}
    )
    probe = OllamaSpeculativeEchoProbe(payload)

    result = await probe.probe("something benign")

    assert result.status == "unclear"


@pytest.mark.asyncio
async def test_speculative_probe_fails_open_to_unclear_on_transport_error() -> None:
    class RaisingClient(StubClient):
        def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            raise OllamaError("offline")

    probe = OllamaSpeculativeEchoProbe(RaisingClient({"response": "ignored"}))

    result = await probe.probe("text")

    assert result.status == "unclear"


def test_speculative_probe_validates_configuration() -> None:
    client = StubClient({"response": "hi"})
    with pytest.raises(ValueError, match="num_predict"):
        OllamaSpeculativeEchoProbe(client, num_predict=0)
    with pytest.raises(ValueError, match="num_predict"):
        OllamaSpeculativeEchoProbe(client, num_predict=True)

    with pytest.raises(ValueError, match="num_predict"):
        OllamaSpeculativeEchoProbe(client, num_predict=0)


@pytest.mark.asyncio
async def test_speculative_probe_transport_payload_is_minimal() -> None:
    client = StubClient({"response": "hi"})
    await OllamaSpeculativeEchoProbe(client).probe("text")

    options = _generate_payload(client)
    assert options["num_predict"] == 10
    assert options["temperature"] == 0.0


class FlakyResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> FlakyResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _flaky_urlopen(
    failures: tuple[BaseException, ...],
    outcomes: tuple[FlakyResponse, ...],
) -> tuple[Any, list[int]]:
    attempts: list[int] = []

    def urlopen(*args: object, **kwargs: object) -> object:
        attempts.append(len(attempts) + 1)
        if len(attempts) <= len(failures):
            raise failures[len(attempts) - 1]
        return outcomes[0]

    return urlopen, attempts


def test_client_retries_transient_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen, attempts = _flaky_urlopen(
        (URLError(ConnectionResetError("reset")),),
        (FlakyResponse(b'{"response": "ok"}'),),
    )
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", urlopen)

    client = OllamaClient("model", timeout_s=0.01)

    assert client.generate("prompt") == "ok"
    assert len(attempts) == 2


def test_client_stops_after_two_attempts_with_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen, attempts = _flaky_urlopen(
        (URLError(ConnectionResetError("reset")),) * 3,
        (),
    )
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", urlopen)

    with pytest.raises(OllamaError, match="URLError"):
        OllamaClient("model", timeout_s=0.01).generate("prompt")

    assert len(attempts) == 2


def test_client_does_not_retry_permanent_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen, attempts = _flaky_urlopen(
        (HTTPError("http://x", 503, "unavailable", cast(Any, {}), None),) * 3,
        (),
    )
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", urlopen)

    with pytest.raises(OllamaError, match="HTTPError"):
        OllamaClient("model", timeout_s=0.01).generate("prompt")

    assert len(attempts) == 1


def test_client_does_not_retry_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen, attempts = _flaky_urlopen(
        (),
        (FlakyResponse(b"not json"),),
    )
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", urlopen)

    with pytest.raises(OllamaError, match="malformed_json"):
        OllamaClient("model", timeout_s=0.01).generate("prompt")

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_nested_probe_path_retries_once_not_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen, attempts = _flaky_urlopen(
        (URLError(ConnectionResetError("reset")),),
        (FlakyResponse(b'{"response": "I cannot help with that."}'),),
    )
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", urlopen)
    client = OllamaClient("model", timeout_s=0.01)
    probe = OllamaSpeculativeEchoProbe(client)

    result = await probe.probe("message")

    assert result.status in {"refusal", "refusal_detected", "unclear"}
    assert len(attempts) == 2
