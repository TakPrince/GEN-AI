from typing import Any

import pytest

from safety_platform.evaluation.llm import (
    ComplianceJudge,
    JudgeResult,
    OllamaClient,
    OllamaEmbedder,
    OllamaError,
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
