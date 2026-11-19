# Tests for AI readers and cache

import json

import pytest
from google.genai import _api_client, errors

from form_digitizer.extraction.base import ReadResult
from form_digitizer.extraction.cache import CachedReader
from form_digitizer.extraction.gemini import GeminiReader
from form_digitizer.extraction.mock import MockReader
from form_digitizer.extraction.schema import FormRead, build_response_schema, empty_read
from form_digitizer.layout import DEFAULT_LAYOUT

PNG = b"\x89PNG fake image bytes"

GOOD_REPLY = {
    "is_safety_form": True,
    "observation_date": {"value": "2024-10-02", "status": "ok"},
    "observer": {"value": "John Davis", "status": "ok"},
    "company_observed": {"value": "Summit Construction", "status": "ok"},
    "timesheet_number": {"value": "25489", "status": "ok"},
    "employee_number": {"value": "7852", "status": "ok"},
    "location": "8000",
    "shift": "Day",
    "checklist_title": "Breaking Containment - Life Saving Action",
    "checklist": [
        {"row": i + 1, "question": f"Question {i + 1}", "mark": m} for i, m in enumerate("IIIRSIII")
    ],
    "expectations": [
        {"question": q, "answer": a}
        for q, a in zip(DEFAULT_LAYOUT.expectation_questions, ["Yes", "No", "N/A", "N/A"], strict=True)
    ],
    "comments": {"value": "Safety measures were strictly followed.", "status": "ok"},
}


class CountingReader:
    def __init__(self, result: ReadResult):
        self.result, self.calls = result, 0

    def read(self, png):
        self.calls += 1
        return self.result


def gemini(**kwargs) -> GeminiReader:
    return GeminiReader("test-key", "test-model", DEFAULT_LAYOUT, sleep=lambda s: None, **kwargs)


def test_mock_reads_nothing_and_invents_nothing():
    result = MockReader().read(PNG)
    assert result.source == "mock" and not result.has_data
    assert result.data.observer.value is None and result.data.observer.status == "not_read"


def test_cache_saves_model_replies_and_reuses_them(tmp_path):
    inner = CountingReader(ReadResult("model", "m", FormRead.model_validate(GOOD_REPLY)))
    reader = CachedReader(inner, tmp_path, "m", "layout")
    first, second = reader.read(PNG), reader.read(PNG)
    assert (first.source, second.source) == ("model", "cache")
    assert inner.calls == 1
    assert second.data == first.data


def test_cache_never_stores_mock_or_error_results(tmp_path):
    inner = CountingReader(ReadResult("error", "m", empty_read(), "boom"))
    reader = CachedReader(inner, tmp_path, "m", "layout")
    reader.read(PNG)
    reader.read(PNG)
    assert inner.calls == 2 and not list(tmp_path.glob("*.json"))


def test_broken_cache_entry_is_ignored(tmp_path):
    inner = CountingReader(ReadResult("model", "m", FormRead.model_validate(GOOD_REPLY)))
    reader = CachedReader(inner, tmp_path, "m", "layout")
    reader.path_for(PNG).write_text("{not json", encoding="utf-8")
    assert reader.read(PNG).source == "model"


def test_gemini_request_is_built_as_documented(monkeypatch):
    captured = {}

    def fake_request(self, http_method, path, request_dict, http_options=None):
        captured.update(path=path, body=request_dict)
        raise errors.ClientError(
            400, {"error": {"code": 400, "message": "captured", "status": "INVALID_ARGUMENT"}}
        )

    monkeypatch.setattr(_api_client.BaseApiClient, "request", fake_request)
    gemini().read(PNG)

    body = captured["body"]
    config = body["generationConfig"]
    assert captured["path"] == "models/test-model:generateContent"
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == build_response_schema(DEFAULT_LAYOUT)
    assert config["temperature"] == 0
    assert "systemInstruction" in body
    assert "inlineData" in body["contents"][0]["parts"][0]


def test_gemini_retries_when_overloaded_then_succeeds():
    reader = gemini()
    replies = iter(
        [errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}})]
    )

    def call(png):
        reply = next(replies, None)
        if reply is not None:
            raise reply
        return json.dumps(GOOD_REPLY)

    reader._call = call
    result = reader.read(PNG)
    assert result.source == "model" and result.data.observer.value == "John Davis"


def test_gemini_does_not_retry_a_bad_key():
    reader = gemini()
    calls = []

    def call(png):
        calls.append(1)
        raise errors.ClientError(
            401, {"error": {"code": 401, "message": "bad key", "status": "UNAUTHENTICATED"}}
        )

    reader._call = call
    result = reader.read(PNG)
    assert result.source == "error" and len(calls) == 1


@pytest.mark.parametrize("reply", ["Sure! Here is the form...", json.dumps({"observer": "John"})])
def test_gemini_rejects_replies_that_break_the_schema(reply):
    reader = gemini(max_attempts=2)
    reader._call = lambda png: reply
    result = reader.read(PNG)
    assert result.source == "error" and "schema" in result.error
    assert result.data == empty_read()
