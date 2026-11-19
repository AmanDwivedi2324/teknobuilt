"""Step 5: read the handwriting with a vision-language model.

For each page image the reader tries, in order:
  1. cache  - a saved model reply for exactly this image + model + prompt version (free, repeatable)
  2. gemini - if GEMINI_API_KEY is set (the free tier is enough for this assessment)
  3. mock   - no key: every field comes back empty and flagged, so the pipeline still runs
It never raises: a failed call returns empty fields plus the error message.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
PROMPT_VERSION = "v1"          # bump when the prompt or schema changes so old cache entries aren't reused
CACHE_DIR = Path("output/cache")
SEND_WIDTH = 1240              # px; plenty for handwriting, fewer image tokens than the full 1654
MAX_ATTEMPTS = 3

LOCATIONS = ["1000", "2000", "3000", "4000", "8000", "5000", "5900", "6000", "7000", "OSLD"]

# ---------------------------------------------------------------- prompt
SYSTEM_PROMPT = """You transcribe scanned, handwritten construction safety observation forms into JSON.

Rules:
1. Copy handwriting exactly as written. Do not fix spelling, expand abbreviations or reword site slang.
2. Never guess. If a field is empty: value null, status "blank".
3. If you cannot read a field with confidence (smudged, muddy, torn, too faint): value null, status "illegible".
4. Ignore crossed-out writing. If nothing readable is left: value null, status "crossed_out".
5. Copy dates exactly as written. Do not reformat them.
6. Checkboxes: report the box that holds a mark (X or tick). No mark: "none". More than one mark,
   or you cannot tell: "unclear".
7. Checklist rows: copy each printed question and its mark, in order. Only include rows that exist.
8. If the image is not this safety form (blank page, other document, test page), set is_safety_form
   to false and leave every other field null / empty."""

USER_PROMPT = "Transcribe this form."

# ---------------------------------------------------------------- output schema (sent to the model)
# Only plain JSON-Schema keywords, so the same schema works with Gemini, OpenAI or Claude.
TEXT_FIELD = {
    "type": "object",
    "properties": {
        "value": {"type": ["string", "null"]},
        "status": {"type": "string", "enum": ["ok", "blank", "illegible", "crossed_out"]},
    },
    "required": ["value", "status"],
}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_safety_form": {"type": "boolean"},
        "observation_date": TEXT_FIELD,
        "observer": TEXT_FIELD,
        "company_observed": TEXT_FIELD,
        "timesheet_number": TEXT_FIELD,
        "employee_number": TEXT_FIELD,
        "location": {"type": "string", "enum": LOCATIONS + ["none", "unclear"]},
        "shift": {"type": "string", "enum": ["Day", "Night", "none", "unclear"]},
        "checklist_title": {"type": ["string", "null"]},
        "checklist": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer"},
                    "question": {"type": "string"},
                    "mark": {"type": "string", "enum": ["S", "R", "I", "N", "none", "unclear"]},
                },
                "required": ["row", "question", "mark"],
            },
        },
        "expectations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string", "enum": ["Yes", "No", "N/A", "none", "unclear"]},
                },
                "required": ["question", "answer"],
            },
        },
        "comments": TEXT_FIELD,
    },
    "required": [
        "is_safety_form", "observation_date", "observer", "company_observed", "timesheet_number",
        "employee_number", "location", "shift", "checklist_title", "checklist", "expectations", "comments",
    ],
}


# ---------------------------------------------------------------- validation (never trust the reply blindly)
class TextField(BaseModel):
    value: str | None
    status: Literal["ok", "blank", "illegible", "crossed_out", "not_read"]


class ChecklistRow(BaseModel):
    row: int
    question: str
    mark: Literal["S", "R", "I", "N", "none", "unclear"]


class ExpectationRow(BaseModel):
    question: str
    answer: Literal["Yes", "No", "N/A", "none", "unclear"]


class FormRead(BaseModel):
    is_safety_form: bool | None
    observation_date: TextField
    observer: TextField
    company_observed: TextField
    timesheet_number: TextField
    employee_number: TextField
    location: Literal["1000", "2000", "3000", "4000", "8000", "5000", "5900", "6000", "7000", "OSLD", "none", "unclear"]
    shift: Literal["Day", "Night", "none", "unclear"]
    checklist_title: str | None
    checklist: list[ChecklistRow]
    expectations: list[ExpectationRow]
    comments: TextField


def empty_read() -> FormRead:
    """What we return when no model reply is available: nothing read, nothing invented."""
    not_read = TextField(value=None, status="not_read")
    return FormRead(
        is_safety_form=None, observation_date=not_read, observer=not_read, company_observed=not_read,
        timesheet_number=not_read, employee_number=not_read, location="none", shift="none",
        checklist_title=None, checklist=[], expectations=[], comments=not_read,
    )


# ---------------------------------------------------------------- model call
def call_gemini(png: bytes) -> str:
    """One request to Gemini. Returns the raw JSON text of the reply."""
    from google import genai          # imported here so the mock path works without the package
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=png, mime_type="image/png"), USER_PROMPT],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
            temperature=0,           # same image in, same answer out
        ),
    )
    return response.text


def encode(image: np.ndarray) -> bytes:
    scale = SEND_WIDTH / image.shape[1]
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    ok, buf = cv2.imencode(".png", small)
    return buf.tobytes()


def read_text(image: np.ndarray) -> dict:
    """Read one page. Returns {source, model, data, error, seconds}."""
    png = encode(image)
    key = hashlib.sha256(png + MODEL.encode() + PROMPT_VERSION.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.json"
    start = time.time()

    def result(source: str, data: FormRead, error: str | None = None) -> dict:
        return {"source": source, "model": MODEL if source in ("cache", "gemini") else None,
                "data": data.model_dump(), "error": error, "seconds": round(time.time() - start, 2)}

    if cache_file.exists():
        try:
            return result("cache", FormRead.model_validate_json(cache_file.read_text(encoding="utf-8")))
        except ValidationError:
            cache_file.unlink()  # stale or broken cache entry: ask the model again

    if not os.environ.get("GEMINI_API_KEY"):
        return result("mock", empty_read(), "no GEMINI_API_KEY set; text fields not read")

    error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = call_gemini(png)
            data = FormRead.model_validate_json(raw)   # reject replies that break the schema
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(data.model_dump_json(indent=2), encoding="utf-8")
            return result("gemini", data)
        except ValidationError as exc:
            error = f"reply did not match the schema: {exc.error_count()} problems"
        except Exception as exc:  # network, quota, server errors
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        if attempt < MAX_ATTEMPTS:
            time.sleep(2 ** attempt)  # 2 s, then 4 s
    return result("error", empty_read(), error)


if __name__ == "__main__":
    from align import align_page
    from quality_gate import check_page

    for path in sys.argv[1:] or sorted(str(p) for p in Path("output/pages").glob("*.png")):
        name = Path(path).name
        gate = check_page(path)
        if gate["status"] == "rejected":
            print(f"== {name}: skipped, no model call ({gate['reasons']})")
            continue
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        aligned, _, _ = align_page(gray, gate["metrics"]["skew_deg"])
        r = read_text(aligned if aligned is not None else gray)
        d = r["data"]
        print(f"== {name}  source={r['source']}  model={r['model']}  {r['seconds']}s")
        if r["error"]:
            print("  error:", r["error"])
        for field in ("observation_date", "observer", "company_observed", "timesheet_number", "employee_number", "comments"):
            print(f"  {field:17} {d[field]['value']!r:40} ({d[field]['status']})")
        print("  checklist title:", d["checklist_title"])