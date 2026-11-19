# Prompt, reply schema and reply checks

from typing import Literal

from pydantic import BaseModel

from ..layout import FormLayout

PROMPT_VERSION = "v1"  # bump to invalidate the cache

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

UNCLEAR = "unclear"
NONE = "none"


# JSON schema the model must follow
def build_response_schema(layout: FormLayout) -> dict:
    text_field = {
        "type": "object",
        "properties": {
            "value": {"type": ["string", "null"]},
            "status": {"type": "string", "enum": ["ok", "blank", "illegible", "crossed_out"]},
        },
        "required": ["value", "status"],
    }

    def choice(options) -> dict:
        return {"type": "string", "enum": [*options, NONE, UNCLEAR]}

    return {
        "type": "object",
        "properties": {
            "is_safety_form": {"type": "boolean"},
            "observation_date": text_field,
            "observer": text_field,
            "company_observed": text_field,
            "timesheet_number": text_field,
            "employee_number": text_field,
            "location": choice(layout.location_boxes),
            "shift": choice(layout.shift_boxes),
            "checklist_title": {"type": ["string", "null"]},
            "checklist": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row": {"type": "integer"},
                        "question": {"type": "string"},
                        "mark": choice(layout.checklist_columns),
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
                        "answer": choice(layout.expectation_columns),
                    },
                    "required": ["question", "answer"],
                },
            },
            "comments": text_field,
        },
        "required": [
            "is_safety_form",
            "observation_date",
            "observer",
            "company_observed",
            "timesheet_number",
            "employee_number",
            "location",
            "shift",
            "checklist_title",
            "checklist",
            "expectations",
            "comments",
        ],
    }


# shape check for the model's reply
class TextField(BaseModel):
    value: str | None
    status: Literal["ok", "blank", "illegible", "crossed_out", "not_read"]


class ChecklistRow(BaseModel):
    row: int
    question: str
    mark: str


class ExpectationRow(BaseModel):
    question: str
    answer: str


class FormRead(BaseModel):
    is_safety_form: bool | None
    observation_date: TextField
    observer: TextField
    company_observed: TextField
    timesheet_number: TextField
    employee_number: TextField
    location: str
    shift: str
    checklist_title: str | None
    checklist: list[ChecklistRow]
    expectations: list[ExpectationRow]
    comments: TextField


# empty reply, nothing invented
def empty_read() -> FormRead:
    not_read = TextField(value=None, status="not_read")
    return FormRead(
        is_safety_form=None,
        observation_date=not_read,
        observer=not_read,
        company_observed=not_read,
        timesheet_number=not_read,
        employee_number=not_read,
        location=NONE,
        shift=NONE,
        checklist_title=None,
        checklist=[],
        expectations=[],
        comments=not_read,
    )
