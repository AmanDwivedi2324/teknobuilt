# Shared data shapes and output JSON

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(str, Enum):
    REJECT = "reject"  # unusable page
    REVIEW = "review"  # person must check
    INFO = "info"  # just a note


class Issue(BaseModel):
    code: str
    message: str
    severity: Severity
    field: str | None = None


PageStatus = Literal["ok", "needs_review", "rejected"]


# worst issue decides page status
def status_from_issues(issues: list[Issue]) -> PageStatus:
    if any(i.severity is Severity.REJECT for i in issues):
        return "rejected"
    if any(i.severity is Severity.REVIEW for i in issues):
        return "needs_review"
    return "ok"


# final output shapes
class ChecklistItem(BaseModel):
    row: int
    question: str | None
    rating: Literal["S", "R", "I", "N"] | None
    rating_label: Literal["Safe", "At Risk", "Imminent Hazard", "N/A"] | None


class ExpectationItem(BaseModel):
    question: str
    answer: Literal["Yes", "No", "N/A"] | None


class SafetyForm(BaseModel):
    observation_date: str | None = Field(description="ISO 8601 (YYYY-MM-DD) when the date could be parsed")
    observation_date_raw: str | None = Field(description="the date exactly as handwritten")
    observer: str | None
    company_observed: str | None
    location_cwa: str | None
    shift: Literal["Day", "Night"] | None
    timesheet_number: str | None
    employee_number: str | None
    checklist_title: str | None
    checklist: list[ChecklistItem]
    expectations: list[ExpectationItem]
    comments: str | None


class ExtractionInfo(BaseModel):
    source: Literal["model", "cache", "mock", "error"]
    model: str | None
    seconds: float
    error: str | None = None


class PageResult(BaseModel):
    page: int
    status: PageStatus
    issues: list[Issue]
    form: SafetyForm | None
    quality: dict[str, float | int] = Field(default_factory=dict)
    alignment: dict[str, float | bool] | None = None
    extraction: ExtractionInfo | None = None


class RunSummary(BaseModel):
    pages: int
    ok: int
    needs_review: int
    rejected: int


class RunResult(BaseModel):
    source_file: str
    processed_at: datetime
    pipeline_version: str
    reader: str
    summary: RunSummary
    pages: list[PageResult]
    errors: list[str] = Field(default_factory=list)
