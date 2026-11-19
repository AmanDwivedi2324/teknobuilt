# Step 6: validate values and cross-check checkboxes

from datetime import date, datetime

from .checkboxes import CheckboxReport, ChoiceReading
from .extraction.base import ReadResult
from .extraction.schema import NONE, UNCLEAR, TextField
from .layout import RATING_LABELS, FormLayout
from .models import ChecklistItem, ExpectationItem, Issue, SafetyForm, Severity

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)
EARLIEST_PLAUSIBLE_DATE = date(2000, 1, 1)
REQUIRED_TEXT_FIELDS = ("observation_date", "observer", "company_observed", "employee_number")


def _review(code: str, message: str, field: str | None = None) -> Issue:
    return Issue(code=code, message=message, severity=Severity.REVIEW, field=field)


# clean text field, flag if missing
def text_value(name: str, field: TextField, issues: list[Issue]) -> str | None:
    if field.status == "not_read":
        return None  # already flagged once per page
    value = (field.value or "").strip()
    if field.status == "ok" and value:
        return value
    if field.status in ("illegible", "crossed_out"):
        issues.append(_review(f"{field.status}_field", f"{name} is {field.status.replace('_', ' ')}", name))
    elif name in REQUIRED_TEXT_FIELDS:
        issues.append(_review("missing_field", f"{name} is blank", name))
    return None


# handwritten date to ISO
def parse_date(raw: str, today: date) -> tuple[str | None, Issue | None]:
    candidates = set()
    for fmt in DATE_FORMATS:
        try:
            candidates.add(datetime.strptime(raw.strip(), fmt).date())
        except ValueError:
            pass
    field = "observation_date"
    if not candidates:
        return None, _review("invalid_date", f"could not read {raw!r} as a date", field)
    if len(candidates) > 1:
        options = ", ".join(sorted(d.isoformat() for d in candidates))
        return None, _review("ambiguous_date", f"{raw!r} could be {options}", field)
    parsed = candidates.pop()
    if not EARLIEST_PLAUSIBLE_DATE <= parsed <= today:
        return parsed.isoformat(), _review(
            "implausible_date", f"{parsed.isoformat()} is in the future or too old", field
        )
    return parsed.isoformat(), None


# ID numbers must be digits
def clean_number(name: str, raw: str | None, issues: list[Issue]) -> str | None:
    if raw is None:
        return None
    digits = raw.replace(" ", "").replace("-", "")
    if not digits.isdigit():
        issues.append(_review("invalid_number", f"{name} {raw!r} is not a number", name))
        return raw
    return digits


# combine pixel and model readings
def merge_choice(
    field: str, pixel: ChoiceReading | None, model_value: str | None, have_model: bool, allowed
) -> tuple[str | None, list[Issue]]:
    issues: list[Issue] = []
    model: str | None = None
    if have_model and model_value not in (None, NONE):
        if model_value == UNCLEAR or model_value in allowed:
            model = model_value
        else:
            issues.append(
                _review("invalid_value", f"model returned {model_value!r}, which is not on the form", field)
            )
            model = UNCLEAR

    # flag any disagreement
    if pixel is not None:
        issues += [_review("checkbox_unclear", flag, field) for flag in pixel.flags]
        if have_model and model != pixel.value:
            issues.append(
                _review(
                    "checkbox_mismatch",
                    f"pixel count says {pixel.value or 'blank'}, model says {model or 'blank'}",
                    field,
                )
            )
    elif have_model and model == UNCLEAR:
        issues.append(_review("checkbox_unclear", "model could not tell which box is marked", field))

    # clean pixels win, else model
    if pixel is not None and not pixel.flags:
        value = pixel.value
    elif model not in (None, UNCLEAR):
        value = model
    else:
        value = pixel.value if pixel is not None else None
    return value, issues


# build the final form record
def build_form(
    pixel: CheckboxReport | None, read: ReadResult, layout: FormLayout, today: date | None = None
) -> tuple[SafetyForm, list[Issue]]:
    today = today or date.today()
    issues: list[Issue] = []
    data, have_model = read.data, read.has_data

    if read.source == "mock":
        issues.append(_review("text_not_read", read.error or "no model configured"))
    elif read.source == "error":
        issues.append(_review("model_error", f"text could not be read: {read.error}"))
    if have_model and data.is_safety_form is False:
        issues.append(
            _review("model_says_not_a_form", "the model does not recognise this page as the safety form")
        )

    # text fields
    raw_date = text_value("observation_date", data.observation_date, issues)
    iso_date = None
    if raw_date is not None:
        iso_date, date_issue = parse_date(raw_date, today)
        if date_issue:
            issues.append(date_issue)
    observer = text_value("observer", data.observer, issues)
    company = text_value("company_observed", data.company_observed, issues)
    timesheet = clean_number(
        "timesheet_number", text_value("timesheet_number", data.timesheet_number, issues), issues
    )
    employee = clean_number(
        "employee_number", text_value("employee_number", data.employee_number, issues), issues
    )
    comments = text_value("comments", data.comments, issues)

    # checkbox groups
    def merge(field, pixel_reading, model_value, allowed):
        value, found = merge_choice(field, pixel_reading, model_value, have_model, allowed)
        issues.extend(found)
        return value

    location = merge("location_cwa", pixel.location if pixel else None, data.location, layout.location_boxes)
    shift = merge("shift", pixel.shift if pixel else None, data.shift, layout.shift_boxes)

    # checklist rows
    pixel_rows = pixel.checklist if pixel else []
    model_rows = data.checklist if have_model else []
    if pixel and have_model and len(pixel_rows) != len(model_rows):
        issues.append(
            _review(
                "row_count_mismatch",
                f"pixels found {len(pixel_rows)} rows, model found {len(model_rows)}",
                "checklist",
            )
        )
    checklist = []
    for i in range(max(len(pixel_rows), len(model_rows))):
        model_row = model_rows[i] if i < len(model_rows) else None
        rating = merge(
            f"checklist[{i + 1}]",
            pixel_rows[i] if i < len(pixel_rows) else None,
            model_row.mark if model_row else None,
            layout.checklist_columns,
        )
        checklist.append(
            ChecklistItem(
                row=i + 1,
                question=model_row.question.strip() if model_row else None,
                rating=rating,
                rating_label=RATING_LABELS.get(rating) if rating else None,
            )
        )

    # Yes / No / N/A questions
    expectations = []
    for i, question in enumerate(layout.expectation_questions):
        model_row = data.expectations[i] if have_model and i < len(data.expectations) else None
        pixel_row = pixel.expectations[i] if pixel and i < len(pixel.expectations) else None
        answer = merge(
            f"expectations[{i + 1}]",
            pixel_row,
            model_row.answer if model_row else None,
            layout.expectation_columns,
        )
        expectations.append(ExpectationItem(question=question, answer=answer))

    title = (data.checklist_title or "").strip() if have_model else ""
    form = SafetyForm(
        observation_date=iso_date,
        observation_date_raw=raw_date,
        observer=observer,
        company_observed=company,
        location_cwa=location,
        shift=shift,
        timesheet_number=timesheet,
        employee_number=employee,
        checklist_title=title or None,
        checklist=checklist,
        expectations=expectations,
        comments=comments,
    )
    return form, issues
