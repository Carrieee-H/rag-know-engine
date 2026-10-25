"""/eval routes. Responses use the {success, message, data, total} wrapper that eval-report.html reads."""

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from know_engine.eval import dataset, report
from know_engine.eval.runner import EvalConfig, run_batch

router = APIRouter(prefix="/eval")


def ok(data=None, total: int | None = None) -> dict:
    return {"success": True, "message": None, "data": data, "total": total}


def fail(message: str) -> dict:
    return {"success": False, "message": message, "data": None, "total": None}


class RunRequest(BaseModel):
    datasetPath: str
    config: dict | None = None


@router.post("/run")
def run(body: RunRequest):
    try:
        run_id = run_batch(body.datasetPath, EvalConfig.from_json(body.config))
    except (ValueError, LookupError) as e:
        return fail(str(e))
    return ok(report.get_report(run_id))


@router.post("/dataset/upload")
def upload_dataset(file: UploadFile = File(...)):
    try:
        return ok({"datasetPath": dataset.save_upload(file.filename or "dataset.jsonl", file.file.read())})
    except (ValueError, UnicodeDecodeError) as e:
        return fail(str(e))


@router.get("/report/{run_id}")
def get_report(run_id: str, includeCases: bool = True):
    try:
        return ok(report.get_report(run_id, includeCases))
    except LookupError as e:
        return fail(str(e))


@router.get("/runs")
def runs(page: int = 1, size: int = 20):
    records, total = report.list_runs(page, size)
    return ok({"records": records, "total": total, "current": page, "size": size}, total)


@router.get("/compare")
def compare(baseline: str, candidate: str):
    try:
        return ok(report.compare(baseline, candidate))
    except LookupError as e:
        return fail(str(e))


@router.get("/report/{run_id}/export.csv")
def export(run_id: str):
    return Response(
        report.export_csv(run_id), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="eval-{run_id}.csv"'},
    )
