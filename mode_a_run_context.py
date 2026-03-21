from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from utils import maybe_get, ensure_list, resolve_qa_path


RUN_STAMP_RE = re.compile(r"^run_stamp__(?P<region>.+?)__(?P<mode>.+?)\.json$", re.IGNORECASE)


def discover_run_stamps(insights_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not insights_dir.exists():
        return out
    for p in insights_dir.glob("run_stamp__*__*.json"):
        m = RUN_STAMP_RE.match(p.name)
        if not m:
            continue
        out.append(
            {
                "region": m.group("region"),
                "mode": m.group("mode"),
                "path": p,
                "mtime": p.stat().st_mtime,
            }
        )
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def safe_read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None, f"JSON is not an object: {path}"
        return payload, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _pick_col(df: pd.DataFrame, *candidates: str) -> str | None:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _coerce_boolish(v: Any) -> bool | None:
    if pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def load_qa_summary(qa_path: Path, insights_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(qa_path)
    if df is None:
        df = pd.DataFrame()

    file_col = _pick_col(df, "FileName", "File", "file", "file_name")
    rows_col = _pick_col(df, "Rows", "rows")
    cols_col = _pick_col(df, "Cols", "cols", "Columns")
    exists_col = _pick_col(df, "Exists", "exists")

    if file_col is not None:
        df["FileName"] = df[file_col].astype(str)
    else:
        df["FileName"] = ""

    if exists_col is not None:
        parsed_exists = df[exists_col].map(_coerce_boolish)
    else:
        parsed_exists = pd.Series([None] * len(df), index=df.index, dtype="object")

    def _exists_from_disk(name: Any) -> bool:
        s = str(name).strip()
        if not s:
            return False
        p = Path(s)
        full = p if p.is_absolute() else (insights_dir / p)
        return full.exists()

    disk_exists = df["FileName"].map(_exists_from_disk)
    df["Exists"] = parsed_exists.where(parsed_exists.notna(), disk_exists).astype(bool)

    if rows_col is not None:
        df["Rows"] = pd.to_numeric(df[rows_col], errors="coerce").fillna(0).astype(int)
    else:
        df["Rows"] = 0

    if cols_col is not None:
        df["Cols"] = pd.to_numeric(df[cols_col], errors="coerce").fillna(0).astype(int)
    else:
        df["Cols"] = 0

    return df


def build_artifact_index(insights_dir: Path, required_files: list[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    missing: list[str] = []
    for raw in required_files:
        p = Path(str(raw))
        full = p if p.is_absolute() else (insights_dir / p)
        artifacts[p.name] = full
        if not full.exists():
            missing.append(str(full))
    if missing:
        msg = "Missing required file(s):\n" + "\n".join(missing)
        raise FileNotFoundError(msg)
    return artifacts


def resolve_artifact(ctx: dict[str, Any], filename: str) -> Path:
    artifacts = ctx.get("artifacts_by_name", {}) if isinstance(ctx, dict) else {}
    if isinstance(artifacts, dict) and filename in artifacts:
        return Path(artifacts[filename])
    raise FileNotFoundError(f"Artifact not found in context: {filename}")


def get_run_context_for_region_mode(insights_dir: Path, region: str, mode: str) -> dict[str, Any]:
    selected: dict[str, Any] | None = None
    for item in discover_run_stamps(insights_dir):
        if str(item.get("region", "")).upper() == region.upper() and str(item.get("mode", "")).upper() == mode.upper():
            selected = item
            break
    if selected is None:
        raise FileNotFoundError(f"run_stamp not found for region/mode under {insights_dir}: {region}/{mode}")

    source_path = Path(selected["path"])
    run_stamp, run_stamp_err = safe_read_json(source_path)
    if run_stamp is None:
        raise RuntimeError(f"Could not read run_stamp: {run_stamp_err}")

    effective_insights_dir = insights_dir
    insights_from_stamp = maybe_get(run_stamp, "insights_dir", "InsightsDir")
    if isinstance(insights_from_stamp, str) and insights_from_stamp.strip():
        candidate = Path(insights_from_stamp.strip())
        if candidate.exists():
            effective_insights_dir = candidate

    required_files = ensure_list(maybe_get(run_stamp, "required_files", "RequiredFiles"))
    if not required_files:
        raise RuntimeError("run_stamp.required_files is missing/empty.")

    qa_path = resolve_qa_path(run_stamp, effective_insights_dir, str(selected["region"]), str(selected["mode"]))
    artifacts_by_name = build_artifact_index(effective_insights_dir, required_files)

    return {
        "source_path": source_path,
        "insights_dir": effective_insights_dir,
        "region": str(selected["region"]),
        "mode": str(selected["mode"]),
        "stamp": maybe_get(run_stamp, "stamp", "Stamp"),
        "git_branch": maybe_get(run_stamp, "git_branch", "GitBranch"),
        "git_commit": maybe_get(run_stamp, "git_commit", "GitCommit"),
        "run_stamp": run_stamp,
        "qa_path": qa_path,
        "required_files": required_files,
        "artifacts_by_name": artifacts_by_name,
    }


def _stamp_is_pass(stamp_item: dict, insights_dir: Path) -> bool:
    stamp_json, _ = safe_read_json(Path(stamp_item["path"]))
    if stamp_json is None:
        return False
    required = ensure_list(maybe_get(stamp_json, "required_files", "RequiredFiles"))
    if not required:
        return False
    for raw in required:
        p = Path(str(raw))
        full = p if p.is_absolute() else (insights_dir / p)
        if not full.exists():
            return False
    return True


def get_or_select_run_context(st, insights_dir: Path) -> dict[str, Any]:
    all_stamps = discover_run_stamps(insights_dir)
    run_stamps = [r for r in all_stamps if _stamp_is_pass(r, insights_dir)]
    if not run_stamps:
        if all_stamps:
            st.error("No PASS scopes available — required files missing for all discovered run stamps.")
        else:
            st.error(f"No run stamp files found under: `{insights_dir}` (pattern: `run_stamp__*__*.json`)")
        st.stop()

    options = [str(r["region"]) for r in run_stamps]
    selected_path_key = "mode_a_selected_run_stamp_path"
    selected_path_prev = str(st.session_state.get(selected_path_key, "")).strip()
    default_idx = 0
    if selected_path_prev:
        for i, r in enumerate(run_stamps):
            if str(r["path"]) == selected_path_prev:
                default_idx = i
                break
    else:
        for i, r in enumerate(run_stamps):
            if str(r["region"]).upper() == "EAT7":
                default_idx = i
                break

    selected_label = st.sidebar.selectbox("Scope", options=options, index=default_idx)
    selected = run_stamps[options.index(selected_label)]
    run_stamp_path = Path(selected["path"])
    st.session_state[selected_path_key] = str(run_stamp_path)

    run_stamp, run_stamp_err = safe_read_json(run_stamp_path)
    if run_stamp is None:
        st.error(f"Could not read run stamp `{run_stamp_path}`")
        st.code(run_stamp_err or "unknown error")
        st.stop()

    region = str(selected["region"])
    mode = str(selected["mode"])
    stamp = maybe_get(run_stamp, "stamp", "Stamp")
    git_commit = maybe_get(run_stamp, "git_commit", "GitCommit")
    git_branch = maybe_get(run_stamp, "git_branch", "GitBranch")

    effective_insights_dir = insights_dir
    insights_from_stamp = maybe_get(run_stamp, "insights_dir", "InsightsDir")
    if isinstance(insights_from_stamp, str) and insights_from_stamp.strip():
        candidate = Path(insights_from_stamp.strip())
        if candidate.exists():
            effective_insights_dir = candidate

    required_files = ensure_list(maybe_get(run_stamp, "required_files", "RequiredFiles"))
    if not required_files:
        st.error("run_stamp.required_files is missing/empty.")
        st.stop()

    qa_path = resolve_qa_path(run_stamp, effective_insights_dir, region, mode)
    # Fallback: if resolved path doesn't exist (e.g. relative path double-nesting on
    # Streamlit Cloud), glob for any qa_summary__*.csv in insights_dir.
    if qa_path is None or not qa_path.exists():
        candidates = sorted(effective_insights_dir.glob("qa_summary__*.csv"))
        if candidates:
            qa_path = candidates[-1]
    qa_df = pd.DataFrame()
    if qa_path is not None and qa_path.exists():
        try:
            qa_df = load_qa_summary(qa_path, effective_insights_dir)
        except Exception as e:
            st.error(f"Could not read qa_summary `{qa_path}`: {type(e).__name__}: {e}")
            st.stop()

    try:
        artifacts_by_name = build_artifact_index(effective_insights_dir, required_files)
    except Exception as e:
        st.error(str(e))
        st.stop()

    return {
        "insights_dir": effective_insights_dir,
        "region": region,
        "mode": mode,
        "stamp": stamp,
        "git_branch": git_branch,
        "git_commit": git_commit,
        "run_stamp_path": run_stamp_path,
        "run_stamp_json_path": run_stamp_path,
        "run_stamp": run_stamp,
        "qa_path": qa_path,
        "qa_df": qa_df,
        "required_files": required_files,
        "artifacts_by_name": artifacts_by_name,
    }
