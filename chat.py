import base64
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


WEBHOOK_URL = "https://chatgpt.id.vn/webhook-test/70ecee2a-c278-461f-a898-52ff907b4fb2"

st.set_page_config(page_title="Chat voi n8n")
st.title("Chat voi n8n (webhook)")


def ensure_session_dir(session_id: str) -> Path:
    session_path = Path(session_id)
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path


def extract_message_types(raw: Any) -> List[str]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        items = [raw]
    else:
        items = []

    parsed: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        for part in item.split(","):
            token = part.strip().lower()
            if token and token not in parsed:
                parsed.append(token)
    return parsed


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def parse_charts(raw_chart: Any, session_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    if not raw_chart:
        return []

    charts: List[Dict[str, Any]] = []
    sources = raw_chart if isinstance(raw_chart, list) else [raw_chart]
    for entry in sources:
        chart_info: Dict[str, Any] = {}

        code: Optional[str] = None
        if isinstance(entry, dict):
            code_value = entry.get("code")
            if isinstance(code_value, list):
                code = "\n".join(str(line) for line in code_value)
            elif isinstance(code_value, str):
                code = code_value
        elif isinstance(entry, str):
            code = entry

        if code:
            chart_info["code"] = code

        if isinstance(entry, dict):
            for key in ("title", "caption", "description"):
                value = entry.get(key)
                if isinstance(value, str):
                    chart_info[key] = value

            file_entry = entry.get("file")
            if file_entry:
                if session_dir is not None:
                    files = process_files(file_entry, session_dir)
                    if files:
                        chart_info["file"] = files if len(files) > 1 else files[0]
                else:
                    chart_info["file"] = file_entry

        if chart_info:
            charts.append(chart_info)
    return charts


def parse_tables(raw_table: Any) -> List[Dict[str, Any]]:
    if not raw_table:
        return []

    tables: List[Dict[str, Any]] = []
    sources = raw_table if isinstance(raw_table, list) else [raw_table]

    for entry in sources:
        if not isinstance(entry, dict):
            continue

        columns = entry.get("columns")
        rows = entry.get("rows")
        caption = entry.get("caption")

        if isinstance(columns, (list, tuple)) and isinstance(rows, (list, tuple)):
            table_info = {
                "columns": list(columns),
                "rows": [
                    list(row) if isinstance(row, (list, tuple)) else row
                    for row in rows
                ],
            }
            if isinstance(caption, str):
                table_info["caption"] = caption
            tables.append(table_info)

    return tables


def resolve_unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        new_candidate = directory / f"{stem}_{counter}{suffix}"
        if not new_candidate.exists():
            return new_candidate
        counter += 1


def process_files(raw_file: Any, session_dir: Path) -> List[Dict[str, Any]]:
    if not raw_file:
        return []

    entries = raw_file if isinstance(raw_file, list) else [raw_file]
    processed: List[Dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        data_b64 = (
            entry.get("data") or entry.get("base64") or entry.get("content") or ""
        )
        if not isinstance(data_b64, str):
            processed.append({"error": "File khong co du lieu hop le."})
            continue

        data_b64 = data_b64.strip()
        if data_b64.startswith("data:"):
            _, _, data_b64 = data_b64.partition(",")

        filename = (
            entry.get("filename")
            or entry.get("name")
            or f"tep_{len(processed) + 1}.bin"
        )
        mime = entry.get("mime") or entry.get("content_type") or "application/octet-stream"
        label = entry.get("label")

        try:
            binary = base64.b64decode(data_b64)
        except Exception as exc:
            processed.append(
                {
                    "filename": filename,
                    "mime": mime,
                    "label": label,
                    "error": f"Loi giai ma tep: {exc}",
                }
            )
            continue

        target_path = resolve_unique_path(session_dir, filename)
        try:
            with target_path.open("wb") as fh:
                fh.write(binary)
        except OSError as exc:
            processed.append(
                {
                    "filename": filename,
                    "mime": mime,
                    "label": label,
                    "error": f"Loi ghi tep: {exc}",
                }
            )
            continue

        processed.append(
            {
                "filename": filename,
                "mime": mime,
                "label": label,
                "bytes": binary,
                "path": target_path,
            }
        )

    return processed


def build_payload(data: Any, session_dir: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "types": [],
        "text": [],
        "charts": [],
        "tables": [],
        "files": [],
    }

    if isinstance(data, dict):
        types = extract_message_types(data.get("type_message"))

        text = normalize_text(data.get("text"))
        if text:
            payload["text"].append(text)

        raw_charts = data.get("chart") or data.get("code_chart")
        charts = parse_charts(raw_charts, session_dir)
        if charts:
            payload["charts"].extend(charts)

        tables = parse_tables(data.get("table") or data.get("code_table"))
        if tables:
            payload["tables"].extend(tables)

        files = process_files(data.get("file"), session_dir)
        if files:
            payload["files"].extend(files)

        for category, key in (
            ("text", "text"),
            ("chart", "charts"),
            ("table", "tables"),
            ("file", "files"),
        ):
            if payload[key] and category not in types:
                types.append(category)

        if not types:
            types = ["text"]

        payload["types"] = types

        if not any(payload[key] for key in ("text", "charts", "tables", "files")):
            fallback = normalize_text(data.get("message") or data.get("reply") or data)
            if fallback:
                payload["text"].append(fallback)
                if "text" not in payload["types"]:
                    payload["types"].append("text")

        return payload

    normalized = normalize_text(data)
    if normalized:
        payload["text"].append(normalized)
        payload["types"] = ["text"]

    return payload


def render_chart(chart: Dict[str, Any], session_dir: Path) -> None:
    code = chart.get("code")
    if not isinstance(code, str) or not code.strip():
        st.info("Khong co code bieu do de hien thi.")
        return

    caption = chart.get("caption") or chart.get("description")
    if isinstance(caption, str):
        st.caption(caption)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        st.warning("Matplotlib chua duoc cai dat.")
        st.code(code, language="python")
        return

    plt.close("all")
    exec_globals: Dict[str, Any] = {
        "plt": plt,
        "session_dir": session_dir,
        "SESSION_DIR": session_dir,
        "session_path": session_dir,
    }

    chart_files_raw = chart.get("file")
    chart_files: List[Dict[str, Any]] = []
    if isinstance(chart_files_raw, dict):
        chart_files = [chart_files_raw]
    elif isinstance(chart_files_raw, list):
        chart_files = [item for item in chart_files_raw if isinstance(item, dict)]

    if chart_files:
        exec_globals["chart_files"] = chart_files
        primary_file = chart_files[0]
        exec_globals["chart_file"] = primary_file

        path_value = primary_file.get("path")
        if isinstance(path_value, Path):
            exec_globals["chart_file_path"] = path_value
        elif isinstance(path_value, str):
            exec_globals["chart_file_path"] = Path(path_value)

        exec_globals["chart_file_bytes"] = primary_file.get("bytes")
        exec_globals["chart_file_name"] = primary_file.get("filename")
        exec_globals["chart_file_paths"] = [
            path
            if isinstance(path, Path)
            else Path(path)
            for path in (info.get("path") for info in chart_files)
            if isinstance(path, (Path, str))
        ]

    exec_locals: Dict[str, Any] = {}

    try:
        exec(code, exec_globals, exec_locals)
        figure = (
            exec_locals.get("fig")
            or exec_globals.get("fig")
            or exec_locals.get("figure")
        )
        if figure is None:
            figure = plt.gcf()
        st.pyplot(figure)
    except Exception as exc:
        st.error(f"Loi thuc thi code bieu do: {exc}")
        st.code(code, language="python")
        return

    if chart_files:
        for idx, file_info in enumerate(chart_files):
            binary = file_info.get("bytes")
            if not isinstance(binary, (bytes, bytearray)):
                continue

            filename = file_info.get("filename") or f"chart_file_{idx + 1}.bin"
            mime = file_info.get("mime") or "application/octet-stream"
            label = file_info.get("label") or f"Tai {filename}"

            download_key = (
                f"chart_download_{session_dir.name}_{filename}_{idx}".replace(" ", "_")
            )

            st.download_button(
                label=label,
                data=binary,
                file_name=filename,
                mime=mime,
                key=download_key,
            )


def render_table(table: Dict[str, Any]) -> None:
    caption = table.get("caption")
    if caption:
        st.caption(caption)

    columns = table.get("columns")
    rows = table.get("rows")

    if not isinstance(columns, list) or not isinstance(rows, list):
        st.info("Khong co du lieu bang de hien thi.")
        return

    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows, columns=columns)
        st.dataframe(df, use_container_width=True)
    except ImportError:
        st.table(rows)
    except Exception as exc:
        st.error(f"Loi hien thi bang: {exc}")


def render_file(
    file_info: Dict[str, Any],
    session_id: str,
    key_prefix: str,
    index: int,
) -> None:
    if "error" in file_info:
        label = file_info.get("filename") or "Tep khong ro ten"
        st.error(f"{label}: {file_info['error']}")
        return

    path = file_info.get("path")
    if isinstance(path, Path):
        st.success(f"Da luu tep tai {path}")

    binary = file_info.get("bytes")
    if not isinstance(binary, (bytes, bytearray)):
        return

    filename = file_info.get("filename") or f"tep_{index}.bin"
    mime = file_info.get("mime") or "application/octet-stream"
    label = file_info.get("label") or f"Tai {filename}"

    st.download_button(
        label=label,
        data=binary,
        file_name=filename,
        mime=mime,
        key=f"download_{session_id}_{key_prefix}_{index}",
    )


def render_payload(
    payload: Dict[str, Any],
    *,
    session_id: str,
    session_dir: Path,
    key_prefix: str,
) -> None:
    if not payload:
        return

    def render_texts() -> None:
        for text in payload.get("text", []):
            st.markdown(text)

    def render_charts() -> None:
        for chart in payload.get("charts", []):
            render_chart(chart, session_dir)

    def render_tables() -> None:
        for table in payload.get("tables", []):
            render_table(table)

    def render_files() -> None:
        for idx, file_info in enumerate(payload.get("files", [])):
            render_file(file_info, session_id, key_prefix, idx)

    handlers = {
        "text": (render_texts, "text"),
        "chart": (render_charts, "charts"),
        "table": (render_tables, "tables"),
        "file": (render_files, "files"),
    }

    rendered: List[str] = []
    for message_type in payload.get("types", []):
        message_type = message_type.lower()
        handler_entry = handlers.get(message_type)
        if handler_entry and message_type not in rendered:
            handler_entry[0]()
            rendered.append(message_type)

    for message_type, (handler_func, data_key) in handlers.items():
        if message_type not in rendered and payload.get(data_key):
            handler_func()


if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())

SESSION_ID = st.session_state.sid
SESSION_DIR = ensure_session_dir(SESSION_ID)

if "messages" not in st.session_state:
    st.session_state.messages = []


for history_index, message in enumerate(st.session_state.messages):
    role = message.get("role", "assistant")
    with st.chat_message(role):
        if role == "user":
            st.markdown(message.get("content", ""))
        else:
            payload = message.get("payload")
            if payload:
                render_payload(
                    payload,
                    session_id=SESSION_ID,
                    session_dir=SESSION_DIR,
                    key_prefix=f"history_{history_index}",
                )
            elif message.get("content"):
                st.markdown(message["content"])
            if message.get("error"):
                st.error(message["error"])


prompt = st.chat_input("Nhap cau hoi...")

if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    payload_request = {"text": prompt, "session_id": SESSION_ID}

    assistant_container = st.chat_message("assistant")
    with assistant_container:
        status_placeholder = st.empty()
        status_placeholder.markdown("Dang soan phan hoi...")

        reply_payload: Optional[Dict[str, Any]] = None
        error_message: Optional[str] = None

        with st.spinner("Dang cho phan hoi tu webhook..."):
            try:
                response = requests.post(WEBHOOK_URL, json=payload_request, timeout=300)
                response.raise_for_status()
                data = response.json()
                reply_payload = build_payload(data, SESSION_DIR)
            except requests.exceptions.RequestException as exc:
                error_message = f"Loi goi webhook: {exc}"
            except ValueError:
                error_message = "Phan hoi webhook khong hop le (khong phai JSON)."
            except Exception as exc:
                error_message = f"Loi khong xac dinh: {exc}"

        if reply_payload is None or not any(
            reply_payload.get(key) for key in ("text", "charts", "tables", "files")
        ):
            fallback_text = error_message or "Khong nhan duoc phan hoi tu webhook."
            reply_payload = {
                "types": ["text"],
                "text": [fallback_text],
                "charts": [],
                "tables": [],
                "files": [],
            }

        status_placeholder.empty()
        render_payload(
            reply_payload,
            session_id=SESSION_ID,
            session_dir=SESSION_DIR,
            key_prefix=f"reply_{len(st.session_state.messages)}",
        )

        if error_message:
            st.error(error_message)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "payload": reply_payload,
            "error": error_message,
        }
    )
