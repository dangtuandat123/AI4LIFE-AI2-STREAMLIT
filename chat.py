import base64
import datetime
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from htbuilder import div, styles
from htbuilder.units import rem

import requests
import streamlit as st


WEBHOOK_URL = "https://chatgpt.id.vn/webhook-test/70ecee2a-c278-461f-a898-52ff907b4fb2"
AGENT_SUGGEST_WEBHOOK_URL = "https://chatgpt.id.vn/webhook/agent-suggest"

SUGGESTIONS_DEBOUNCE_SECONDS = 0.5

FALLBACK_SUGGESTIONS = [
    "n8n là gì và cách bắt đầu một workflow đơn giản?",
    "Giải thích cách sử dụng webhook trigger và cách kiểm tra payload đầu vào.",
    "Hướng dẫn các bước thử nghiệm và debug workflow n8n khi gặp lỗi.",
    "Có những cách nào để lưu trữ kết quả workflow (Google Sheet, DB, Snowflake...)?",
    "Gợi ý các kỹ thuật xác thực và giải mã để bảo vệ webhook của tôi.",
]

st.set_page_config(page_title="Chat với n8n", page_icon="✨")

st.html(div(style=styles(font_size=rem(5), line_height=1))["❉"])

title_row = st.container(horizontal=True, vertical_alignment="bottom")


@st.dialog("Lưu ý")
def show_disclaimer_dialog():
    st.caption(
        """
        Đây là demo chatbot kết nối đến một webhook n8n. Các câu trả lời có thể
        chứa thông tin chưa chính xác. Vui lòng tránh chia sẻ dữ liệu nhạy cảm
        và hãy kiểm chứng lại các hành động thực tế trước khi thực hiện.
        """
    )


def clear_conversation():
    st.session_state.messages = []
    st.session_state.sid = str(uuid.uuid4())
    st.session_state.suggestion_seed = ""
    st.session_state.chat_prompt = ""
    st.session_state.last_input_change = datetime.datetime.now()
    st.session_state.last_suggest_text = None
    st.session_state.last_suggest_timestamp = datetime.datetime.fromtimestamp(0)
    st.session_state.agent_suggestions = list(FALLBACK_SUGGESTIONS)
    st.session_state.suggestions_loading = False
    st.session_state.prefill_prompt = None


def parse_agent_suggestions(payload: Any) -> List[str]:
    """Extracts suggestion strings from the agent webhook payload."""
    suggestions: List[str] = []

    if isinstance(payload, list):
        for item in payload:
            suggestions.extend(parse_agent_suggestions(item))
    elif isinstance(payload, dict):
        if "output" in payload:
            suggestions.extend(parse_agent_suggestions(payload["output"]))
        raw = payload.get("suggestions")
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, str):
                    clean = entry.strip()
                    if clean:
                        suggestions.append(clean)
    elif isinstance(payload, str):
        clean = payload.strip()
        if clean:
            suggestions.append(clean)

    unique: List[str] = []
    seen = set()
    for suggestion in suggestions:
        if suggestion not in seen:
            seen.add(suggestion)
            unique.append(suggestion)

    return unique


def fetch_agent_suggestions(query_text: str) -> List[str]:
    """Calls the suggestion webhook and returns a cleaned list of prompts."""
    try:
        response = requests.post(
            AGENT_SUGGEST_WEBHOOK_URL,
            json={"text": query_text or ""},
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        st.warning(f"Lỗi lấy gợi ý: {exc}")
        return []

    try:
        payload = response.json()
    except ValueError:
        st.warning("Phản hồi gợi ý không phải JSON.")
        return []

    suggestions = parse_agent_suggestions(payload)
    return suggestions


def refresh_agent_suggestions(query_text: str) -> None:
    """Updates session state with the newest suggestions for `query_text`."""
    if st.session_state.get("suggestions_loading"):
        return

    st.session_state.suggestions_loading = True
    try:
        suggestions = fetch_agent_suggestions(query_text)
    finally:
        st.session_state.suggestions_loading = False

    if not suggestions:
        suggestions = list(FALLBACK_SUGGESTIONS)

    st.session_state.agent_suggestions = suggestions
    st.session_state.last_suggest_text = query_text
    st.session_state.last_suggest_timestamp = datetime.datetime.now()


def mark_suggestion_seed_changed() -> None:
    """Records edits to the suggestion seed text."""
    st.session_state.last_input_change = datetime.datetime.now()


def prefill_chat_input(text: str) -> None:
    """Queues a suggestion to be inserted into the chat input box."""
    st.session_state.prefill_prompt = text


with title_row:
    st.title("BeeBox", anchor=False, width="stretch")
    st.button(
        "Khởi động lại",
        icon=":material/refresh:",
        on_click=clear_conversation,
        type="secondary",
    )


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
            processed.append({"error": "Tệp không có dữ liệu hợp lệ."})
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
                    "error": f"Lỗi giải mã tệp: {exc}",
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
                    "error": f"Lỗi ghi tệp: {exc}",
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
        st.info("Không có code biểu đồ để hiển thị.")
        return

    caption = chart.get("caption") or chart.get("description")
    if isinstance(caption, str):
        st.caption(caption)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        st.warning("Matplotlib chưa được cài đặt.")
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
        st.error(f"Lỗi thực thi code biểu đồ: {exc}")
        st.code(code, language="python")
        return

    if chart_files:
        for idx, file_info in enumerate(chart_files):
            binary = file_info.get("bytes")
            if not isinstance(binary, (bytes, bytearray)):
                continue

            filename = file_info.get("filename") or f"chart_file_{idx + 1}.bin"
            mime = file_info.get("mime") or "application/octet-stream"
            label = file_info.get("label") or f"Tải {filename}"

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
        st.info("Không có dữ liệu bảng để hiển thị.")
        return

    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows, columns=columns)
        st.dataframe(df, use_container_width=True)
    except ImportError:
        st.table(rows)
    except Exception as exc:
        st.error(f"Lỗi hiển thị bảng: {exc}")


def render_file(
    file_info: Dict[str, Any],
    session_id: str,
    key_prefix: str,
    index: int,
) -> None:
    if "error" in file_info:
        label = file_info.get("filename") or "Tệp không rõ tên"
        st.error(f"{label}: {file_info['error']}")
        return

    path = file_info.get("path")
    if isinstance(path, Path):
        st.success(f"Đã lưu tệp tại {path}")

    binary = file_info.get("bytes")
    if not isinstance(binary, (bytes, bytearray)):
        return

    filename = file_info.get("filename") or f"tep_{index}.bin"
    mime = file_info.get("mime") or "application/octet-stream"
    label = file_info.get("label") or f"Tải {filename}"

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

if "suggestion_seed" not in st.session_state:
    st.session_state.suggestion_seed = ""

if "chat_prompt" not in st.session_state:
    st.session_state.chat_prompt = ""

if "prefill_prompt" not in st.session_state:
    st.session_state.prefill_prompt = None

if "agent_suggestions" not in st.session_state:
    st.session_state.agent_suggestions = list(FALLBACK_SUGGESTIONS)

if "last_suggest_text" not in st.session_state:
    st.session_state.last_suggest_text = None

if "last_suggest_timestamp" not in st.session_state:
    st.session_state.last_suggest_timestamp = datetime.datetime.fromtimestamp(0)

if "suggestions_loading" not in st.session_state:
    st.session_state.suggestions_loading = False

if "last_input_change" not in st.session_state:
    st.session_state.last_input_change = datetime.datetime.now()

now = datetime.datetime.now()
current_prompt_value = st.session_state.suggestion_seed.strip()

if (
    st.session_state.last_suggest_text is None
    and not st.session_state.suggestions_loading
):
    refresh_agent_suggestions(current_prompt_value)
else:
    debounce_delta = datetime.timedelta(seconds=SUGGESTIONS_DEBOUNCE_SECONDS)
    idle_time = now - st.session_state.last_input_change
    if (
        idle_time >= debounce_delta
        and current_prompt_value != (st.session_state.last_suggest_text or "")
        and not st.session_state.suggestions_loading
    ):
        refresh_agent_suggestions(current_prompt_value)

with st.sidebar:
    st.subheader("Bảng điều khiển")
    st.caption("Chatbot này gửi câu hỏi đến webhook n8n bên dưới.")
    st.code(WEBHOOK_URL, language="text")
    st.text_input("Mã phiên", value=SESSION_ID, disabled=True)
    st.metric("Số tin nhắn", len(st.session_state.messages))
    st.button(
        ":material/balance: Thông tin pháp lý",
        key="sidebar_disclaimer",
        use_container_width=True,
        on_click=show_disclaimer_dialog,
        type="secondary",
    )
    st.button(
        ":material/refresh: Tạo phiên mới",
        key="sidebar_restart",
        use_container_width=True,
        on_click=clear_conversation,
    )
    st.divider()
    st.markdown("### Gợi ý hiện tại")
    if st.session_state.suggestions_loading:
        st.caption("Đang cập nhật gợi ý dựa trên nội dung bạn nhập…")
    sidebar_suggestions = st.session_state.agent_suggestions[:3]
    if sidebar_suggestions:
        for idx, suggestion in enumerate(sidebar_suggestions, start=1):
            st.markdown(f"**#{idx}** {suggestion}")
    else:
        st.caption("Chưa có gợi ý nào khả dụng.")
    st.divider()
    st.caption("Gợi ý sẽ tự cập nhật sau ~1 giây kể từ lần gõ cuối cùng.")

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


st.divider()
with st.container():
    st.markdown("### Gợi ý từ Agent")
    st.text_area(
        "Nhập mô tả để lấy gợi ý",
        key="suggestion_seed",
        placeholder="Ví dụ: Phân tích doanh thu theo tháng, so sánh theo vùng...",
        height=100,
        on_change=mark_suggestion_seed_changed,
    )
    st.caption("Gợi ý sẽ tự cập nhật ~1 giây sau khi bạn dừng nhập.")
    with st.container(border=True):
        if st.session_state.suggestions_loading:
            st.caption("Đang cập nhật gợi ý dựa trên nội dung bạn nhập…")
        main_suggestions = st.session_state.agent_suggestions[:6]
        if not main_suggestions:
            st.caption("Chưa có gợi ý phù hợp. Hãy nhập yêu cầu cụ thể hơn.")
        else:
            for idx, suggestion in enumerate(main_suggestions, start=1):
                row_cols = st.columns([12, 2], gap="small")
                with row_cols[0]:
                    st.markdown(f":small[#{idx}] {suggestion}")
                with row_cols[1]:
                    st.button(
                        ":material/input:",
                        key=f"suggestion_insert_{idx}",
                        use_container_width=True,
                        type="secondary",
                        on_click=lambda text=suggestion: prefill_chat_input(text),
                    )

if st.session_state.prefill_prompt:
    st.session_state.chat_prompt = st.session_state.prefill_prompt
    st.session_state.prefill_prompt = None

user_message = st.chat_input("Nhập câu hỏi và nhấn Enter...", key="chat_prompt")

if user_message:
    raw_prompt = user_message
    display_prompt = raw_prompt.replace("$", r"\$")

    user_record = {"role": "user", "content": raw_prompt}
    st.session_state.messages.append(user_record)

    with st.chat_message("user"):
        st.markdown(display_prompt)

    payload_request = {"text": raw_prompt, "session_id": SESSION_ID}

    assistant_container = st.chat_message("assistant")
    with assistant_container:
        status_placeholder = st.empty()
        status_placeholder.markdown("Đang soạn phản hồi...")

        reply_payload: Optional[Dict[str, Any]] = None
        error_message: Optional[str] = None

        with st.spinner("Đang chờ phản hồi từ webhook..."):
            try:
                response = requests.post(WEBHOOK_URL, json=payload_request, timeout=300)
                response.raise_for_status()
                data = response.json()
                reply_payload = build_payload(data, SESSION_DIR)
            except requests.exceptions.RequestException as exc:
                error_message = f"Lỗi gọi webhook: {exc}"
            except ValueError:
                error_message = "Phản hồi webhook không hợp lệ (không phải JSON)."
            except Exception as exc:
                error_message = f"Lỗi không xác định: {exc}"

        if reply_payload is None or not any(
            reply_payload.get(key) for key in ("text", "charts", "tables", "files")
        ):
            fallback_text = error_message or "Không nhận được phản hồi từ webhook."
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
