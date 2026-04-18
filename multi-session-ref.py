"""
멀티세션 RAG 챗봇 — Supabase 세션/벡터 저장, OpenAI 임베딩, 스트리밍 답변.
실행: streamlit run multi-session-ref.py (7.MultiService/code 권장)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from postgrest.exceptions import APIError
from supabase import Client, create_client

# --- 경로 및 환경 ---
ROOT = Path(__file__).resolve().parent.parent.parent
SQL_SETUP_PATH = Path(__file__).resolve().parent / "multi-session-ref.sql"
ENV_PATH = ROOT / ".env"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"

load_dotenv(ENV_PATH)

for _name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("multi_session_rag")

MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
VECTOR_BATCH = 10
RAG_K = 10

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~[^~]+~~", "", text)
    text = re.sub(r"(?m)^\s*(---+|===+|___+)\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_supabase() -> Optional[Client]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def _is_missing_schema_error(exc: BaseException) -> bool:
    raw = str(exc)
    low = raw.lower()
    if "pgrst205" in low:
        return True
    if "could not find the table" in low and "sessions" in low:
        return True
    if isinstance(exc, APIError):
        d = getattr(exc, "args", None)
        if d and isinstance(d[0], dict):
            return d[0].get("code") == "PGRST205"
    return False


def verify_supabase_schema(supabase: Client) -> tuple[bool, str]:
    """sessions 테이블 존재 여부 확인 (PostgREST 스키마 캐시에 노출되어 있어야 함)."""
    try:
        supabase.table("sessions").select("id").limit(1).execute()
        return True, ""
    except APIError as e:
        if _is_missing_schema_error(e):
            return (
                False,
                (
                    f"Supabase에 `public.sessions` 테이블이 없습니다. "
                    f"Dashboard → **SQL Editor**에서 아래 파일 내용을 **전부 실행**하세요.\n\n"
                    f"`{SQL_SETUP_PATH}`\n\n"
                    "실행 후에도 같은 오류면 **Project Settings → API → Reload schema** 또는 잠시 후 다시 시도하세요."
                ),
            )
        return False, f"Supabase 오류: {e}"
    except Exception as e:
        if _is_missing_schema_error(e):
            return (
                False,
                (
                    f"Supabase에 필요한 테이블이 없거나 스키마가 반영되지 않았습니다.\n\n"
                    f"`{SQL_SETUP_PATH}` 를 SQL Editor에서 실행하세요."
                ),
            )
        return False, f"Supabase 연결 오류: {e}"


def get_embeddings() -> Optional[OpenAIEmbeddings]:
    if not OPENAI_API_KEY:
        return None
    return OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)


def get_llm(streaming: bool = True, temperature: float = 0.7) -> Optional[ChatOpenAI]:
    if not OPENAI_API_KEY:
        return None
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        streaming=streaming,
        openai_api_key=OPENAI_API_KEY,
    )


def touch_session_updated(supabase: Client, session_id: str) -> None:
    supabase.table("sessions").update({"updated_at": datetime.utcnow().isoformat() + "Z"}).eq(
        "id", session_id
    ).execute()


def insert_session(supabase: Client, title: str, session_id: Optional[str] = None) -> str:
    sid = session_id or str(uuid.uuid4())
    payload = {"id": sid, "title": title}
    supabase.table("sessions").insert(payload).execute()
    return sid


def save_messages_to_db(supabase: Client, session_id: str, messages: List[dict]) -> None:
    supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
    rows = []
    for i, m in enumerate(messages):
        rows.append(
            {
                "session_id": session_id,
                "role": m["role"],
                "content": m["content"],
                "sort_order": i,
            }
        )
    if rows:
        supabase.table("chat_messages").insert(rows).execute()
    touch_session_updated(supabase, session_id)


def load_messages_from_db(supabase: Client, session_id: str) -> List[dict]:
    res = (
        supabase.table("chat_messages")
        .select("role, content, sort_order")
        .eq("session_id", session_id)
        .order("sort_order")
        .execute()
    )
    rows = res.data or []
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def list_sessions(supabase: Client) -> List[dict]:
    res = (
        supabase.table("sessions")
        .select("id, title, updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return res.data or []


def parse_embedding(raw: Any) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            return [float(x) for x in json.loads(s)]
    return []


def cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_by_rpc(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    query: str,
    k: int,
) -> str:
    q_emb = embeddings.embed_query(query)
    try:
        res = supabase.rpc(
            "match_vector_documents",
            {
                "query_embedding": q_emb,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        rows = res.data or []
        parts = []
        for r in rows:
            fn = r.get("file_name", "")
            c = r.get("content", "")
            parts.append(f"[{fn}]\n{c}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning("RPC retrieve failed: %s", e)
        return retrieve_fallback(supabase, embeddings, session_id, query, k)


def retrieve_fallback(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    query: str,
    k: int,
) -> str:
    res = (
        supabase.table("vector_documents")
        .select("content, file_name, embedding")
        .eq("session_id", session_id)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return ""
    q_emb = embeddings.embed_query(query)
    scored = []
    for r in rows:
        emb = parse_embedding(r.get("embedding"))
        if len(emb) != len(q_emb):
            continue
        scored.append((cosine_sim(q_emb, emb), r))
    scored.sort(key=lambda x: -x[0])
    top = [t[1] for t in scored[:k]]
    parts = [f"[{t.get('file_name','')}]\n{t.get('content','')}" for t in top]
    return "\n\n".join(parts)


def embed_and_insert_pdf_chunks(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    file_name: str,
    chunks: List[str],
) -> None:
    for i in range(0, len(chunks), VECTOR_BATCH):
        batch = chunks[i : i + VECTOR_BATCH]
        vecs = embeddings.embed_documents(batch)
        rows = []
        for text, vec in zip(batch, vecs):
            rows.append(
                {
                    "session_id": session_id,
                    "file_name": file_name,
                    "content": text,
                    "metadata": {"source": file_name},
                    "embedding": vec,
                }
            )
        if rows:
            supabase.table("vector_documents").insert(rows).execute()


def process_uploaded_pdfs(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    files: List[Any],
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    processed = []
    for uf in files:
        path = getattr(uf, "name", "upload.pdf")
        base = os.path.basename(path)
        with st.spinner(f"처리 중: {base}"):
            tmp = ROOT / "logs" / f"_tmp_{uuid.uuid4().hex}_{base}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp.write_bytes(uf.getvalue())
                loader = PyPDFLoader(str(tmp))
                docs = loader.load()
                texts = splitter.split_documents(docs)
                chunks: List[str] = []
                for d in texts:
                    d.metadata = d.metadata or {}
                    d.metadata["file_name"] = base
                    chunks.append(d.page_content)
                if chunks:
                    embed_and_insert_pdf_chunks(supabase, embeddings, session_id, base, chunks)
                    processed.append(base)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
    return processed


def generate_session_title_llm(first_q: str, first_a: str) -> str:
    llm = get_llm(streaming=False, temperature=0.3)
    if not llm:
        return "새 세션"
    prompt = (
        "다음은 사용자의 첫 질문과 어시스턴트의 첫 답변입니다. "
        "이 대화를 대표하는 짧은 세션 제목을 한 줄로만 출력하세요 (25자 이내, 따옴표 없음).\n\n"
        f"질문: {first_q}\n\n답변: {first_a[:1200]}"
    )
    out = llm.invoke([HumanMessage(content=prompt)])
    title = (out.content or "").strip().split("\n")[0][:80]
    return title or "새 세션"


def build_system_instruction(include_rag_context: bool) -> str:
    base = (
        "당신은 친절한 AI 어시스턴트입니다. 답변은 한국어 존댓말로 하고, "
        "# ## ### 마크다운 헤딩으로 구조화하세요. "
        "구분선(---, ===, ___)과 취소선(~~)은 쓰지 마세요. "
        "출처·참조 문구는 넣지 마세요.\n"
        "답변 마지막에 반드시 다음 섹션을 추가하세요:\n"
        "### 💡 다음에 물어볼 수 있는 질문들\n"
        "1. …\n2. …\n3. …\n"
        "질문은 사용자가 문서·주제를 더 깊게 탐구할 수 있게 구체적으로 작성하세요."
    )
    if include_rag_context:
        base += (
            "\n아래 [참고 문맥]에 나온 내용을 우선 활용하고, 없는 내용은 추측하지 말고 모른다고 하세요."
        )
    return base


def stream_answer(
    llm: ChatOpenAI,
    system_text: str,
    history_lc: List[Any],
    user_text: str,
) -> Generator[str, None, None]:
    msgs = [SystemMessage(content=system_text)] + history_lc + [HumanMessage(content=user_text)]
    for chunk in llm.stream(msgs):
        if chunk.content:
            yield chunk.content


def duplicate_session_snapshot(
    supabase: Client,
    source_session_id: str,
    title: str,
) -> str:
    new_id = str(uuid.uuid4())
    supabase.table("sessions").insert({"id": new_id, "title": title}).execute()
    msgs = load_messages_from_db(supabase, source_session_id)
    if msgs:
        rows = []
        for i, m in enumerate(msgs):
            rows.append(
                {
                    "session_id": new_id,
                    "role": m["role"],
                    "content": m["content"],
                    "sort_order": i,
                }
            )
        supabase.table("chat_messages").insert(rows).execute()

    offset = 0
    page_size = 500
    while True:
        res = (
            supabase.table("vector_documents")
            .select("file_name, content, metadata, embedding")
            .eq("session_id", source_session_id)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = res.data or []
        if not batch:
            break
        ins = []
        for r in batch:
            emb = r.get("embedding")
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                except json.JSONDecodeError:
                    continue
            ins.append(
                {
                    "session_id": new_id,
                    "file_name": r["file_name"],
                    "content": r["content"],
                    "metadata": r.get("metadata") or {},
                    "embedding": emb,
                }
            )
        if ins:
            for i in range(0, len(ins), VECTOR_BATCH):
                supabase.table("vector_documents").insert(ins[i : i + VECTOR_BATCH]).execute()
        if len(batch) < page_size:
            break
        offset += page_size

    touch_session_updated(supabase, new_id)
    return new_id


def delete_session_cascade(supabase: Client, session_id: str) -> None:
    supabase.table("sessions").delete().eq("id", session_id).execute()


def list_vector_file_names(supabase: Client, session_id: str) -> List[str]:
    res = (
        supabase.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = sorted({r["file_name"] for r in (res.data or []) if r.get("file_name")})
    return names


def count_vectors(supabase: Client, session_id: str) -> int:
    res = (
        supabase.table("vector_documents")
        .select("id", count="exact")
        .eq("session_id", session_id)
        .execute()
    )
    return getattr(res, "count", None) or len(res.data or [])


def render_line_chart_panel() -> None:
    """CSV 또는 쉼표 구분 숫자로 꺾은 선 그래프 표시."""
    st.markdown("#### 꺾은 선 그래프")
    t_csv, t_manual = st.tabs(["CSV 파일", "숫자 직접 입력"])
    with t_csv:
        f = st.file_uploader(
            "CSV (첫 번째 열: 가로축, 두 번째 열: 세로축)",
            type=["csv"],
            key="line_chart_csv",
        )
        if f is not None:
            try:
                df = pd.read_csv(f)
                if df.shape[1] < 2:
                    st.warning("CSV는 가로·세로에 해당하는 열이 최소 2개 있어야 합니다.")
                else:
                    plot_df = df.iloc[:, :2].copy()
                    plot_df.columns = ["x", "y"]
                    try:
                        plot_df["y"] = pd.to_numeric(plot_df["y"], errors="coerce")
                    except Exception:
                        pass
                    if plot_df["y"].isna().all():
                        st.warning("세로축 열을 숫자로 변환할 수 없습니다.")
                    else:
                        plot_df = plot_df.set_index("x")
                        st.line_chart(plot_df, height=320)
            except Exception as e:
                st.error(f"CSV를 읽는 중 오류: {e}")
    with t_manual:
        y_str = st.text_input(
            "Y 값 (쉼표로 구분)",
            "10, 25, 15, 30, 20",
            key="line_chart_y_vals",
        )
        x_str = st.text_input(
            "X 라벨 (쉼표로 구분, 비우면 0부터 번호)",
            "",
            key="line_chart_x_vals",
        )
        if st.button("그래프 그리기", key="line_chart_draw_btn"):
            ys: Optional[List[float]] = None
            try:
                ys = [float(s.strip()) for s in y_str.split(",") if s.strip()]
            except ValueError:
                st.warning("Y 값은 숫자만 쉼표로 구분해 입력하세요.")
            if ys is not None:
                if not ys:
                    st.warning("Y 값을 하나 이상 입력하세요.")
                elif x_str.strip():
                    xs = [s.strip() for s in x_str.split(",") if s.strip()]
                    if len(xs) != len(ys):
                        st.warning("X 라벨 개수와 Y 값 개수가 같아야 합니다.")
                    else:
                        dfp = pd.DataFrame({"값": ys}, index=xs)
                        st.line_chart(dfp, height=320)
                else:
                    dfp = pd.DataFrame({"값": ys})
                    st.line_chart(dfp, height=320)


def apply_header_css() -> None:
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div[data-testid="stButton"] button {
    background-color: #ff69b4 !important;
    color: #ffffff !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_top_header() -> None:
    logo_path = ROOT / "logo.png"
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if logo_path.is_file():
            st.image(str(logo_path), width=180)
        else:
            st.markdown("### 📚")
    with col2:
        st.markdown(
            """
<div style="text-align:center;">
  <span style="font-size:4rem !important; font-weight:800;">
    <span style="color:#1f77b4;">멀티세션</span>
    <span style="color:#ffd700;"> RAG 챗봇</span>
  </span>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.empty()


def ensure_session_in_db(supabase: Client) -> None:
    if "session_id" not in st.session_state:
        return
    sid = st.session_state.session_id
    chk = supabase.table("sessions").select("id").eq("id", sid).execute()
    if not (chk.data or []):
        insert_session(supabase, "새 세션", session_id=sid)


def maybe_update_title_after_first_turn(supabase: Client) -> None:
    msgs = st.session_state.messages
    if len(msgs) < 2:
        return
    if st.session_state.get("title_auto_done"):
        return
    u0 = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a0 = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    if not u0 or not a0:
        return
    title = generate_session_title_llm(u0, a0)
    supabase.table("sessions").update({"title": title}).eq("id", st.session_state.session_id).execute()
    st.session_state.title_auto_done = True
    touch_session_updated(supabase, st.session_state.session_id)


def main() -> None:
    st.set_page_config(page_title="멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
    apply_header_css()
    render_top_header()

    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing.append("SUPABASE_ANON_KEY")

    if missing:
        st.error(f"환경 변수가 없습니다: {', '.join(missing)}. {ENV_PATH} 를 확인하세요.")
        return

    supabase = get_supabase()
    embeddings = get_embeddings()
    if not supabase or not embeddings:
        st.error("Supabase 또는 OpenAI 임베딩 초기화에 실패했습니다.")
        return

    schema_ok, schema_msg = verify_supabase_schema(supabase)
    if not schema_ok:
        st.error(schema_msg)
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        insert_session(supabase, "새 세션", session_id=st.session_state.session_id)
    else:
        ensure_session_in_db(supabase)

    if "title_auto_done" not in st.session_state:
        st.session_state.title_auto_done = False
    if "show_vectordb_panel" not in st.session_state:
        st.session_state.show_vectordb_panel = False
    if "show_line_chart_panel" not in st.session_state:
        st.session_state.show_line_chart_panel = False

    # --- 사이드바 ---
    with st.sidebar:
        st.markdown("#### LLM 모델")
        st.radio("모델", [MODEL_NAME], index=0, disabled=True, help="고정: gpt-4o-mini")
        st.markdown("#### RAG (PDF)")
        rag_mode = st.radio("RAG", ["사용 안 함", "RAG 사용"], index=1, horizontal=False)
        st.markdown("#### PDF 업로드")
        uploads = st.file_uploader(
            "PDF (다중 선택 가능)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기"):
            if not uploads:
                st.warning("PDF를 선택하세요.")
            else:
                with st.spinner("PDF를 벡터 DB에 저장하는 중…"):
                    done = process_uploaded_pdfs(supabase, embeddings, st.session_state.session_id, uploads)
                if done:
                    st.success(f"처리 완료: {', '.join(done)}")
                    save_messages_to_db(supabase, st.session_state.session_id, st.session_state.messages)
                    st.rerun()

        st.markdown("#### 세션 관리")
        all_sess = list_sessions(supabase)

        def _on_session_pick() -> None:
            sid = st.session_state.get("session_pick")
            if not sid:
                return
            st.session_state.messages = load_messages_from_db(supabase, sid)
            st.session_state.session_id = sid
            st.session_state.title_auto_done = True
            st.session_state.show_vectordb_panel = False

        ids = [s["id"] for s in all_sess]

        def _fmt(sid: str) -> str:
            row = next((x for x in all_sess if x["id"] == sid), None)
            t = (row or {}).get("title", sid)
            return f"{t} ({str(sid)[:8]}…)"

        if ids:
            try:
                _idx = ids.index(st.session_state.session_id)
            except ValueError:
                _idx = 0
            st.selectbox(
                "세션 선택 (선택 시 자동 로드)",
                ids,
                index=_idx,
                format_func=_fmt,
                key="session_pick",
                on_change=_on_session_pick,
            )
        else:
            st.caption("저장된 세션이 없습니다.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("세션저장"):
                msgs = st.session_state.messages
                users = [m for m in msgs if m["role"] == "user"]
                asst = [m for m in msgs if m["role"] == "assistant"]
                if not users or not asst:
                    st.warning("첫 질문과 답변이 있어야 세션을 저장할 수 있습니다.")
                else:
                    title = generate_session_title_llm(users[0]["content"], asst[0]["content"])
                    new_sid = duplicate_session_snapshot(supabase, st.session_state.session_id, title)
                    st.success(f"새 세션으로 저장했습니다. 제목: {title}")
                    st.caption(f"새 세션 ID: {new_sid[:8]}…")
                    st.rerun()

        with c2:
            if st.button("세션로드"):
                sid = st.session_state.get("session_pick")
                if not sid:
                    st.warning("세션을 선택하세요.")
                else:
                    st.session_state.messages = load_messages_from_db(supabase, sid)
                    st.session_state.session_id = sid
                    st.session_state.title_auto_done = True
                    st.rerun()

        c3, c4 = st.columns(2)
        with c3:
            if st.button("세션삭제"):
                sid = st.session_state.session_id
                if sid:
                    delete_session_cascade(supabase, sid)
                    new_id = str(uuid.uuid4())
                    insert_session(supabase, "새 세션", session_id=new_id)
                    st.session_state.session_id = new_id
                    st.session_state.messages = []
                    st.session_state.title_auto_done = False
                    st.success("세션을 삭제하고 새 세션을 시작했습니다.")
                    st.rerun()

        with c4:
            if st.button("화면초기화"):
                new_id = str(uuid.uuid4())
                insert_session(supabase, "새 세션", session_id=new_id)
                st.session_state.session_id = new_id
                st.session_state.messages = []
                st.session_state.title_auto_done = False
                st.session_state.show_vectordb_panel = False
                st.session_state.show_line_chart_panel = False
                st.info("화면을 초기화했습니다.")
                st.rerun()

        if st.button("vectordb"):
            st.session_state.show_vectordb_panel = not st.session_state.show_vectordb_panel
            st.rerun()

        if st.button("꺾은선 그래프"):
            st.session_state.show_line_chart_panel = not st.session_state.show_line_chart_panel
            st.rerun()

        n_vec = count_vectors(supabase, st.session_state.session_id)
        st.text(
            f"모델: {MODEL_NAME}\n"
            f"RAG: {rag_mode}\n"
            f"현재 세션 벡터 청크 수: {n_vec}\n"
            f"대화 메시지 수: {len(st.session_state.messages)}"
        )

    if st.session_state.show_vectordb_panel:
        names = list_vector_file_names(supabase, st.session_state.session_id)
        st.markdown("#### 현재 세션 vectordb 파일 목록")
        if names:
            for n in names:
                st.markdown(f"- `{n}`")
        else:
            st.caption("저장된 파일이 없습니다.")

    if st.session_state.show_line_chart_panel:
        render_line_chart_panel()

    llm = get_llm(streaming=True)
    if not llm:
        st.error("LLM을 초기화할 수 없습니다.")
        return

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(remove_separators(m["content"]))

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_input = st.session_state.messages[-1]["content"]
        history_lc: List[Any] = []
        for m in st.session_state.messages[:-1]:
            if m["role"] == "user":
                history_lc.append(HumanMessage(content=m["content"]))
            else:
                history_lc.append(AIMessage(content=m["content"]))

        use_rag = rag_mode == "RAG 사용" and count_vectors(supabase, st.session_state.session_id) > 0
        ctx = ""
        if use_rag:
            ctx = retrieve_by_rpc(supabase, embeddings, st.session_state.session_id, user_input, RAG_K)

        sys_text = build_system_instruction(bool(ctx))
        if ctx:
            user_block = f"[참고 문맥]\n{ctx}\n\n[질문]\n{user_input}"
        else:
            user_block = user_input

        full = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for piece in stream_answer(llm, sys_text, history_lc, user_block):
                full += piece
                placeholder.markdown(remove_separators(full) + "▌")
            placeholder.markdown(remove_separators(full))

        final_text = remove_separators(full)
        st.session_state.messages.append({"role": "assistant", "content": final_text})
        save_messages_to_db(supabase, st.session_state.session_id, st.session_state.messages)
        maybe_update_title_after_first_turn(supabase)
        st.rerun()

    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_messages_to_db(supabase, st.session_state.session_id, st.session_state.messages)
        st.rerun()


if __name__ == "__main__":
    main()
