import logging
import os
import re
import time
import fitz
import numpy as np
import streamlit as st
from annoy import AnnoyIndex
try:
    from mistralai import Mistral
except ImportError:  # mistralai v2.x namespace
    from mistralai.client import Mistral
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

EMBEDDING_MODEL = "mistral-embed"
# Smaller chunks work better for resumes (sections, bullets, job entries)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
MIN_CHUNK_CHARS = 40
PAGES_PER_BATCH = 20
EMBED_BATCH_SIZE = 8
EMBED_RETRY_DELAY = 2
MAX_EMBED_RETRIES = 5

HYBRID_METHODS = ("annoy", "tfidf", "bm25", "word2vec")
METHOD_LABELS = {
    "annoy": "Semantic (Mistral + Annoy)",
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "word2vec": "Word vectors",
}
# Hybrid fusion (Step 13): sum of per-method scores (each normalized 0–1 across chunks)
HYBRID_METHOD_ORDER = ("bm25", "tfidf", "word2vec", "annoy")

# Relevance gate: skip Mistral when the question does not match indexed PDFs
MIN_HYBRID_SUM_FOR_ANSWER = 1.55  # max possible ≈ 4.0
MIN_TOKEN_OVERLAP = 0.18  # share of query keywords found in top chunk
MIN_TOP_TFIDF = 0.11
MIN_TOP_ANNOY = 0.38

STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might must shall can to of in for on "
    "with at by from as and or but if not this that these those it its "
    "i you he she they we my your how what when where why who which".split()
)

NOT_IN_DOCUMENTS_ONLY_MSG = (
    "**Not found in your uploaded documents.**\n\n"
    "Your question does not match the content in the PDF(s) you uploaded. "
    "Enable *General answer when not in documents* in Advanced settings to get a "
    "Mistral AI answer as well."
)


def get_mistral_api_key() -> str:
    try:
        return st.secrets["MISTRAL_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    key = os.environ.get("MISTRAL_API_KEY")
    if key:
        return key
    st.error(
        "Mistral API key not configured. Add `MISTRAL_API_KEY` in "
        "Streamlit Cloud → Settings → Secrets, or set it locally in "
        "`.streamlit/secrets.toml`."
    )
    st.stop()


@st.cache_resource
def get_mistral_client(api_key: str) -> Mistral:
    return Mistral(api_key=api_key)


def clean_extracted_text(text: str) -> str:
    """Fix common PDF extraction artifacts (resumes, columns, hyphenation)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join hyphenated line breaks: "experi-\nence" -> "experience"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse spaces but keep paragraph breaks
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_page_text_robust(page: fitz.Page) -> str:
    """Try several PyMuPDF extractors; keep the longest result (most complete)."""
    candidates: list[str] = []

    def add(text: str | None):
        if text and text.strip():
            candidates.append(clean_extracted_text(text))

    try:
        add(page.get_text("text", sort=True))
    except (TypeError, ValueError):
        pass

    try:
        blocks = page.get_text("blocks", sort=True)
        parts = []
        for block in blocks:
            if len(block) >= 7 and block[6] == 0 and block[4].strip():
                parts.append(block[4].strip())
        if parts:
            add("\n".join(parts))
    except (TypeError, ValueError):
        pass

    try:
        data = page.get_text("dict")
        lines = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                span_text = "".join(s.get("text", "") for s in line.get("spans", []))
                if span_text.strip():
                    lines.append(span_text)
        if lines:
            add("\n".join(lines))
    except (TypeError, ValueError, KeyError):
        pass

    try:
        add(page.get_text())
    except (TypeError, ValueError):
        pass

    if not candidates:
        return ""
    return max(candidates, key=len)


def extract_pdf_with_stats(pdf_bytes: bytes) -> tuple[str, list[str], dict]:
    """
    Extract all pages. Returns full text, per-page texts, and extraction stats.
  Empty pages are usually scanned/image-only (no selectable text in PDF).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count
    page_texts: list[str] = []
    pages_with_text = 0

    for page_num in range(page_count):
        raw = extract_page_text_robust(doc.load_page(page_num))
        page_texts.append(raw)
        if raw.strip():
            pages_with_text += 1

    doc.close()

    parts = []
    for i, text in enumerate(page_texts):
        if text.strip():
            parts.append(f"--- Page {i + 1} ---\n{text}")

    full_text = "\n\n".join(parts)
    stats = {
        "page_count": page_count,
        "pages_with_text": pages_with_text,
        "pages_empty": page_count - pages_with_text,
        "char_count": len(full_text),
    }
    return full_text, page_texts, stats


def extract_full_text_from_pdf(pdf_bytes: bytes) -> str:
    full_text, _, _ = extract_pdf_with_stats(pdf_bytes)
    return full_text


def split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines; skip page marker lines only."""
    parts = re.split(r"\n\s*\n+", text)
    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Drop standalone page markers (content lives on other parts / pages)
        if re.fullmatch(r"--- Page \d+ ---", part):
            continue
        if part.startswith("--- Page "):
            part = re.sub(r"^--- Page \d+ ---\s*", "", part).strip()
            if not part:
                continue
        lines = part.split("\n")
        buffer: list[str] = []
        for line in lines:
            line = line.strip()
            if not line or re.fullmatch(r"--- Page \d+ ---", line):
                continue
            is_bullet = bool(re.match(r"^[\u2022\-\*\u25cf•]\s*", line)) or re.match(
                r"^\d+[\.\)]\s+", line
            )
            if is_bullet and buffer:
                sections.append("\n".join(buffer))
                buffer = [line]
            else:
                buffer.append(line)
        if buffer:
            sections.append("\n".join(buffer))
    return [s for s in sections if len(s.strip()) >= MIN_CHUNK_CHARS]


def split_long_section(section: str, chunk_size: int, overlap: int) -> list[str]:
    if len(section) <= chunk_size:
        return [section]
    chunks = []
    start = 0
    while start < len(section):
        end = min(start + chunk_size, len(section))
        chunk = section[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def dedupe_exact_chunks(chunks: list[str]) -> list[str]:
    """Remove only exact duplicate chunks (keep pages with similar headers)."""
    seen: set[str] = set()
    unique: list[str] = []
    for c in chunks:
        key = c.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def chunk_single_page_text(text: str) -> list[str]:
    """Chunk one page's text without cross-page deduplication."""
    text = clean_extracted_text(text)
    if not text.strip():
        return []
    paragraphs = split_into_paragraphs(text)
    chunks: list[str] = []
    for para in paragraphs:
        chunks.extend(split_long_section(para, CHUNK_SIZE, CHUNK_OVERLAP))
    if not chunks:
        chunks = split_long_section(text, CHUNK_SIZE, CHUNK_OVERLAP)
    return dedupe_exact_chunks(chunks)


def chunk_document(text: str, page_texts: list[str] | None = None) -> list[str]:
    """
    Chunk full document. Prefer per-page chunking so every page contributes
    (avoids losing pages when headers repeat across pages).
    """
    if page_texts:
        chunks: list[str] = []
        for page_text in page_texts:
            chunks.extend(chunk_single_page_text(page_text))
        return dedupe_exact_chunks(chunks)

    text = clean_extracted_text(text)
    paragraphs = split_into_paragraphs(text)
    chunks = []
    for para in paragraphs:
        chunks.extend(split_long_section(para, CHUNK_SIZE, CHUNK_OVERLAP))
    if not chunks and text:
        chunks = split_long_section(text, CHUNK_SIZE, CHUNK_OVERLAP)
    return dedupe_exact_chunks(chunks)


def extract_chunks_from_pdf(pdf_bytes: bytes) -> tuple[list[str], str, dict]:
    full_text, page_texts, pdf_stats = extract_pdf_with_stats(pdf_bytes)
    chunks = chunk_document(full_text, page_texts=page_texts)
    meta = {
        "char_count": len(full_text),
        "chunk_count": len(chunks),
        "page_count": pdf_stats["page_count"],
        "pages_with_text": pdf_stats["pages_with_text"],
        "preview": full_text[:4000] + ("…" if len(full_text) > 4000 else ""),
        "chunks_indexed": [
            {"Chunk": chunk_label(i), "Preview": truncate_preview(c, 100)}
            for i, c in enumerate(chunks)
        ],
    }
    return chunks, full_text, meta


def truncate_for_embedding(text: str, max_tokens: int = 8000) -> str:
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens])
    return text


def normalize_array(scores: np.ndarray) -> np.ndarray:
    """Min–max normalize scores to 0–1 across all chunks (for hybrid sum)."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    if arr.max() == arr.min():
        return np.ones_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def chunk_label(index: int, source_doc: str | None = None, local_index: int | None = None) -> str:
    num = (local_index + 1) if local_index is not None else (index + 1)
    if source_doc:
        short = source_doc if len(source_doc) <= 28 else source_doc[:25] + "…"
        return f"{short} · Chunk {num}"
    return f"Chunk {num}"


def build_corpus_from_pdfs(
    files: list[tuple[str, bytes]],
) -> tuple[list[str], list[str], dict]:
    """Merge multiple PDFs into one searchable corpus."""
    all_chunks: list[str] = []
    all_sources: list[str] = []
    chunks_indexed: list[dict] = []
    documents: list[dict] = []
    previews: list[str] = []
    total_chars = 0

    for filename, pdf_bytes in files:
        full_text, page_texts, pdf_stats = extract_pdf_with_stats(pdf_bytes)
        chunks = chunk_document(full_text, page_texts=page_texts)
        if not chunks:
            documents.append(
                {
                    "Document": filename,
                    "Pages": pdf_stats["page_count"],
                    "Pages with text": pdf_stats["pages_with_text"],
                    "Sections": 0,
                    "Characters": 0,
                    "Status": "No text found",
                }
            )
            continue

        total_chars += len(full_text)
        previews.append(
            f"### {filename}\n"
            f"Pages: {pdf_stats['page_count']} total · "
            f"{pdf_stats['pages_with_text']} with extractable text · "
            f"{pdf_stats['pages_empty']} empty (likely scanned images)\n\n"
            f"{full_text[:4000]}{'…' if len(full_text) > 4000 else ''}"
        )
        for local_i, chunk in enumerate(chunks):
            global_i = len(all_chunks)
            all_chunks.append(chunk)
            all_sources.append(filename)
            chunks_indexed.append(
                {
                    "Document": filename,
                    "Chunk": f"Chunk {local_i + 1}",
                    "Global #": global_i + 1,
                    "Preview": truncate_preview(chunk, 80),
                }
            )
        documents.append(
            {
                "Document": filename,
                "Pages": pdf_stats["page_count"],
                "Pages with text": pdf_stats["pages_with_text"],
                "Sections": len(chunks),
                "Characters": len(full_text),
                "Status": "Indexed",
            }
        )

    meta = {
        "char_count": total_chars,
        "chunk_count": len(all_chunks),
        "document_count": len([d for d in documents if d.get("Status") == "Indexed"]),
        "documents": documents,
        "preview": "\n\n---\n\n".join(previews),
        "chunks_indexed": chunks_indexed,
    }
    return all_chunks, all_sources, meta


def truncate_preview(text: str, max_len: int = 120) -> str:
    t = " ".join(text.split())
    return t[:max_len] + ("…" if len(t) > max_len else "")


def query_token_overlap(user_query: str, chunk_text: str) -> float:
    """Fraction of meaningful query words that appear in the chunk."""
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]+", user_query.lower())
        if t not in STOPWORDS and len(t) > 2
    ]
    if not tokens:
        return 0.0
    chunk_lower = chunk_text.lower()
    hits = sum(1 for t in tokens if t in chunk_lower)
    return hits / len(tokens)


def assess_document_relevance(user_query: str, top_chunk: dict | None) -> dict:
    """
    Decide if the question is answerable from uploaded PDFs.
    Returns dict with passed (bool) and diagnostic scores for the UI.
    """
    if not top_chunk:
        return {
            "passed": False,
            "reason": "no_chunks",
            "hybrid_sum": 0.0,
            "token_overlap": 0.0,
            "top_tfidf": 0.0,
            "top_annoy": 0.0,
            "top_bm25": 0.0,
        }

    hybrid_sum = float(top_chunk.get("hybrid_sum", 0))
    raw = top_chunk.get("raw", {})
    top_tfidf = float(raw.get("tfidf", 0))
    top_annoy = float(raw.get("annoy", 0))
    top_bm25 = float(raw.get("bm25", 0))
    overlap = query_token_overlap(user_query, top_chunk.get("text", ""))

    # Clear match — allow Mistral
    strong = (
        hybrid_sum >= 2.1
        or overlap >= 0.32
        or (top_tfidf >= 0.22 and top_annoy >= 0.42)
        or (top_bm25 >= 2.0 and overlap >= 0.12)
    )
    if strong:
        return {
            "passed": True,
            "reason": "strong_match",
            "hybrid_sum": hybrid_sum,
            "token_overlap": overlap,
            "top_tfidf": top_tfidf,
            "top_annoy": top_annoy,
            "top_bm25": top_bm25,
        }

    # Clear mismatch — do not call Mistral (e.g. "boat neck blouse" vs database PDF)
    weak = (
        hybrid_sum < MIN_HYBRID_SUM_FOR_ANSWER
        and overlap < MIN_TOKEN_OVERLAP
        and top_tfidf < MIN_TOP_TFIDF
        and top_annoy < MIN_TOP_ANNOY
    )
    if weak:
        return {
            "passed": False,
            "reason": "low_relevance",
            "hybrid_sum": hybrid_sum,
            "token_overlap": overlap,
            "top_tfidf": top_tfidf,
            "top_annoy": top_annoy,
            "top_bm25": top_bm25,
        }

    # Borderline: require modest keyword or TF-IDF signal
    borderline_ok = overlap >= 0.12 or top_tfidf >= 0.16 or hybrid_sum >= 1.85
    return {
        "passed": borderline_ok,
        "reason": "borderline_pass" if borderline_ok else "low_relevance",
        "hybrid_sum": hybrid_sum,
        "token_overlap": overlap,
        "top_tfidf": top_tfidf,
        "top_annoy": top_annoy,
        "top_bm25": top_bm25,
    }


def embed_text_chunks(client: Mistral, chunks: list[str], progress_bar, status) -> np.ndarray:
    all_embeddings = []
    total = len(chunks)
    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = [
            truncate_for_embedding(chunks[i])
            for i in range(batch_start, min(batch_start + EMBED_BATCH_SIZE, total))
        ]
        retries = 0
        delay = EMBED_RETRY_DELAY
        while retries < MAX_EMBED_RETRIES:
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=batch,
                )
                all_embeddings.extend([item.embedding for item in response.data])
                done = min(batch_start + len(batch), total)
                progress_bar.progress(0.7 * (done / total))
                status.caption(f"Generating embeddings… {done}/{total} sections")
                time.sleep(0.3)
                break
            except Exception as e:
                retries += 1
                if retries >= MAX_EMBED_RETRIES:
                    raise RuntimeError(f"Failed to embed document: {e}") from e
                status.caption(f"Rate limited, retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2

    if len(all_embeddings) != total:
        raise RuntimeError("Embedding count does not match text chunks.")
    return np.array(all_embeddings, dtype=np.float32)


def build_annoy_index(embeddings: np.ndarray, num_trees: int = 10) -> AnnoyIndex:
    dim = embeddings.shape[1]
    index = AnnoyIndex(dim, "angular")
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding.tolist())
    index.build(num_trees)
    return index


class Word2VecIndex:
    def __init__(self, texts: list[str], vector_size: int = 100):
        self.vector_size = vector_size
        corpus = [" ".join(text.lower().split()) for text in texts]
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
        doc_term = self.vectorizer.fit_transform(corpus)
        n_terms = doc_term.shape[1]
        n_components = min(
            vector_size,
            max(1, n_terms - 1),
            max(1, doc_term.shape[0] - 1),
        )

        term_doc = doc_term.T.tocsr()
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        term_embeddings = self.svd.fit_transform(term_doc)

        vocab = self.vectorizer.get_feature_names_out()
        self.word_vectors = {
            vocab[i]: self._pad(term_embeddings[i]) for i in range(len(vocab))
        }
        self._zero = np.zeros(vector_size, dtype=np.float32)

    def _pad(self, vector: np.ndarray) -> np.ndarray:
        out = np.zeros(self.vector_size, dtype=np.float32)
        out[: min(len(vector), self.vector_size)] = vector[: self.vector_size]
        return out

    def average_vector(self, tokens: list[str]) -> np.ndarray:
        vectors = [
            self.word_vectors[t.lower()]
            for t in tokens
            if t.lower() in self.word_vectors
        ]
        if not vectors:
            return self._zero.copy()
        return np.mean(vectors, axis=0).astype(np.float32)

    def document_vectors(self, texts: list[str]) -> np.ndarray:
        return np.array([self.average_vector(text.split()) for text in texts])


class MistralRAGChatbot:
    def __init__(
        self,
        client: Mistral,
        embeddings: np.ndarray,
        texts: list[str],
        doc_meta: dict | None = None,
        chunk_sources: list[str] | None = None,
    ):
        self.client = client
        self.embeddings = embeddings
        self.texts = texts
        self.chunk_sources = chunk_sources or ["Document"] * len(texts)
        self.doc_meta = doc_meta or {}
        self.annoy_index = build_annoy_index(embeddings)
        self.tfidf_matrix, self.tfidf_vectorizer = self._build_tfidf(texts)
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.word2vec_index = Word2VecIndex(texts)
        self.doc_w2v_vectors = self.word2vec_index.document_vectors(texts)

    @classmethod
    def from_pdfs(
        cls,
        client: Mistral,
        files: list[tuple[str, bytes]],
        progress_bar,
        status,
    ) -> "MistralRAGChatbot":
        if not files:
            raise ValueError("Upload at least one PDF.")

        status.caption(f"Reading {len(files)} document(s)…")
        chunks, sources, meta = build_corpus_from_pdfs(files)
        if not chunks:
            raise ValueError(
                "No readable text in the uploaded PDFs. Use text-based PDFs "
                "(not scanned images only)."
            )
        status.caption(
            f"{meta['document_count']} file(s), {len(chunks)} sections, "
            f"{meta['char_count']:,} characters. Embedding…"
        )
        progress_bar.progress(0.05)
        embeddings = embed_text_chunks(client, chunks, progress_bar, status)
        status.caption("Building indexes (BM25, TF-IDF, Word2Vec, Annoy)…")
        progress_bar.progress(0.85)
        chatbot = cls(client, embeddings, chunks, meta, sources)
        progress_bar.progress(1.0)
        return chatbot

    @classmethod
    def from_pdf(
        cls,
        client: Mistral,
        pdf_bytes: bytes,
        progress_bar,
        status,
        filename: str = "document.pdf",
    ) -> "MistralRAGChatbot":
        return cls.from_pdfs(client, [(filename, pdf_bytes)], progress_bar, status)

    def _build_tfidf(self, texts: list[str]):
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        return matrix, vectorizer

    def get_text_embedding(self, text: str, model: str = EMBEDDING_MODEL):
        """Sync API — avoids 'Event loop is closed' when calling Mistral repeatedly in Streamlit."""
        response = self.client.embeddings.create(
            model=model,
            inputs=[text],
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _generate_chat_response(self, model: str, prompt: str) -> str:
        """Sync chat API (stable on Streamlit Cloud; no asyncio.run per message)."""
        messages = [{"role": "user", "content": prompt}]
        try:
            result = self.client.chat.complete(model=model, messages=messages)
            content = result.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as e:
            logging.error("Chat complete failed: %s", e)
            raise
        raise RuntimeError(
            "Mistral returned an empty response. Check your API key and try again."
        )

    def score_all_chunks(
        self, user_query: str, query_embedding: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Step 10–12: score every chunk with BM25, TF-IDF, Word2Vec, Annoy."""
        n = len(self.texts)
        tokens = user_query.lower().split()

        bm25_raw = np.clip(self.bm25.get_scores(tokens), 0, None)

        query_tfidf = self.tfidf_vectorizer.transform([user_query])
        tfidf_raw = np.clip(
            cosine_similarity(query_tfidf, self.tfidf_matrix).flatten(), 0, None
        )

        query_w2v = self.word2vec_index.average_vector(tokens).reshape(1, -1)
        w2v_raw = np.clip(
            cosine_similarity(query_w2v, self.doc_w2v_vectors).flatten(), 0, None
        )

        annoy_raw = np.zeros(n, dtype=float)
        indices, distances = self.annoy_index.get_nns_by_vector(
            query_embedding.tolist(), n, include_distances=True
        )
        for idx, dist in zip(indices, distances):
            annoy_raw[int(idx)] = 1.0 / (1.0 + float(dist))

        return {
            "bm25": bm25_raw,
            "tfidf": tfidf_raw,
            "word2vec": w2v_raw,
            "annoy": annoy_raw,
        }

    def hybrid_retrieve(
        self, user_query: str, query_embedding: np.ndarray, top_k: int
    ) -> tuple[list[dict], dict]:
        """
        Steps 10–14: per-method scores → hybrid sum → rerank → top chunks.
        Returns (top_chunks, retrieval_report for UI).
        """
        raw_scores = self.score_all_chunks(user_query, query_embedding)
        norm_scores = {m: normalize_array(raw_scores[m]) for m in HYBRID_METHOD_ORDER}

        n = len(self.texts)
        hybrid_sum = sum(norm_scores[m] for m in HYBRID_METHOD_ORDER)

        # Build per-chunk records
        all_chunks = []
        for i in range(n):
            source_doc = self.chunk_sources[i]
            row = {
                "index": i,
                "chunk_id": i + 1,
                "source_doc": source_doc,
                "label": chunk_label(i, source_doc),
                "text": self.texts[i],
                "raw": {m: float(raw_scores[m][i]) for m in HYBRID_METHOD_ORDER},
                "norm": {m: float(norm_scores[m][i]) for m in HYBRID_METHOD_ORDER},
                "hybrid_sum": float(hybrid_sum[i]),
            }
            all_chunks.append(row)

        # Step 14: rerank by hybrid_sum (descending)
        reranked = sorted(all_chunks, key=lambda c: c["hybrid_sum"], reverse=True)
        for rank, chunk in enumerate(reranked, start=1):
            chunk["rerank_position"] = rank

        top_chunks = []
        for rank, chunk in enumerate(reranked[:top_k], start=1):
            top_chunks.append(
                {
                    "rank": rank,
                    "index": chunk["index"],
                    "chunk_id": chunk["chunk_id"],
                    "source_doc": chunk["source_doc"],
                    "label": chunk["label"],
                    "text": chunk["text"],
                    "hybrid_sum": chunk["hybrid_sum"],
                    "raw": chunk["raw"],
                    "norm": chunk["norm"],
                    "rerank_position": chunk["rerank_position"],
                }
            )

        query_tokens = user_query.lower().split()
        method_winners = {}
        method_tables = {}
        step_titles = {
            "bm25": "Step 10 — BM25 retrieval (keyword search)",
            "tfidf": "Step 11 — TF-IDF retrieval (cosine similarity)",
            "word2vec": "Step 12 — Word2Vec retrieval (semantic word vectors)",
            "annoy": "Step 13a — Annoy retrieval (Mistral embeddings)",
        }
        for method in HYBRID_METHOD_ORDER:
            sorted_by_method = sorted(
                all_chunks, key=lambda c: c["raw"][method], reverse=True
            )
            winner = sorted_by_method[0] if sorted_by_method else None
            method_winners[method] = winner["label"] if winner else "—"
            method_tables[method] = [
                {
                    "Document": c["source_doc"],
                    "Chunk": c["label"],
                    "Score": round(c["raw"][method], 4),
                    "Preview": truncate_preview(c["text"]),
                }
                for c in sorted_by_method
            ]

        hybrid_table = [
            {
                "Rank": c["rerank_position"],
                "Document": c["source_doc"],
                "Chunk": c["label"],
                "BM25": round(c["raw"]["bm25"], 4),
                "TF-IDF": round(c["raw"]["tfidf"], 4),
                "Word2Vec": round(c["raw"]["word2vec"], 4),
                "Annoy": round(c["raw"]["annoy"], 4),
                "Hybrid sum": round(c["hybrid_sum"], 4),
            }
            for c in reranked
        ]

        winner = reranked[0] if reranked else None
        retrieval_report = {
            "query": user_query,
            "query_tokens": query_tokens,
            "total_chunks": n,
            "document_count": self.doc_meta.get("document_count", 1),
            "step_titles": step_titles,
            "method_tables": method_tables,
            "method_winners": method_winners,
            "hybrid_formula": (
                "hybrid_sum = norm(BM25) + norm(TF-IDF) + norm(Word2Vec) + norm(Annoy) "
                "(each normalized 0–1 across all chunks)"
            ),
            "hybrid_table": hybrid_table,
            "winner_label": winner["label"] if winner else "—",
            "winner_hybrid_sum": round(winner["hybrid_sum"], 4) if winner else 0,
            "top_chunks": top_chunks,
        }
        return top_chunks, retrieval_report

    def _style_instruction(self, response_style: str) -> str:
        style_prompts = {
            "Detailed": "Give a detailed, practical answer in 3–6 sentences.",
            "Concise": "Give a very brief answer in 1–2 sentences only.",
            "Creative": "Give a clear, engaging answer in 2–4 sentences.",
            "Technical": "Give a precise technical answer in 3–5 sentences.",
        }
        return style_prompts.get(response_style, style_prompts["Detailed"])

    def _document_section_rules(self, response_style: str) -> str:
        if response_style == "Concise":
            return (
                "Answer in **1–2 short sentences maximum**. State only the direct fact "
                "the user asked for (e.g. a year, grade, school name, or one project title). "
                "**Never** paste full bullet lists, long paragraphs, or entire resume blocks."
            )
        if response_style == "Detailed":
            return (
                "Answer using ONLY the passages. Be specific; you may use a short paragraph "
                "or a few bullets if needed."
            )
        return (
            "Answer using ONLY the passages. "
            + self._style_instruction(response_style)
        )

    def _build_dual_section_prompt(
        self,
        user_query: str,
        response_style: str,
        *,
        in_documents: bool,
        context: str,
    ) -> str:
        """Two-part answer: document status + Mistral general knowledge when needed."""
        style = self._style_instruction(response_style)
        headers = (
            "Use exactly these two markdown headers:\n"
            "### From your uploaded documents\n"
            "### General answer (Mistral AI)\n"
        )
        doc_rules = self._document_section_rules(response_style)
        if in_documents:
            return (
                "You are a helpful assistant. The user's question may be answered from "
                "their uploaded PDF passages below.\n\n"
                f"{headers}\n"
                f"In **From your uploaded documents**: {doc_rules} "
                "If the passages do not contain the answer, say so clearly.\n\n"
                "In **General answer (Mistral AI)**: If the passages fully answered the "
                f"question, write only: *Not needed — fully covered in your documents above.* "
                f"Otherwise give a helpful answer ({style}).\n\n"
                f"--- Passages from PDFs ---\n{context}\n\n"
                f"--- Question ---\n{user_query}"
            )
        return (
            "The user asked a question. Their uploaded PDFs were searched but do NOT "
            "contain relevant information (retrieval match was too weak). "
            "Do not pretend the unrelated excerpts answer the question.\n\n"
            f"{headers}\n"
            "In **From your uploaded documents**: State clearly in 1–3 sentences that "
            "this topic is **not covered** in their uploaded PDF(s).\n\n"
            "In **General answer (Mistral AI)**: Give a correct, helpful, practical "
            f"answer to the question using your general knowledge. {style}\n\n"
            f"--- Question ---\n{user_query}\n\n"
            f"--- Unrelated excerpts (do NOT use as facts) ---\n{context or '(no text indexed)'}"
        )

    def generate_response_with_rag(
        self,
        user_query: str,
        model: str = "mistral-small-latest",
        top_k: int = 5,
        response_style: str = "Detailed",
        add_general_answer: bool = True,
    ):
        retrieval_report = {}
        try:
            query_embedding = self.get_text_embedding(user_query)
            ranked_chunks, retrieval_report = self.hybrid_retrieve(
                user_query, query_embedding, top_k
            )

            source_info = {
                "retrieval_report": retrieval_report,
                "context_chunks": ranked_chunks,
            }

            if not ranked_chunks:
                retrieval_report["relevance_check"] = assess_document_relevance(
                    user_query, None
                )
                if not add_general_answer:
                    return NOT_IN_DOCUMENTS_ONLY_MSG, [], source_info
                prompt = self._build_dual_section_prompt(
                    user_query, response_style, in_documents=False, context=""
                )
                retrieval_report["answer_mode"] = "general_only"
                response = self._generate_chat_response(model, prompt)
                return response, [], source_info

            relevance = assess_document_relevance(user_query, ranked_chunks[0])
            retrieval_report["relevance_check"] = relevance
            in_documents = relevance["passed"]

            context = "\n\n".join(
                f"From [{d.get('source_doc', 'document')}]:\n{d['text']}"
                for d in ranked_chunks[:3]
            )

            if not add_general_answer and not in_documents:
                retrieval_report["answer_blocked"] = True
                retrieval_report["block_reason"] = (
                    "Not in your PDFs. Enable general answer in settings for Mistral help."
                )
                return NOT_IN_DOCUMENTS_ONLY_MSG, [], source_info

            retrieval_report["answer_mode"] = (
                "document_plus_general" if in_documents else "not_in_doc_plus_general"
            )
            full_prompt = self._build_dual_section_prompt(
                user_query,
                response_style,
                in_documents=in_documents,
                context=context,
            )
            response = self._generate_chat_response(model, full_prompt)
            return response, [doc["text"] for doc in ranked_chunks], source_info

        except Exception as e:
            logging.error("Error generating response: %s", e)
            return (
                f"Error generating response: {e}",
                [],
                {"retrieval_report": retrieval_report} if retrieval_report else {},
            )


def render_hybrid_retrieval_sources(report: dict, key_suffix: str = "0"):
    """UI matching the RAG pipeline: Steps 8–15."""
    st.markdown("##### Pipeline for this question")
    st.markdown(
        f"**Step 8 — User query:** `{report['query']}`  \n"
        f"**Step 9 — Query tokens:** `{' | '.join(report['query_tokens']) or '(none)'}`  \n"
        f"**Documents indexed:** {report.get('document_count', 1)} · "
        f"**Total chunks:** {report['total_chunks']}"
    )

    rel = report.get("relevance_check")
    mode = report.get("answer_mode", "")
    if mode == "not_in_doc_plus_general":
        st.info(
            "Topic **not in your PDFs** — response includes a **general Mistral AI** answer."
        )
    elif mode == "document_plus_general":
        st.success("Matched your documents — document section uses PDF passages.")
    elif mode == "general_only":
        st.warning("No indexed text — answer is from Mistral general knowledge only.")

    if rel:
        if rel.get("passed"):
            st.caption(
                f"Relevance: **high** (hybrid={rel['hybrid_sum']:.2f}, "
                f"keywords={rel['token_overlap']:.0%})"
            )
        else:
            st.caption(
                f"Relevance: **low** (hybrid={rel['hybrid_sum']:.2f}, "
                f"keywords={rel['token_overlap']:.0%}) — PDFs likely do not cover this question."
            )
    if report.get("answer_blocked"):
        st.error(report.get("block_reason", ""))

    step_options = {
        report["step_titles"][m]: m for m in HYBRID_METHOD_ORDER
    }
    selected_step = st.selectbox(
        "View retrieval step (Steps 10–13)",
        options=list(step_options.keys()),
        index=0,
        key=f"retrieval_step_{key_suffix}",
    )
    method = step_options[selected_step]
    st.success(f"Winner: **{report['method_winners'][method]}**")
    st.dataframe(
        report["method_tables"][method],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("##### Step 13 — Hybrid scoring")
    st.code(report["hybrid_formula"], language=None)
    st.info(
        f"Top chunk after fusion: **{report['winner_label']}** "
        f"(hybrid sum = **{report['winner_hybrid_sum']}**)"
    )

    st.markdown("##### Step 14 — Reranking (all chunks, highest hybrid sum first)")
    st.dataframe(report["hybrid_table"], use_container_width=True, hide_index=True)

    st.markdown("##### Step 15 — Context sent to Mistral (top passages)")
    for chunk in report["top_chunks"]:
        doc = chunk.get("source_doc", "")
        st.markdown(
            f"**{chunk['label']}** · `{doc}` · hybrid sum `{chunk['hybrid_sum']:.4f}`"
        )
        with st.container(border=True):
            st.text(chunk["text"])
        cols = st.columns(4)
        for col, method in zip(cols, HYBRID_METHOD_ORDER):
            col.metric(
                METHOD_LABELS[method].split("(")[0].strip(),
                f"{chunk['raw'][method]:.3f}",
            )


def render_chat_history():
    for msg_idx, (role, message, source_info) in enumerate(st.session_state.chat_history):
        if role == "user":
            with st.chat_message("user"):
                st.markdown(message)
        else:
            with st.chat_message("assistant"):
                if message and message.strip():
                    st.markdown(message)
                else:
                    st.warning("No answer was generated. Check your API key or try again.")

                report = None
                if isinstance(source_info, dict):
                    report = source_info.get("retrieval_report")

                if report:
                    n_chunks = len(report.get("top_chunks", []))
                    rel = report.get("relevance_check") or {}
                    rel_label = "matched PDFs" if rel.get("passed") else "not in PDFs"
                    mode = report.get("answer_mode", "")
                    if mode == "not_in_doc_plus_general":
                        rel_label = "not in PDFs + general AI"
                    expander_title = (
                        f"Retrieval process (hybrid pipeline) — "
                        f"{n_chunks} chunk(s) · {rel_label}"
                    )
                    with st.expander(expander_title, expanded=False):
                        render_hybrid_retrieval_sources(report, key_suffix=str(msg_idx))


def main():
    st.set_page_config(
        page_title="Document Chat",
        page_icon=":speech_balloon:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .message-user {
            background-color: #0057D9;
            color: white;
            padding: 12px 16px;
            border-radius: 12px;
            margin: 8px 5px 8px 18%;
            text-align: left;
            line-height: 1.5;
        }
        .message-assistant {
            background-color: #1a3d34;
            color: white;
            padding: 12px 16px;
            border-radius: 12px;
            margin: 8px 18% 8px 5px;
            text-align: left;
            line-height: 1.5;
        }
        .fixed-header {
            position: sticky;
            top: 0;
            background: linear-gradient(135deg, #1e2a4a 0%, #2d3a6b 100%);
            padding: 14px 16px;
            z-index: 100;
            border-radius: 12px;
            text-align: center;
            color: white;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="fixed-header"><h2 style="margin:0;">Chat with your documents</h2>'
        '<p style="font-size:0.85rem;margin:4px 0 0;opacity:0.9;">'
        "Upload one or more PDFs · hybrid search across all files"
        "</p></div>",
        unsafe_allow_html=True,
    )

    api_key = get_mistral_api_key()
    client = get_mistral_client(api_key)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []
    if "files_id" not in st.session_state:
        st.session_state.files_id = None
    if "doc_preview" not in st.session_state:
        st.session_state.doc_preview = None
    if "doc_stats" not in st.session_state:
        st.session_state.doc_stats = None

    model = "mistral-small-latest"
    top_k = 5
    response_style = "Detailed"
    add_general_answer = True

    with st.sidebar:
        st.header("Your documents")
        pdf_files = st.file_uploader(
            "Upload PDFs (one or more)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select multiple PDFs to search across all of them in one query.",
        )

        if pdf_files:
            files_id = tuple(sorted((f.name, f.size) for f in pdf_files))
            if st.session_state.get("files_id") != files_id:
                try:
                    file_data = [(f.name, f.getvalue()) for f in pdf_files]
                    with st.status(
                        f"Indexing {len(file_data)} document(s)…", expanded=True
                    ) as idx_status:
                        progress = st.progress(0.0)
                        status_line = st.empty()
                        st.session_state.chatbot = MistralRAGChatbot.from_pdfs(
                            client,
                            file_data,
                            progress,
                            status_line,
                        )
                        idx_status.update(label="Indexing complete", state="complete")
                    meta = st.session_state.chatbot.doc_meta
                    st.session_state.files_id = files_id
                    st.session_state.uploaded_names = [f.name for f in pdf_files]
                    st.session_state.doc_preview = meta.get("preview", "")
                    st.session_state.doc_stats = {
                        "chunks": meta.get("chunk_count", 0),
                        "chars": meta.get("char_count", 0),
                        "documents": meta.get("document_count", 0),
                    }
                    st.session_state.chunks_indexed = meta.get("chunks_indexed", [])
                    st.session_state.documents_table = meta.get("documents", [])
                    st.session_state.chat_history = []
                except Exception as e:
                    st.session_state.chatbot = None
                    st.error(str(e))

            if st.session_state.chatbot is not None:
                n_docs = len(st.session_state.uploaded_names)
                st.success(f"Ready — **{n_docs}** document(s) indexed")
                for name in st.session_state.uploaded_names:
                    st.caption(f"📄 {name}")
                meta = st.session_state.chatbot.doc_meta
                empty_pages = sum(
                    d.get("Pages", 0) - d.get("Pages with text", 0)
                    for d in meta.get("documents", [])
                    if d.get("Status") == "Indexed"
                )
                if empty_pages > 0:
                    st.warning(
                        f"**{empty_pages} page(s)** had no selectable text (common with "
                        "scanned PDFs). Only text-based pages are indexed. "
                        "Re-export with real text or use OCR."
                    )

        stats = st.session_state.doc_stats
        if stats:
            st.caption(
                f"**{stats.get('documents', 1)}** files · "
                f"**{stats['chunks']}** sections · "
                f"**{stats['chars']:,}** characters"
            )

        if st.session_state.get("documents_table"):
            with st.expander("Indexed files", expanded=False):
                st.dataframe(
                    st.session_state.documents_table,
                    use_container_width=True,
                    hide_index=True,
                )

        if st.session_state.get("chunks_indexed"):
            with st.expander("Steps 1–2 — All chunks (all documents)", expanded=False):
                st.caption("**Step 1** — Text extracted from each PDF (preview)")
                st.markdown(st.session_state.doc_preview or "_No preview_")
                st.caption("**Step 2** — Combined chunk index")
                st.dataframe(
                    st.session_state.chunks_indexed,
                    use_container_width=True,
                    hide_index=True,
                )
        elif st.session_state.doc_preview:
            with st.expander("Preview extracted text", expanded=False):
                st.text_area(
                    "Extracted content",
                    st.session_state.doc_preview,
                    height=220,
                    disabled=True,
                    label_visibility="collapsed",
                )

        with st.expander("Advanced settings", expanded=False):
            model = st.selectbox(
                "Model",
                ["mistral-small-latest", "mistral-large-latest"],
                key="setting_model",
            )
            top_k = st.slider(
                "Passages used per answer", 3, 12, 5, key="setting_top_k"
            )
            response_style = st.selectbox(
                "Answer style",
                ["Detailed", "Concise", "Creative", "Technical"],
                key="setting_response_style",
            )
            add_general_answer = st.checkbox(
                "General answer when not in documents (recommended)",
                value=True,
                key="setting_general_answer",
                help="If your PDFs do not cover the question: say so clearly, "
                "then Mistral gives a correct general answer (e.g. sewing, cooking).",
            )

    if st.session_state.chatbot is None:
        st.info(
            "**Upload one or more PDFs** in the sidebar, wait for indexing, then ask a question. "
            "The app searches **all uploaded files** together (e.g. *Compare skills in my resume and cover letter*)."
        )

    render_chat_history()

    user_message = st.chat_input("Ask about your documents…")
    if user_message:
        if st.session_state.chatbot is None:
            st.warning("Please upload at least one PDF first.")
            st.stop()

        st.session_state.chat_history.append(("user", user_message, None))

        try:
            with st.spinner("Running hybrid retrieval and generating answer…"):
                response, _, source_info = (
                    st.session_state.chatbot.generate_response_with_rag(
                        user_message,
                        model=model,
                        top_k=top_k,
                        response_style=response_style,
                        add_general_answer=add_general_answer,
                    )
                )
        except Exception as e:
            logging.exception("Query failed")
            st.session_state.chat_history.append(
                ("assistant", f"Something went wrong: {e}", {})
            )
            st.rerun()

        if isinstance(response, str) and response.startswith("Error generating"):
            st.session_state.chat_history.append(("assistant", response, source_info or {}))
        elif not (response and str(response).strip()):
            st.session_state.chat_history.append(
                (
                    "assistant",
                    "The model returned an empty answer. Please try again.",
                    source_info or {},
                )
            )
        else:
            st.session_state.chat_history.append(("assistant", response, source_info or {}))

        st.rerun()


if __name__ == "__main__":
    main()
