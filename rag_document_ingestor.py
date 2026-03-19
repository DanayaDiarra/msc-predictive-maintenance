"""
RAG Document Ingestor — Upload-to-Corpus Pipeline
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | March 2026

Handles runtime ingestion of uploaded documents (PDFs, TXT, HTML, CSV)
into the RAG knowledge base without restarting the pipeline.

SUPPORTED INPUT TYPES:
  .pdf   → text extraction via pdfminer.six (if installed) or PyPDF2 fallback
  .txt   → direct read
  .html  → BeautifulSoup text extraction (if installed) or regex fallback
  .csv   → pandas + structured text serialisation
  .json  → structured text serialisation (alarm dicts, CMDB exports)
  .md    → direct read

CHUNKING STRATEGY:
  Recursive character splitting — mirrors LangChain's RecursiveCharacterTextSplitter:
    chunk_size=512 tokens (~400 chars), overlap=64 tokens (~50 chars)
  Each chunk tagged with full provenance metadata.

INTEGRATION:
  1. User uploads files via Streamlit sidebar
  2. Ingestor extracts text and chunks each file
  3. Chunks appended to corpus.json
  4. RAGIndex rebuilt incrementally (sparse + dense matrices updated)
  5. New chunks immediately searchable

USAGE:
  from rag_document_ingestor import DocumentIngestor
  ingestor = DocumentIngestor(corpus_dir="data/rag_corpus",
                               index_dir="data/rag_index")
  n = ingestor.ingest_bytes(file_bytes, filename="my_sop.pdf",
                             doc_type="sop", subsystem="power_subsystem")
"""

import os, json, hashlib, re, io
from dataclasses import dataclass, asdict
from typing import Optional, List
import pickle
import numpy as np

CORPUS_DIR = "data/rag_corpus"
INDEX_DIR  = "data/rag_index"

# Default chunking parameters (mirrors 512-token chunks)
CHUNK_CHARS    = 1800
CHUNK_OVERLAP  = 200

# Auto-detect subsystem from filename / content
SUBSYSTEM_HINTS = {
    "power":     "power_subsystem",
    "rectifier": "power_subsystem",
    "battery":   "power_subsystem",
    "bbu":       "power_subsystem",
    "thermal":   "thermal_management",
    "cooling":   "thermal_management",
    "fan":       "thermal_management",
    "hvac":      "thermal_management",
    "rf":        "rf_antenna",
    "antenna":   "rf_antenna",
    "vswr":      "rf_antenna",
    "feeder":    "rf_antenna",
    "backhaul":  "backhaul_connectivity",
    "fibre":     "backhaul_connectivity",
    "microwave": "backhaul_connectivity",
    "latency":   "backhaul_connectivity",
    "baseband":  "baseband_processing",
    "cpu":       "baseband_processing",
    "alarm":     "power_subsystem",  # most alarms relate to power first
}

DOC_TYPE_MAP = {
    "sop":        "sop",
    "manual":     "manual",
    "alarm":      "alarm_dict",
    "ticket":     "ticket",
    "spec":       "spec",
    "fmea":       "fmea",
    "procedure":  "sop",
    "guide":      "manual",
}


@dataclass
class IngestedChunk:
    chunk_id:         str
    doc_id:           str
    doc_type:         str
    equipment_family: str
    subsystem:        str
    alarm_category:   Optional[str]
    software_release: Optional[str]
    title:            str
    text:             str
    keywords:         List[str]
    source_file:      str   # original filename
    user_uploaded:    bool  # flag for provenance


# ── Text extractors ───────────────────────────────────────────────────────

def _extract_pdf(data: bytes) -> str:
    """Extract text from PDF bytes. Tries pdfminer, then PyPDF2, then raw."""
    # Try pdfminer.six
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        out = io.StringIO()
        extract_text_to_fp(io.BytesIO(data), out, laparams=LAParams())
        text = out.getvalue()
        if len(text.strip()) > 50:
            return text
    except ImportError:
        pass

    # Try PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        if len(text.strip()) > 50:
            return text
    except ImportError:
        pass

    # Raw decode fallback — works for text-based PDFs
    try:
        raw = data.decode("latin-1", errors="ignore")
        # Extract visible text between stream markers
        texts = re.findall(r"\((.*?)\)", raw)
        return " ".join(t for t in texts if len(t) > 3 and t.isascii())
    except Exception:
        return ""


def _extract_html(data: bytes) -> str:
    """Extract readable text from HTML."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(data, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    except ImportError:
        pass
    # Regex fallback
    text = data.decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)


def _extract_csv(data: bytes) -> str:
    """Serialise CSV rows into readable text."""
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(data))
        lines = []
        for _, row in df.iterrows():
            parts = [f"{col}: {val}" for col, val in row.items()
                     if str(val).strip() not in ("", "nan", "NaN")]
            lines.append(" | ".join(parts))
        return "\n".join(lines)
    except Exception:
        return data.decode("utf-8", errors="ignore")


def extract_text(data: bytes, filename: str) -> str:
    """Route extraction by file extension."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return _extract_pdf(data)
    elif ext in (".html", ".htm"):
        return _extract_html(data)
    elif ext == ".csv":
        return _extract_csv(data)
    elif ext == ".json":
        try:
            obj = json.loads(data)
            return json.dumps(obj, indent=2)
        except Exception:
            return data.decode("utf-8", errors="ignore")
    else:
        # .txt, .md, raw text
        return data.decode("utf-8", errors="ignore")


# ── Chunker ───────────────────────────────────────────────────────────────

def _recursive_split(text: str, chunk_size=CHUNK_CHARS,
                     overlap=CHUNK_OVERLAP) -> List[str]:
    """
    Recursive character splitter — prefers splitting at paragraph boundaries,
    then sentence boundaries, then character boundaries.
    Mirrors LangChain RecursiveCharacterTextSplitter behaviour.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    def split(s, seps):
        if not s.strip():
            return
        if len(s) <= chunk_size:
            chunks.append(s.strip())
            return
        sep = next((x for x in seps if x in s), "")
        if not sep:
            # Hard split
            for i in range(0, len(s), chunk_size - overlap):
                chunks.append(s[i:i + chunk_size].strip())
            return
        parts = s.split(sep)
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    split(current, seps[seps.index(sep)+1:] if sep in seps else [])
                current = part
        if current:
            split(current, seps[seps.index(sep)+1:] if sep in seps else [])

    split(text, separators)
    return [c for c in chunks if len(c) > 40]


# ── Metadata inference ────────────────────────────────────────────────────

def _infer_subsystem(filename: str, text: str) -> str:
    combined = (filename + " " + text[:500]).lower()
    for hint, sub in SUBSYSTEM_HINTS.items():
        if hint in combined:
            return sub
    return "general"


def _infer_doc_type(filename: str, user_doc_type: str) -> str:
    if user_doc_type and user_doc_type.lower() in DOC_TYPE_MAP:
        return DOC_TYPE_MAP[user_doc_type.lower()]
    name_lower = filename.lower()
    for hint, dt in DOC_TYPE_MAP.items():
        if hint in name_lower:
            return dt
    return "manual"


def _make_id(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _extract_title(filename: str, text: str) -> str:
    """Use first non-empty line as title, fallback to filename."""
    for line in text.split("\n"):
        line = line.strip()
        if 8 < len(line) < 120:
            return line[:100]
    return os.path.splitext(filename)[0].replace("_", " ").title()


# ── Main ingestor class ───────────────────────────────────────────────────

class DocumentIngestor:
    """
    Handles ingestion of uploaded documents into the RAG knowledge base.
    
    Workflow:
      1. Extract text from file bytes
      2. Chunk into 512-token segments
      3. Infer metadata (subsystem, doc_type, title)
      4. Append to corpus.json
      5. Rebuild RAG index incrementally
    """

    def __init__(self, corpus_dir=CORPUS_DIR, index_dir=INDEX_DIR):
        self.corpus_dir = corpus_dir
        self.index_dir  = index_dir
        os.makedirs(corpus_dir, exist_ok=True)
        os.makedirs(index_dir,  exist_ok=True)

    def _load_corpus(self) -> list:
        path = os.path.join(self.corpus_dir, "corpus.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return []

    def _save_corpus(self, chunks: list):
        path = os.path.join(self.corpus_dir, "corpus.json")
        with open(path, "w") as f:
            json.dump(chunks, f, indent=2)

    def _rebuild_index(self):
        """Rebuild RAGIndex from the current corpus.json."""
        try:
            from rag_pipeline import RAGIndex
            idx = RAGIndex()
            idx.load_corpus(os.path.join(self.corpus_dir, "corpus.json"))
            idx.build()
            idx.save(self.index_dir)
            return True
        except Exception as e:
            print(f"  [Ingestor] Index rebuild failed: {e}")
            return False

    def ingest_bytes(self,
                     file_bytes:  bytes,
                     filename:    str,
                     doc_type:    str = "auto",
                     subsystem:   str = "auto",
                     equipment_family: str = "bts_outdoor") -> int:
        """
        Ingest a single document from bytes.
        
        Args:
          file_bytes:  raw file content
          filename:    original filename (used for extension detection + metadata)
          doc_type:    "sop" | "manual" | "alarm" | "ticket" | "spec" | "fmea" | "auto"
          subsystem:   subsystem tag or "auto" to infer from content
          equipment_family: equipment family tag
        
        Returns:
          Number of chunks added to corpus
        """
        print(f"  [Ingestor] Processing: {filename} ({len(file_bytes)//1024}KB)")

        # Extract text
        text = extract_text(file_bytes, filename)
        if not text or len(text.strip()) < 100:
            print(f"  [Ingestor] WARNING: extracted text too short (<100 chars). Skipping.")
            return 0

        # Infer metadata
        resolved_subsystem = (subsystem if subsystem != "auto"
                              else _infer_subsystem(filename, text))
        resolved_doc_type  = _infer_doc_type(filename, doc_type)
        title              = _extract_title(filename, text)
        doc_id             = f"USR-{_make_id(filename + text[:100])[:8].upper()}"

        # Chunk
        raw_chunks = _recursive_split(text)
        print(f"  [Ingestor] {len(raw_chunks)} chunks from {filename}")

        # Build chunk objects
        new_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            cid = _make_id(doc_id + str(i) + chunk_text[:30])
            chunk_title = f"{title} (Part {i+1})" if len(raw_chunks) > 1 else title
            kw  = [resolved_subsystem, resolved_doc_type, equipment_family, "user_uploaded"]
            new_chunks.append(asdict(IngestedChunk(
                chunk_id=cid, doc_id=doc_id, doc_type=resolved_doc_type,
                equipment_family=equipment_family, subsystem=resolved_subsystem,
                alarm_category=None, software_release=None,
                title=chunk_title, text=chunk_text, keywords=kw,
                source_file=filename, user_uploaded=True,
            )))

        # Append to corpus
        existing = self._load_corpus()
        # Deduplicate: remove any previously uploaded version of same file
        existing = [c for c in existing if c.get("source_file") != filename]
        updated  = existing + new_chunks
        self._save_corpus(updated)
        print(f"  [Ingestor] Corpus: {len(existing)} → {len(updated)} chunks")

        # Rebuild index
        ok = self._rebuild_index()
        if ok:
            print(f"  [Ingestor] Index rebuilt successfully.")
        return len(new_chunks)

    def list_user_documents(self) -> list:
        """Return list of user-uploaded documents in the corpus."""
        corpus = self._load_corpus()
        seen, docs = set(), []
        for c in corpus:
            if c.get("user_uploaded") and c.get("source_file") not in seen:
                seen.add(c["source_file"])
                docs.append({
                    "filename":  c["source_file"],
                    "doc_id":    c["doc_id"],
                    "doc_type":  c["doc_type"],
                    "subsystem": c["subsystem"],
                    "n_chunks":  sum(1 for x in corpus if x.get("source_file") == c["source_file"])
                })
        return docs

    def remove_document(self, filename: str) -> int:
        """Remove all chunks from a user-uploaded document."""
        corpus  = self._load_corpus()
        before  = len(corpus)
        corpus  = [c for c in corpus if c.get("source_file") != filename]
        removed = before - len(corpus)
        if removed:
            self._save_corpus(corpus)
            self._rebuild_index()
            print(f"  [Ingestor] Removed {removed} chunks from '{filename}'.")
        return removed


# ── Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingestor = DocumentIngestor()
    sample_text = b"""
SOP: Emergency Generator Startup - Telecom BTS

Trigger: PWR-004 (mains failure) active and generator present at site.

Step 1: Verify mains failure via OMC remote telemetry.
Step 2: Issue remote generator start command via SCADA.
Step 3: Monitor generator output voltage (nominal 230V AC, +/-5%).
Step 4: Confirm BTS transfers to generator within 30 seconds.
Step 5: Check generator fuel level - alert if below 50% capacity.
Step 6: Log event in CMDB with mains failure timestamp.
Step 7: Contact grid operator with estimated repair timeline.

Expected generator autonomy: 8 hours at full load, 20 hours at 50%.
"""
    n = ingestor.ingest_bytes(sample_text, "generator_sop.txt",
                               doc_type="sop", subsystem="power_subsystem")
    print(f"Added {n} chunks.")
    print("User documents:", ingestor.list_user_documents())
