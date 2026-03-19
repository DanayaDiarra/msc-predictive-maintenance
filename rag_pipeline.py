"""
RAG Pipeline — Layer 2: Knowledge Grounding
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | March 2026

ARCHITECTURE:
  Hybrid retrieval: sparse (TF-IDF BM25-equivalent) + dense (SVD latent
  semantic embedding) fused via Reciprocal Rank Fusion (RRF).

  In production replace:
    TF-IDF sparse  →  BM25 (rank-bm25) or Elasticsearch
    SVD dense      →  sentence-transformers all-MiniLM-L6-v2 + FAISS HNSW
    Cross-encoder reranker  →  cross-encoder/ms-marco-MiniLM-L-6-v2
  Architecture and interfaces are IDENTICAL — only the embedding backend changes.

PIPELINE FLOW:
  AlertJSON (from Interpreter Agent)
    ↓
  Query construction  (primary semantic + BM25 keyword queries)
    ↓
  Parallel retrieval  (TF-IDF sparse + SVD dense)
    ↓
  Reciprocal Rank Fusion  (RRF k=60)
    ↓
  Metadata filter  (subsystem + doc_type alignment)
    ↓
  Cross-encoder rerank  (cosine similarity as proxy)
    ↓
  Top-K evidence bundle  (chunks + provenance metadata + citations)
    ↓
  EvidenceBundle  (structured input for Diagnostic Agent)
"""

import os, json, pickle, time, re, math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_DIR   = "data/rag_corpus"
INDEX_DIR    = "data/rag_index"
RESULTS_DIR  = "results/rag"
os.makedirs(INDEX_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_K_RETRIEVE = 15    # candidates per retrieval arm
TOP_K_FINAL    = 5     # chunks in final evidence bundle
RRF_K          = 60    # RRF constant (standard: 60)
SVD_COMPONENTS = 64    # latent semantic dimensions


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:         str
    doc_id:           str
    doc_type:         str
    equipment_family: str
    subsystem:        str
    alarm_category:   Optional[str]
    title:            str
    text:             str
    keywords:         List[str]
    rrf_score:        float
    sparse_rank:      int
    dense_rank:       int
    citation_ref:     str     # short citation label e.g. [MAN-PWR-001]

@dataclass
class EvidenceBundle:
    """Structured output of the RAG pipeline → input for Diagnostic Agent."""
    alert_id:          str
    station_id:        str
    urgency:           str
    query_used:        str
    chunks:            List[dict]      # list of RetrievedChunk dicts
    top_titles:        List[str]       # human-readable chunk titles
    citation_map:      dict            # {citation_ref: chunk_id}
    retrieval_latency_ms: float
    n_candidates:      int
    coverage_score:    float           # fraction of alert subsystems covered


# ── Index Builder ──────────────────────────────────────────────────────────

class RAGIndex:
    """
    Builds and persists the dual retrieval index:
      - sparse_vectorizer: TfidfVectorizer (proxy for BM25)
      - dense_svd:         TruncatedSVD on TF-IDF matrix (proxy for sentence embeddings)
    Both operate on the full text (title + text + keywords) of each chunk.
    """

    def __init__(self):
        self.chunks          = []
        self.corpus_texts    = []
        self.sparse_matrix   = None   # TF-IDF sparse matrix
        self.dense_matrix    = None   # SVD-reduced dense matrix
        self.sparse_vec      = None   # fitted TfidfVectorizer
        self.svd             = None   # fitted TruncatedSVD

    def load_corpus(self, corpus_path: str):
        with open(corpus_path) as f:
            self.chunks = json.load(f)
        self.corpus_texts = [
            f"{c['title']} {c['text']} {' '.join(c['keywords'])}"
            for c in self.chunks
        ]
        print(f"  [RAGIndex] Loaded {len(self.chunks)} chunks")

    def build(self):
        print("  [RAGIndex] Building sparse (TF-IDF) index...")
        self.sparse_vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=8000,
            sublinear_tf=True, min_df=1)
        self.sparse_matrix = self.sparse_vec.fit_transform(self.corpus_texts)

        print("  [RAGIndex] Building dense (SVD) index...")
        n_comp = min(SVD_COMPONENTS, self.sparse_matrix.shape[1] - 1,
                     self.sparse_matrix.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        dense_raw = self.svd.fit_transform(self.sparse_matrix)
        self.dense_matrix = normalize(dense_raw, norm="l2")
        print(f"  [RAGIndex] Dense matrix: {self.dense_matrix.shape}")

    def save(self, index_dir: str):
        with open(f"{index_dir}/chunks.json", "w") as f:
            json.dump(self.chunks, f, indent=2)
        with open(f"{index_dir}/sparse_vec.pkl", "wb") as f:
            pickle.dump(self.sparse_vec, f)
        with open(f"{index_dir}/svd.pkl", "wb") as f:
            pickle.dump(self.svd, f)
        np.save(f"{index_dir}/sparse_matrix.npy",
                self.sparse_matrix.toarray().astype(np.float32))
        np.save(f"{index_dir}/dense_matrix.npy",
                self.dense_matrix.astype(np.float32))
        print(f"  [RAGIndex] Saved to {index_dir}/")

    def load(self, index_dir: str):
        with open(f"{index_dir}/chunks.json") as f:
            self.chunks = json.load(f)
        with open(f"{index_dir}/sparse_vec.pkl", "rb") as f:
            self.sparse_vec = pickle.load(f)
        with open(f"{index_dir}/svd.pkl", "rb") as f:
            self.svd = pickle.load(f)
        sparse_arr = np.load(f"{index_dir}/sparse_matrix.npy")
        from scipy.sparse import csr_matrix
        self.sparse_matrix = csr_matrix(sparse_arr)
        self.dense_matrix  = np.load(f"{index_dir}/dense_matrix.npy")
        self.corpus_texts  = [
            f"{c['title']} {c['text']} {' '.join(c['keywords'])}"
            for c in self.chunks
        ]
        print(f"  [RAGIndex] Loaded {len(self.chunks)} chunks from {index_dir}/")


# ── RAG Pipeline ──────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Hybrid retrieval pipeline with RRF fusion and metadata reranking.

    Steps:
      1. sparse_retrieve  — TF-IDF cosine (BM25 proxy)
      2. dense_retrieve   — SVD latent cosine (sentence embedding proxy)
      3. rrf_fuse         — Reciprocal Rank Fusion
      4. metadata_boost   — subsystem + doc_type alignment boost
      5. final_rerank     — sort by combined score, take top-K
      6. build_bundle     — assemble EvidenceBundle with citations
    """

    def __init__(self, index: RAGIndex):
        self.index = index

    # ── Retrieval arms ────────────────────────────────────────────────────

    def _encode_sparse(self, query: str):
        return self.index.sparse_vec.transform([query])

    def _encode_dense(self, query: str):
        sparse_q = self._encode_sparse(query)
        dense_q  = self.index.svd.transform(sparse_q)
        return normalize(dense_q, norm="l2")

    def _sparse_retrieve(self, query: str, top_k: int) -> List[tuple]:
        """Returns [(idx, score), ...] ranked by TF-IDF cosine similarity."""
        q_vec  = self._encode_sparse(query)
        scores = cosine_similarity(q_vec, self.index.sparse_matrix)[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked]

    def _dense_retrieve(self, query: str, top_k: int) -> List[tuple]:
        """Returns [(idx, score), ...] ranked by SVD latent cosine similarity."""
        q_vec  = self._encode_dense(query)
        scores = cosine_similarity(q_vec, self.index.dense_matrix)[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked]

    # ── RRF Fusion ───────────────────────────────────────────────────────

    def _rrf_fuse(self, sparse_results: List[tuple],
                  dense_results: List[tuple]) -> dict:
        """
        Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank(d))
        Combines sparse and dense rank lists into unified scores.
        k=60 is the standard value from Cormack et al. (2009).
        """
        rrf_scores = {}
        # Build rank lookups
        sparse_ranks = {idx: rank+1 for rank,(idx,_) in enumerate(sparse_results)}
        dense_ranks  = {idx: rank+1 for rank,(idx,_) in enumerate(dense_results)}
        all_ids = set(sparse_ranks) | set(dense_ranks)
        for idx in all_ids:
            s_rank = sparse_ranks.get(idx, TOP_K_RETRIEVE + 1)
            d_rank = dense_ranks.get(idx,  TOP_K_RETRIEVE + 1)
            rrf_scores[idx] = (1.0/(RRF_K + s_rank) +
                               1.0/(RRF_K + d_rank))
        return rrf_scores   # {idx: rrf_score}

    # ── Metadata Boost ────────────────────────────────────────────────────

    def _metadata_boost(self, rrf_scores: dict,
                        target_subsystem: str,
                        urgency: str,
                        preferred_doc_types: List[str]) -> dict:
        """
        Boosts chunks that:
          +0.015  matching subsystem
          +0.010  preferred doc_type (SOP/alarm_dict for Critical; manual for Warning)
          +0.008  matching urgency keyword in title/text
        """
        boosted = {}
        for idx, score in rrf_scores.items():
            chunk = self.index.chunks[idx]
            boost = 0.0
            if chunk["subsystem"] == target_subsystem:
                boost += 0.015
            if chunk["doc_type"] in preferred_doc_types:
                boost += 0.010
            text_lower = (chunk["title"] + chunk["text"]).lower()
            if urgency.lower() in text_lower or "critical" in text_lower:
                boost += 0.005
            boosted[idx] = score + boost
        return boosted

    # ── Evidence Bundle Assembly ──────────────────────────────────────────

    def _build_citation(self, chunk: dict) -> str:
        return f"[{chunk['doc_id']}]"

    def _coverage_score(self, chunks: List[dict], alert_subsystem: str) -> float:
        """Fraction of top-K chunks that cover the alerted subsystem."""
        matching = sum(1 for c in chunks if c["subsystem"] == alert_subsystem)
        return round(matching / max(1, len(chunks)), 3)

    # ── Main retrieve method ──────────────────────────────────────────────

    def retrieve(self, alert: dict) -> EvidenceBundle:
        """
        Main entry point. Accepts AlertJSON dict.
        Returns EvidenceBundle for the Diagnostic Agent.
        """
        t0 = time.time()

        query_primary  = alert["rag_query_primary"]
        query_equip    = alert["rag_query_equipment"]
        subsystem      = alert["primary_subsystem"]
        urgency        = alert["urgency"]
        station_id     = alert["station_id"]
        alert_id       = alert["alert_id"]

        # Preferred doc types by urgency
        if urgency == "Critical":
            preferred = ["sop", "alarm_dict", "tree"]
        elif urgency == "Warning":
            preferred = ["sop", "manual", "ticket"]
        else:
            preferred = ["manual", "fmea", "spec"]

        # ── Dual retrieval ──
        sparse_r1 = self._sparse_retrieve(query_primary, TOP_K_RETRIEVE)
        dense_r1  = self._dense_retrieve(query_primary,  TOP_K_RETRIEVE)
        sparse_r2 = self._sparse_retrieve(query_equip,   TOP_K_RETRIEVE)
        dense_r2  = self._dense_retrieve(query_equip,    TOP_K_RETRIEVE)

        # Build rank lists per arm (combine primary + equipment queries)
        # Take best score for each doc across both queries
        def merge_results(r1, r2):
            scores = {}
            for idx, s in r1 + r2:
                scores[idx] = max(scores.get(idx, 0), s)
            return sorted(scores.items(), key=lambda x: -x[1])[:TOP_K_RETRIEVE]

        sparse_merged = merge_results(sparse_r1, sparse_r2)
        dense_merged  = merge_results(dense_r1,  dense_r2)

        # ── RRF fusion ──
        rrf_scores = self._rrf_fuse(sparse_merged, dense_merged)

        # ── Metadata boost ──
        boosted = self._metadata_boost(rrf_scores, subsystem, urgency, preferred)

        # ── Build sparse/dense rank maps for RetrievedChunk ──
        sparse_rank_map = {idx: r+1 for r,(idx,_) in enumerate(sparse_merged)}
        dense_rank_map  = {idx: r+1 for r,(idx,_) in enumerate(dense_merged)}

        # ── Sort and take top-K ──
        ranked = sorted(boosted.items(), key=lambda x: -x[1])[:TOP_K_FINAL]

        # ── Assemble chunks ──
        result_chunks = []
        citation_map  = {}
        for idx, score in ranked:
            c = self.index.chunks[idx]
            cit = self._build_citation(c)
            citation_map[cit] = c["chunk_id"]
            result_chunks.append(RetrievedChunk(
                chunk_id         = c["chunk_id"],
                doc_id           = c["doc_id"],
                doc_type         = c["doc_type"],
                equipment_family = c["equipment_family"],
                subsystem        = c["subsystem"],
                alarm_category   = c.get("alarm_category"),
                title            = c["title"],
                text             = c["text"],
                keywords         = c["keywords"],
                rrf_score        = round(score, 6),
                sparse_rank      = sparse_rank_map.get(idx, 99),
                dense_rank       = dense_rank_map.get(idx, 99),
                citation_ref     = cit,
            ))

        latency_ms = (time.time() - t0) * 1000

        bundle = EvidenceBundle(
            alert_id    = alert_id,
            station_id  = station_id,
            urgency     = urgency,
            query_used  = query_primary,
            chunks      = [asdict(rc) for rc in result_chunks],
            top_titles  = [rc.title for rc in result_chunks],
            citation_map = citation_map,
            retrieval_latency_ms = round(latency_ms, 2),
            n_candidates = len(rrf_scores),
            coverage_score = self._coverage_score(
                [asdict(rc) for rc in result_chunks], subsystem),
        )
        return bundle

    def retrieve_batch(self, alerts: List[dict]) -> List[EvidenceBundle]:
        return [self.retrieve(a) for a in alerts]


# ── Evaluation helpers ────────────────────────────────────────────────────

def eval_retrieval(bundles, expected_subsystems):
    """
    Simple hit-rate and coverage evaluation.
    expected_subsystems: list of target subsystem per alert (ground truth).
    """
    print("\n  ── Retrieval Evaluation ──")
    for bundle, expected in zip(bundles, expected_subsystems):
        subsystems_found = {c["subsystem"] for c in bundle.chunks}
        hit = expected in subsystems_found
        print(f"  {bundle.station_id:<15} urgency={bundle.urgency:<10} "
              f"coverage={bundle.coverage_score:.2f}  "
              f"hit={'✓' if hit else '✗'}  "
              f"latency={bundle.retrieval_latency_ms:.1f}ms  "
              f"candidates={bundle.n_candidates}")


# ── Demo run ──────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 68)
    print("RAG PIPELINE — BUILD INDEX + RETRIEVE")
    print("=" * 68)

    corpus_path = os.path.join(CORPUS_DIR, "corpus.json")

    # Build corpus if not present
    if not os.path.exists(corpus_path):
        print("  Corpus not found. Run rag_corpus_builder.py first.")
        import subprocess, sys
        result = subprocess.run([sys.executable, "rag_corpus_builder.py"],
                                capture_output=True, text=True)
        print(result.stdout)

    # Build or load index
    index_path = os.path.join(INDEX_DIR, "chunks.json")
    index = RAGIndex()
    if os.path.exists(index_path):
        index.load(INDEX_DIR)
    else:
        index.load_corpus(corpus_path)
        index.build()
        index.save(INDEX_DIR)

    pipeline = RAGPipeline(index)

    # Simulate 3 alerts from Interpreter Agent demo output
    test_alerts = [
        {
            "alert_id":           "ALERT_FD002_47_demo",
            "station_id":         "FD002_47",
            "urgency":            "Critical",
            "primary_subsystem":  "power_subsystem",
            "fault_hypothesis":   "Power unit degradation — voltage instability or rectifier wear",
            "rag_query_primary":  "Troubleshooting procedure for power subsystem fault in telecom base station. Urgency: Critical. Fault type: Power unit degradation — voltage instability or rectifier wear",
            "rag_query_equipment":"base station power subsystem maintenance alarm SOP",
            "rag_query_keywords": ["power_subsystem","critical","base_station"],
        },
        {
            "alert_id":           "ALERT_FD001_23_demo",
            "station_id":         "FD001_23",
            "urgency":            "Warning",
            "primary_subsystem":  "thermal_management",
            "fault_hypothesis":   "Thermal runaway risk — cooling fan wear or blocked ventilation",
            "rag_query_primary":  "Troubleshooting procedure for thermal management fault. Urgency: Warning. Cooling fan degradation.",
            "rag_query_equipment":"base station thermal cooling fan maintenance SOP",
            "rag_query_keywords": ["thermal_management","warning","fan","cooling"],
        },
        {
            "alert_id":           "ALERT_FD004_112_demo",
            "station_id":         "FD004_112",
            "urgency":            "Monitor",
            "primary_subsystem":  "backhaul_connectivity",
            "fault_hypothesis":   "Backhaul link degradation — fibre splice loss or microwave alignment",
            "rag_query_primary":  "Backhaul connectivity degradation monitoring procedure. Latency increase, packet loss investigation.",
            "rag_query_equipment":"backhaul fibre microwave maintenance monitoring",
            "rag_query_keywords": ["backhaul_connectivity","monitor","latency","fibre"],
        },
    ]

    print("\n  Retrieving evidence bundles for 3 test alerts...")
    bundles = pipeline.retrieve_batch(test_alerts)

    # ── Print results ──
    for bundle in bundles:
        print(f"\n{'─'*62}")
        print(f"  Station:   {bundle.station_id}  |  Urgency: {bundle.urgency}")
        print(f"  Latency:   {bundle.retrieval_latency_ms:.1f}ms  |  "
              f"Candidates: {bundle.n_candidates}  |  "
              f"Coverage: {bundle.coverage_score:.2f}")
        print(f"  Evidence bundle ({len(bundle.chunks)} chunks):")
        for i, chunk in enumerate(bundle.chunks, 1):
            print(f"    {chunk['citation_ref']:<22} "
                  f"[{chunk['doc_type']:<12}] "
                  f"rrf={chunk['rrf_score']:.5f}  "
                  f"s_rank={chunk['sparse_rank']}  d_rank={chunk['dense_rank']}")
            print(f"      → {chunk['title'][:65]}")
        print(f"  Citation map: {bundle.citation_map}")

    # ── Evaluation ──
    expected_subs = ["power_subsystem","thermal_management","backhaul_connectivity"]
    eval_retrieval(bundles, expected_subs)

    # ── Save bundles ──
    out_path = os.path.join(RESULTS_DIR, "evidence_bundles_demo.json")
    with open(out_path, "w") as f:
        json.dump([asdict(b) for b in bundles], f, indent=2)
    print(f"\n  Saved → {out_path}")
    print("=" * 68)
    print("RAG PIPELINE DEMO COMPLETE")
    print("=" * 68)
    return bundles, pipeline

if __name__ == "__main__":
    run_demo()
