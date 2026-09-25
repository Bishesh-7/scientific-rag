import re
from time import perf_counter

import numpy as np

from src.chunking import chunk_paper
from src.retrieval import retrieve


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _evidence_for_question(answers):
    evidence = []
    for annotation in answers.get("answer", []):
        if annotation.get("unanswerable"):
            continue
        evidence.extend(annotation.get("highlighted_evidence") or annotation.get("evidence") or [])
    return [item for item in evidence if item and item.strip()]


def _matches_evidence(chunk_text, evidence_text, threshold=0.5):
    evidence_terms = _tokens(evidence_text)
    if not evidence_terms:
        return False
    return len(_tokens(chunk_text) & evidence_terms) / len(evidence_terms) >= threshold


def evaluate_dense_retrieval(papers, model, top_k=5, max_questions=30):
    """Measure evidence hit rate and reciprocal rank on answerable QASPER QA pairs."""
    papers_selected = len(papers) if hasattr(papers, "__len__") else None
    evaluated = hits = 0
    reciprocal_ranks = []
    latencies = []
    papers_considered = 0
    papers_evaluated = set()

    for paper in papers:
        papers_considered += 1
        chunks = chunk_paper(paper)
        if not chunks:
            continue
        embeddings = model.encode([item["embedding_text"] for item in chunks], show_progress=False)
        qas = paper["qas"]
        for index, question in enumerate(qas["question"]):
            evidence = _evidence_for_question(qas["answers"][index])
            if not evidence:
                continue
            started = perf_counter()
            results = retrieve(question, chunks, embeddings, model, top_k)
            latencies.append((perf_counter() - started) * 1000)
            rank = None
            for result_index, result in enumerate(results, start=1):
                if any(_matches_evidence(result["text"], item) for item in evidence):
                    rank = result_index
                    break
            hits += rank is not None
            reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
            evaluated += 1
            papers_evaluated.add(paper["id"])
            if evaluated >= max_questions:
                break
        if evaluated >= max_questions:
            break

    if not evaluated:
        raise ValueError("No answerable questions with annotated evidence were found")
    return {
        "questions": evaluated,
        "papers_selected": papers_selected,
        "papers_considered": papers_considered,
        "papers_with_evaluated_questions": len(papers_evaluated),
        "top_k": top_k,
        "evidence_hit_rate": hits / evaluated,
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_query_latency_ms": float(np.mean(latencies)),
    }
