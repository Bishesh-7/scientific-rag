import re
from time import perf_counter

import numpy as np

from src.chunking import chunk_paper
from src.retrieval import BM25Retriever, hybrid_retrieve, retrieve


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


def _first_evidence_rank(results, evidence):
    for result_index, result in enumerate(results, start=1):
        if any(_matches_evidence(result["text"], item) for item in evidence):
            return result_index
    return None


def evaluate_retrievers(
    papers, model, top_k=5, max_questions=100, candidate_pool=20,
    dense_weight=0.5, reranker=None,
):
    """Compare BM25, dense, hybrid, and optional cross-encoder retrieval."""
    method_names = ["bm25", "dense", "hybrid"]
    if reranker is not None:
        method_names.append("hybrid_reranked")
    stats = {
        name: {"hits": 0, "reciprocal_ranks": [], "latencies": []}
        for name in method_names
    }
    evaluated = 0
    papers_considered = 0
    papers_evaluated = set()
    question_results = []

    for paper in papers:
        papers_considered += 1
        chunks = chunk_paper(paper)
        if not chunks:
            continue
        embeddings = model.encode(
            [item["embedding_text"] for item in chunks], show_progress=False
        )
        bm25_index = BM25Retriever(chunks)
        qas = paper["qas"]
        for index, question in enumerate(qas["question"]):
            evidence = _evidence_for_question(qas["answers"][index])
            if not evidence:
                continue

            started = perf_counter()
            bm25_results = bm25_index.retrieve(question, top_k)
            stats["bm25"]["latencies"].append((perf_counter() - started) * 1000)

            started = perf_counter()
            dense_results = retrieve(question, chunks, embeddings, model, top_k)
            stats["dense"]["latencies"].append((perf_counter() - started) * 1000)

            started = perf_counter()
            hybrid_results = hybrid_retrieve(
                question, chunks, embeddings, model, bm25_index, top_k,
                candidate_pool, dense_weight,
            )
            stats["hybrid"]["latencies"].append((perf_counter() - started) * 1000)

            method_results = {
                "bm25": bm25_results,
                "dense": dense_results,
                "hybrid": hybrid_results,
            }
            if reranker is not None:
                started = perf_counter()
                candidates = hybrid_retrieve(
                    question, chunks, embeddings, model, bm25_index,
                    candidate_pool, candidate_pool, dense_weight,
                )
                reranked = reranker.rerank(question, candidates, top_k)
                stats["hybrid_reranked"]["latencies"].append(
                    (perf_counter() - started) * 1000
                )
                method_results["hybrid_reranked"] = reranked

            for name, results in method_results.items():
                rank = _first_evidence_rank(results, evidence)
                stats[name]["hits"] += rank is not None
                stats[name]["reciprocal_ranks"].append(
                    0.0 if rank is None else 1.0 / rank
                )

            question_results.append(
                {
                    "paper_id": paper["id"],
                    "question": question,
                    "first_evidence_rank": {
                        name: _first_evidence_rank(results, evidence)
                        for name, results in method_results.items()
                    },
                }
            )

            evaluated += 1
            papers_evaluated.add(paper["id"])
            if evaluated >= max_questions:
                break
        if evaluated >= max_questions:
            break

    if not evaluated:
        raise ValueError("No answerable questions with annotated evidence were found")
    results = {}
    for name, values in stats.items():
        latencies = np.asarray(values["latencies"], dtype=np.float64)
        reciprocal_ranks = np.asarray(values["reciprocal_ranks"], dtype=np.float64)
        top_one_hits = int(np.count_nonzero(reciprocal_ranks == 1.0))
        top_three_hits = int(np.count_nonzero(reciprocal_ranks >= (1.0 / 3.0)))
        results[name] = {
            "hits_at_k": int(values["hits"]),
            "misses_at_k": evaluated - int(values["hits"]),
            "evidence_hit_rate_at_k": values["hits"] / evaluated,
            "top_1_accuracy": top_one_hits / evaluated,
            "top_3_accuracy": top_three_hits / evaluated,
            "mrr": float(np.mean(reciprocal_ranks)),
            "mean_query_latency_ms": float(np.mean(latencies)),
            "median_query_latency_ms": float(np.median(latencies)),
            "p95_query_latency_ms": float(np.percentile(latencies, 95)),
        }
    return {
        "questions": evaluated,
        "papers_selected": len(papers) if hasattr(papers, "__len__") else None,
        "papers_considered": papers_considered,
        "papers_with_evaluated_questions": len(papers_evaluated),
        "top_k": top_k,
        "candidate_pool": candidate_pool,
        "dense_weight": dense_weight,
        "methods": results,
        "question_results": question_results,
    }
