import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


GENERATOR_MODEL = "google/flan-t5-small"
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with",
}
EVALUATION_WORDS = {
    "accuracy", "agreement", "assess", "evaluate", "evaluation", "f1", "kappa",
    "measure", "metric", "performance", "precision", "recall", "result", "score",
    "scores", "test",
}


def _sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _terms(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _focused_evidence(question, passages, max_items=3):
    question_terms = _terms(question) - STOP_WORDS
    asks_evaluation = any(term.startswith(("evaluat", "assess", "measur")) for term in question_terms)
    candidates = []
    for passage_rank, passage in enumerate(passages):
        for sentence_rank, sentence in enumerate(_sentences(passage["text"])):
            terms = _terms(sentence)
            overlap = len(question_terms & terms) / max(1, len(question_terms))
            evaluation_bonus = 0.12 * len(terms & EVALUATION_WORDS) if asks_evaluation else 0.0
            score = overlap + evaluation_bonus + 0.05 * float(passage.get("score", 0))
            candidates.append((score, passage_rank, sentence_rank, sentence))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))

    chosen = []
    seen = set()
    for score, passage_rank, _, sentence in candidates:
        normalized = " ".join(sentence.lower().split())
        if normalized in seen:
            continue
        chosen.append({"score": score, "passage_rank": passage_rank, "text": sentence})
        seen.add(normalized)
        if len(chosen) == max_items:
            break
    return chosen


class LocalRAGGenerator:
    """Generate a concise answer from retrieved evidence using local FLAN-T5."""

    def __init__(self, model_name=GENERATOR_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question, passages, max_new_tokens=80):
        if not passages:
            return {
                "answer": "I could not find enough evidence to answer this question.",
                "citations": [], "model": self.model_name, "evidence": [],
            }
        evidence = _focused_evidence(question, passages)
        source_numbers = {}
        citations = []
        context_lines = []
        for item in evidence:
            passage_rank = item["passage_rank"]
            if passage_rank not in source_numbers:
                number = len(source_numbers) + 1
                source_numbers[passage_rank] = number
                passage = passages[passage_rank]
                citations.append(
                    {
                        "number": number, "paper_id": passage["paper_id"],
                        "title": passage["title"], "section": passage["section"],
                        "chunk_id": passage["chunk_id"],
                    }
                )
            context_lines.append(f"[{source_numbers[passage_rank]}] {item['text']}")

        prompt = (
            "Answer the question in one concise sentence using only the evidence. "
            "Do not discuss the annotation procedure unless it answers the question.\n"
            f"Question: {question}\nEvidence:\n" + "\n".join(context_lines) + "\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=4, early_stopping=True,
        )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if not answer:
            answer = evidence[0]["text"]
        cited = " ".join(f"[{item['number']}]" for item in citations)
        if not re.search(r"\[\d+\]", answer):
            answer = f"{answer} {cited}".strip()
        return {
            "answer": answer, "citations": citations, "model": self.model_name,
            "evidence": [item["text"] for item in evidence],
        }


def generate_grounded_answer(question, passages, max_sentences=1):
    """Focused extractive fallback used for tests or constrained environments."""
    if not passages:
        return {"answer": "I could not find enough evidence to answer this question.", "citations": []}
    evidence = _focused_evidence(question, passages, max_sentences)
    citations = []
    answer_parts = []
    for item in evidence:
        passage = passages[item["passage_rank"]]
        number = len(citations) + 1
        citations.append(
            {
                "number": number, "paper_id": passage["paper_id"],
                "title": passage["title"], "section": passage["section"],
                "chunk_id": passage["chunk_id"],
            }
        )
        answer_parts.append(f"{item['text']} [{number}]")
    return {"answer": " ".join(answer_parts), "citations": citations}
