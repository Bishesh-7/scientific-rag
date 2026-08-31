def split_text(text, max_words=180, overlap=30):
    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("max_words must be greater than overlap")

    words = text.split()
    step = max_words - overlap
    pieces = []

    for start in range(0, len(words), step):
        part = words[start : start + max_words]
        if part:
            pieces.append(" ".join(part))
        if start + max_words >= len(words):
            break

    return pieces


def chunk_paper(paper, max_words=180, overlap=30):
    paper_id = paper["id"]
    title = paper["title"]
    sections = []

    abstract = paper.get("abstract", "").strip()
    if abstract:
        sections.append(("Abstract", abstract))

    full_text = paper.get("full_text") or {}
    names = full_text.get("section_name") or []
    paragraphs = full_text.get("paragraphs") or []

    for name, section_paragraphs in zip(names, paragraphs):
        text = " ".join(p.strip() for p in section_paragraphs if p and p.strip())
        if text:
            sections.append((name or "Untitled section", text))

    chunks = []
    for section, text in sections:
        for content in split_text(text, max_words, overlap):
            chunk_id = f"{paper_id}:{len(chunks)}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "title": title,
                    "section": section,
                    "text": content,
                    "embedding_text": f"{title}\n{section}\n{content}",
                }
            )

    return chunks


def chunk_papers(papers, max_words=180, overlap=30):
    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunk_paper(paper, max_words, overlap))
    return all_chunks
