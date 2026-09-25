# Scientific RAG on QASPER

This project builds a retrieval-augmented generation (RAG) pipeline over the QASPER scientific paper dataset. It chunks papers, embeds them with a sentence-transformers model, stores them in a persistent vector database, retrieves relevant passages for a question, and evaluates evidence coverage on QASPER QA pairs.

The current workflow is:

```text
QASPER papers -> chunking -> embeddings -> ChromaDB -> retrieval -> generation/evaluation
```

## Project goals

- Load QASPER train/validation/test splits from the published parquet files
- Extract paper metadata and text sections for retrieval
- Build chunk-level embeddings for semantic search
- Store and query them efficiently with ChromaDB
- Evaluate dense retrieval against answer evidence
- Generate grounded answers from retrieved evidence using a local model

## Repository structure

```text
.
├── checkpoint1.py          # starter retrieval demo
├── checkpoint2.py          # dense RAG baseline with evaluation
├── data/                   # local data and vector store artifacts
├── reports/                # evaluation outputs written by checkpoint2.py
├── src/
│   ├── chunking.py         # chunking and paper preprocessing
│   ├── embeddings.py       # embedding model wrapper
│   ├── evaluation.py       # evidence-hit-rate and MRR evaluation
│   ├── generation.py       # local answer generation and extractive fallback
│   ├── load_data.py        # QASPER dataset loading
│   ├── retrieval.py        # dense cosine similarity retrieval helper
│   └── vector_store.py     # persistent ChromaDB vector store
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project depends on:

- `datasets`
- `numpy`
- `sentence-transformers`
- `chromadb`
- `transformers` (used by the local generation pipeline)

## Dataset loading

The project uses the converted parquet files from the fixed QASPER revision instead of the older Python loading script that is incompatible with some `datasets` versions.

You can verify the dataset loads correctly with:

```bash
python src/load_data.py
```

This prints the dataset split sizes and a sample paper.

## Checkpoint 1: semantic retrieval demo

The first checkpoint demonstrates chunking + embedding + cosine retrieval without the vector database layer.

Run it with the defaults:

```bash
python checkpoint1.py
```

Customize the number of papers, query, or top-k result count:

```bash
python checkpoint1.py --papers 50 --query "What datasets were used?" --top-k 10
```

Outputs are stored in:

```text
data/checkpoint1_index
```

This contains the embedding matrix and chunk metadata for the selected papers.

## Checkpoint 2: dense RAG baseline

The second checkpoint builds a persistent ChromaDB collection, retrieves the most relevant passages, and optionally generates a grounded answer.

Typical run:

```bash
python checkpoint2.py --papers 25 --evaluation-papers 10 --questions 20 --top-k 5
```

Useful options:

```bash
python checkpoint2.py \
  --papers 50 \
  --evaluation-papers 12 \
  --questions 30 \
  --top-k 5 \
  --query "How is the annotation experiment evaluated?" \
  --rebuild
```

Optional extractive fallback:

```bash
python checkpoint2.py --extractive --query "What is the main contribution?"
```

### What checkpoint 2 produces

- A persistent ChromaDB database at `data/checkpoint2_chroma`
- Retrieval metrics written to `reports/checkpoint2_results.json`
- A sample generated answer with citations for the requested query
- Evidence-hit-rate and MRR scores over validation papers

## Evaluation metrics

The evaluation in `src/evaluation.py` measures:

- evidence hit rate: whether a retrieved chunk contains matching evidence from the QASPER answer annotations
- MRR (mean reciprocal rank): the average reciprocal rank of the first evidence-matching result
- mean query latency in milliseconds

These metrics are reported in the JSON result file for the chosen validation set.

## Notes

- The QASPER dataset is public; the `HF_TOKEN` warning is usually informational and not required for access.
- If a vector database already exists and the embedding model matches, `checkpoint2.py` will reuse it unless `--rebuild` is passed.
- The local generation model is implemented in `src/generation.py` and can be swapped by passing `--generator-model`.

## Example result workflow

```bash
python checkpoint2.py --papers 10 --evaluation-papers 8 --questions 20 --top-k 5
```

This is a lightweight local baseline for experimenting with dense retrieval and answer generation on scientific papers.
