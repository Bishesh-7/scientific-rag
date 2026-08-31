# Scientific RAG - Checkpoint 1

This is the first checkpoint of my scientific document question answering
project. At this stage I have focused on loading the QASPER papers, processing
their text and building a basic semantic search system.

The current pipeline is:

```text
QASPER papers -> text chunks -> embeddings -> cosine similarity -> top results
```

## What I have completed

- Loaded the QASPER train, validation and test sets
- Extracted the abstract and full-text sections from each paper
- Split the sections into chunks of 180 words with a 30-word overlap
- Kept the paper ID, title and section name for each chunk
- Generated embeddings with `all-MiniLM-L6-v2`
- Saved the embeddings and chunk information locally
- Retrieved the most similar chunks using cosine similarity

## Running the project

Create and activate a virtual environment, then install the packages:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Check that the dataset loads:

```bash
python src/load_data.py
```

Run the checkpoint demonstration:

```bash
python checkpoint1.py
```

It uses 25 papers by default so it does not take too long. A different number
of papers or question can also be used:

```bash
python checkpoint1.py --papers 100 --query "What datasets were used?"
```

The output is saved in `data/checkpoint1_index`. It contains the embeddings,
the chunks and a small metadata file describing the index.

## Problem I had with QASPER

The original `load_dataset("allenai/qasper", "qasper")` command failed because
my version of the `datasets` library does not support the old QASPER Python
loading script. I fixed this by loading the converted parquet files from a fixed
Hugging Face revision.

The `HF_TOKEN` message is only a warning because the dataset and model are
public. `hf auth login` can be used for higher download limits, but it is not
required.

## What I will demonstrate

For Checkpoint 1 I will show the dataset split sizes, a sample paper, the number
of chunks, the embedding shape and the top passages found for a question. BM25,
hybrid retrieval, answer generation and multiple LLMs will be added later.
