# Retriever
A retrieval demo built on LlamaIndex, with optional reranking support.

## Basic Usage

```python
import pandas as pd
from core.retriever import Retriever

r = Retriever(
    csv_path="data/data.csv",
    model_name="Qwen/Qwen3-Embedding-0.6B",
    retrieve_k=64,
    reranker_name="BAAI/bge-reranker-v2-m3",  # optional
    top_k=8,
    rebuild_index=False,
)

r.load()

query = "what causes hiccups?"
nodes = r.search(query)

pd.DataFrame(
    [{"id": n.metadata["id"], "text": n.text, "score": n.score} for n in nodes]
).head(3)
