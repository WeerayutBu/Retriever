# Retriever
A retrieval demo built on LlamaIndex, with optional reranking support.

## Basic Usage

#### Build Docker
```bash
docker build -t retriever .

# Using Image Env
docker run --rm -it --gpus all -p 8000:8000 \
    retriever uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Dev
docker run --rm -it --gpus all -p 8000:8000 retriever bash 
```

#### Startup API
```bash
python -m uvicorn api.main:app --reload

# Test inference
curl --get --data-urlencode "q=what causes hiccups?" http://127.0.0.1:8000/retrieve
```

#### Retrieval code
```python
import pandas as pd
from core.retriever import Retriever

r = Retriever(
    csv_path="data/demo.csv",
    model_name="Qwen/Qwen3-Embedding-0.6B",
    retrieve_k=64,
    reranker_name="BAAI/bge-reranker-v2-m3",  # optional
    top_k=8,
    rebuild_index=True,
)

r.load()

query = "what causes hiccups?"
nodes = r.search(query)

pd.DataFrame(
    [{"id": n.metadata["id"], "text": n.text, "score": n.score} for n in nodes]
).head(3)
```