"""Curriculum ingestion, vectorisation and retrieval with a lexical fallback."""
from __future__ import annotations
import json, math, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any
from src.config import CHROMA_DIR, CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, CURRICULUM_DIR, EMBED_MODEL, RETRIEVAL_TIMEOUT_MS, llm_available
from src.logic import _STOPWORDS
COLLECTION="sprints_curriculum"

def load_curriculum_documents(docs_dir: Path | str=CURRICULUM_DIR) -> list[dict[str,str]]:
    """Load Markdown and text curriculum files."""
    folder=Path(docs_dir)
    if not folder.is_dir(): raise FileNotFoundError(f"Curriculum folder not found: {folder}")
    docs=[{"source":p.name,"text":p.read_text(encoding="utf-8")} for p in sorted(folder.iterdir()) if p.suffix.lower() in {".md",".txt"}]
    if not docs: raise FileNotFoundError(f"No curriculum documents in {folder}")
    return docs

class _WhitespaceTokenizer:
    """Lossless fallback tokenizer."""
    def encode(self,text:str)->list[str]: return re.findall(r"\S+\s*",text)
    def decode(self,tokens:list[str])->str: return "".join(tokens)

def get_tokenizer():
    """Return tiktoken tokenizer or whitespace fallback."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return _WhitespaceTokenizer()

def chunk_text(text:str,chunk_tokens:int=CHUNK_TOKENS,overlap_tokens:int=CHUNK_OVERLAP_TOKENS,tokenizer:Any=None)->list[str]:
    """Split text into overlapping token chunks."""
    if overlap_tokens>=chunk_tokens: raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    tok=tokenizer or get_tokenizer(); tokens=tok.encode(text); step=chunk_tokens-overlap_tokens; chunks=[]
    for start in range(0,max(len(tokens),1),step):
        piece=tokens[start:start+chunk_tokens]
        if not piece: break
        chunks.append(tok.decode(piece).strip())
        if start+chunk_tokens>=len(tokens): break
    return [c for c in chunks if c]

def build_chunks(docs:list[dict[str,str]],tokenizer:Any=None)->list[dict[str,Any]]:
    """Chunk documents while retaining source metadata."""
    out=[]
    for doc in docs:
        for i,piece in enumerate(chunk_text(doc["text"],tokenizer=tokenizer)):
            out.append({"source":doc["source"],"chunk_id":i,"text":piece})
    return out

def _terms(text:str)->list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-]+",text.lower()) if w not in _STOPWORDS]

class LexicalIndex:
    """Small in-memory BM25 index used when vector search is unavailable."""
    def __init__(self,chunks:list[dict[str,Any]],k1:float=1.5,b:float=0.75)->None:
        self.chunks,self.k1,self.b=chunks,k1,b
        self.tf=[Counter(_terms(c["text"])) for c in chunks]
        self.len=[sum(t.values()) for t in self.tf]
        self.avg=(sum(self.len)/len(self.len)) if self.len else 1.0
        df:Counter=Counter()
        for t in self.tf: df.update(t.keys())
        n=len(chunks); self.idf={w:math.log(1+(n-d+0.5)/(d+0.5)) for w,d in df.items()}
    def search(self,query:str,k:int=3)->list[dict[str,Any]]:
        """Return top-k chunks ranked by BM25."""
        q=_terms(query); scored=[]
        for i,tf in enumerate(self.tf):
            s=0.0
            for w in q:
                if w in tf:
                    f=tf[w]; s+=self.idf.get(w,0)*f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*self.len[i]/self.avg))
            if s>0: scored.append((s,i))
        scored.sort(reverse=True)
        return [{**self.chunks[i],"score":round(s,4)} for s,i in scored[:k]]

def build_chroma_vectorstore(docs_dir:Path|str=CURRICULUM_DIR,persist_dir:Path|str=CHROMA_DIR,force_rebuild:bool=False):
    """Create or reuse the persistent ChromaDB vector store."""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import TokenTextSplitter
    persist=Path(persist_dir); embeddings=OpenAIEmbeddings(model=EMBED_MODEL)
    if persist.exists() and any(persist.iterdir()) and not force_rebuild:
        return Chroma(collection_name=COLLECTION,persist_directory=str(persist),embedding_function=embeddings)
    raw=[Document(page_content=d["text"],metadata={"source":d["source"]}) for d in load_curriculum_documents(docs_dir)]
    splitter=TokenTextSplitter(encoding_name="cl100k_base",chunk_size=CHUNK_TOKENS,chunk_overlap=CHUNK_OVERLAP_TOKENS)
    pieces=splitter.split_documents(raw)
    if force_rebuild and persist.exists():
        import shutil
        shutil.rmtree(persist)
    return Chroma.from_documents(pieces,embeddings,collection_name=COLLECTION,persist_directory=str(persist))

class CurriculumRetriever:
    """Retrieval facade with a latency budget and automatic lexical fallback."""
    def __init__(self,docs_dir:Path|str=CURRICULUM_DIR,persist_dir:Path|str=CHROMA_DIR,use_chroma:bool|None=None,timeout_ms:int=RETRIEVAL_TIMEOUT_MS)->None:
        self.timeout_s=timeout_ms/1000.0
        self.chunks=build_chunks(load_curriculum_documents(docs_dir))
        self.lexical=LexicalIndex(self.chunks); self.store=None; self._pool=ThreadPoolExecutor(max_workers=2)
        want=llm_available() if use_chroma is None else use_chroma
        if want:
            try: self.store=build_chroma_vectorstore(docs_dir,persist_dir)
            except Exception: self.store=None
    @property
    def backend(self)->str: return "chroma" if self.store is not None else "lexical"
    def _chroma_search(self,query:str,k:int)->list[dict[str,Any]]:
        results=self.store.similarity_search_with_score(query,k=k)
        return [{"source":d.metadata.get("source","?"),"text":d.page_content,"score":round(1.0/(1.0+float(dist)),4)} for d,dist in results]
    def search(self,query:str,k:int=3)->dict[str,Any]:
        """Search curriculum and return hits, timing, backend and fallback metadata."""
        start=time.perf_counter(); hits=[]; backend=self.backend; fallback=False
        if self.store is not None:
            try: hits=self._pool.submit(self._chroma_search,query,k).result(timeout=self.timeout_s)
            except (FutureTimeout,Exception): fallback,backend=True,"lexical"
        if self.store is None or fallback: hits=self.lexical.search(query,k)
        latency=(time.perf_counter()-start)*1000
        return {"hits":hits,"latency_ms":round(latency,2),"backend":backend,"fallback_used":fallback,"within_budget":latency<=self.timeout_s*1000+50}

def benchmark_latency(retriever:CurriculumRetriever,queries:list[str],k:int=3)->dict[str,float]:
    """Report mean, p95 and max retrieval latency."""
    times=sorted(retriever.search(q,k)["latency_ms"] for q in queries)
    p95=times[min(len(times)-1,int(math.ceil(.95*len(times)))-1)]
    return {"queries":len(times),"mean_ms":round(sum(times)/len(times),2),"p95_ms":p95,"max_ms":times[-1]}

def get_retriever_tool(retriever:CurriculumRetriever):
    """Create the LangChain curriculum search tool."""
    from langchain_core.tools import tool
    @tool
    def search_curriculum(query:str)->str:
        """Search Sprints curriculum passages and return JSON with source module."""
        return json.dumps(retriever.search(query),ensure_ascii=False)
    return search_curriculum
