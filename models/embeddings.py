import math
import re
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple

from pypdf import PdfReader


@dataclass
class RetrievedChunk:
	text: str
	score: float


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
	clean = " ".join(text.split())
	if not clean:
		return []

	chunks = []
	start = 0
	while start < len(clean):
		end = min(len(clean), start + chunk_size)
		chunks.append(clean[start:end])
		if end == len(clean):
			break
		start = max(0, end - overlap)
	return chunks


def _extract_pdf_text(file_like) -> str:
	reader = PdfReader(file_like)
	pages = [(page.extract_text() or "") for page in reader.pages]
	return "\n".join(pages)


def _tokenize(text: str) -> List[str]:
	return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _build_idf(all_tokens: List[List[str]]) -> Dict[str, float]:
	doc_count = len(all_tokens)
	df_counter = Counter()
	for tokens in all_tokens:
		df_counter.update(set(tokens))

	idf = {}
	for term, doc_freq in df_counter.items():
		idf[term] = math.log((doc_count + 1) / (doc_freq + 1)) + 1.0
	return idf


def _to_tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
	if not tokens:
		return {}
	tf = Counter(tokens)
	length = float(len(tokens))
	vec = {}
	for term, freq in tf.items():
		vec[term] = (freq / length) * idf.get(term, 0.0)
	return vec


def _cosine_sparse(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
	if not vec_a or not vec_b:
		return 0.0
	common = set(vec_a.keys()) & set(vec_b.keys())
	numerator = sum(vec_a[t] * vec_b[t] for t in common)
	norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
	norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
	if norm_a == 0 or norm_b == 0:
		return 0.0
	return numerator / (norm_a * norm_b)


def build_vector_store(uploaded_files) -> Dict:
	all_chunks: List[str] = []
	for file in uploaded_files:
		text = _extract_pdf_text(file)
		all_chunks.extend(_chunk_text(text))

	if not all_chunks:
		return {"chunks": [], "idf": {}, "chunk_vectors": []}

	all_tokens = [_tokenize(chunk) for chunk in all_chunks]
	idf = _build_idf(all_tokens)
	chunk_vectors = [_to_tfidf_vector(tokens, idf) for tokens in all_tokens]
	return {"chunks": all_chunks, "idf": idf, "chunk_vectors": chunk_vectors}


def retrieve_chunks(vector_store: Dict, query: str, top_k: int = 4) -> List[RetrievedChunk]:
	if not vector_store or not vector_store.get("chunks"):
		return []

	idf = vector_store.get("idf", {})
	chunk_vectors = vector_store.get("chunk_vectors", [])
	query_vector = _to_tfidf_vector(_tokenize(query), idf)

	results: List[RetrievedChunk] = []
	scored = []
	for idx, vec in enumerate(chunk_vectors):
		scored.append((idx, _cosine_sparse(query_vector, vec)))
	scored.sort(key=lambda item: item[1], reverse=True)

	for idx, score in scored[:top_k]:
		score = float(score)
		if score <= 0:
			continue
		results.append(RetrievedChunk(text=vector_store["chunks"][idx], score=score))
	return results


def format_context(chunks: List[RetrievedChunk]) -> str:
	if not chunks:
		return ""
	parts = []
	for i, ch in enumerate(chunks, start=1):
		parts.append(f"[{i}] score={ch.score:.3f}\n{ch.text}")
	return "\n\n".join(parts)
