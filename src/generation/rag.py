import logging
import os
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.vector_database.milvus_vector_db import MilvusVectorDB
from src.embeddings.embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


@dataclass
class RAGResult:
    query: str
    response: str
    sources_used: List[Dict[str, Any]]
    retrieval_count: int
    generation_tokens: Optional[int] = None

    def get_citation_summary(self) -> str:
        if not self.sources_used:
            return "No sources cited"
        source_summary = []
        for source in self.sources_used:
            source_info = f"• {source.get('source_file', 'Unknown')} ({source.get('source_type', 'unknown')})"
            if source.get('page_number'):
                source_info += f" - Page {source['page_number']}"
            source_summary.append(source_info)
        return "\n".join(source_summary)


class RAGGenerator:
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_db: MilvusVectorDB,
        openai_api_key: str = None,
        model_name: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL
        logger.info(f"RAG Generator initialized with {self.model_name} via Ollama")

    def _call_llm(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]

    def generate_response(
        self,
        query: str,
        max_chunks: int = 8,
        max_context_chars: int = 4000,
        top_k: int = 10,
    ) -> RAGResult:
        if not query.strip():
            return RAGResult(query=query, response="Please provide a valid question.",
                             sources_used=[], retrieval_count=0)
        try:
            logger.info(f"Generating response for: '{query[:50]}...'")
            query_vector = self.embedding_generator.generate_query_embedding(query)
            search_results = self.vector_db.search(query_vector=query_vector.tolist(), limit=top_k)

            if not search_results:
                return RAGResult(
                    query=query,
                    response="I couldn't find any relevant information in the available documents.",
                    sources_used=[], retrieval_count=0
                )

            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, max_context_chars)
            prompt = self._create_rag_prompt(query, context)
            response = self._call_llm(prompt)

            rag_result = RAGResult(query=query, response=response,
                                   sources_used=sources_info, retrieval_count=len(search_results))
            logger.info(f"Response generated using {len(sources_info)} sources")
            return rag_result

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return RAGResult(query=query,
                             response=f"I encountered an error: {str(e)}",
                             sources_used=[], retrieval_count=0)

    def _format_context_with_citations(
        self,
        search_results: List[Dict[str, Any]],
        max_chunks: int,
        max_context_chars: int
    ) -> Tuple[str, List[Dict[str, Any]]]:
        context_parts = []
        sources_info = []
        total_chars = 0

        for i, result in enumerate(search_results[:max_chunks]):
            citation_info = result['citation']
            citation_ref = f"[{i+1}]"
            chunk_text = f"{citation_ref} {result['content']}"

            if total_chars + len(chunk_text) > max_context_chars and context_parts:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
            sources_info.append({
                'reference': citation_ref,
                'source_file': citation_info.get('source_file', 'Unknown Source'),
                'source_type': citation_info.get('source_type', 'unknown'),
                'page_number': citation_info.get('page_number'),
                'chunk_id': result['id'],
                'relevance_score': result['score']
            })

        return '\n\n'.join(context_parts), sources_info

    def _create_rag_prompt(self, query: str, context: str) -> str:
        return f"""Context:
{context}

Question: {query}

Answer the question using only the context above. Be direct and concise. Cite sources as [1], [2] etc. after each claim.
Answer:"""

    def generate_summary(
        self,
        max_chunks: int = 15,
        summary_length: str = "medium"
    ) -> RAGResult:
        try:
            summary_query = "main topics key findings important information overview"
            query_vector = self.embedding_generator.generate_query_embedding(summary_query)
            search_results = self.vector_db.search(query_vector=query_vector.tolist(), limit=max_chunks)

            if not search_results:
                return RAGResult(query="Document Summary",
                                 response="No documents available for summarization.",
                                 sources_used=[], retrieval_count=0)

            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, 6000)

            length_instructions = {
                'short': "Provide a concise 2-3 paragraph summary of the most important points.",
                'medium': "Provide a comprehensive 4-5 paragraph summary covering key topics and findings.",
                'long': "Provide a detailed summary with multiple sections covering all major topics."
            }

            summary_prompt = f"""Context:
{context}

{length_instructions.get(summary_length, length_instructions['medium'])} Cite sources as [1], [2] etc.
Summary:"""

            response = self._call_llm(summary_prompt)
            return RAGResult(query="Document Summary", response=response,
                             sources_used=sources_info, retrieval_count=len(search_results))

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return RAGResult(query="Document Summary",
                             response=f"Error generating summary: {str(e)}",
                             sources_used=[], retrieval_count=0)


if __name__ == "__main__":
    embedding_gen = EmbeddingGenerator()
    vector_db = MilvusVectorDB()
    rag = RAGGenerator(embedding_generator=embedding_gen, vector_db=vector_db)
    result = rag.generate_response("What are the main findings discussed in the documents?")
    print(f"Response: {result.response}")
    print(result.get_citation_summary())