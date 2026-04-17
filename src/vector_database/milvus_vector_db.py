import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from src.embeddings.embedding_generator import EmbeddedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusVectorDB:
    def __init__(
        self,
        db_path: str = "./data/chroma",
        collection_name: str = "notebook_lm",
        embedding_dim: int = 384
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = None
        self.collection = None
        self.collection_exists = False

        self._initialize_client()
        self._setup_collection()

    def _initialize_client(self):
        try:
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"ChromaDB client initialized at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise

    def _setup_collection(self):
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "l2"}
            )
            self.collection_exists = True
            logger.info(f"Collection '{self.collection_name}' ready")
        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise

    def create_index(self, **kwargs):
        # ChromaDB manages indexing automatically (HNSW)
        logger.info("ChromaDB uses automatic HNSW indexing — no manual index creation needed")

    def insert_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> List[str]:
        if not embedded_chunks:
            return []
        try:
            ids, embeddings, metadatas, documents = [], [], [], []

            for embedded_chunk in embedded_chunks:
                chunk_data = embedded_chunk.to_vector_db_format()

                chunk_id = chunk_data["id"]
                vector = chunk_data["vector"]
                content = chunk_data["content"]

                meta = {
                    "source_file": chunk_data.get("source_file", ""),
                    "source_type": chunk_data.get("source_type", ""),
                    "page_number": chunk_data.get("page_number") or -1,
                    "chunk_index": chunk_data.get("chunk_index") or 0,
                    "start_char": chunk_data.get("start_char") or -1,
                    "end_char": chunk_data.get("end_char") or -1,
                    "embedding_model": chunk_data.get("embedding_model", ""),
                    "metadata_json": json.dumps(chunk_data.get("metadata", {}))
                }

                ids.append(chunk_id)
                embeddings.append(vector)
                metadatas.append(meta)
                documents.append(content)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            logger.info(f"Inserted {len(ids)} embeddings into ChromaDB")
            return ids

        except Exception as e:
            logger.error(f"Error inserting embeddings: {str(e)}")
            raise

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_expr: Optional[str] = None,
        **kwargs  # absorbs nprobe, rbq_query_bits, etc.
    ) -> List[Dict[str, Any]]:
        try:
            where = None
            # Basic filter support: convert simple milvus expr to chroma where clause
            # e.g. 'source_type == "pdf"' -> {"source_type": {"$eq": "pdf"}}
            # For complex filters, extend this block as needed
            if filter_expr:
                logger.warning(f"filter_expr '{filter_expr}' ignored — add manual where clause if needed")

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"]
            )

            formatted_results = []
            if not results or not results["ids"][0]:
                return formatted_results

            for i, chunk_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                content = results["documents"][0][i]
                distance = results["distances"][0][i]

                try:
                    metadata = json.loads(meta.get("metadata_json", "{}"))
                except Exception:
                    metadata = {}

                formatted_results.append({
                    "id": chunk_id,
                    "score": distance,
                    "content": content,
                    "citation": {
                        "source_file": meta.get("source_file", ""),
                        "source_type": meta.get("source_type", ""),
                        "page_number": meta.get("page_number") if meta.get("page_number") != -1 else None,
                        "chunk_index": meta.get("chunk_index"),
                        "start_char": meta.get("start_char") if meta.get("start_char") != -1 else None,
                        "end_char": meta.get("end_char") if meta.get("end_char") != -1 else None,
                    },
                    "metadata": metadata,
                    "embedding_model": meta.get("embedding_model", "")
                })

            logger.info(f"Search completed: {len(formatted_results)} results found")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection_exists = False
            logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.collection_exists:
                logger.warning("Collection does not exist")
                return None

            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )

            if not results["ids"]:
                logger.warning(f"No chunk found with ID: {chunk_id}")
                return None

            meta = results["metadatas"][0]
            content = results["documents"][0]

            try:
                metadata = json.loads(meta.get("metadata_json", "{}"))
            except Exception:
                metadata = {}

            return {
                "id": results["ids"][0],
                "content": content,
                "metadata": metadata,
                "source_file": meta.get("source_file", ""),
                "source_type": meta.get("source_type", ""),
                "page_number": meta.get("page_number") if meta.get("page_number") != -1 else None,
                "chunk_index": meta.get("chunk_index")
            }

        except Exception as e:
            logger.error(f"Error retrieving chunk by ID {chunk_id}: {str(e)}")
            return None

    def close(self):
        # ChromaDB PersistentClient has no explicit close — data is auto-persisted
        logger.info("ChromaDB client closed (data persisted automatically)")


if __name__ == "__main__":
    from src.document_processing.doc_processor import DocumentProcessor
    from src.embeddings.embedding_generator import EmbeddingGenerator

    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_db = MilvusVectorDB()

    try:
        chunks = doc_processor.process_document("data/raft.pdf")
        embedded_chunks = embedding_generator.generate_embeddings(chunks)
        vector_db.create_index()

        inserted_ids = vector_db.insert_embeddings(embedded_chunks)
        print(f"Inserted {len(inserted_ids)} embeddings")

        query_text = "What is the main topic?"
        query_vector = embedding_generator.generate_query_embedding(query_text)

        search_results = vector_db.search(query_vector.tolist(), limit=5)

        for i, result in enumerate(search_results):
            print(f"\nResult {i+1}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content'][:200]}...")
            print(f"Citation: {result['citation']}")

    except Exception as e:
        print(f"Error in example: {e}")

    finally:
        vector_db.close()