import logging
import os
import time
import json
import sqlite3
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.generation.rag import RAGResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    user_query: str
    assistant_response: str
    sources_used: List[Dict[str, Any]]
    timestamp: str
    session_id: str


class NotebookMemoryLayer:
    def __init__(
        self,
        user_id: str,
        session_id: str,
        zep_api_key: Optional[str] = None,  # kept for signature compat, unused
        mode: str = "summary",
        indexing_wait_time: int = 0,        # no async indexing needed
        create_new_session: bool = False,
        db_path: str = "./data/memory.db"
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.indexing_wait_time = indexing_wait_time

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

        if create_new_session:
            self.clear_session()

        logger.info(f"NotebookMemoryLayer initialized for user {user_id}, session {session_id}")

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                metadata TEXT,
                timestamp TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session ON memory(session_id)
        """)
        self.conn.commit()

    def _save(self, content: str, role: str, metadata: Dict[str, Any]):
        self.conn.execute(
            "INSERT INTO memory (user_id, session_id, role, content, metadata, timestamp) VALUES (?,?,?,?,?,?)",
            (self.user_id, self.session_id, role, content, json.dumps(metadata), datetime.now().isoformat())
        )
        self.conn.commit()

    def save_conversation_turn(
        self,
        rag_result: RAGResult,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None
    ):
        try:
            self._save(
                content=rag_result.query,
                role="user",
                metadata={
                    "type": "message",
                    "session_id": self.session_id,
                    **(user_metadata or {})
                }
            )
            self._save(
                content=rag_result.response,
                role="assistant",
                metadata={
                    "type": "message",
                    "sources_count": len(rag_result.sources_used),
                    "retrieval_count": rag_result.retrieval_count,
                    "model_used": getattr(rag_result, "model_name", "unknown"),
                    "sources_summary": self._create_sources_summary(rag_result.sources_used),
                    "session_id": self.session_id,
                    **(assistant_metadata or {})
                }
            )
            self._save_source_context(rag_result.sources_used)
            logger.info(f"Saved conversation turn with {len(rag_result.sources_used)} sources")

        except Exception as e:
            logger.error(f"Error saving conversation turn: {str(e)}")
            raise

    def _create_sources_summary(self, sources_used: List[Dict[str, Any]]) -> str:
        if not sources_used:
            return "No sources used"
        source_files = list(set(s.get("source_file", "Unknown") for s in sources_used))
        source_types = list(set(s.get("source_type", "unknown") for s in sources_used))
        summary = f"{len(source_files)} files ({', '.join(source_types)}): {', '.join(source_files[:3])}"
        if len(source_files) > 3:
            summary += f" and {len(source_files) - 3} more"
        return summary

    def _save_source_context(self, sources_used: List[Dict[str, Any]]):
        if not sources_used:
            return
        context = {
            "referenced_documents": [
                {
                    "file": s.get("source_file", "Unknown"),
                    "type": s.get("source_type", "unknown"),
                    "page": s.get("page_number"),
                    "relevance": s.get("relevance_score", 0)
                }
                for s in sources_used
            ],
            "document_types": list(set(s.get("source_type", "unknown") for s in sources_used))
        }
        self._save(
            content=f"Document sources referenced: {context}",
            role="system",
            metadata={"type": "source_context", "session_id": self.session_id}
        )

    def save_user_preferences(self, preferences: Dict[str, Any]):
        try:
            self._save(
                content=f"User preferences: {preferences}",
                role="system",
                metadata={"type": "preferences", "session_id": self.session_id}
            )
            logger.info("User preferences saved to memory")
        except Exception as e:
            logger.error(f"Error saving preferences: {str(e)}")

    def save_document_metadata(self, document_info: Dict[str, Any]):
        try:
            self._save(
                content=f"Document processed: {document_info}",
                role="system",
                metadata={"type": "document_metadata", "session_id": self.session_id}
            )
            logger.info(f"Document metadata saved: {document_info.get('name', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error saving document metadata: {str(e)}")

    def get_conversation_context(self) -> str:
        try:
            rows = self.conn.execute(
                "SELECT role, content FROM memory WHERE session_id=? AND role IN ('user','assistant') ORDER BY id",
                (self.session_id,)
            ).fetchall()
            if not rows:
                return ""
            return "\n".join(f"{role.upper()}: {content}" for role, content in rows[-20:])
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return "No conversation context available"

    def get_relevant_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword search — no external API needed."""
        try:
            keywords = query.lower().split()
            rows = self.conn.execute(
                "SELECT role, content, metadata, timestamp FROM memory WHERE user_id=? ORDER BY id DESC LIMIT 200",
                (self.user_id,)
            ).fetchall()

            scored = []
            for role, content, metadata, timestamp in rows:
                score = sum(1 for kw in keywords if kw in content.lower())
                if score > 0:
                    scored.append((score, role, content, metadata, timestamp))

            scored.sort(key=lambda x: -x[0])
            results = []
            for score, role, content, metadata, timestamp in scored[:limit]:
                results.append({
                    "content": content,
                    "role": role,
                    "relevance_score": score,
                    "timestamp": timestamp,
                    "session_id": self.session_id,
                    "thread_id": self.session_id,
                })
            logger.info(f"Retrieved {len(results)} relevant memories for query")
            return results
        except Exception as e:
            logger.error(f"Error getting relevant memory: {str(e)}")
            return []

    def wait_for_indexing(self):
        # No async indexing — SQLite writes are synchronous
        if self.indexing_wait_time > 0:
            time.sleep(self.indexing_wait_time)

    def get_session_summary(self) -> Dict[str, Any]:
        try:
            rows = self.conn.execute(
                "SELECT role, timestamp FROM memory WHERE session_id=?",
                (self.session_id,)
            ).fetchall()
            if not rows:
                return {"message_count": 0, "summary": "No messages in session"}
            user_msgs = [r for r in rows if r[0] == "user"]
            asst_msgs = [r for r in rows if r[0] == "assistant"]
            return {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "total_messages": len(rows),
                "user_messages": len(user_msgs),
                "assistant_messages": len(asst_msgs),
                "context_available": len(rows) > 0,
                "last_interaction": rows[-1][1] if rows else None
            }
        except Exception as e:
            logger.error(f"Error getting session summary: {str(e)}")
            return {"error": str(e)}

    def clear_session(self):
        try:
            self.conn.execute("DELETE FROM memory WHERE session_id=?", (self.session_id,))
            self.conn.commit()
            logger.info(f"Session {self.session_id} cleared")
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}")
            raise

    def close(self):
        self.conn.close()