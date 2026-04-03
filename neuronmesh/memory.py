"""
NeuronMesh Memory - Intelligent Memory Layer

Provides persistent, searchable memory for agents:
- Short-term memory (recent context)
- Long-term memory (vector RAG)
- Structured memory (facts, preferences)
"""

import os
import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory"""
    SHORT_TERM = "stm"      # Recent conversation
    LONG_TERM = "ltm"       # Vector stored memories
    WORKING = "working"    # Current task context
    PROCEDURAL = "procedural"  # Skills and procedures


@dataclass
class MemoryEntry:
    """A single memory entry"""
    id: str
    content: str
    entry_type: str = "fact"  # "fact", "preference", "conversation", "skill"
    memory_type: MemoryType = MemoryType.LONG_TERM
    importance: float = 0.5   # 0.0 - 1.0
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self):
        """Update access time and count"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "entry_type": self.entry_type,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data["id"],
            content=data["content"],
            entry_type=data.get("entry_type", "fact"),
            memory_type=MemoryType(data.get("memory_type", "ltm")),
            importance=data.get("importance", 0.5),
            embedding=data.get("embedding"),
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


class VectorStore:
    """Simple vector store for embeddings (production would use Qdrant/Pinecone)"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.expanduser("~/.neuronmesh/memory/")
        os.makedirs(self.storage_path, exist_ok=True)
        self._index_file = os.path.join(self.storage_path, "vectors.json")
        self._load_index()
    
    def _load_index(self):
        """Load existing index"""
        if os.path.exists(self._index_file):
            with open(self._index_file, "r") as f:
                self.vectors = json.load(f)
        else:
            self.vectors = {}
    
    def _save_index(self):
        """Save index to disk"""
        with open(self._index_file, "w") as f:
            json.dump(self.vectors, f)
    
    def add(self, entry: MemoryEntry):
        """Add an entry to the store"""
        self.vectors[entry.id] = entry.to_dict()
        self._save_index()
    
    def get(self, id: str) -> Optional[MemoryEntry]:
        """Get an entry by ID"""
        data = self.vectors.get(id)
        if data:
            return MemoryEntry.from_dict(data)
        return None
    
    def delete(self, id: str):
        """Delete an entry"""
        if id in self.vectors:
            del self.vectors[id]
            self._save_index()
    
    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
    ) -> List[tuple[str, float]]:
        """
        Search for similar vectors.
        Uses simple cosine similarity for now.
        
        Returns list of (id, similarity_score) tuples
        """
        results = []
        
        for id, data in self.vectors.items():
            stored_embedding = data.get("embedding")
            if not stored_embedding:
                continue
            
            # Simple cosine similarity
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            results.append((id, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


class Memory:
    """
    Intelligent memory for agents.
    
    Features:
    - Automatic importance scoring
    - Memory consolidation (STM -> LTM)
    - Semantic search via embeddings
    - Configurable retention policies
    
    Example:
        memory = Memory()
        
        # Store something
        memory.add("I prefer dark mode", entry_type="preference")
        
        # Retrieve relevant memories
        relevant = memory.retrieve("What are my preferences?")
        for entry in relevant:
            print(entry.content)
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_stm_size: int = 100,
        consolidation_threshold: float = 0.6,
    ):
        self.storage_path = storage_path
        self.max_stm_size = max_stm_size
        self.consolidation_threshold = consolidation_threshold
        
        # Initialize stores
        self.vector_store = VectorStore(storage_path)
        
        # Short-term memory (in-memory)
        self.stm: List[MemoryEntry] = []
        
        # Working memory (current task)
        self.working: List[MemoryEntry] = []
        
        # Load persisted memories
        self._load_memories()
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for a memory"""
        return hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:16]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding for text.
        Production should use sentence-transformers or OpenAI embeddings.
        """
        # Simple TF-IDF-like embedding (for demo purposes)
        # In production, use: sentence-transformers or OpenAI embeddings
        words = text.lower().split()
        embedding = [0.0] * 128
        
        for i, word in enumerate(words[:128]):
            embedding[i % 128] += hash(word) % 100 / 100.0
        
        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def add(
        self,
        content: str,
        entry_type: str = "fact",
        importance: Optional[float] = None,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        Add a new memory entry.
        
        Args:
            content: The memory content
            entry_type: Type of memory ("fact", "preference", etc.)
            importance: 0.0-1.0 importance score (auto-calculated if not provided)
            memory_type: Where to store the memory
            metadata: Additional metadata
            
        Returns:
            The created MemoryEntry
        """
        # Auto-calculate importance if not provided
        if importance is None:
            importance = self._calculate_importance(content, entry_type)
        
        entry = MemoryEntry(
            id=self._generate_id(content),
            content=content,
            entry_type=entry_type,
            memory_type=memory_type,
            importance=importance,
            embedding=self._generate_embedding(content),
            metadata=metadata or {},
        )
        
        if memory_type == MemoryType.SHORT_TERM:
            self.stm.append(entry)
            self._maybe_consolidate()
        elif memory_type == MemoryType.WORKING:
            self.working.append(entry)
        else:
            self.vector_store.add(entry)
        
        return entry
    
    def _calculate_importance(self, content: str, entry_type: str) -> float:
        """Calculate importance score for a memory"""
        score = 0.5  # Base score
        
        # Higher importance for certain types
        if entry_type == "preference":
            score += 0.2
        elif entry_type == "skill":
            score += 0.3
        elif entry_type == "fact":
            score += 0.1
        
        # Longer content might be more important
        if len(content) > 100:
            score += 0.1
        
        return min(1.0, score)
    
    def _maybe_consolidate(self):
        """
        Consolidate short-term memories to long-term.
        Called when STM is getting full.
        """
        if len(self.stm) > self.max_stm_size:
            # Move important memories to LTM
            self.stm.sort(key=lambda e: e.importance, reverse=True)
            
            to_move = self.stm[:self.max_stm_size // 2]
            for entry in to_move:
                self.vector_store.add(entry)
            
            self.stm = self.stm[self.max_stm_size // 2:]
    
    def retrieve(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: Query text
            limit: Maximum number of memories to return
            memory_types: Which memory types to search (None = all)
            
        Returns:
            List of relevant MemoryEntry objects, sorted by relevance
        """
        results = []
        memory_types = memory_types or [MemoryType.LONG_TERM, MemoryType.SHORT_TERM]
        
        query_embedding = self._generate_embedding(query)
        
        # Search vector store
        if MemoryType.LONG_TERM in memory_types:
            vector_results = self.vector_store.search(query_embedding, limit)
            for id, score in vector_results:
                entry = self.vector_store.get(id)
                if entry:
                    entry.touch()
                    results.append(entry)
        
        # Search STM
        if MemoryType.SHORT_TERM in memory_types:
            stm_scores = []
            for entry in self.stm:
                if entry.embedding:
                    score = self._cosine_similarity(query_embedding, entry.embedding)
                    stm_scores.append((entry, score))
            
            stm_scores.sort(key=lambda x: x[1], reverse=True)
            for entry, _ in stm_scores[:limit]:
                entry.touch()
                results.append(entry)
        
        # Search working memory
        if MemoryType.WORKING in memory_types:
            for entry in self.working:
                entry.touch()
                results.append(entry)
        
        return results[:limit]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        if not a or not b:
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def forget(self, id: str):
        """Remove a memory by ID"""
        self.vector_store.delete(id)
        self.stm = [e for e in self.stm if e.id != id]
        self.working = [e for e in self.working if e.id != id]
    
    def clear(self, memory_type: Optional[MemoryType] = None):
        """Clear memories"""
        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            self.stm.clear()
        if memory_type is None or memory_type == MemoryType.WORKING:
            self.working.clear()
        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            self.vector_store.vectors.clear()
            self.vector_store._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "stm_size": len(self.stm),
            "working_size": len(self.working),
            "ltm_size": len(self.vector_store.vectors),
            "total_memories": len(self.vector_store.vectors) + len(self.stm) + len(self.working),
        }
    
    def _load_memories(self):
        """Load memories from storage"""
        # This would load from disk in a full implementation
        pass
    
    def _save_memories(self):
        """Save memories to storage"""
        # This would save to disk in a full implementation
        pass
