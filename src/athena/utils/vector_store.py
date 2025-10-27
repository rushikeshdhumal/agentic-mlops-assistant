"""
Vector store for semantic search over experiments.

Uses Chroma for efficient similarity search with embeddings.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from athena.config.storage import get_storage_config

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for experiment metadata and documentation.

    Uses Chroma for storage and sentence-transformers for embeddings.
    """

    def __init__(
        self,
        collection_name: str = "athena_experiments",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[Path] = None,
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the Chroma collection.
            embedding_model: Name of the sentence-transformer model.
            persist_directory: Directory to persist Chroma data.
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Setup persist directory
        if persist_directory is None:
            storage_config = get_storage_config()
            persist_directory = storage_config.get_all_paths()["vector_db"]

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dimension}")

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "ATHENA experiment metadata and documentation"},
        )

        logger.info(f"Vector store initialized with collection: {collection_name}")
        logger.info(f"Persist directory: {self.persist_directory}")
        logger.info(f"Collection count: {self.collection.count()}")

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of text documents to embed.
            metadatas: List of metadata dictionaries.
            ids: List of unique document IDs.
        """
        if not documents:
            logger.warning("No documents to add")
            return

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)

        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist(),
        )

        logger.info(f"Added {len(documents)} documents to vector store")

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query: Search query text.
            n_results: Number of results to return.
            where: Optional metadata filter.

        Returns:
            Dictionary with ids, documents, metadatas, and distances.
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
        )

        logger.info(f"Found {len(results['ids'][0])} results for query: {query[:50]}...")
        return results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete.
        """
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")

    def update_document(
        self,
        document_id: str,
        document: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Update a document.

        Args:
            document_id: Document ID.
            document: Updated document text.
            metadata: Updated metadata.
        """
        # Generate new embedding
        embedding = self.embedding_model.encode([document])[0]

        # Update
        self.collection.update(
            ids=[document_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[embedding.tolist()],
        )

        logger.info(f"Updated document: {document_id}")

    def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.

        Args:
            document_id: Document ID.

        Returns:
            Document data or None if not found.
        """
        results = self.collection.get(ids=[document_id])
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0],
                "metadata": results["metadatas"][0],
            }
        return None

    def count(self) -> int:
        """
        Get total number of documents.

        Returns:
            Document count.
        """
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete collection and recreate
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "ATHENA experiment metadata and documentation"},
        )
        logger.info(f"Cleared collection: {self.collection_name}")


def initialize_vector_store(
    collection_name: str = "athena_experiments",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> VectorStore:
    """
    Initialize vector store with default configuration.

    Args:
        collection_name: Name of the Chroma collection.
        embedding_model: Name of the sentence-transformer model.

    Returns:
        Configured VectorStore instance.
    """
    store = VectorStore(collection_name=collection_name, embedding_model=embedding_model)
    logger.info("Vector store initialized successfully")
    return store
