from collections.abc import Callable, Sequence

from sqlalchemy import Select, bindparam, func, literal_column, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.embeddings.base import EmbeddingProvider
from app.knowledge.embedding.contracts import (
    ChunkForEmbedding,
    RetrievalFilters,
    StoredEmbeddingHit,
    StoredLexicalHit,
)
from app.knowledge.embedding.errors import (
    DatabaseUnavailable,
    IncompatibleEmbeddingDimension,
    VectorExtensionUnavailable,
)
from app.models.embedding import EmbeddingSpace, KnowledgeEmbedding

_ENGLISH_TEXT_CONFIG = literal_column("'english'::regconfig")


class EmbeddingRepository:
    """PostgreSQL/pgvector persistence for vectors and small provenance references."""

    def __init__(self, session_factory: Callable[[], Session] = SessionLocal) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _raise_database_error(exc: SQLAlchemyError) -> None:
        detail = str(exc).lower()
        if "vector" in detail and any(
            marker in detail for marker in ("extension", "does not exist", "no operator")
        ):
            raise VectorExtensionUnavailable(
                "PostgreSQL pgvector is unavailable. Apply the Alembic migration and use the "
                "pgvector-enabled PostgreSQL image."
            ) from exc
        raise DatabaseUnavailable(
            "PostgreSQL vector storage is unavailable. Check DATABASE_URL and database health."
        ) from exc

    @staticmethod
    def _ensure_space(session: Session, provider: str, model: str, dimension: int) -> None:
        insert_space = pg_insert(EmbeddingSpace).values(
            provider=provider,
            model=model,
            dimension=dimension,
        )
        session.execute(
            insert_space.on_conflict_do_nothing(index_elements=["provider", "model"])
        )
        space = session.execute(
            select(EmbeddingSpace).where(
                EmbeddingSpace.provider == provider,
                EmbeddingSpace.model == model,
            )
        ).scalar_one()
        if space.dimension != dimension:
            raise IncompatibleEmbeddingDimension(
                f"Embedding space {provider}/{model} is registered at {space.dimension} dimensions, "
                f"but this operation supplied {dimension}. Use a new model identity or restore the "
                "matching dimension."
            )

    def existing_content_hashes(
        self,
        chunk_ids: Sequence[str],
        provider: str,
        model: str,
        dimension: int,
    ) -> dict[str, str]:
        if not chunk_ids:
            return {}
        try:
            with self._session_factory() as session, session.begin():
                self._ensure_space(session, provider, model, dimension)
                rows = session.execute(
                    select(KnowledgeEmbedding.chunk_id, KnowledgeEmbedding.content_sha256).where(
                        KnowledgeEmbedding.chunk_id.in_(chunk_ids),
                        KnowledgeEmbedding.embedding_provider == provider,
                        KnowledgeEmbedding.embedding_model == model,
                    )
                )
                return {chunk_id: digest for chunk_id, digest in rows}
        except IncompatibleEmbeddingDimension:
            raise
        except SQLAlchemyError as exc:
            self._raise_database_error(exc)

    @staticmethod
    def _row_values(
        record: ChunkForEmbedding,
        provider: str,
        model: str,
        dimension: int,
    ) -> dict[str, object]:
        chunk = record.chunk
        source = record.source
        if len(record.vector) != dimension:
            raise IncompatibleEmbeddingDimension(
                f"Chunk {chunk.chunk_id} has a {len(record.vector)}-dimension vector; expected {dimension}."
            )
        return {
            "chunk_id": chunk.chunk_id,
            "source_id": source.source_id,
            "source_type": source.source_type,
            "title": source.title,
            "author": source.author,
            "source_url": source.source_url,
            "source_file_path": source.file_path,
            "processed_file_path": str(record.processed_file_path.resolve()),
            "published_at": source.published_at,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "content_sha256": record.content_sha256,
            "embedding": record.vector,
            "search_vector": func.to_tsvector(_ENGLISH_TEXT_CONFIG, chunk.text),
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_dimension": dimension,
        }

    def upsert_batch(
        self,
        records: Sequence[ChunkForEmbedding],
        provider: str,
        model: str,
        dimension: int,
    ) -> tuple[int, int, int]:
        """Insert new vectors, replace changed chunk content, and skip unchanged rows."""
        if not records:
            return 0, 0, 0
        rows = [self._row_values(record, provider, model, dimension) for record in records]
        ids = [str(record.chunk.chunk_id) for record in records]
        try:
            with self._session_factory() as session, session.begin():
                self._ensure_space(session, provider, model, dimension)
                existing_rows = session.execute(
                    select(KnowledgeEmbedding.chunk_id, KnowledgeEmbedding.content_sha256).where(
                        KnowledgeEmbedding.chunk_id.in_(ids),
                        KnowledgeEmbedding.embedding_provider == provider,
                        KnowledgeEmbedding.embedding_model == model,
                    )
                )
                existing = {chunk_id: digest for chunk_id, digest in existing_rows}
                statement = pg_insert(KnowledgeEmbedding).values(rows)
                excluded = statement.excluded
                changed = session.execute(
                    statement.on_conflict_do_update(
                        constraint="uq_knowledge_embedding_chunk_space",
                        set_={
                            "source_id": excluded.source_id,
                            "source_type": excluded.source_type,
                            "title": excluded.title,
                            "author": excluded.author,
                            "source_url": excluded.source_url,
                            "source_file_path": excluded.source_file_path,
                            "processed_file_path": excluded.processed_file_path,
                            "published_at": excluded.published_at,
                            "page_number": excluded.page_number,
                            "chunk_index": excluded.chunk_index,
                            "char_start": excluded.char_start,
                            "char_end": excluded.char_end,
                            "content_sha256": excluded.content_sha256,
                            "embedding": excluded.embedding,
                            "search_vector": excluded.search_vector,
                            "embedding_dimension": excluded.embedding_dimension,
                            "created_at": func.now(),
                        },
                        where=KnowledgeEmbedding.content_sha256 != excluded.content_sha256,
                    ).returning(KnowledgeEmbedding.chunk_id)
                ).scalars().all()
                changed_ids = set(changed)
                created = sum(chunk_id not in existing for chunk_id in ids)
                updated = sum(
                    chunk_id in existing and chunk_id in changed_ids for chunk_id in ids
                )
                skipped = len(ids) - created - updated
                return created, updated, skipped
        except (IncompatibleEmbeddingDimension, VectorExtensionUnavailable, DatabaseUnavailable):
            raise
        except SQLAlchemyError as exc:
            self._raise_database_error(exc)

    def refresh_provenance(
        self,
        records: Sequence[ChunkForEmbedding],
        provider: str,
        model: str,
        dimension: int,
    ) -> None:
        """Refresh JSON-file references for unchanged chunks without regenerating vectors."""
        if not records:
            return
        try:
            with self._session_factory() as session, session.begin():
                self._ensure_space(session, provider, model, dimension)
                for record in records:
                    chunk = record.chunk
                    source = record.source
                    session.execute(
                        update(KnowledgeEmbedding)
                        .where(
                            KnowledgeEmbedding.chunk_id == chunk.chunk_id,
                            KnowledgeEmbedding.embedding_provider == provider,
                            KnowledgeEmbedding.embedding_model == model,
                            KnowledgeEmbedding.content_sha256 == record.content_sha256,
                        )
                        .values(
                            source_id=source.source_id,
                            source_type=source.source_type,
                            title=source.title,
                            author=source.author,
                            source_url=source.source_url,
                            source_file_path=source.file_path,
                            processed_file_path=str(record.processed_file_path.resolve()),
                            published_at=source.published_at,
                            page_number=chunk.page_number,
                            chunk_index=chunk.chunk_index,
                            char_start=chunk.char_start,
                            char_end=chunk.char_end,
                            search_vector=func.to_tsvector(_ENGLISH_TEXT_CONFIG, chunk.text),
                        )
                    )
        except (IncompatibleEmbeddingDimension, VectorExtensionUnavailable, DatabaseUnavailable):
            raise
        except SQLAlchemyError as exc:
            self._raise_database_error(exc)

    def refresh_lexical_vectors(self, records: Sequence[ChunkForEmbedding]) -> int:
        """Backfill derived tsvectors without loading or changing embedding vectors."""
        if not records:
            return 0
        statement = (
            update(KnowledgeEmbedding)
            .where(
                KnowledgeEmbedding.chunk_id == bindparam("target_chunk_id"),
                KnowledgeEmbedding.source_id == bindparam("target_source_id"),
                KnowledgeEmbedding.content_sha256 == bindparam("target_content_sha256"),
            )
            .values(
                search_vector=func.to_tsvector(
                    _ENGLISH_TEXT_CONFIG, bindparam("canonical_chunk_text")
                )
            )
        )
        parameters = [
            {
                "target_chunk_id": record.chunk.chunk_id,
                "target_source_id": record.source.source_id,
                "target_content_sha256": record.content_sha256,
                "canonical_chunk_text": record.chunk.text,
            }
            for record in records
        ]
        try:
            with self._session_factory() as session, session.begin():
                # Core execution keeps SQLAlchemy's ORM bulk-update-by-primary-key
                # behavior out of this natural-key batch update.
                result = session.connection().execute(statement, parameters)
                return max(result.rowcount or 0, 0)
        except SQLAlchemyError as exc:
            self._raise_database_error(exc)

    def search(
        self,
        vector: Sequence[float],
        provider: EmbeddingProvider,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[StoredEmbeddingHit]:
        try:
            with self._session_factory() as session:
                space = session.execute(
                    select(EmbeddingSpace).where(
                        EmbeddingSpace.provider == provider.provider_name,
                        EmbeddingSpace.model == provider.model_name,
                    )
                ).scalar_one_or_none()
                if space is None:
                    return []
                if space.dimension != provider.dimension or len(vector) != space.dimension:
                    raise IncompatibleEmbeddingDimension(
                        f"Query vector has {len(vector)} dimensions; stored model space "
                        f"{provider.provider_name}/{provider.model_name} has {space.dimension}."
                    )

                distance = KnowledgeEmbedding.embedding.cosine_distance(vector).label("distance")
                statement: Select[tuple[KnowledgeEmbedding, float]] = (
                    select(KnowledgeEmbedding, distance)
                    .where(
                        KnowledgeEmbedding.embedding_provider == provider.provider_name,
                        KnowledgeEmbedding.embedding_model == provider.model_name,
                        KnowledgeEmbedding.embedding_dimension == provider.dimension,
                    )
                    .order_by(distance, KnowledgeEmbedding.chunk_id.asc())
                    .limit(top_k)
                )
                if filters is not None:
                    statement = self._apply_filters(statement, filters)
                rows = session.execute(statement).all()
                return [
                    StoredEmbeddingHit(
                        chunk_id=row.chunk_id,
                        source_id=row.source_id,
                        source_type=row.source_type,
                        title=row.title,
                        author=row.author,
                        source_url=row.source_url,
                        source_file_path=row.source_file_path,
                        processed_file_path=row.processed_file_path,
                        published_at=row.published_at,
                        page_number=row.page_number,
                        chunk_index=row.chunk_index,
                        char_start=row.char_start,
                        char_end=row.char_end,
                        content_sha256=row.content_sha256,
                        distance=float(score),
                    )
                    for row, score in rows
                ]
        except IncompatibleEmbeddingDimension:
            raise
        except SQLAlchemyError as exc:
            self._raise_database_error(exc)

    def search_lexical(
        self,
        query_text: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[StoredLexicalHit]:
        """Search derived tsvectors with PostgreSQL web-style full-text syntax."""
        if not query_text or not query_text.strip() or top_k <= 0:
            return []
        try:
            with self._session_factory() as session:
                tsquery = func.websearch_to_tsquery(_ENGLISH_TEXT_CONFIG, query_text.strip())
                rank = func.ts_rank_cd(KnowledgeEmbedding.search_vector, tsquery).label("lexical_score")
                candidate_rows = select(
                    KnowledgeEmbedding.id.label("embedding_id"),
                    rank.label("lexical_score"),
                    func.row_number().over(
                        partition_by=KnowledgeEmbedding.chunk_id,
                        order_by=(
                            KnowledgeEmbedding.embedding_provider.asc(),
                            KnowledgeEmbedding.embedding_model.asc(),
                            KnowledgeEmbedding.id.asc(),
                        ),
                    ).label("model_choice"),
                ).where(
                    KnowledgeEmbedding.search_vector.is_not(None),
                    KnowledgeEmbedding.search_vector.op("@@")(tsquery),
                )
                if filters is not None:
                    candidate_rows = self._apply_filters(candidate_rows, filters)
                candidates = candidate_rows.subquery()
                statement: Select = (
                    select(KnowledgeEmbedding, candidates.c.lexical_score)
                    .join(candidates, KnowledgeEmbedding.id == candidates.c.embedding_id)
                    .where(
                        candidates.c.model_choice == 1,
                    )
                    .order_by(candidates.c.lexical_score.desc(), KnowledgeEmbedding.chunk_id.asc())
                    .limit(top_k)
                )
                rows = session.execute(statement).all()
                return [
                    StoredLexicalHit(
                        chunk_id=row.chunk_id,
                        source_id=row.source_id,
                        source_type=row.source_type,
                        title=row.title,
                        author=row.author,
                        source_url=row.source_url,
                        source_file_path=row.source_file_path,
                        processed_file_path=row.processed_file_path,
                        published_at=row.published_at,
                        page_number=row.page_number,
                        chunk_index=row.chunk_index,
                        char_start=row.char_start,
                        char_end=row.char_end,
                        content_sha256=row.content_sha256,
                        lexical_score=float(score),
                    )
                    for row, score in rows
                ]
        except SQLAlchemyError as exc:
            self._raise_database_error(exc)

    @staticmethod
    def _apply_filters(statement: Select, filters: RetrievalFilters) -> Select:
        source_id = filters.effective_source_id
        if source_id is not None:
            statement = statement.where(KnowledgeEmbedding.source_id == source_id)
        if filters.source_type is not None:
            statement = statement.where(KnowledgeEmbedding.source_type == filters.source_type)
        if filters.file_path is not None:
            statement = statement.where(KnowledgeEmbedding.source_file_path == filters.file_path)
        if filters.page_number is not None:
            statement = statement.where(KnowledgeEmbedding.page_number == filters.page_number)
        return statement
