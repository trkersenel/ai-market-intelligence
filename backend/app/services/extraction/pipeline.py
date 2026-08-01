"""Running extraction over the stored corpus and recording proposals."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.db.mongo import Collection, MongoDatabase
from app.repositories.graph import EntityRepository, ProposalRepository
from app.services.extraction.extractor import ExtractionReport, RelationshipExtractor
from app.services.rag.llm import LlmClient

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineReport:
    """What one pipeline run did."""

    report: ExtractionReport
    stored: int


class ExtractionPipeline:
    """Reads recent articles, proposes edges, and files them for review."""

    def __init__(
        self,
        *,
        mongo: MongoDatabase,
        entities: EntityRepository,
        proposals: ProposalRepository,
        llm: LlmClient,
    ) -> None:
        """Wire the pipeline to its collaborators."""
        self._mongo = mongo
        self._entities = entities
        self._proposals = proposals
        self._llm = llm

    async def run(self, *, limit: int = 40) -> PipelineReport:
        """Extract from the most recent articles.

        Args:
            limit: Articles to consider. Kept modest because local inference is
                seconds per article and this runs on a schedule, not on demand.

        Returns:
            The extraction statistics and how many proposals were newly stored.
        """
        known = list(await self._entities.list_all())
        documents = (
            await self._mongo.collection(Collection.NEWS_ARTICLES)
            .find({}, {"title": 1, "summary": 1, "content": 1, "url": 1})
            .sort("published_at", -1)
            .limit(limit)
            .to_list(length=limit)
        )

        extractor = RelationshipExtractor(llm=self._llm, entities=known)
        proposed, report = await extractor.extract(documents)

        by_slug = {entity.slug: entity.id for entity in known}
        rows: list[dict[str, object]] = [
            {
                "source_id": by_slug[proposal.source_slug],
                "target_id": by_slug[proposal.target_slug],
                "kind": proposal.kind,
                "confidence": proposal.confidence,
                "quote": proposal.quote,
                "document_id": proposal.document_id,
                "document_title": proposal.document_title,
                "document_url": proposal.document_url,
            }
            for proposal in proposed
            if proposal.source_slug in by_slug and proposal.target_slug in by_slug
        ]
        stored = await self._proposals.upsert_many(rows)

        logger.info("extraction_pipeline_complete", proposed=report.proposed, stored=stored)
        return PipelineReport(report=report, stored=stored)
