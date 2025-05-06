#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge base models for the LangGraph 멀티 에이전트 시스템
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, root_validator

from models.base import (ContentType, IdentifiedModel, SourceType,
                       TemporalRelevance, VerificationStatus)


class Source(BaseModel):
    """Information source metadata"""
    type: SourceType
    url: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None


class TemporalInfo(BaseModel):
    """Temporal information about knowledge"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    temporal_relevance: TemporalRelevance = TemporalRelevance.CURRENT


class QualityMetrics(BaseModel):
    """Quality and reliability metrics for knowledge"""
    reliability_score: float = 0.7  # Default reasonable reliability
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_method: Optional[str] = None


class Relationships(BaseModel):
    """Relationships between knowledge chunks"""
    parent_id: Optional[str] = None
    related_ids: List[str] = Field(default_factory=list)
    follows: Optional[str] = None
    precedes: Optional[str] = None


class UsageStats(BaseModel):
    """Usage statistics for knowledge chunks"""
    retrieval_count: int = 0
    relevance_feedback: float = 0.5  # Default neutral feedback
    last_accessed: Optional[datetime] = None


class KnowledgeChunk(IdentifiedModel):
    """
    A chunk of knowledge stored in the system
    Based on the metadata schema from docs/지식베이스명세서.md
    """
    document_id: str
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    content: str
    embedding: Optional[List[float]] = None
    
    source: Source
    temporal: TemporalInfo = Field(default_factory=TemporalInfo)
    content_type: ContentType = ContentType.TEXT
    domain: List[str] = Field(default_factory=list)
    language: str = "ko"  # Default to Korean
    quality: QualityMetrics = Field(default_factory=QualityMetrics)
    relationships: Relationships = Field(default_factory=Relationships)
    usage_stats: UsageStats = Field(default_factory=UsageStats)
    
    @root_validator
    def update_last_modified(cls, values):
        """Update the last modified timestamp"""
        values["temporal"].updated_at = datetime.utcnow()
        return values


class Document(IdentifiedModel):
    """A complete document which can be chunked for the knowledge base"""
    title: str
    content: str
    source: Source
    language: str = "ko"
    content_type: ContentType = ContentType.TEXT
    domain: List[str] = Field(default_factory=list)
    
    # Document-specific metadata
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    
    # Processing status
    is_processed: bool = False
    chunks: List[str] = Field(default_factory=list)  # IDs of generated chunks


class VectorDBCollection(BaseModel):
    """Configuration for a vector database collection"""
    name: str
    description: str
    embedding_model: str = "text-embedding-3-large"
    dimensions: int = 1536
    metric: str = "cosine"
    
    class Config:
        arbitrary_types_allowed = True


class KnowledgeBaseConfig(BaseModel):
    """Configuration for the knowledge base system"""
    collections: List[VectorDBCollection] = Field(default_factory=list)
    default_chunk_size: int = 400  # Default token size for chunks
    default_chunk_overlap: int = 50  # Default token overlap
    
    # Default collections if not specified
    @root_validator(pre=True)
    def set_default_collections(cls, values):
        if "collections" not in values or not values["collections"]:
            values["collections"] = [
                VectorDBCollection(
                    name="general_knowledge",
                    description="일반적인 사실 및 개념"
                ),
                VectorDBCollection(
                    name="domain_knowledge",
                    description="도메인별 전문 지식"
                ),
                VectorDBCollection(
                    name="user_data",
                    description="사용자 제공 데이터 및 분석"
                ),
                VectorDBCollection(
                    name="generated_content",
                    description="시스템 생성 결과물"
                )
            ]
        return values


class SearchQuery(BaseModel):
    """A search query for the knowledge base"""
    query_text: str
    top_k: int = 5
    similarity_threshold: float = 0.7
    filter_criteria: Dict[str, Any] = Field(default_factory=dict)
    collections: List[str] = Field(default=["general_knowledge", "domain_knowledge"])


class SearchResult(BaseModel):
    """A search result from the knowledge base"""
    chunks: List[KnowledgeChunk]
    similarity_scores: List[float]
    total_results: int
    query: str
    search_time: float  # in seconds
    collections_searched: List[str] 