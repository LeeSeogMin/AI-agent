#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base data models for the LangGraph 멀티 에이전트 시스템
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class Priority(str, Enum):
    """Message priority levels"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class MessageType(str, Enum):
    """Message types for agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class AgentType(str, Enum):
    """Types of agents in the system"""
    SUPERVISOR = "supervisor"
    RAG = "rag"
    WEB_SEARCH = "web_search"
    DATA_ANALYSIS = "data_analysis"
    PSYCHOLOGICAL = "psychological"
    REPORT_WRITER = "report_writer"


class TaskStatus(str, Enum):
    """Status values for tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    """Stage of query processing"""
    PLANNING = "planning"
    INFORMATION_GATHERING = "information_gathering"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REPORTING = "reporting"


class ContentType(str, Enum):
    """Types of content in the knowledge base"""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    MIXED = "mixed"


class SourceType(str, Enum):
    """Types of information sources"""
    WEB = "web"
    INTERNAL = "internal"
    USER = "user"
    GENERATED = "generated"


class TemporalRelevance(str, Enum):
    """Temporal relevance of information"""
    STATIC = "static"
    CURRENT = "current"
    HISTORICAL = "historical"


class VerificationStatus(str, Enum):
    """Verification status of information"""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"


class TimestampedModel(BaseModel):
    """Base model with creation and update timestamps"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class IdentifiedModel(TimestampedModel):
    """Base model with ID and timestamps"""
    id: UUID = Field(default_factory=uuid4) 