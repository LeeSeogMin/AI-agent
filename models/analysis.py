#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data analysis models for the data analysis agent in the LangGraph 멀티 에이전트 시스템
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from models.base import IdentifiedModel


class DataType(str, Enum):
    """Types of data for analysis"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    BOOLEAN = "boolean"
    MIXED = "mixed"


class AnalysisType(str, Enum):
    """Types of analysis that can be performed"""
    DESCRIPTIVE = "descriptive"
    EXPLORATORY = "exploratory"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


class VisualizationType(str, Enum):
    """Types of visualizations that can be generated"""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    PIE_CHART = "pie_chart"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"
    NETWORK_GRAPH = "network_graph"
    WORD_CLOUD = "word_cloud"
    CUSTOM = "custom"


class Dataset(IdentifiedModel):
    """A dataset for analysis"""
    name: str
    description: Optional[str] = None
    source: str
    # Data storage is flexible - could be file path, API reference, or actual data
    data_location: str
    schema: Dict[str, DataType] = Field(default_factory=dict)
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    time_period: Optional[str] = None
    
    # Metadata for the dataset
    tags: List[str] = Field(default_factory=list)
    has_missing_values: bool = False
    has_outliers: bool = False
    is_preprocessed: bool = False


class DataColumn(BaseModel):
    """Information about a column in a dataset"""
    name: str
    data_type: DataType
    description: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    missing_count: int = 0
    unique_count: Optional[int] = None


class AnalysisRequest(BaseModel):
    """A request for data analysis"""
    dataset_id: UUID
    analysis_type: AnalysisType
    target_columns: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    visualizations: List[VisualizationType] = Field(default_factory=list)
    description: Optional[str] = None


class AnalysisResult(IdentifiedModel):
    """Results of a data analysis"""
    dataset_id: UUID
    analysis_type: AnalysisType
    target_columns: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Results
    summary: Dict[str, Any] = Field(default_factory=dict)
    statistical_results: Dict[str, Any] = Field(default_factory=dict)
    insights: List[str] = Field(default_factory=list)
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    execution_time: float  # in seconds
    status: str = "completed"
    error_message: Optional[str] = None


class Visualization(IdentifiedModel):
    """A data visualization"""
    title: str
    visualization_type: VisualizationType
    dataset_id: UUID
    columns_used: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # The actual visualization data or reference
    plot_data: Dict[str, Any] = Field(default_factory=dict)
    image_path: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    insights: List[str] = Field(default_factory=list)


class DataInsight(IdentifiedModel):
    """An insight derived from data analysis"""
    dataset_id: UUID
    analysis_id: UUID
    description: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0  # 0.0 to 1.0
    
    # Categorization
    insight_type: str  # e.g., "trend", "anomaly", "correlation"
    importance: int = 1  # 1 to 5, with 5 being most important
    
    # Related visualizations
    visualization_ids: List[UUID] = Field(default_factory=list)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v
    
    @validator('importance')
    def validate_importance(cls, v):
        """Validate importance is between 1 and 5"""
        if not 1 <= v <= 5:
            raise ValueError('Importance must be between 1 and 5')
        return v 