#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Report generation models for the report writer agent in the LangGraph 멀티 에이전트 시스템
Primarily focused on prompt-based report generation approaches
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from models.base import IdentifiedModel


class ReportFormat(str, Enum):
    """Supported report output formats"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PDF = "pdf"


class ReportSection(str, Enum):
    """Standard report sections"""
    EXECUTIVE_SUMMARY = "executive_summary"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODOLOGY = "methodology"
    FINDINGS = "findings"
    ANALYSIS = "analysis"
    DISCUSSION = "discussion"
    RECOMMENDATIONS = "recommendations"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"


class ReportType(str, Enum):
    """Types of reports that can be generated"""
    ANALYSIS_REPORT = "analysis_report"
    RESEARCH_SUMMARY = "research_summary"
    TECHNICAL_DOCUMENT = "technical_document"
    INSIGHTS_BRIEF = "insights_brief"
    RECOMMENDATION_MEMO = "recommendation_memo"
    STATUS_UPDATE = "status_update"
    CUSTOM = "custom"


class AudienceLevel(str, Enum):
    """Target audience expertise levels"""
    GENERAL = "general"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    TECHNICAL = "technical"
    EXECUTIVE = "executive"


class ContentElement(BaseModel):
    """A content element in a report"""
    element_type: str  # e.g., "text", "image", "table", "chart", "code"
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReportSectionContent(IdentifiedModel):
    """Content of a report section"""
    section_type: ReportSection
    title: str
    order: int
    content_elements: List[ContentElement] = Field(default_factory=list)
    
    # Section-specific data
    key_points: List[str] = Field(default_factory=list)
    references: List[Dict[str, str]] = Field(default_factory=list)
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add text content to the section"""
        self.content_elements.append(
            ContentElement(
                element_type="text",
                content=text,
                metadata=metadata or {}
            )
        )
    
    def add_image(self, image_path: str, caption: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add image content to the section"""
        self.content_elements.append(
            ContentElement(
                element_type="image",
                content={"path": image_path, "caption": caption},
                metadata=metadata or {}
            )
        )
    
    def add_table(self, table_data: Dict[str, Any], caption: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add table content to the section"""
        self.content_elements.append(
            ContentElement(
                element_type="table",
                content={"data": table_data, "caption": caption},
                metadata=metadata or {}
            )
        )


class ReportTemplate(BaseModel):
    """Template for a report"""
    name: str
    description: str
    report_type: ReportType
    sections: List[ReportSection]
    default_format: ReportFormat = ReportFormat.MARKDOWN
    
    # Style guidance
    style_guide: Dict[str, Any] = Field(default_factory=dict)
    audience_level: AudienceLevel = AudienceLevel.GENERAL
    
    # Prompt templates for each section
    section_prompts: Dict[ReportSection, str] = Field(default_factory=dict)
    
    def get_section_prompt(self, section: ReportSection) -> Optional[str]:
        """Get the prompt template for a section"""
        return self.section_prompts.get(section)


class Report(IdentifiedModel):
    """A complete report"""
    title: str
    report_type: ReportType
    format: ReportFormat
    audience_level: AudienceLevel
    
    # Report metadata
    author: str = "AI Agent System"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    
    # Report content
    sections: Dict[ReportSection, ReportSectionContent] = Field(default_factory=dict)
    
    # References and sources
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[Dict[str, str]] = Field(default_factory=list)
    
    # Output information
    output_path: Optional[str] = None
    output_size: Optional[int] = None  # in bytes
    
    def add_section(self, section: ReportSectionContent) -> None:
        """Add a section to the report"""
        self.sections[section.section_type] = section
    
    def get_section(self, section_type: ReportSection) -> Optional[ReportSectionContent]:
        """Get a section from the report"""
        return self.sections.get(section_type)


class ReportGenerationRequest(BaseModel):
    """Request to generate a report"""
    title: str
    report_type: ReportType
    format: ReportFormat = ReportFormat.MARKDOWN
    audience_level: AudienceLevel = AudienceLevel.GENERAL
    
    # Content sources
    analysis_results: List[Dict[str, Any]] = Field(default_factory=list)
    psychological_insights: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_sources: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Report specifications
    required_sections: List[ReportSection] = Field(default_factory=list)
    max_length: Optional[int] = None  # in words
    emphasis_areas: List[str] = Field(default_factory=list)
    
    # Special requirements
    special_instructions: Optional[str] = None


class PromptTemplateLibrary(BaseModel):
    """Library of prompt templates for report generation"""
    report_templates: Dict[ReportType, ReportTemplate] = Field(default_factory=dict)
    section_templates: Dict[ReportSection, List[str]] = Field(default_factory=dict)
    
    # Special purpose templates
    summarization_templates: List[str] = Field(default_factory=list)
    insight_extraction_templates: List[str] = Field(default_factory=list)
    recommendation_templates: List[str] = Field(default_factory=list)
    
    def get_report_template(self, report_type: ReportType) -> Optional[ReportTemplate]:
        """Get a report template by type"""
        return self.report_templates.get(report_type)
    
    def get_section_templates(self, section: ReportSection) -> List[str]:
        """Get templates for a specific section"""
        return self.section_templates.get(section, []) 