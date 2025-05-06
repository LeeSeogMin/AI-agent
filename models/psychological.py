#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Psychological analysis models for the psychological agent in the LangGraph 멀티 에이전트 시스템
Primarily focused on prompt-based psychological analysis approaches
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from models.base import IdentifiedModel


class EmotionCategory(str, Enum):
    """Categories of emotions for sentiment analysis"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class EmotionIntensity(str, Enum):
    """Intensity levels for emotions"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PersonalityDimension(str, Enum):
    """Big Five personality dimensions"""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class CommunicationStyle(str, Enum):
    """Communication styles for empathetic responses"""
    SUPPORTIVE = "supportive"
    ANALYTICAL = "analytical"
    DIRECTIVE = "directive"
    REFLECTIVE = "reflective"
    EXPRESSIVE = "expressive"
    SYSTEMATIC = "systematic"


class EmotionAnalysis(IdentifiedModel):
    """Emotion analysis results"""
    text: str  # The analyzed text
    dominant_emotion: EmotionCategory
    emotion_scores: Dict[EmotionCategory, float] = Field(default_factory=dict)
    intensity: EmotionIntensity
    confidence: float
    context_factors: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional detailed analysis
    emotional_triggers: List[str] = Field(default_factory=list)
    emotional_patterns: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v


class PersonalityAnalysis(IdentifiedModel):
    """Personality analysis based on text"""
    text: str  # The analyzed text
    big_five_scores: Dict[PersonalityDimension, float] = Field(default_factory=dict)
    dominant_traits: List[PersonalityDimension] = Field(default_factory=list)
    confidence: float
    
    # Optional additional personality frameworks
    mbti_type: Optional[str] = None  # e.g., "INTJ"
    disc_profile: Optional[Dict[str, float]] = None  # e.g., {"D": 0.7, "I": 0.2, ...}
    
    behavioral_observations: List[str] = Field(default_factory=list)
    communication_preferences: List[str] = Field(default_factory=list)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v


class BehavioralPattern(IdentifiedModel):
    """Behavioral pattern identified in text"""
    pattern_name: str
    description: str
    evidence: List[str] = Field(default_factory=list)
    frequency: float  # 0.0 to 1.0
    confidence: float
    
    implications: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    @validator('confidence', 'frequency')
    def validate_probability(cls, v):
        """Validate value is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError('Value must be between 0 and 1')
        return v


class EmpatheticResponse(IdentifiedModel):
    """Empathetic response generation"""
    original_text: str
    emotion_analysis: EmotionAnalysis
    personality_analysis: Optional[PersonalityAnalysis] = None
    
    response_style: CommunicationStyle
    response_text: str
    
    # Response characteristics
    empathy_level: float  # 0.0 to 1.0
    personalization_level: float  # 0.0 to 1.0
    effectiveness_score: Optional[float] = None  # To be filled after feedback
    
    # Rationale behind the response
    reasoning: Optional[str] = None
    targeted_emotions: List[EmotionCategory] = Field(default_factory=list)
    
    @validator('empathy_level', 'personalization_level')
    def validate_level(cls, v):
        """Validate level is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError('Level must be between 0 and 1')
        return v


class PsychologicalInsight(IdentifiedModel):
    """Psychological insight derived from analysis"""
    subject: str  # What/who the insight is about
    insight_text: str
    supporting_evidence: List[str] = Field(default_factory=list)
    confidence: float
    
    # Categorization
    categories: List[str] = Field(default_factory=list)  # e.g., "motivation", "communication", "stress"
    relevance_score: float  # 0.0 to 1.0
    
    # Application of the insight
    practical_applications: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    
    @validator('confidence', 'relevance_score')
    def validate_score(cls, v):
        """Validate score is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


class PromptTemplate(BaseModel):
    """Prompt template for psychological analysis"""
    name: str
    description: str
    template: str
    
    # Metadata for prompt usage
    analysis_type: str  # e.g., "emotion", "personality", "behavioral"
    parameters: List[str] = Field(default_factory=list)
    example_inputs: Dict[str, str] = Field(default_factory=dict)
    example_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    accuracy_score: Optional[float] = None
    consistency_score: Optional[float] = None
    
    def format(self, **kwargs) -> str:
        """Format the template with provided parameters"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(f"Missing required parameter: {missing_key}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt template: {str(e)}") 