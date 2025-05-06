#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 심리 분석 에이전트 모듈
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union

from agents.base import Agent
from core.config import get_config
from core.error_handler import handle_agent_error, TaskExecutionError
from services.llm_service import generate_text, generate_json

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

class PsychologicalAgent(Agent):
    """심리 분석 에이전트 클래스"""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        system_message: Optional[str] = None
    ):
        """
        심리 분석 에이전트 초기화
        
        Args:
            agent_id: 에이전트 ID (없으면 자동 생성)
            system_message: 시스템 메시지
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="psychological",
            system_message=system_message
        )
        
        # 시스템 메시지 로드
        if not self.system_message:
            self.system_message = self.load_system_message()
        
        # 작업 핸들러 등록
        self._register_task_handlers()
        
        logger.info(f"Psychological agent initialized with ID: {self.agent_id}")
    
    def _register_task_handlers(self) -> None:
        """작업 핸들러 등록"""
        self.task_handlers = {
            "process_task": self._handle_process_task,
            "analyze_personality": self._handle_analyze_personality,
            "analyze_sentiment": self._handle_analyze_sentiment,
            "analyze_cognitive_biases": self._handle_analyze_cognitive_biases,
            "provide_psychological_insight": self._handle_provide_psychological_insight
        }
    
    @handle_agent_error()
    def _handle_process_task(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        작업 처리 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        task_description = message_content.get("data", {}).get("task_description", "")
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for psychological analysis",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 분석 유형 결정
        analysis_type = self._determine_analysis_type(query, task_description)
        
        result = {}
        
        # 분석 유형에 따른 처리
        if analysis_type == "personality":
            result = self._analyze_personality(query)
        elif analysis_type == "sentiment":
            result = self._analyze_sentiment(query)
        elif analysis_type == "cognitive_biases":
            result = self._analyze_cognitive_biases(query)
        else:
            # 일반적인 심리학적 인사이트
            result = self._provide_psychological_insight(query, task_description)
        
        return {
            "status": "success",
            "result": result
        }
    
    @handle_agent_error()
    def _handle_analyze_personality(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        성격 분석 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        text = message_content.get("data", {}).get("text", "")
        
        if not text:
            raise TaskExecutionError(
                message="No text provided for personality analysis",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 성격 분석 수행
        result = self._analyze_personality(text)
        
        return {
            "status": "success",
            "result": result
        }
    
    @handle_agent_error()
    def _handle_analyze_sentiment(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        감정 분석 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        text = message_content.get("data", {}).get("text", "")
        
        if not text:
            raise TaskExecutionError(
                message="No text provided for sentiment analysis",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 감정 분석 수행
        result = self._analyze_sentiment(text)
        
        return {
            "status": "success",
            "result": result
        }
    
    @handle_agent_error()
    def _handle_analyze_cognitive_biases(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        인지 편향 분석 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        text = message_content.get("data", {}).get("text", "")
        
        if not text:
            raise TaskExecutionError(
                message="No text provided for cognitive bias analysis",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 인지 편향 분석 수행
        result = self._analyze_cognitive_biases(text)
        
        return {
            "status": "success",
            "result": result
        }
    
    @handle_agent_error()
    def _handle_provide_psychological_insight(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        심리학적 인사이트 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        context = message_content.get("data", {}).get("context", "")
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for psychological insight",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 심리학적 인사이트 제공
        result = self._provide_psychological_insight(query, context)
        
        return {
            "status": "success",
            "result": result
        }
    
    def _determine_analysis_type(self, query: str, task_description: str) -> str:
        """
        쿼리 분석을 통한 분석 유형 결정
        
        Args:
            query: 사용자 쿼리
            task_description: 작업 설명
        
        Returns:
            str: 분석 유형 (personality, sentiment, cognitive_biases, general)
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": ["personality", "sentiment", "cognitive_biases", "general"]
                },
                "reasoning": {"type": "string"}
            },
            "required": ["analysis_type", "reasoning"]
        }
        
        # 프롬프트 준비
        prompt = f"""사용자의 쿼리를 분석하여 가장 적합한 심리 분석 유형을 결정하세요:

사용자 쿼리: "{query}"
작업 설명: "{task_description}"

다음 분석 유형 중에서 선택하세요:
1. personality: 성격 특성 분석이 필요한 경우
2. sentiment: 감정 또는 정서 분석이 필요한 경우
3. cognitive_biases: 인지 편향 또는 사고 패턴 분석이 필요한 경우
4. general: 일반적인 심리학적 인사이트가 필요한 경우

JSON 스키마에 맞춰 응답하세요."""
        
        # LLM으로 분석 유형 결정
        try:
            analysis_type_result = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            analysis_type = analysis_type_result.get("analysis_type", "general")
            logger.info(f"Determined analysis type: {analysis_type}")
            return analysis_type
            
        except Exception as e:
            logger.error(f"Error determining analysis type: {str(e)}")
            return "general"
    
    def _analyze_personality(self, text: str) -> Dict[str, Any]:
        """
        성격 분석 수행
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            Dict[str, Any]: 성격 분석 결과
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "big_five": {
                    "type": "object",
                    "properties": {
                        "openness": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number", "minimum": 0, "maximum": 100},
                                "explanation": {"type": "string"}
                            }
                        },
                        "conscientiousness": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number", "minimum": 0, "maximum": 100},
                                "explanation": {"type": "string"}
                            }
                        },
                        "extraversion": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number", "minimum": 0, "maximum": 100},
                                "explanation": {"type": "string"}
                            }
                        },
                        "agreeableness": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number", "minimum": 0, "maximum": 100},
                                "explanation": {"type": "string"}
                            }
                        },
                        "neuroticism": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number", "minimum": 0, "maximum": 100},
                                "explanation": {"type": "string"}
                            }
                        }
                    }
                },
                "personality_insights": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "areas_for_growth": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["big_five", "personality_insights", "strengths", "areas_for_growth"]
        }
        
        # 프롬프트 준비
        prompt = f"""다음 텍스트를 분석하여 BIG 5 성격 특성 모델에 따른 성격 분석을 제공하세요:

텍스트: "{text}"

분석 결과에는 다음을 포함하세요:
1. BIG 5 성격 특성 각각에 대한 0-100 점수 및 설명:
   - 개방성 (Openness)
   - 성실성 (Conscientiousness)
   - 외향성 (Extraversion)
   - 친화성 (Agreeableness)
   - 신경증 (Neuroticism)
2. 주요 성격 인사이트 목록
3. 강점 목록
4. 개선 가능한 영역 목록

제공된 정보가 제한적이므로, 텍스트에서 추론할 수 있는 내용에 기반하여 분석하세요.
확실하지 않은 부분에 대해서는 적절히 설명하세요.

JSON 스키마에 맞춰 응답하세요."""
        
        # LLM으로 성격 분석
        try:
            personality_analysis = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info("Personality analysis completed")
            return personality_analysis
            
        except Exception as e:
            logger.error(f"Error in personality analysis: {str(e)}")
            # 기본 분석 결과 반환
            return {
                "big_five": {
                    "openness": {"score": 50, "explanation": "정보가 부족하여 정확한 평가가 어렵습니다."},
                    "conscientiousness": {"score": 50, "explanation": "정보가 부족하여 정확한 평가가 어렵습니다."},
                    "extraversion": {"score": 50, "explanation": "정보가 부족하여 정확한 평가가 어렵습니다."},
                    "agreeableness": {"score": 50, "explanation": "정보가 부족하여 정확한 평가가 어렵습니다."},
                    "neuroticism": {"score": 50, "explanation": "정보가 부족하여 정확한 평가가 어렵습니다."}
                },
                "personality_insights": ["정보가 부족하여 정확한 인사이트를 제공하기 어렵습니다."],
                "strengths": ["정보가 부족하여 강점을 파악하기 어렵습니다."],
                "areas_for_growth": ["정보가 부족하여 개선 영역을 파악하기 어렵습니다."]
            }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        감정 분석 수행
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            Dict[str, Any]: 감정 분석 결과
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "sentiment_score": {"type": "number", "minimum": -1, "maximum": 1},
                "primary_emotion": {"type": "string"},
                "secondary_emotions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "emotional_valence": {"type": "string"},
                "emotional_arousal": {"type": "string"},
                "sentiment_analysis": {"type": "string"}
            },
            "required": ["sentiment_score", "primary_emotion", "secondary_emotions", "sentiment_analysis"]
        }
        
        # 프롬프트 준비
        prompt = f"""다음 텍스트에 대해 감정 분석을 수행하세요:

텍스트: "{text}"

분석 결과에는 다음을 포함하세요:
1. 감정 점수: -1(매우 부정적)에서 1(매우 긍정적) 사이의 값
2. 주요 감정: 가장 두드러진 감정 하나
3. 부수적 감정: 함께 나타나는 부수적 감정들
4. 감정의 원인: 이 감정들이 발생한 가능한 원인
5. 감정적 균형: 높음(긍정적), 중립, 낮음(부정적) 중 하나
6. 감정적 활성화: 높음(활성화된), 중간, 낮음(차분한) 중 하나
7. 전반적인 감정 분석: 텍스트에서 표현된 전반적인 감정 상태에 대한 설명

제공된 정보가 제한적이므로, 텍스트에서 추론할 수 있는 내용에 기반하여 분석하세요.

JSON 스키마에 맞춰 응답하세요."""
        
        # LLM으로 감정 분석
        try:
            sentiment_analysis = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info("Sentiment analysis completed")
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            # 기본 분석 결과 반환
            return {
                "sentiment_score": 0.0,
                "primary_emotion": "중립",
                "secondary_emotions": [],
                "emotional_valence": "중립",
                "emotional_arousal": "중간",
                "sentiment_analysis": "텍스트에서 감정을 분석하기 위한 충분한 정보가 없습니다."
            }
    
    def _analyze_cognitive_biases(self, text: str) -> Dict[str, Any]:
        """
        인지 편향 분석 수행
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            Dict[str, Any]: 인지 편향 분석 결과
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "identified_biases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bias_name": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "explanation": {"type": "string"},
                            "evidence": {"type": "string"}
                        }
                    }
                },
                "thinking_patterns": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "decision_making_analysis": {"type": "string"},
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["identified_biases", "thinking_patterns", "decision_making_analysis"]
        }
        
        # 프롬프트 준비
        prompt = f"""다음 텍스트를 분석하여 인지 편향과 사고 패턴을 식별하세요:

텍스트: "{text}"

분석 결과에는 다음을 포함하세요:
1. 식별된 인지 편향 목록 (각각에 대해 편향 이름, 확신도, 설명, 텍스트에서의 증거 포함)
2. 식별된 사고 패턴 목록
3. 의사 결정 프로세스 분석
4. 더 균형 잡힌 사고를 위한 권장사항 목록

흔한 인지 편향에는 다음이 포함됩니다:
- 확증 편향 (Confirmation Bias)
- 가용성 휴리스틱 (Availability Heuristic)
- 선택적 인지 (Selective Perception)
- 기본값 효과 (Default Effect)
- 프레이밍 효과 (Framing Effect)
- 집단 사고 (Groupthink)
- 후광 효과 (Halo Effect)
- 과신 (Overconfidence Bias)
- 자기 중심적 편향 (Self-serving Bias)
- 현상 유지 편향 (Status Quo Bias)

제공된 정보가 제한적이므로, 텍스트에서 추론할 수 있는 내용에 기반하여 분석하세요.

JSON 스키마에 맞춰 응답하세요."""
        
        # LLM으로 인지 편향 분석
        try:
            bias_analysis = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info("Cognitive bias analysis completed")
            return bias_analysis
            
        except Exception as e:
            logger.error(f"Error in cognitive bias analysis: {str(e)}")
            # 기본 분석 결과 반환
            return {
                "identified_biases": [],
                "thinking_patterns": ["정보가 부족하여 사고 패턴을 식별하기 어렵습니다."],
                "decision_making_analysis": "제공된 텍스트에서는 의사 결정 과정을 분석하기 위한 충분한 정보가 없습니다.",
                "recommendations": ["더 많은 정보가 필요합니다."]
            }
    
    def _provide_psychological_insight(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        일반적인 심리학적 인사이트 제공
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
        
        Returns:
            Dict[str, Any]: 심리학적 인사이트
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "psychological_perspective": {"type": "string"},
                "behavioral_insights": {"type": "string"},
                "cognitive_aspects": {"type": "string"},
                "emotional_dimensions": {"type": "string"},
                "developmental_factors": {"type": "string"},
                "social_implications": {"type": "string"},
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "research_foundation": {"type": "string"}
            },
            "required": [
                "psychological_perspective", 
                "behavioral_insights", 
                "cognitive_aspects", 
                "emotional_dimensions"
            ]
        }
        
        # 프롬프트 준비
        prompt = f"""다음 쿼리에 대해 심리학적 관점에서 인사이트를 제공하세요:

쿼리: "{query}"
컨텍스트: "{context}"

다음 측면을 포함하세요:
1. 심리학적 관점: 주요 심리학 이론 관점에서의 해석
2. 행동적 인사이트: 행동 패턴이나 동기에 대한 분석
3. 인지적 측면: 사고 과정, 인식, 신념에 대한 분석
4. 정서적 측면: 감정적 요소와 그 영향
5. 발달적 요인: 발달 심리학 관점에서의 분석 (관련된 경우)
6. 사회적 함의: 대인 관계 또는 사회적 맥락에서의 의미 (관련된 경우)
7. 권장사항: 심리학적 관점에서의 실용적인 제안
8. 연구 기반: 관련 심리학 연구나 이론에 대한 간략한 언급

심리학적으로 정확하고 과학적으로 타당한 인사이트를 제공하세요.
확실하지 않은 부분에 대해서는 그 한계를 인정하세요.

JSON 스키마에 맞춰 응답하세요."""
        
        # LLM으로 심리학적 인사이트 생성
        try:
            insights = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info("Psychological insights generated")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating psychological insights: {str(e)}")
            # 기본 인사이트 반환
            return {
                "psychological_perspective": "제공된 정보로는 심층적인 심리학적 분석을 제공하기 어렵습니다.",
                "behavioral_insights": "행동 패턴을 분석하기 위한 충분한 정보가 없습니다.",
                "cognitive_aspects": "인지적 측면을 분석하기 위한 충분한 정보가 없습니다.",
                "emotional_dimensions": "정서적 측면을 분석하기 위한 충분한 정보가 없습니다.",
                "recommendations": ["더 구체적인 정보가 필요합니다."]
            }

    @handle_agent_error()
    def process_user_query(self, query: str, task_description: str = "") -> Dict[str, Any]:
        """
        사용자 쿼리 처리를 위한 추상 메소드 구현
        
        Args:
            query: 사용자 쿼리
            task_description: 작업 설명
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            logger.info(f"Processing user query: {query[:50]}...")
            
            # 분석 유형 결정
            analysis_type = self._determine_analysis_type(query, task_description)
            
            result = {}
            
            # 분석 유형에 따른 처리
            if analysis_type == "personality":
                result = self._analyze_personality(query)
            elif analysis_type == "sentiment":
                result = self._analyze_sentiment(query)
            elif analysis_type == "cognitive_biases":
                result = self._analyze_cognitive_biases(query)
            else:
                # 일반적인 심리학적 인사이트
                result = self._provide_psychological_insight(query, task_description)
            
            return {
                "status": "success",
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Error processing user query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }


# 전역 심리 분석 에이전트 인스턴스
psychological_agent = PsychologicalAgent()

def get_psychological_agent() -> PsychologicalAgent:
    """심리 분석 에이전트 인스턴스 가져오기"""
    return psychological_agent
