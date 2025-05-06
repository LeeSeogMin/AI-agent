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
        한국 노인 정신건강 분석 인사이트 제공
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
        
        Returns:
            Dict[str, Any]: 정신건강 분석 인사이트
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "mental_health_indicators": {"type": "string"},
                "cognitive_function": {"type": "string"},
                "somatic_expressions": {"type": "string"},
                "emotional_state": {"type": "string"},
                "social_connection": {"type": "string"},
                "role_identity": {"type": "string"},
                "family_dynamics": {"type": "string"},
                "spiritual_wellbeing": {"type": "string"},
                "coping_mechanisms": {"type": "string"},
                "support_resources": {"type": "string"},
                "risk_factors": {"type": "array", "items": {"type": "string"}},
                "protective_factors": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "clinical_considerations": {"type": "string"}
            },
            "required": [
                "mental_health_indicators", 
                "emotional_state", 
                "social_connection",
                "family_dynamics",
                "risk_factors",
                "protective_factors",
                "recommendations"
            ]
        }
        
        # 프롬프트 준비
        prompt = f"""다음 텍스트를 분석하여 한국 노인의 정신건강 상태에 대한 인사이트를 제공하세요:

텍스트: "{query}"
컨텍스트: "{context}"

다음 측면을 포함하세요:
1. 정신건강 지표: 우울, 불안, 외로움 등 정신건강 문제의 잠재적 신호 분석
2. 인지기능: 의사소통 패턴에서 나타나는 인지기능 관련 관찰 (있는 경우)
3. 신체화 표현: 신체 증상(두통, 소화불량, 수면장애 등)으로 표현되는 정신건강 호소
4. 정서 상태: 표현된 감정과 정서적 건강 상태 평가
5. 사회적 연결: 사회적 관계, 고립감, 소속감 분석
6. 역할 정체성: 은퇴, 역할 상실, 사회적 위치 변화와 관련된 정신건강 이슈
7. 가족 역학: 가족 관계, 부양 부담, 세대 간 갈등이 정신건강에 미치는 영향
8. 영적 웰빙: 종교, 의미, 목적 등 영적 측면이 정신건강에 미치는 영향
9. 대처 메커니즘: 스트레스나 어려움에 대응하는 방식과 회복력
10. 지지 자원: 활용 가능하거나 부족한 지지체계와 서비스
11. 위험요인: 정신건강 악화 가능성이 있는 위험요인 목록
12. 보호요인: 정신건강을 지키는 데 도움이 되는 요소 목록
13. 권장사항: 정신건강 지원을 위한 비임상적 제안
14. 임상적 고려사항: 전문가 상담이 필요할 수 있는 징후 (진단이 아닌 참고 정보로만 제공)

분석 시 다음 사항을 고려하세요:
- 한국 노인은 정신건강 문제를 직접적으로 표현하기보다 신체 증상이나 상황적 어려움으로 표현하는 경향이 있습니다
- 문화적으로 정신건강 문제에 대한 낙인이 있어 우회적 표현을 사용할 수 있습니다
- 세대 차이와 개인차를 인식하고, 노인에 대한 고정관념을 피하세요
- 분석은 정보 제공 목적이며, 전문적 진단이나 치료를 대체할 수 없음을 명시하세요

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
