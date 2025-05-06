from typing import Dict, List, Any, Optional
import json
from prompts.base import PromptTemplate, load_system_message

# Load the system message for Psychological Agent
SYSTEM_MESSAGE = load_system_message("psychological")

class PsychologicalTemplates:
    """
    Collection of templates for the Psychological Agent.
    """

    @staticmethod
    def emotional_analysis_template() -> PromptTemplate:
        """
        Template for performing emotional analysis on text.
        """
        template = """
        # 감정 분석 작업

        ## 분석 대상 텍스트
        {input_text}

        ## 요청 작업
        제공된 텍스트의 감정적 요소를 분석하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 주요 감정 및 강도
        2. 감정 변화 패턴
        3. 표현 방식 (명시적/암시적)
        4. 문화적/상황적 맥락 고려사항
        """

        return PromptTemplate(
            template=template,
            required_variables=["input_text"],
            output_format={
                "emotional_analysis": {
                    "primary_emotions": ["emotion_type"],
                    "emotional_intensity": {
                        "emotion_type": "high|medium|low"
                    },
                    "emotional_valence": "positive|neutral|negative",
                    "emotional_patterns": "text",
                    "expression_mode": "explicit|implicit|mixed"
                },
                "contextual_factors": ["contextual_factor"],
                "confidence": "high|medium|low",
                "limitations": ["limitation"]
            }
        )

    @staticmethod
    def personality_traits_template() -> PromptTemplate:
        """
        Template for identifying personality traits from text.
        """
        template = """
        # 성격 특성 식별 작업

        ## 분석 대상 텍스트
        {input_text}

        ## 추가 컨텍스트
        {additional_context}

        ## 요청 작업
        제공된 텍스트와 컨텍스트를 기반으로 성격 특성을 식별하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 주요 성격 특성 및 해당 근거
        2. 행동 패턴 및 선호도
        3. 성격 모델 프레임워크에 따른 분류(선택사항)
        4. 분석의 한계 및 불확실성
        """

        return PromptTemplate(
            template=template,
            required_variables=["input_text"],
            optional_variables=["additional_context"],
            output_format={
                "personality_traits": {
                    "dominant_traits": ["trait"],
                    "evidence": {
                        "trait": ["text_evidence"]
                    },
                    "behavioral_patterns": ["pattern"],
                    "preferences": ["preference"]
                },
                "framework_analysis": {
                    "model_name": "text",
                    "classifications": {
                        "dimension": "score_or_category"
                    }
                },
                "confidence": "high|medium|low",
                "limitations": ["limitation"]
            }
        )

    @staticmethod
    def communication_style_template() -> PromptTemplate:
        """
        Template for analyzing communication style.
        """
        template = """
        # 커뮤니케이션 스타일 분석 작업

        ## 분석 대상 텍스트
        {input_text}

        ## 상황적 컨텍스트
        {situational_context}

        ## 요청 작업
        제공된 텍스트의 커뮤니케이션 스타일을 분석하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 주요 커뮤니케이션 특성
        2. 명시적/암시적 메시지
        3. 설득 전략 및 영향력 패턴
        4. 대인관계 역학 패턴
        """

        return PromptTemplate(
            template=template,
            required_variables=["input_text"],
            optional_variables=["situational_context"],
            output_format={
                "communication_style": {
                    "primary_style": "text",
                    "directness": "direct|indirect|mixed",
                    "formality": "formal|casual|mixed",
                    "assertiveness": "high|medium|low",
                    "openness": "high|medium|low"
                },
                "message_patterns": {
                    "explicit_messages": ["message"],
                    "implicit_messages": ["message"]
                },
                "persuasion_strategies": ["strategy"],
                "interpersonal_dynamics": "text",
                "recommendations": ["recommendation"],
                "limitations": ["limitation"]
            }
        )

    @staticmethod
    def empathetic_response_template() -> PromptTemplate:
        """
        Template for generating empathetic responses.
        """
        template = """
        # 공감적 응답 생성 작업

        ## 원본 텍스트
        {input_text}

        ## 감정 분석 결과
        {emotional_analysis}

        ## 대상 독자
        {target_audience}

        ## 요청 작업
        원본 텍스트 및 감정 분석을 기반으로 공감적 응답을 생성하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 공감적 응답 텍스트
        2. 적용된 공감 기법
        3. 고려된 심리적 요소
        4. 효과적인 후속 대응에 대한 제안
        """

        return PromptTemplate(
            template=template,
            required_variables=["input_text", "emotional_analysis"],
            optional_variables=["target_audience"],
            output_format={
                "empathetic_response": "text",
                "empathy_techniques": ["technique"],
                "psychological_considerations": ["consideration"],
                "follow_up_suggestions": ["suggestion"],
                "tone": "warm|neutral|professional|encouraging|supportive"
            }
        )

    @staticmethod
    def behavioral_pattern_template() -> PromptTemplate:
        """
        Template for identifying behavioral patterns.
        """
        template = """
        # 행동 패턴 식별 작업

        ## 분석 대상 텍스트/데이터
        {input_data}

        ## 시간적 컨텍스트
        {temporal_context}

        ## 요청 작업
        제공된 데이터에서 반복적인 행동 패턴을 식별하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 식별된 주요 행동 패턴
        2. 패턴의 트리거 및 강화 요인
        3. 패턴의 기능적 의미
        4. 관련 심리학적 이론 또는 모델
        """

        return PromptTemplate(
            template=template,
            required_variables=["input_data"],
            optional_variables=["temporal_context"],
            output_format={
                "behavioral_patterns": [
                    {
                        "pattern": "text",
                        "evidence": ["evidence"],
                        "triggers": ["trigger"],
                        "reinforcers": ["reinforcer"],
                        "functional_meaning": "text"
                    }
                ],
                "psychological_framework": {
                    "relevant_theories": ["theory"],
                    "explanation": "text"
                },
                "confidence": "high|medium|low",
                "limitations": ["limitation"],
                "recommendations": ["recommendation"]
            }
        )

    @staticmethod
    def cultural_consideration_template() -> PromptTemplate:
        """
        Template for analyzing cultural aspects in psychological context.
        """
        template = """
        # 문화적 고려사항 분석 작업

        ## 분석 대상 텍스트
        {input_text}

        ## 문화적 컨텍스트
        {cultural_context}

        ## 요청 작업
        제공된 텍스트의 문화적 요소와 그 심리적 영향을 분석하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 식별된 문화적 요소
        2. 문화적 요소의 심리적 영향
        3. 문화 간 차이점 및 유사점
        4. 다문화적 관점에서의 권장사항
        """

        return PromptTemplate(
            template=template,
            required_variables=["input_text"],
            optional_variables=["cultural_context"],
            output_format={
                "cultural_elements": ["element"],
                "psychological_influences": [
                    {
                        "cultural_element": "text",
                        "psychological_impact": "text",
                        "evidence": "text"
                    }
                ],
                "cross_cultural_perspective": {
                    "similarities": ["similarity"],
                    "differences": ["difference"],
                    "universal_aspects": ["aspect"]
                },
                "recommendations": ["recommendation"],
                "limitations": ["limitation"]
            }
        )

psychological_templates = {
    "emotional_analysis": PsychologicalTemplates.emotional_analysis_template(),
    "personality_traits": PsychologicalTemplates.personality_traits_template(),
    "communication_style": PsychologicalTemplates.communication_style_template(),
    "empathetic_response": PsychologicalTemplates.empathetic_response_template(),
    "behavioral_pattern": PsychologicalTemplates.behavioral_pattern_template(),
    "cultural_consideration": PsychologicalTemplates.cultural_consideration_template()
} 