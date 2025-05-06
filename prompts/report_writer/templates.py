from typing import Dict, List, Any, Optional
import json
from prompts.base import PromptTemplate, load_system_message

# Load the system message for Report Writer Agent
SYSTEM_MESSAGE = load_system_message("report_writer")

class ReportWriterTemplates:
    """
    Collection of templates for the Report Writer Agent.
    """

    @staticmethod
    def report_structure_template() -> PromptTemplate:
        """
        Template for planning the structure of a report.
        """
        template = """
        # 보고서 구조 설계 작업

        ## 보고서 주제
        {report_topic}

        ## 가용 정보
        {available_information}

        ## 대상 독자
        {target_audience}

        ## 요청 작업
        제공된 주제와 정보를 기반으로 보고서의 논리적 구조를 설계하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 보고서 제목
        2. 섹션 및 하위 섹션 구조
        3. 각 섹션의 주요 포함 내용
        4. 필요한 시각적 요소
        """

        return PromptTemplate(
            template=template,
            required_variables=["report_topic", "available_information"],
            optional_variables=["target_audience"],
            output_format={
                "report_title": "text",
                "structure": [
                    {
                        "section_title": "text",
                        "level": "int",
                        "key_content": ["content_point"],
                        "subsections": [
                            {
                                "section_title": "text",
                                "level": "int",
                                "key_content": ["content_point"]
                            }
                        ]
                    }
                ],
                "visual_elements": [
                    {
                        "element_type": "chart|table|diagram|image|other",
                        "purpose": "text",
                        "placement": "section_title",
                        "required_data": ["data_point"]
                    }
                ],
                "estimated_length": "text"
            }
        )

    @staticmethod
    def content_integration_template() -> PromptTemplate:
        """
        Template for integrating content from multiple sources.
        """
        template = """
        # 콘텐츠 통합 작업

        ## 섹션 주제
        {section_topic}

        ## 정보 소스
        {information_sources}

        ## 요청 작업
        다양한 소스의 정보를 통합하여 일관된 내용의 섹션을 작성하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 통합된 섹션 내용
        2. 사용된 주요 소스
        3. 불일치 또는 모순된 정보 해결 방법
        4. 추가 필요한 정보
        """

        return PromptTemplate(
            template=template,
            required_variables=["section_topic", "information_sources"],
            output_format={
                "integrated_content": "text",
                "key_sources": [
                    {
                        "source": "text",
                        "contribution": "text"
                    }
                ],
                "information_conflicts": [
                    {
                        "conflict": "text",
                        "resolution_approach": "text"
                    }
                ],
                "missing_information": ["information_need"],
                "confidence": "high|medium|low"
            }
        )

    @staticmethod
    def executive_summary_template() -> PromptTemplate:
        """
        Template for creating an executive summary.
        """
        template = """
        # 주요 요약 작성 작업

        ## 보고서 전체 내용
        {report_content}

        ## 대상 독자
        {target_audience}

        ## 요약 길이
        {summary_length}

        ## 요청 작업
        보고서 전체 내용을 간결하고 정확하게 요약하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 주요 요약 내용
        2. 핵심 발견사항
        3. 중요한 결론 및 권장사항
        """

        return PromptTemplate(
            template=template,
            required_variables=["report_content"],
            optional_variables=["target_audience", "summary_length"],
            output_format={
                "executive_summary": "text",
                "key_findings": ["finding"],
                "main_conclusions": ["conclusion"],
                "key_recommendations": ["recommendation"]
            }
        )

    @staticmethod
    def visualization_integration_template() -> PromptTemplate:
        """
        Template for integrating visualizations into reports.
        """
        template = """
        # 시각화 요소 통합 작업

        ## 텍스트 내용
        {text_content}

        ## 시각화 설명
        {visualization_description}

        ## 요청 작업
        제공된 텍스트와 시각화 요소를 효과적으로 통합하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 시각화가 통합된 콘텐츠
        2. 시각화에 대한 설명 텍스트
        3. 시각화를 통해 강조되는 핵심 포인트
        """

        return PromptTemplate(
            template=template,
            required_variables=["text_content", "visualization_description"],
            output_format={
                "integrated_content": "text",
                "visualization_caption": "text",
                "visual_reference_text": "text",
                "key_points_emphasized": ["point"],
                "placement_recommendation": "text"
            }
        )

    @staticmethod
    def style_adaptation_template() -> PromptTemplate:
        """
        Template for adapting content to target audience.
        """
        template = """
        # 스타일 적응 작업

        ## 원본 내용
        {original_content}

        ## 대상 독자
        {target_audience}

        ## 요청 작업
        원본 내용을 대상 독자에 맞게 스타일을 조정하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 조정된 내용
        2. 적용된 스타일 변경 사항
        3. 제거되거나 추가된 주요 내용
        """

        return PromptTemplate(
            template=template,
            required_variables=["original_content", "target_audience"],
            output_format={
                "adapted_content": "text",
                "style_adaptations": [
                    {
                        "adaptation_type": "terminology|complexity|tone|structure|other",
                        "explanation": "text",
                        "example": "text"
                    }
                ],
                "content_changes": {
                    "removed": ["item"],
                    "added": ["item"],
                    "modified": ["item"]
                },
                "readability_level": "text"
            }
        )

    @staticmethod
    def references_formatting_template() -> PromptTemplate:
        """
        Template for formatting references and citations.
        """
        template = """
        # 참고문헌 형식화 작업

        ## 인용된 소스
        {cited_sources}

        ## 인용 스타일
        {citation_style}

        ## 요청 작업
        지정된 인용 스타일에 맞게 참고문헌 목록을 형식화하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 형식화된 참고문헌 목록
        2. 본문 내 인용 형식 예시
        3. 형식화 시 발생한 이슈 또는 누락된 정보
        """

        return PromptTemplate(
            template=template,
            required_variables=["cited_sources"],
            optional_variables=["citation_style"],
            output_format={
                "formatted_references": ["reference"],
                "in_text_citation_examples": {
                    "single_author": "text",
                    "multiple_authors": "text",
                    "no_author": "text",
                    "multiple_works": "text"
                },
                "formatting_issues": ["issue"],
                "missing_information": ["missing_item"]
            }
        )

report_writer_templates = {
    "report_structure": ReportWriterTemplates.report_structure_template(),
    "content_integration": ReportWriterTemplates.content_integration_template(),
    "executive_summary": ReportWriterTemplates.executive_summary_template(),
    "visualization_integration": ReportWriterTemplates.visualization_integration_template(),
    "style_adaptation": ReportWriterTemplates.style_adaptation_template(),
    "references_formatting": ReportWriterTemplates.references_formatting_template()
} 