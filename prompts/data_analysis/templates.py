from typing import Dict, List, Any, Optional
import json
from prompts.base import PromptTemplate, load_system_message

# Load the system message for Data Analysis Agent
SYSTEM_MESSAGE = load_system_message("data_analysis")

class DataAnalysisTemplates:
    """
    Collection of templates for the Data Analysis Agent.
    """

    @staticmethod
    def dataset_review_template() -> PromptTemplate:
        """
        Template for reviewing and summarizing a dataset.
        """
        template = """
        # 데이터셋 검토 작업

        ## 입력 데이터
        {dataset_description}

        ## 요청 작업
        제공된 데이터셋의 구조, 품질, 기본 특성을 평가하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 데이터셋 요약 (행 수, 열 수, 데이터 타입)
        2. 결측치 및 이상치 분석
        3. 주요 변수의 기술 통계
        4. 데이터 품질 이슈
        5. 추가 분석을 위한 권장사항
        """

        return PromptTemplate(
            template=template,
            required_variables=["dataset_description"],
            output_format={
                "dataset_summary": {
                    "rows": "int",
                    "columns": ["column_name"],
                    "data_types": {"column_name": "data_type"},
                    "missing_values": {"column_name": "percentage"}
                },
                "descriptive_stats": {
                    "variable_name": {
                        "mean": "float",
                        "median": "float",
                        "std": "float",
                        "min": "float",
                        "max": "float"
                    }
                },
                "quality_issues": ["issue_description"],
                "recommendations": ["recommendation"]
            }
        )

    @staticmethod
    def correlation_analysis_template() -> PromptTemplate:
        """
        Template for performing correlation analysis.
        """
        template = """
        # 상관관계 분석 작업

        ## 입력 데이터
        {dataset_description}

        ## 대상 변수
        {target_variables}

        ## 요청 작업
        지정된 변수들 간의 상관관계를 분석하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 변수 쌍 간의 상관계수
        2. 주요 상관관계에 대한 해석
        3. 시각화 설명
        4. 제한사항 및 주의점
        """

        return PromptTemplate(
            template=template,
            required_variables=["dataset_description", "target_variables"],
            output_format={
                "correlation_matrix": {
                    "var1_var2": "correlation_coefficient"
                },
                "significant_correlations": [
                    {
                        "variables": ["var1", "var2"],
                        "coefficient": "float",
                        "p_value": "float",
                        "interpretation": "text"
                    }
                ],
                "visualization_description": "text",
                "limitations": ["limitation"]
            }
        )

    @staticmethod
    def time_series_analysis_template() -> PromptTemplate:
        """
        Template for performing time series analysis.
        """
        template = """
        # 시계열 분석 작업

        ## 입력 데이터
        {dataset_description}

        ## 시간 변수
        {time_variable}

        ## 대상 변수
        {target_variables}

        ## 요청 작업
        지정된 시간 변수와 대상 변수를 사용하여 시계열 분석을 수행하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 시간적 트렌드 분석
        2. 계절성 패턴
        3. 이상치 및 변화점
        4. 예측 가능성 평가
        """

        return PromptTemplate(
            template=template,
            required_variables=["dataset_description", "time_variable", "target_variables"],
            output_format={
                "time_series_properties": {
                    "variable_name": {
                        "trend": "text",
                        "seasonality": "text",
                        "stationarity": "boolean",
                        "outliers": ["timestamp"]
                    }
                },
                "trend_analysis": "text",
                "forecasting_potential": {
                    "recommendation": "text",
                    "suggested_models": ["model_name"],
                    "limitations": ["limitation"]
                },
                "visualization_description": "text"
            }
        )

    @staticmethod
    def data_visualization_template() -> PromptTemplate:
        """
        Template for generating data visualization recommendations.
        """
        template = """
        # 데이터 시각화 작업

        ## 입력 데이터
        {dataset_description}

        ## 분석 목적
        {analysis_purpose}

        ## 대상 변수
        {target_variables}

        ## 요청 작업
        분석 목적과 대상 변수에 적합한 시각화 방법을 제안하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 권장 시각화 유형
        2. 각 시각화에 대한 설명 및 이점
        3. 필요한 데이터 전처리 단계
        4. 시각화 구현을 위한 코드 가이드라인
        """

        return PromptTemplate(
            template=template,
            required_variables=["dataset_description", "analysis_purpose", "target_variables"],
            output_format={
                "recommended_visualizations": [
                    {
                        "type": "visualization_type",
                        "variables": ["variable_name"],
                        "purpose": "text",
                        "benefits": ["benefit"],
                        "preprocessing_steps": ["step"]
                    }
                ],
                "implementation_guidelines": {
                    "visualization_type": "code_guidelines"
                },
                "additional_suggestions": ["suggestion"]
            }
        )

    @staticmethod
    def statistical_testing_template() -> PromptTemplate:
        """
        Template for performing statistical hypothesis testing.
        """
        template = """
        # 통계적 가설 검정 작업

        ## 입력 데이터
        {dataset_description}

        ## 검정 유형
        {test_type}

        ## 가설
        {hypothesis}

        ## 대상 변수
        {target_variables}

        ## 요청 작업
        지정된 가설과 변수를 사용하여 통계적 검정을 수행하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 검정 결과 및 통계량
        2. p-값 및 유의성 판단
        3. 결과 해석
        4. 제한사항 및 주의점
        """

        return PromptTemplate(
            template=template,
            required_variables=["dataset_description", "test_type", "hypothesis", "target_variables"],
            output_format={
                "test_results": {
                    "test_name": "text",
                    "test_statistic": "float",
                    "p_value": "float",
                    "significant": "boolean",
                    "confidence_interval": ["lower_bound", "upper_bound"]
                },
                "interpretation": "text",
                "effect_size": {
                    "measure": "text",
                    "value": "float",
                    "interpretation": "text"
                },
                "limitations": ["limitation"],
                "additional_tests_recommended": ["test_name"]
            }
        )

    @staticmethod
    def insight_generation_template() -> PromptTemplate:
        """
        Template for generating insights from analyzed data.
        """
        template = """
        # 인사이트 도출 작업

        ## 분석 결과
        {analysis_results}

        ## 도메인 컨텍스트
        {domain_context}

        ## 요청 작업
        분석 결과와 도메인 컨텍스트를 기반으로 실용적인 인사이트를 도출하세요.

        ## 출력 요구사항
        다음 정보를 포함하는 JSON 응답을 제공하세요:
        1. 주요 발견사항
        2. 비즈니스 또는 실용적 의미
        3. 인사이트의 신뢰도 및 한계
        4. 추가 데이터 요구사항
        """

        return PromptTemplate(
            template=template,
            required_variables=["analysis_results", "domain_context"],
            output_format={
                "key_insights": [
                    {
                        "finding": "text",
                        "evidence": "text",
                        "business_implications": "text",
                        "confidence_level": "high|medium|low",
                        "limitations": ["limitation"]
                    }
                ],
                "recommendations": ["recommendation"],
                "additional_data_needs": ["data_need"],
                "follow_up_questions": ["question"]
            }
        )

data_analysis_templates = {
    "dataset_review": DataAnalysisTemplates.dataset_review_template(),
    "correlation_analysis": DataAnalysisTemplates.correlation_analysis_template(),
    "time_series_analysis": DataAnalysisTemplates.time_series_analysis_template(),
    "data_visualization": DataAnalysisTemplates.data_visualization_template(),
    "statistical_testing": DataAnalysisTemplates.statistical_testing_template(),
    "insight_generation": DataAnalysisTemplates.insight_generation_template()
} 