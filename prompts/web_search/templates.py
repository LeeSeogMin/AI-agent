#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt templates for the Web Search Agent in the LangGraph 멀티 에이전트 시스템
"""

from typing import Dict, List, Any
import os
from pathlib import Path

from prompts.base import PromptTemplateManager


# Create template manager for Web Search agent
template_dir = Path(os.path.dirname(os.path.abspath(__file__)))
manager = PromptTemplateManager(str(template_dir))


# Define prompt templates as strings
QUERY_FORMULATION_TEMPLATE = """
# 웹 검색 쿼리 최적화

다음 원본 쿼리를 분석하여 웹 검색에 최적화된 검색 쿼리를 생성하세요:

## 원본 쿼리:
```
{original_query}
```

## 검색 요구사항:
```json
{search_requirements}
```

최적화된 웹 검색 전략을 다음 JSON 형식으로 출력하세요:

```json
{
  "optimized_queries": [
    {
      "query_text": "최적화된 검색 쿼리",
      "search_engine": "google|bing|duckduckgo|scholar|arxiv",
      "query_type": "general|news|academic|image|specific",
      "rationale": "이 쿼리를 선택한 이유"
    }
  ],
  "search_parameters": {
    "time_range": "all|day|week|month|year",
    "language": "ko|en|all",
    "region": "kr|global|specific",
    "content_type": "all|news|blogs|academic|images"
  },
  "expected_information_types": ["기대하는 정보 유형"],
  "priority_sources": ["우선적으로 검색할 소스 도메인"],
  "avoid_sources": ["제외할 소스 도메인"]
}
```
"""


SEARCH_RESULT_EVALUATION_TEMPLATE = """
# 검색 결과 평가

다음 웹 검색 결과를 검토하고 관련성, 신뢰성, 최신성을 평가하세요:

## 원본 쿼리:
```
{original_query}
```

## 검색 쿼리:
```
{search_query}
```

## 검색 결과:
```json
{search_results}
```

검색 결과 평가를 다음 JSON 형식으로 출력하세요:

```json
{
  "evaluation": {
    "overall_relevance": 0.85,
    "coverage": 0.75,
    "reliability": 0.9,
    "recency": 0.8,
    "missing_aspects": ["누락된 정보 측면"],
    "information_quality": "high|medium|low"
  },
  "ranked_results": [
    {
      "result_id": 3,
      "url": "결과 URL",
      "relevance_score": 0.95,
      "reliability_score": 0.9,
      "recency_score": 0.8,
      "key_information": ["이 결과의 핵심 정보"],
      "rationale": "이 결과를 높게 평가한 이유"
    }
  ],
  "content_to_extract": [
    {
      "url": "추출할 콘텐츠 URL",
      "extraction_priority": "high|medium|low",
      "extraction_method": "full_page|specific_section|metadata_only",
      "expected_content_type": "text|table|image|pdf"
    }
  ],
  "needs_additional_search": true|false,
  "follow_up_queries": ["제안된 후속 검색 쿼리"]
}
```
"""


CONTENT_EXTRACTION_TEMPLATE = """
# 웹 콘텐츠 추출 및 정제

다음 웹 페이지 콘텐츠를 추출하고 정제하여 관련 정보만 유지하세요:

## 원본 쿼리:
```
{original_query}
```

## 추출 요구사항:
```json
{extraction_requirements}
```

## 웹 페이지 콘텐츠:
```
{web_content}
```

추출 및 정제된 콘텐츠를 다음 JSON 형식으로 출력하세요:

```json
{
  "extracted_content": {
    "title": "페이지 제목",
    "url": "페이지 URL",
    "main_content": "핵심 콘텐츠 텍스트",
    "key_sections": [
      {
        "section_title": "섹션 제목",
        "section_content": "섹션 내용",
        "relevance": 0.9
      }
    ],
    "key_facts": ["추출된 핵심 사실"],
    "metadata": {
      "author": "콘텐츠 작성자",
      "published_date": "출판일",
      "last_updated": "최종 업데이트일",
      "source_type": "news|blog|academic|government|company|social"
    }
  },
  "excluded_content": ["제외된 섹션 또는 콘텐츠 요약"],
  "content_quality": {
    "reliability": 0.85,
    "objectivity": 0.8,
    "completeness": 0.9,
    "citations": true|false,
    "has_advertising": true|false
  },
  "extraction_notes": ["추출 과정에서의 주요 결정이나 문제점"]
}
```
"""


MULTI_SOURCE_SYNTHESIS_TEMPLATE = """
# 다중 소스 정보 통합

다음 여러 웹 소스에서 추출한 정보를 분석하고 통합하세요:

## 원본 쿼리:
```
{original_query}
```

## 추출된 정보:
```json
{extracted_information}
```

통합된 정보를 다음 JSON 형식으로 출력하세요:

```json
{
  "synthesized_information": {
    "summary": "모든 소스의 주요 정보를 통합한 요약",
    "key_findings": ["주요 발견사항"],
    "consistent_information": ["모든/대부분의 소스가 동의하는 정보"],
    "conflicting_information": [
      {
        "topic": "충돌 주제",
        "viewpoints": ["다양한 관점"],
        "assessment": "대립되는 정보에 대한 평가"
      }
    ],
    "fact_vs_opinion": {
      "facts": ["검증된 사실"],
      "opinions": ["주관적 의견 또는 해석"]
    }
  },
  "source_analysis": {
    "most_reliable": ["가장 신뢰할 수 있는 소스"],
    "least_reliable": ["가장 신뢰성이 낮은 소스"],
    "bias_assessment": {
      "source_url": "편향성 평가"
    },
    "recency_assessment": {
      "source_url": "최신성 평가"
    }
  },
  "information_gaps": ["식별된 정보 격차"],
  "confidence": 0.85
}
```
"""


ACADEMIC_SEARCH_TEMPLATE = """
# 학술 자료 검색 및 평가

다음 학술 검색 요청에 따라 관련 학술 논문과 자료를 검색하고 평가하세요:

## 검색 주제:
```
{search_topic}
```

## 학술 검색 요구사항:
```json
{academic_requirements}
```

학술 검색 결과 및 평가를 다음 JSON 형식으로 출력하세요:

```json
{
  "optimized_queries": [
    {
      "query_text": "arXiv/Google Scholar용 최적화된 쿼리",
      "platform": "arxiv|scholar|pubmed|specific_journal",
      "filters": {
        "year_range": "2020-2023",
        "categories": ["cs.AI", "cs.CL"],
        "sort_by": "relevance|date|citations"
      }
    }
  ],
  "recommended_papers": [
    {
      "title": "논문 제목",
      "authors": ["저자 목록"],
      "year": 2022,
      "venue": "학회/저널명",
      "url": "논문 URL",
      "abstract": "초록 요약",
      "relevance_score": 0.95,
      "citation_count": 45,
      "key_findings": ["주요 발견사항"],
      "methodology": "연구 방법론 요약",
      "limitations": ["연구의 한계점"]
    }
  ],
  "key_researchers": ["해당 분야 주요 연구자"],
  "emerging_trends": ["식별된 최신 연구 동향"],
  "recommended_downloads": [
    {
      "url": "다운로드할 PDF URL",
      "paper_id": "논문 ID",
      "priority": "high|medium|low"
    }
  ],
  "search_assessment": {
    "coverage": 0.85,
    "recency": 0.9,
    "depth": 0.8,
    "limitations": ["검색 제한사항"]
  }
}
```
"""


DOWNLOAD_MANAGEMENT_TEMPLATE = """
# 자료 다운로드 관리

다음 자료 다운로드 요청을 처리하고 다운로드된 콘텐츠를 구조화하세요:

## 다운로드 요청:
```json
{download_requests}
```

## 시스템 제약조건:
```json
{system_constraints}
```

다운로드 계획 및 처리를 다음 JSON 형식으로 출력하세요:

```json
{
  "download_plan": [
    {
      "url": "다운로드할 URL",
      "file_type": "pdf|image|html|dataset",
      "expected_size": "예상 파일 크기",
      "priority": "high|medium|low",
      "storage_path": "저장 경로",
      "download_method": "direct|api|scraping",
      "requires_authentication": true|false
    }
  ],
  "preprocessing_steps": [
    {
      "file_path": "처리할 파일 경로",
      "processing_type": "extract_text|compress|convert|crop",
      "parameters": {
        "매개변수명": "값"
      },
      "output_format": "출력 형식"
    }
  ],
  "metadata_extraction": [
    {
      "file_path": "메타데이터를 추출할 파일 경로",
      "metadata_fields": ["author", "date", "title", "keywords"]
    }
  ],
  "compliance_check": {
    "copyright_status": "open|restricted|unknown",
    "robots_txt_checked": true|false,
    "usage_restrictions": ["사용 제한사항"],
    "attribution_required": true|false
  },
  "resource_usage": {
    "estimated_bandwidth": "예상 대역폭 사용량",
    "estimated_storage": "예상 저장공간 사용량",
    "estimated_time": "예상 소요시간(초)"
  }
}
```
"""


# Create a dictionary of templates for easy access
TEMPLATES = {
    "query_formulation": QUERY_FORMULATION_TEMPLATE,
    "search_result_evaluation": SEARCH_RESULT_EVALUATION_TEMPLATE,
    "content_extraction": CONTENT_EXTRACTION_TEMPLATE,
    "multi_source_synthesis": MULTI_SOURCE_SYNTHESIS_TEMPLATE,
    "academic_search": ACADEMIC_SEARCH_TEMPLATE,
    "download_management": DOWNLOAD_MANAGEMENT_TEMPLATE
}


def get_query_formulation_prompt(
    original_query: str, 
    search_requirements: Dict[str, Any]
) -> str:
    """Get a prompt for query formulation"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "query_formulation", 
        original_query=original_query,
        search_requirements=format_json_for_prompt(search_requirements)
    )


def get_search_result_evaluation_prompt(
    original_query: str, 
    search_query: str, 
    search_results: List[Dict[str, Any]]
) -> str:
    """Get a prompt for search result evaluation"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "search_result_evaluation", 
        original_query=original_query,
        search_query=search_query,
        search_results=format_json_for_prompt(search_results)
    )


def get_content_extraction_prompt(
    original_query: str,
    extraction_requirements: Dict[str, Any],
    web_content: str
) -> str:
    """Get a prompt for content extraction"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "content_extraction", 
        original_query=original_query,
        extraction_requirements=format_json_for_prompt(extraction_requirements),
        web_content=web_content
    )


def get_multi_source_synthesis_prompt(
    original_query: str,
    extracted_information: List[Dict[str, Any]]
) -> str:
    """Get a prompt for multi-source synthesis"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "multi_source_synthesis", 
        original_query=original_query,
        extracted_information=format_json_for_prompt(extracted_information)
    )


def get_academic_search_prompt(
    search_topic: str,
    academic_requirements: Dict[str, Any]
) -> str:
    """Get a prompt for academic search"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "academic_search", 
        search_topic=search_topic,
        academic_requirements=format_json_for_prompt(academic_requirements)
    )


def get_download_management_prompt(
    download_requests: List[Dict[str, Any]],
    system_constraints: Dict[str, Any]
) -> str:
    """Get a prompt for download management"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "download_management", 
        download_requests=format_json_for_prompt(download_requests),
        system_constraints=format_json_for_prompt(system_constraints)
    ) 