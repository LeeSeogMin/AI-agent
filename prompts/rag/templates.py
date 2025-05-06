#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt templates for the RAG Agent in the LangGraph 멀티 에이전트 시스템
"""

from typing import Dict, List, Any
import os
from pathlib import Path

from prompts.base import PromptTemplateManager


# Create template manager for RAG agent
template_dir = Path(os.path.dirname(os.path.abspath(__file__)))
manager = PromptTemplateManager(str(template_dir))


# Define prompt templates as strings
QUERY_UNDERSTANDING_TEMPLATE = """
# 검색 쿼리 이해 및 분석

다음 원본 쿼리를 분석하여 최적의 검색 쿼리와 전략을 도출하세요:

## 원본 쿼리:
```
{original_query}
```

검색 전략을 다음 JSON 형식으로 출력하세요:

```json
{
  "search_queries": [
    {
      "query_text": "최적화된 검색 쿼리 텍스트",
      "collections": ["검색할 컬렉션 목록"],
      "rationale": "이 쿼리와 컬렉션을 선택한 이유"
    }
  ],
  "metadata_filters": {
    "필터명1": "필터값1",
    "필터명2": "필터값2"
  },
  "expected_information": ["기대하는 정보 유형"],
  "key_concepts": ["쿼리에서 식별된 핵심 개념"],
  "semantic_search_weight": 0.8,
  "keyword_search_weight": 0.2
}
```
"""


RETRIEVAL_EVALUATION_TEMPLATE = """
# 검색 결과 평가

다음 검색 결과를 검토하고 관련성 및 품질을 평가하세요:

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
{retrieved_chunks}
```

검색 결과 평가를 다음 JSON 형식으로 출력하세요:

```json
{
  "evaluation": {
    "overall_relevance": 0.75,
    "coverage": 0.8,
    "diversity": 0.6,
    "information_quality": 0.85,
    "missing_aspects": ["누락된 측면 목록"]
  },
  "ranked_chunks": [
    {
      "chunk_id": "청크 ID",
      "relevance_score": 0.95,
      "key_information": ["이 청크의 핵심 정보 목록"],
      "rationale": "이 청크가 관련성 있는 이유"
    }
  ],
  "follow_up_queries": ["제안되는 후속 검색 쿼리 목록"],
  "needs_additional_retrieval": true,
  "collection_recommendations": ["추가 검색할 컬렉션 추천"]
}
```
"""


INFORMATION_SYNTHESIS_TEMPLATE = """
# 검색 정보 통합 및 합성

다음 검색 결과 및 평가를 바탕으로 정보를 통합하여 포괄적인 응답을 생성하세요:

## 원본 쿼리:
```
{original_query}
```

## 평가된 검색 결과:
```json
{evaluated_results}
```

통합된 정보를 다음 JSON 형식으로 출력하세요:

```json
{
  "synthesized_information": {
    "summary": "모든 관련 정보를 통합한 요약",
    "key_facts": ["핵심 사실 목록"],
    "detailed_explanation": "상세한 정보 설명",
    "conflicting_information": [
      {
        "topic": "충돌 주제",
        "viewpoints": ["다양한 관점 목록"],
        "resolution": "가능한 해결 또는 설명"
      }
    ]
  },
  "information_gaps": ["확인된 정보 격차 목록"],
  "sources": [
    {
      "source_id": "출처 ID",
      "source_type": "출처 유형",
      "reliability": 0.9,
      "contribution": "이 출처가 제공한 핵심 정보"
    }
  ],
  "confidence": 0.85,
  "temporal_context": "정보의 시간적 컨텍스트"
}
```
"""


RESPONSE_GENERATION_TEMPLATE = """
# 최종 RAG 응답 생성

합성된 정보를 바탕으로 원본 쿼리에 대한 최종 응답을 생성하세요:

## 원본 쿼리:
```
{original_query}
```

## 합성된 정보:
```json
{synthesized_information}
```

## 요청 컨텍스트:
```json
{request_context}
```

최종 RAG 응답을 다음 JSON 형식으로 출력하세요:

```json
{
  "response": {
    "answer": "쿼리에 대한 포괄적인 답변",
    "confidence": 0.9,
    "supporting_evidence": ["응답을 뒷받침하는 증거 목록"],
    "limitations": ["응답의 한계 목록"],
    "sources": ["참조된 출처 목록"]
  },
  "knowledge_gaps": ["식별된 지식 갭 목록"],
  "follow_up_suggestions": ["제안되는 후속 질문 목록"],
  "metadata": {
    "collections_used": ["사용된 컬렉션 목록"],
    "chunks_retrieved": 15,
    "chunks_used": 8,
    "processing_time": "처리 시간(초)"
  }
}
```
"""


HYBRID_SEARCH_TEMPLATE = """
# 하이브리드 검색 전략 수립

다음 검색 요구사항을 바탕으로 시맨틱 검색과 키워드 검색을 결합한 최적의 하이브리드 검색 전략을 수립하세요:

## 검색 요구사항:
```json
{search_requirements}
```

하이브리드 검색 전략을 다음 JSON 형식으로 출력하세요:

```json
{
  "semantic_queries": [
    {
      "query_text": "시맨틱 검색을 위한 쿼리",
      "collections": ["검색할 컬렉션"],
      "top_k": 10,
      "similarity_threshold": 0.7
    }
  ],
  "keyword_queries": [
    {
      "query_text": "키워드 검색을 위한 쿼리",
      "collections": ["검색할 컬렉션"],
      "filter_criteria": {
        "필터명1": "필터값1"
      }
    }
  ],
  "reranking_strategy": {
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "recency_weight": 0.2,
    "authority_weight": 0.4
  },
  "rationale": "이 검색 전략을 선택한 이유와 기대 효과"
}
```
"""


CONTEXT_OPTIMIZATION_TEMPLATE = """
# LLM 컨텍스트 최적화

다음 검색 결과와 쿼리 컨텍스트를 바탕으로 LLM 컨텍스트를 최적화하세요:

## 원본 쿼리:
```
{original_query}
```

## 검색 결과:
```json
{search_results}
```

## 컨텍스트 제약조건:
```json
{context_constraints}
```

최적화된 LLM 컨텍스트를 다음 JSON 형식으로 출력하세요:

```json
{
  "optimized_context": [
    {
      "content": "컨텍스트에 포함할 내용",
      "source_id": "출처 ID",
      "importance": 0.95,
      "inclusion_rationale": "이 내용을 포함시킨 이유"
    }
  ],
  "excluded_content": [
    {
      "content_summary": "제외된 내용 요약",
      "source_id": "출처 ID",
      "exclusion_rationale": "이 내용을 제외한 이유"
    }
  ],
  "context_structure": {
    "introduction": "컨텍스트 도입부",
    "main_content": "컨텍스트 주요 내용",
    "supporting_details": "뒷받침하는 세부 정보",
    "conclusion": "컨텍스트 결론"
  },
  "optimization_metrics": {
    "token_count": 1500,
    "information_density": 0.85,
    "relevance_score": 0.92
  }
}
```
"""


# Create a dictionary of templates for easy access
TEMPLATES = {
    "query_understanding": QUERY_UNDERSTANDING_TEMPLATE,
    "retrieval_evaluation": RETRIEVAL_EVALUATION_TEMPLATE,
    "information_synthesis": INFORMATION_SYNTHESIS_TEMPLATE,
    "response_generation": RESPONSE_GENERATION_TEMPLATE,
    "hybrid_search": HYBRID_SEARCH_TEMPLATE,
    "context_optimization": CONTEXT_OPTIMIZATION_TEMPLATE
}


def get_query_understanding_prompt(original_query: str) -> str:
    """Get a prompt for query understanding"""
    return manager.format_template("query_understanding", original_query=original_query)


def get_retrieval_evaluation_prompt(
    original_query: str, 
    search_query: str, 
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    """Get a prompt for retrieval evaluation"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "retrieval_evaluation", 
        original_query=original_query,
        search_query=search_query,
        retrieved_chunks=format_json_for_prompt(retrieved_chunks)
    )


def get_information_synthesis_prompt(
    original_query: str,
    evaluated_results: Dict[str, Any]
) -> str:
    """Get a prompt for information synthesis"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "information_synthesis", 
        original_query=original_query,
        evaluated_results=format_json_for_prompt(evaluated_results)
    )


def get_response_generation_prompt(
    original_query: str,
    synthesized_information: Dict[str, Any],
    request_context: Dict[str, Any]
) -> str:
    """Get a prompt for response generation"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "response_generation", 
        original_query=original_query,
        synthesized_information=format_json_for_prompt(synthesized_information),
        request_context=format_json_for_prompt(request_context)
    )


def get_hybrid_search_prompt(search_requirements: Dict[str, Any]) -> str:
    """Get a prompt for hybrid search strategy"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "hybrid_search", 
        search_requirements=format_json_for_prompt(search_requirements)
    )


def get_context_optimization_prompt(
    original_query: str,
    search_results: Dict[str, Any],
    context_constraints: Dict[str, Any]
) -> str:
    """Get a prompt for LLM context optimization"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "context_optimization", 
        original_query=original_query,
        search_results=format_json_for_prompt(search_results),
        context_constraints=format_json_for_prompt(context_constraints)
    ) 