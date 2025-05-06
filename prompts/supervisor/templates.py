#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt templates for the Supervisor Agent in the LangGraph 멀티 에이전트 시스템
"""

from typing import Dict, List, Any
import os
from pathlib import Path

from prompts.base import PromptTemplateManager


# Create template manager for supervisor agent
template_dir = Path(os.path.dirname(os.path.abspath(__file__)))
manager = PromptTemplateManager(str(template_dir))


# Define prompt templates as strings
QUERY_ANALYSIS_TEMPLATE = """
# 사용자 쿼리 분석

다음 사용자 쿼리를 분석하여 필요한 작업과 전문 에이전트를 식별하세요:
```
{user_query}
```

쿼리 분석 결과를 다음 JSON 형식으로 출력하세요:

```json
{
  "parsed_intent": "사용자의 주요 의도와 목표에 대한 해석",
  "required_information": ["필요한 정보 목록"],
  "required_actions": ["필요한 작업 목록"],
  "recommended_agents": ["필요한 에이전트 목록"],
  "complexity": "low|medium|high",
  "priority_tasks": ["가장 중요한 작업 목록"]
}
```
"""


TASK_PLANNING_TEMPLATE = """
# 작업 계획 수립

다음 사용자 쿼리와 분석 결과를 바탕으로 상세한 작업 계획을 수립하세요:

## 사용자 쿼리:
```
{user_query}
```

## 쿼리 분석 결과:
```json
{query_analysis}
```

각 단계와 작업 흐름을 명확히 설명하는 작업 계획을 JSON 형식으로 출력하세요:

```json
{
  "plan_id": "고유 계획 ID",
  "stages": [
    {
      "stage_id": "stage_1",
      "stage_name": "단계 이름",
      "description": "단계 설명",
      "tasks": [
        {
          "task_id": "task_1_1",
          "agent": "작업을 수행할 에이전트",
          "description": "작업 설명",
          "input_required": ["필요한 입력"],
          "dependencies": ["의존하는 태스크 ID"] 
        }
      ]
    }
  ],
  "expected_duration": "계획 완료까지 예상 시간(초)",
  "parallel_tasks": ["병렬로 실행할 수 있는 태스크 ID"]
}
```
"""


AGENT_ASSIGNMENT_TEMPLATE = """
# 에이전트 작업 할당

다음 작업 정보에 기반하여 가장 적합한 특화 에이전트를 선택하고 작업 요청 메시지를 작성하세요:

## 작업 정보:
```json
{task_info}
```

## 현재 시스템 상태:
```json
{system_state}
```

에이전트 작업 요청을 다음 JSON 형식으로 출력하세요:

```json
{
  "message_id": "{task_id}",
  "timestamp": "현재 시간 ISO 형식",
  "sender": "supervisor",
  "receiver": "선택한 에이전트 ID",
  "message_type": "request",
  "priority": "high|normal|low",
  "content": {
    "action": "수행할 작업 유형",
    "parameters": {
      "작업 매개변수 1": "값",
      "작업 매개변수 2": "값"
    },
    "data": {
      "관련 데이터 1": "값",
      "관련 데이터 2": "값"
    },
    "metadata": {
      "task_id": "{task_id}",
      "plan_id": "{plan_id}",
      "stage_id": "{stage_id}"
    }
  },
  "references": []
}
```
"""


TASK_MONITORING_TEMPLATE = """
# 작업 진행 상황 모니터링

다음 작업 상태 정보를 분석하고 적절한 후속 조치를 결정하세요:

## 현재 작업 상태:
```json
{task_status}
```

## 전체 계획 정보:
```json
{plan_info}
```

작업 모니터링 평가 결과와 필요한 조치를 다음 JSON 형식으로 출력하세요:

```json
{
  "assessment": {
    "overall_progress": 0.75,
    "on_schedule": true|false,
    "bottlenecks": ["병목 지점"],
    "issues": ["발견된 문제점"],
    "completed_tasks": ["완료된 작업 ID"]
  },
  "actions_needed": [
    {
      "action_type": "reassign|retry|cancel|notify",
      "task_id": "관련 작업 ID",
      "description": "필요한 조치 설명",
      "priority": "high|normal|low"
    }
  ],
  "plan_adjustments": ["필요한 계획 조정 사항"]
}
```
"""


RESULT_INTEGRATION_TEMPLATE = """
# 결과 통합

다음 여러 에이전트의 작업 결과를 분석하고 통합하세요:

## 작업 결과:
```json
{task_results}
```

## 원본 사용자 쿼리:
```
{user_query}
```

모든 결과를 종합적으로 분석하고 통합하여 일관성 있는 응답을 다음 JSON 형식으로 출력하세요:

```json
{
  "integrated_result": {
    "summary": "주요 결과 요약",
    "key_findings": ["주요 발견 사항"],
    "conflicting_information": ["불일치 정보"],
    "confidence": 0.85,
    "sources": ["정보 출처"]
  },
  "next_steps": {
    "required_clarification": ["필요한 추가 정보"],
    "recommended_follow_up": ["권장 후속 조치"]
  }
}
```
"""


RESULT_VALIDATION_TEMPLATE = """
# 결과 검증

다음 통합 결과를 검증하고 품질, 완전성, 정확성을 평가하세요:

## 통합 결과:
```json
{integrated_result}
```

## 원본 사용자 쿼리:
```
{user_query}
```

## 모든 에이전트의 개별 결과:
```json
{all_agent_results}
```

검증 결과와 제안 사항을 다음 JSON 형식으로 출력하세요:

```json
{
  "validation_result": {
    "accuracy_score": 0.9,
    "completeness_score": 0.85,
    "consistency_score": 0.95,
    "overall_quality": "high|medium|low",
    "missing_aspects": ["누락된 측면"],
    "factual_errors": ["사실 오류"]
  },
  "recommendations": {
    "improvements_needed": ["필요한 개선 사항"],
    "additional_verification": ["추가 검증 필요 항목"]
  },
  "final_approval": true|false,
  "approval_comments": "승인 또는 거부 이유"
}
```
"""


FINAL_RESPONSE_TEMPLATE = """
# 최종 사용자 응답 생성

다음 검증된 결과를 바탕으로 사용자에게 보낼 최종 응답을 작성하세요:

## 검증된 결과:
```json
{validated_result}
```

## 원본 사용자 쿼리:
```
{user_query}
```

사용자 친화적인 언어로 명확하고 유용한 최종 응답을 다음 JSON 형식으로 출력하세요:

```json
{
  "final_response": "사용자를 위한 친화적이고 포괄적인 응답",
  "recommendations": ["사용자를 위한 추천 사항"],
  "limitations": ["생성된 답변의 제한 사항"],
  "sources": ["사용된 주요 정보 출처"]
}
```
"""


# Create a dictionary of templates for easy access
TEMPLATES = {
    "query_analysis": QUERY_ANALYSIS_TEMPLATE,
    "task_planning": TASK_PLANNING_TEMPLATE,
    "agent_assignment": AGENT_ASSIGNMENT_TEMPLATE,
    "task_monitoring": TASK_MONITORING_TEMPLATE,
    "result_integration": RESULT_INTEGRATION_TEMPLATE,
    "result_validation": RESULT_VALIDATION_TEMPLATE,
    "final_response": FINAL_RESPONSE_TEMPLATE
}


def get_query_analysis_prompt(user_query: str) -> str:
    """Get a prompt for query analysis"""
    return manager.format_template("query_analysis", user_query=user_query)


def get_task_planning_prompt(user_query: str, query_analysis: Dict[str, Any]) -> str:
    """Get a prompt for task planning"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "task_planning", 
        user_query=user_query,
        query_analysis=format_json_for_prompt(query_analysis)
    )


def get_agent_assignment_prompt(task_info: Dict[str, Any], system_state: Dict[str, Any]) -> str:
    """Get a prompt for agent assignment"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "agent_assignment",
        task_info=format_json_for_prompt(task_info),
        system_state=format_json_for_prompt(system_state)
    )


def get_task_monitoring_prompt(task_status: Dict[str, Any], plan_info: Dict[str, Any]) -> str:
    """Get a prompt for task monitoring"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "task_monitoring",
        task_status=format_json_for_prompt(task_status),
        plan_info=format_json_for_prompt(plan_info)
    )


def get_result_integration_prompt(task_results: Dict[str, Any], user_query: str) -> str:
    """Get a prompt for result integration"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "result_integration",
        task_results=format_json_for_prompt(task_results),
        user_query=user_query
    )


def get_result_validation_prompt(
    integrated_result: Dict[str, Any], 
    user_query: str,
    all_agent_results: Dict[str, Any]
) -> str:
    """Get a prompt for result validation"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "result_validation",
        integrated_result=format_json_for_prompt(integrated_result),
        user_query=user_query,
        all_agent_results=format_json_for_prompt(all_agent_results)
    )


def get_final_response_prompt(validated_result: Dict[str, Any], user_query: str) -> str:
    """Get a prompt for final response generation"""
    from prompts.base import format_json_for_prompt
    return manager.format_template(
        "final_response",
        validated_result=format_json_for_prompt(validated_result),
        user_query=user_query
    ) 