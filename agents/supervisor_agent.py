#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 감독 에이전트 모듈
"""

import json
import logging
import uuid
import os
from typing import Dict, List, Any, Optional, Union, TypedDict
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langchain_core.runnables import chain, RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver

from agents.base import Agent
from core.config import get_config, DATA_DIR
from core.state_manager import get_state_manager, State
from core.communication import (
    get_communication_manager, Message, 
    create_request_message, create_response_message
)
from core.event_bus import (
    get_event_bus, publish_agent_started_event, 
    publish_agent_stopped_event, publish_task_started_event, 
    publish_task_completed_event
)
from core.error_handler import handle_agent_error, TaskExecutionError
from services.llm_service import generate_text, generate_json

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

# 감독 에이전트 상태 스키마
class SupervisorState(TypedDict):
    """감독 에이전트 상태 스키마"""
    user_query: str
    query_analysis: Dict[str, Any]
    task_plan: List[Dict[str, Any]]
    agent_assignments: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    validation_result: str
    current_stage: str
    metadata: Dict[str, Any]

class SupervisorAgent(Agent):
    """감독 에이전트 클래스"""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        system_message: Optional[str] = None
    ):
        """
        감독 에이전트 초기화
        
        Args:
            agent_id: 에이전트 ID (없으면 자동 생성)
            system_message: 시스템 메시지
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="supervisor",
            system_message=system_message
        )
        
        # 시스템 메시지 로드
        if not self.system_message:
            self.system_message = self.load_system_message()
        
        # 사용 가능한 에이전트 목록
        self.available_agents = {
            "rag": {"enabled": config.agents["rag"]["enabled"]},
            "web_search": {"enabled": config.agents["web_search"]["enabled"]},
            "data_analysis": {"enabled": config.agents["data_analysis"]["enabled"]},
            "psychological": {"enabled": config.agents["psychological"]["enabled"]},
            "report_writer": {"enabled": config.agents["report_writer"]["enabled"]}
        }
        
        # 그래프 및 체크포인트 설정
        self._setup_workflow_graph()
        
        logger.info(f"Supervisor agent initialized with ID: {self.agent_id}")
    
    def _register_task_handlers(self) -> None:
        """작업 핸들러 등록"""
        self.task_handlers = {
            "analyze_query": self._handle_analyze_query,
            "plan_tasks": self._handle_plan_tasks,
            "assign_agents": self._handle_assign_agents,
            "monitor_progress": self._handle_monitor_progress,
            "integrate_results": self._handle_integrate_results,
            "validate_final_result": self._handle_validate_final_result
        }
    
    def _setup_workflow_graph(self) -> None:
        """워크플로우 그래프 설정"""
        # 체크포인트 디렉토리 생성
        checkpoint_dir = os.path.join(DATA_DIR, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 간단한 인메모리 체크포인터 사용
        self.checkpointer = InMemorySaver()
        
        # 그래프 노드 (상태 함수)
        workflow = {
            "analyze_query": self._analyze_query_node,
            "plan_tasks": self._plan_tasks_node,
            "assign_agents": self._assign_agents_node,
            "execute_tasks": self._execute_tasks_node,
            "integrate_results": self._integrate_results_node,
            "validate_result": self._validate_result_node
        }
        
        # 그래프 생성 (상태 스키마 사용)
        self.workflow_graph = StateGraph(SupervisorState)
        
        # 노드 추가
        for node_name, node_func in workflow.items():
            self.workflow_graph.add_node(node_name, node_func)
        
        # 일반 에지 추가
        self.workflow_graph.add_edge("analyze_query", "plan_tasks")
        self.workflow_graph.add_edge("plan_tasks", "assign_agents")
        self.workflow_graph.add_edge("assign_agents", "execute_tasks")
        self.workflow_graph.add_edge("execute_tasks", "integrate_results")
        self.workflow_graph.add_edge("integrate_results", "validate_result")
        
        # 조건부 에지 추가 (검증 결과에 따라)
        def validation_router(state: SupervisorState) -> str:
            """검증 결과에 따라 다음 노드 결정"""
            if state.get("validation_result") == "valid":
                return "valid"
            else:
                return "invalid"
        
        self.workflow_graph.add_conditional_edges(
            "validate_result",
            validation_router,
            {
                "valid": END,
                "invalid": "plan_tasks"
            }
        )
        
        # 시작점 설정
        self.workflow_graph.set_entry_point("analyze_query")
        
        # 컴파일
        self.workflow = self.workflow_graph.compile(checkpointer=self.checkpointer)
    
    def _analyze_query_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 쿼리 분석 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            Dict[str, Any]: 업데이트된 상태
        """
        user_query = state.get("user_query", "")
        
        # 쿼리 분석 수행
        analysis = self._analyze_query(user_query)
        
        # 상태 업데이트
        state["query_analysis"] = analysis
        state["current_stage"] = "planning"
        
        return state
    
    def _plan_tasks_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        작업 계획 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            Dict[str, Any]: 업데이트된 상태
        """
        user_query = state.get("user_query", "")
        query_analysis = state.get("query_analysis", {})
        
        # 작업 계획 수립
        task_plan = self._plan_tasks(user_query, query_analysis)
        
        # 상태 업데이트
        state["task_plan"] = task_plan
        
        return state
    
    def _assign_agents_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트 할당 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            Dict[str, Any]: 업데이트된 상태
        """
        task_plan = state.get("task_plan", [])
        
        # 에이전트 할당
        agent_assignments = self._assign_agents(task_plan)
        
        # 상태 업데이트
        state["agent_assignments"] = agent_assignments
        state["results"] = []
        state["current_stage"] = "information_gathering"
        
        return state
    
    def _execute_tasks_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        작업 실행 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            Dict[str, Any]: 업데이트된 상태
        """
        agent_assignments = state.get("agent_assignments", [])
        user_query = state.get("user_query", "")
        
        # 작업 실행
        task_results = self._execute_tasks(agent_assignments, user_query)
        
        # 상태 업데이트
        state["task_results"] = task_results
        state["current_stage"] = "analysis"
        
        return state
    
    def _integrate_results_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        결과 통합 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            Dict[str, Any]: 업데이트된 상태
        """
        task_results = state.get("task_results", [])
        user_query = state.get("user_query", "")
        
        # 결과 통합
        integrated_result = self._integrate_results(task_results, user_query)
        
        # 상태 업데이트
        state["integrated_result"] = integrated_result
        state["current_stage"] = "synthesis"
        
        return state
    
    def _validate_result_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        결과 검증 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            Dict[str, Any]: 업데이트된 상태
        """
        integrated_result = state.get("integrated_result", {})
        user_query = state.get("user_query", "")
        
        # 결과 검증
        validation_result = self._validate_result(integrated_result, user_query)
        
        # 상태 업데이트
        state["validation_result"] = "valid" if validation_result.get("is_valid") else "invalid"
        state["validation_details"] = validation_result
        
        if state["validation_result"] == "valid":
            state["current_stage"] = "reporting"
            state["final_result"] = integrated_result
        
        return state
    
    @handle_agent_error()
    def _handle_analyze_query(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        쿼리 분석 작업 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for analysis",
                agent_id=self.agent_id,
                task_id="analyze_query"
            )
        
        # 쿼리 분석 실행
        analysis = self._analyze_query(query)
        
        return {
            "status": "success",
            "analysis": analysis
        }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        사용자 쿼리 분석
        
        Args:
            query: 사용자 쿼리
        
        Returns:
            Dict[str, Any]: 분석 결과
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "intents": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "사용자 쿼리에서 파악된 주요 의도 목록"
                },
                "topics": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "쿼리에서 파악된 주요 주제 또는 키워드"
                },
                "information_needs": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "쿼리를 해결하기 위해 필요한 정보 유형"
                },
                "complexity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "쿼리의 복잡도 (낮음, 중간, 높음)"
                },
                "recommended_agents": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["rag", "web_search", "data_analysis", "psychological", "report_writer"]
                    },
                    "description": "쿼리 해결에 적합한 에이전트 목록"
                }
            },
            "required": ["intents", "topics", "information_needs", "complexity", "recommended_agents"]
        }
        
        # 프롬프트 준비
        prompt = f"""사용자의 쿼리를 분석하여 아래 JSON 형식으로 결과를 제공하세요:

사용자 쿼리: "{query}"

분석 시 다음 측면을 고려하세요:
1. 쿼리의 주요 의도(예: 정보 요청, 비교, 설명 요구, 문제 해결 등)
2. 주요 주제와 키워드
3. 필요한 정보 유형
4. 쿼리의 복잡도(낮음, 중간, 높음)
5. 해결에 적합한 에이전트 유형(rag, web_search, data_analysis, psychological, report_writer)

제공된 스키마에 맞는 정확한 JSON을 생성하세요."""
        
        # LLM으로 분석 실행
        try:
            analysis = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info(f"Query analysis completed: {json.dumps(analysis, ensure_ascii=False)[:100]}...")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in query analysis: {str(e)}")
            # 기본 분석 결과 반환
            return {
                "intents": ["information_request"],
                "topics": [],
                "information_needs": ["factual_information"],
                "complexity": "medium",
                "recommended_agents": ["rag", "web_search"]
            }
    
    @handle_agent_error()
    def _handle_plan_tasks(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        작업 계획 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        analysis = message_content.get("data", {}).get("analysis", {})
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for task planning",
                agent_id=self.agent_id,
                task_id="plan_tasks"
            )
        
        # 작업 계획 실행
        task_plan = self._plan_tasks(query, analysis)
        
        return {
            "status": "success",
            "task_plan": task_plan
        }
    
    def _plan_tasks(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        작업 계획 수립
        
        Args:
            query: 사용자 쿼리
            analysis: 쿼리 분석 결과
        
        Returns:
            List[Dict[str, Any]]: 작업 계획
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "description": {"type": "string"},
                            "required_agent_type": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["task_id", "description", "required_agent_type", "priority"]
                    }
                }
            },
            "required": ["tasks"]
        }
        
        # 프롬프트 준비
        prompt = f"""사용자의 쿼리를 해결하기 위한 세부 작업 계획을 수립하세요:

사용자 쿼리: "{query}"

쿼리 분석 결과:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

다음 에이전트 유형이 사용 가능합니다:
- rag: 지식 베이스에서 정보 검색
- web_search: 웹에서 정보 검색
- data_analysis: 데이터 분석 및 시각화
- psychological: 심리학적 분석 제공
- report_writer: 최종 보고서 작성

각 작업에 다음 정보를 포함하세요:
1. 작업 ID (고유 식별자)
2. 작업 설명 (수행할 작업)
3. 필요한 에이전트 유형
4. 우선순위 (high, medium, low)
5. 의존성 (이 작업 전에 완료되어야 하는 작업 ID 목록)

제공된 JSON 스키마에 맞게 응답하세요."""
        
        # LLM으로 작업 계획 생성
        try:
            plan_data = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info(f"Task planning completed: {len(plan_data.get('tasks', []))} tasks")
            return plan_data.get("tasks", [])
            
        except Exception as e:
            logger.error(f"Error in task planning: {str(e)}")
            # 기본 작업 계획 반환
            return [
                {
                    "task_id": "info_retrieval",
                    "description": "Retrieve relevant information",
                    "required_agent_type": "rag",
                    "priority": "high",
                    "dependencies": []
                }
            ]
    
    @handle_agent_error()
    def _handle_assign_agents(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트 할당 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        task_plan = message_content.get("data", {}).get("task_plan", [])
        
        if not task_plan:
            raise TaskExecutionError(
                message="No task plan provided for agent assignment",
                agent_id=self.agent_id,
                task_id="assign_agents"
            )
        
        # 에이전트 할당 실행
        assignments = self._assign_agents(task_plan)
        
        return {
            "status": "success",
            "assignments": assignments
        }
    
    def _assign_agents(self, task_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        에이전트 할당
        
        Args:
            task_plan: 작업 계획
        
        Returns:
            List[Dict[str, Any]]: 에이전트 할당 결과
        """
        assignments = []
        
        # 사용 가능한 에이전트 필터링
        available_agent_types = [
            agent_type for agent_type, info in self.available_agents.items() 
            if info.get("enabled")
        ]
        
        # 작업별 에이전트 할당
        for task in task_plan:
            required_agent_type = task.get("required_agent_type")
            
            # 필요한 에이전트가 사용 가능한지 확인
            if required_agent_type in available_agent_types:
                assignment = {
                    "task_id": task.get("task_id"),
                    "agent_type": required_agent_type,
                    "task_description": task.get("description"),
                    "priority": task.get("priority", "medium"),
                    "status": "pending",
                    "dependencies": task.get("dependencies", [])
                }
                assignments.append(assignment)
            else:
                logger.warning(f"Required agent type {required_agent_type} not available for task {task.get('task_id')}")
                # 대체 에이전트 할당 (예: RAG나 웹 검색)
                fallback_agent = next(iter(available_agent_types), None)
                if fallback_agent:
                    assignment = {
                        "task_id": task.get("task_id"),
                        "agent_type": fallback_agent,
                        "task_description": task.get("description"),
                        "priority": task.get("priority", "medium"),
                        "status": "pending",
                        "dependencies": task.get("dependencies", []),
                        "note": f"Fallback from {required_agent_type} to {fallback_agent}"
                    }
                    assignments.append(assignment)
        
        # 우선순위 기준 정렬
        priority_values = {"high": 0, "medium": 1, "low": 2}
        assignments.sort(key=lambda x: priority_values.get(x.get("priority"), 1))
        
        logger.info(f"Agent assignments completed: {len(assignments)} assignments")
        return assignments
    
    @handle_agent_error()
    def _handle_monitor_progress(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        진행 상황 모니터링 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        session_id = message_content.get("metadata", {}).get("session_id")
        
        if not session_id:
            raise TaskExecutionError(
                message="No session ID provided for progress monitoring",
                agent_id=self.agent_id,
                task_id="monitor_progress"
            )
        
        # 상태 조회
        state = self.get_state(session_id)
        
        if not state:
            return {
                "status": "error",
                "message": f"No state found for session {session_id}"
            }
        
        # 작업 상태 요약
        status_summary = {
            "current_stage": state.current_stage,
            "completed_tasks": len([t for t in state.task_statuses if t.get("status") == "completed"]),
            "pending_tasks": len([t for t in state.task_statuses if t.get("status") == "pending"]),
            "failed_tasks": len([t for t in state.task_statuses if t.get("status") == "failed"]),
            "active_agents": state.active_agents
        }
        
        return {
            "status": "success",
            "progress": status_summary
        }
    
    def _execute_tasks(self, assignments: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        에이전트에 작업 실행 요청
        
        Args:
            assignments: 에이전트 할당
            query: 사용자 쿼리
        
        Returns:
            List[Dict[str, Any]]: 작업 결과
        """
        results = []
        
        # 의존성 그래프 생성
        dependency_map = {}
        for assignment in assignments:
            task_id = assignment.get("task_id")
            dependencies = assignment.get("dependencies", [])
            dependency_map[task_id] = dependencies
        
        # 의존성 없는 작업부터 실행
        while assignments:
            executable = []
            for assignment in assignments:
                task_id = assignment.get("task_id")
                dependencies = dependency_map.get(task_id, [])
                
                # 의존성 검사 (모든 의존 작업이 완료되었는지)
                completed_deps = [r.get("task_id") for r in results]
                if all(dep in completed_deps for dep in dependencies):
                    executable.append(assignment)
            
            if not executable:
                logger.warning("Circular dependency or unresolvable dependencies found")
                break
            
            # 실행 가능한 작업 실행
            for assignment in executable:
                assignments.remove(assignment)
                
                agent_type = assignment.get("agent_type")
                task_id = assignment.get("task_id")
                task_description = assignment.get("task_description")
                
                try:
                    # 에이전트에 요청 전송
                    message_id = self.send_request(
                        receiver=f"{agent_type}_agent",  # 에이전트 ID는 에이전트 유형에 따름
                        action="process_task",
                        data={
                            "query": query,
                            "task_description": task_description
                        },
                        metadata={
                            "task_id": task_id
                        }
                    )
                    
                    # 응답 대기
                    response = self.wait_for_response(timeout=60.0)
                    
                    if response and response.message_type == "response":
                        # 결과 저장
                        result = {
                            "task_id": task_id,
                            "agent_type": agent_type,
                            "status": "completed",
                            "result": response.content.get("result", {})
                        }
                    else:
                        # 응답 실패
                        result = {
                            "task_id": task_id,
                            "agent_type": agent_type,
                            "status": "failed",
                            "error": "No response or error from agent"
                        }
                
                except Exception as e:
                    logger.error(f"Error executing task {task_id}: {str(e)}")
                    # 실패 결과
                    result = {
                        "task_id": task_id,
                        "agent_type": agent_type,
                        "status": "failed",
                        "error": str(e)
                    }
                
                results.append(result)
        
        return results
    
    @handle_agent_error()
    def _handle_integrate_results(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        결과 통합 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        results = message_content.get("data", {}).get("results", [])
        query = message_content.get("data", {}).get("query", "")
        
        if not results:
            raise TaskExecutionError(
                message="No results provided for integration",
                agent_id=self.agent_id,
                task_id="integrate_results"
            )
        
        # 결과 통합 실행
        integrated = self._integrate_results(results, query)
        
        return {
            "status": "success",
            "integrated_result": integrated
        }
    
    def _integrate_results(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        작업 결과 통합
        
        Args:
            results: 작업 결과 목록
            query: 사용자 쿼리
        
        Returns:
            Dict[str, Any]: 통합된 결과
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "explanation": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "information": {"type": "string"}
                        }
                    }
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["answer", "explanation", "sources", "confidence"]
        }
        
        # 결과 데이터 정리
        result_sections = []
        for i, result in enumerate(results):
            if result.get("status") == "completed":
                agent_type = result.get("agent_type", "unknown")
                task_id = result.get("task_id", f"task_{i}")
                data = result.get("result", {})
                
                result_sections.append(f"### Result from {agent_type} ({task_id}):\n{json.dumps(data, ensure_ascii=False, indent=2)}")
        
        # 프롬프트 준비
        prompt = f"""사용자의 쿼리에 대해 여러 에이전트의 결과를 통합하여 최종 응답을 생성하세요:

사용자 쿼리: "{query}"

에이전트 결과:
{''.join(result_sections)}

위 결과를 활용하여 사용자 쿼리에 대한 종합적인 답변을 제공하세요. 여러 소스의 정보를 조합하고, 충돌하는 정보가 있다면 신뢰도를 평가하여 해결하세요.

요약본이 아닌 완전한 응답을 제공하세요. 
중요한 세부 사항이 누락되지 않도록 하세요.
모든 정보 출처를 명확히 표시하세요.

제공된 JSON 스키마에 맞게 응답하세요."""
        
        # LLM으로 결과 통합
        try:
            integrated_result = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info(f"Results integration completed: {len(integrated_result.get('sources', []))} sources")
            return integrated_result
            
        except Exception as e:
            logger.error(f"Error in results integration: {str(e)}")
            # 기본 통합 결과 반환
            return {
                "answer": "죄송합니다, 결과 통합 중 오류가 발생했습니다.",
                "explanation": "제공된 결과를 통합할 수 없습니다.",
                "sources": [],
                "confidence": 0.0
            }
    
    @handle_agent_error()
    def _handle_validate_final_result(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        최종 결과 검증 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        result = message_content.get("data", {}).get("result", {})
        query = message_content.get("data", {}).get("query", "")
        
        if not result:
            raise TaskExecutionError(
                message="No result provided for validation",
                agent_id=self.agent_id,
                task_id="validate_final_result"
            )
        
        # 결과 검증 실행
        validation = self._validate_result(result, query)
        
        return {
            "status": "success",
            "validation": validation
        }
    
    def _validate_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        최종 결과 검증
        
        Args:
            result: 통합된 결과
            query: 사용자 쿼리
        
        Returns:
            Dict[str, Any]: 검증 결과
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0, "maximum": 10},
                "feedback": {"type": "string"},
                "missing_information": {"type": "array", "items": {"type": "string"}},
                "factual_errors": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["is_valid", "score", "feedback"]
        }
        
        # 프롬프트 준비
        prompt = f"""다음 쿼리와 결과를 검토하고 결과의 품질을 평가하세요:

사용자 쿼리: "{query}"

생성된 결과:
{json.dumps(result, ensure_ascii=False, indent=2)}

다음 기준으로 결과를 평가하세요:
1. 완전성: 쿼리의 모든 측면이 다루어졌는가?
2. 정확성: 사실적 오류가 없는가?
3. 일관성: 내부적으로 일관성이 있는가?
4. 관련성: 쿼리와 직접적으로 관련이 있는가?
5. 출처: 정보 출처가 명확하게 제시되었는가?

0점(매우 나쁨)~10점(매우 좋음) 척도로 평가하고, 7점 이상이면 유효한 것으로 간주하세요.
개선이 필요한 부분에 대한 구체적인 피드백을 제공하세요.

제공된 JSON 스키마에 맞게 응답하세요."""
        
        # LLM으로 결과 검증
        try:
            validation = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info(f"Result validation completed: {validation.get('is_valid')}, score: {validation.get('score')}")
            return validation
            
        except Exception as e:
            logger.error(f"Error in result validation: {str(e)}")
            # 기본 검증 결과 반환 (유효하지 않음)
            return {
                "is_valid": False,
                "score": 5.0,
                "feedback": "검증 과정에서 오류가 발생했습니다.",
                "missing_information": ["결과 검증 불가"],
                "factual_errors": []
            }
    
    def process_user_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        사용자 쿼리 처리
        
        Args:
            session_id: 세션 ID
            query: 사용자 쿼리
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        logger.info(f"Processing user query: {query[:100]}... [session: {session_id}]")
        
        # 세션 상태 초기화 또는 가져오기
        state = self.get_state(session_id)
        if not state:
            state = State(session_id=session_id, user_query=query)
            self.state_manager.add_state(state)
        
        # 대화 기록에 쿼리 추가
        self.add_to_conversation_history(session_id, query)
        
        # 초기 상태 설정
        initial_state = {
            "session_id": session_id,
            "user_query": query,
            "current_stage": "planning",
            "conversation_history": state.conversation_history,
            "timestamp": datetime.now().isoformat()
        }
        
        # 워크플로우 실행
        try:
            config = RunnableConfig(
                configurable={
                    "checkpointer": self.checkpointer
                }
            )
            
            # 체크포인트 키 생성
            checkpoint_key = f"supervisor_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 워크플로우 실행
            final_state = self.workflow.invoke(
                initial_state,
                config={"configurable": {"checkpointer": {"key": checkpoint_key}}}
            )
            
            # 최종 결과 추출
            final_result = final_state.get("final_result", {})
            
            # 세션 상태 업데이트
            self.update_state(session_id, {
                "current_stage": "completed",
                "query_analysis": final_state.get("query_analysis", {}),
                "task_plan": final_state.get("task_plan", []),
                "agent_assignments": final_state.get("agent_assignments", []),
                "task_results": final_state.get("task_results", []),
                "final_result": final_result
            })
            
            # 대화 기록에 응답 추가
            answer = final_result.get("answer", "응답을 생성할 수 없습니다.")
            self.add_to_conversation_history(session_id, answer, is_user=False)
            
            return {
                "status": "success",
                "answer": answer,
                "explanation": final_result.get("explanation", ""),
                "sources": final_result.get("sources", []),
                "confidence": final_result.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            
            # 오류 시 기본 응답
            self.add_to_conversation_history(
                session_id, 
                "죄송합니다, 쿼리 처리 중 오류가 발생했습니다.", 
                is_user=False
            )
            
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}",
                "answer": "죄송합니다, 쿼리 처리 중 오류가 발생했습니다."
            }


# 전역 감독 에이전트 인스턴스
supervisor_agent = SupervisorAgent()

def get_supervisor_agent() -> SupervisorAgent:
    """감독 에이전트 인스턴스 가져오기"""
    return supervisor_agent 