#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 감독 에이전트 모듈
"""

import json
import logging
import uuid
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

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
        
        # 작업 핸들러 등록
        self._register_task_handlers()
        
        logger.info(f"Supervisor agent initialized with ID: {self.agent_id}")
    
    def _register_task_handlers(self) -> None:
        """작업 핸들러 등록"""
        self.task_handlers = {
            "analyze_query": self._handle_analyze_query,
            "process_query": self._handle_process_query
        }
    
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
        
        # 쿼리 분석
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
                }
            },
            "required": ["intents", "topics", "information_needs"]
        }
        
        # 프롬프트 준비
        prompt = f"""사용자의 쿼리를 분석하여 아래 JSON 형식으로 결과를 제공하세요:

사용자 쿼리: "{query}"

분석 시 다음 측면을 고려하세요:
1. 쿼리의 주요 의도(예: 정보 요청, 비교, 설명 요구, 문제 해결 등)
2. 주요 주제와 키워드
3. 필요한 정보 유형

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
                "information_needs": ["general_information"]
            }
    
    @handle_agent_error()
    def _handle_process_query(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        쿼리 처리 작업 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        session_id = message_content.get("metadata", {}).get("session_id")
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for processing",
                agent_id=self.agent_id,
                task_id="process_query"
            )
        
        # 쿼리 처리 실행
        result = self.process_user_query(session_id, query)
        
        return {
            "status": "success",
            "result": result
        }

    def process_user_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        사용자 쿼리 처리 - 워크플로우 없이 직접 처리하는 간소화된 버전
        
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
            self.state_manager.create_state(user_query=query, session_id=session_id)
        
        # 대화 기록에 쿼리 추가
        self.state_manager.add_conversation_entry(session_id, "user", query)
        
        try:
            # 1. 쿼리 분석
            query_analysis = self._analyze_query(query)
            logger.info(f"Query analysis completed for {session_id}")
            
            # 2. 간단한 응답 생성
            prompt = f"""다음 사용자 쿼리에 대한 응답을 생성해주세요:
            
사용자 쿼리: "{query}"

쿼리 분석:
{json.dumps(query_analysis, ensure_ascii=False, indent=2)}

이 쿼리에 대해 명확하고 정보가 풍부한 답변을 제공해주세요.
필요한 경우 관련 정보와 설명을 포함하세요.
"""
            
            response_text = generate_text(
                prompt=prompt,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            # 최종 결과 구성
            final_result = {
                "answer": response_text,
                "explanation": "AI 생성 응답입니다.",
                "sources": [],
                "confidence": 0.8
            }
            
            # 세션 상태 업데이트
            self.update_state(session_id, {
                "current_stage": "completed",
                "query_analysis": query_analysis,
                "final_result": final_result
            })
            
            # 대화 기록에 응답 추가
            self.state_manager.add_conversation_entry(session_id, self.agent_id, response_text)
            
            return {
                "status": "success",
                "answer": response_text,
                "explanation": "AI 생성 응답입니다.",
                "sources": [],
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            
            # 오류 시 기본 응답
            error_message = "죄송합니다, 쿼리 처리 중 오류가 발생했습니다."
            self.state_manager.add_conversation_entry(
                session_id, 
                self.agent_id, 
                error_message
            )
            
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}",
                "answer": error_message
            }


# 전역 감독 에이전트 인스턴스
supervisor_agent = SupervisorAgent()

def get_supervisor_agent() -> SupervisorAgent:
    """감독 에이전트 인스턴스 가져오기"""
    return supervisor_agent 