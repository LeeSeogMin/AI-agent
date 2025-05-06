#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 기본 에이전트 모듈
"""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
import logging
import json
import os
from pathlib import Path

from core.config import get_config
from core.state_manager import get_state_manager, State
from core.communication import (
    get_communication_manager, Message, 
    create_request_message, create_response_message, create_error_message
)
from core.event_bus import (
    get_event_bus, publish_agent_started_event, 
    publish_agent_stopped_event, publish_task_started_event, 
    publish_task_completed_event, publish_task_failed_event
)
from core.error_handler import (
    get_error_handler, AgentError, TaskExecutionError, 
    LLMError, handle_agent_error, retry
)
from core.logging_manager import get_agent_logger

# 설정 가져오기
config = get_config()

class Agent(ABC):
    """기본 에이전트 추상 클래스"""
    
    def __init__(
        self, 
        agent_id: Optional[str] = None, 
        agent_type: str = "base",
        system_message: Optional[str] = None
    ):
        """
        에이전트 초기화
        
        Args:
            agent_id: 에이전트 ID (없으면 자동 생성)
            agent_type: 에이전트 유형
            system_message: 시스템 메시지
        """
        self.agent_id = agent_id or f"{agent_type}_{str(uuid.uuid4())[:8]}"
        self.agent_type = agent_type
        self.system_message = system_message
        
        # 서비스 매니저
        self.state_manager = get_state_manager()
        self.communication_manager = get_communication_manager()
        self.event_bus = get_event_bus()
        self.error_handler = get_error_handler()
        
        # 로거 설정
        self.logger = get_agent_logger(self.agent_id)
        
        # 통신 관리자에 등록
        self.communication_manager.register_agent(self.agent_id)
        
        # 메시지 콜백 등록
        self.communication_manager.subscribe(
            self.agent_id, 
            self._handle_incoming_message
        )
        
        # 태스크 매핑
        self.task_handlers: Dict[str, Callable] = {}
        self._register_task_handlers()
        
        self.logger.info(f"{self.agent_type} agent initialized with ID: {self.agent_id}")
    
    def _register_task_handlers(self) -> None:
        """작업 핸들러 등록 (하위 클래스에서 구현)"""
        pass
    
    def start(self) -> None:
        """에이전트 시작"""
        self.logger.info(f"Starting {self.agent_type} agent: {self.agent_id}")
        publish_agent_started_event(self.agent_id, self.agent_type)
    
    def stop(self) -> None:
        """에이전트 정지"""
        self.logger.info(f"Stopping {self.agent_type} agent: {self.agent_id}")
        publish_agent_stopped_event(self.agent_id, self.agent_type)
    
    @handle_agent_error()
    def _handle_incoming_message(self, message: Message) -> None:
        """
        수신 메시지 처리
        
        Args:
            message: 수신 메시지
        """
        self.logger.debug(f"Received message: {message.message_id} from {message.sender}")
        
        if message.message_type == "request":
            self._handle_request(message)
        elif message.message_type == "response":
            self._handle_response(message)
        elif message.message_type == "error":
            self._handle_error(message)
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
    
    def _handle_request(self, message: Message) -> None:
        """
        요청 메시지 처리
        
        Args:
            message: 요청 메시지
        """
        action = message.content.get("action")
        if not action:
            self._send_error_response(
                message, 
                "Invalid request: missing action", 
                "INVALID_REQUEST"
            )
            return
        
        handler = self.task_handlers.get(action)
        if not handler:
            self._send_error_response(
                message, 
                f"Unsupported action: {action}", 
                "UNSUPPORTED_ACTION"
            )
            return
        
        # 작업 시작 이벤트 발행
        task_id = message.content.get("metadata", {}).get("task_id", message.message_id)
        publish_task_started_event(self.agent_id, task_id, action)
        
        try:
            # 작업 실행
            result = handler(message.content)
            
            # 작업 완료 이벤트 발행
            publish_task_completed_event(self.agent_id, task_id, action, result)
            
            # 응답 전송
            response = create_response_message(
                sender=self.agent_id,
                receiver=message.sender,
                request_id=message.message_id,
                result=result,
                status="completed"
            )
            self.communication_manager.send_message(response)
            
        except Exception as e:
            # 작업 실패 이벤트 발행
            error_data = {"message": str(e), "type": type(e).__name__}
            publish_task_failed_event(self.agent_id, task_id, action, error_data)
            
            # 오류 응답 전송
            self._send_error_response(
                message, 
                str(e), 
                "TASK_EXECUTION_ERROR"
            )
            
            # 로깅
            self.logger.error(f"Error executing task {action}: {e}")
            
            # 에이전트 오류면 오류 핸들러로 처리
            if isinstance(e, AgentError):
                self.error_handler.handle_error(e)
            else:
                # 일반 오류는 에이전트 오류로 변환하여 처리
                agent_error = TaskExecutionError(
                    message=str(e),
                    agent_id=self.agent_id,
                    task_id=task_id,
                    task_type=action
                )
                self.error_handler.handle_error(agent_error)
    
    def _handle_response(self, message: Message) -> None:
        """
        응답 메시지 처리 (하위 클래스에서 구현)
        
        Args:
            message: 응답 메시지
        """
        self.logger.debug(f"Received response: {message.message_id}")
    
    def _handle_error(self, message: Message) -> None:
        """
        오류 메시지 처리 (하위 클래스에서 구현)
        
        Args:
            message: 오류 메시지
        """
        error_type = message.content.get("error_type", "Unknown")
        description = message.content.get("description", "No description")
        self.logger.error(f"Received error from {message.sender}: {error_type} - {description}")
    
    def _send_error_response(
        self, 
        request: Message, 
        error_description: str, 
        error_type: str = "AGENT_ERROR",
        severity: str = "error"
    ) -> None:
        """
        오류 응답 전송
        
        Args:
            request: 원본 요청 메시지
            error_description: 오류 설명
            error_type: 오류 유형
            severity: 심각도
        """
        error_message = create_error_message(
            sender=self.agent_id,
            receiver=request.sender,
            error_type=error_type,
            description=error_description,
            severity=severity,
            related_message_id=request.message_id
        )
        self.communication_manager.send_message(error_message)
    
    def send_request(
        self, 
        receiver: str, 
        action: str, 
        parameters: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        priority: str = "normal"
    ) -> str:
        """
        요청 메시지 전송
        
        Args:
            receiver: 수신자 ID
            action: 요청 작업
            parameters: 작업 매개변수
            data: 작업 데이터
            metadata: 메타데이터
            priority: 우선순위
        
        Returns:
            str: 메시지 ID
        """
        message = create_request_message(
            sender=self.agent_id,
            receiver=receiver,
            action=action,
            parameters=parameters or {},
            data=data or {},
            metadata=metadata or {},
            priority=priority
        )
        
        success = self.communication_manager.send_message(message)
        if not success:
            self.logger.error(f"Failed to send message to {receiver}")
        
        return message.message_id
    
    def wait_for_response(
        self, 
        timeout: Optional[float] = None,
        expected_message_type: str = "response"
    ) -> Optional[Message]:
        """
        응답 대기
        
        Args:
            timeout: 최대 대기 시간 (초)
            expected_message_type: 기대하는 메시지 유형
        
        Returns:
            Optional[Message]: 수신한 메시지 (시간 초과시 None)
        """
        message = self.communication_manager.get_message(
            self.agent_id,
            block=True,
            timeout=timeout
        )
        
        if not message:
            return None
        
        if message.message_type != expected_message_type:
            self.logger.warning(
                f"Expected {expected_message_type} message, but got {message.message_type}"
            )
        
        return message
    
    def load_system_message(self) -> Optional[str]:
        """
        시스템 메시지 로드
        
        Returns:
            Optional[str]: 시스템 메시지
        """
        if self.system_message:
            return self.system_message
        
        # 프롬프트 파일 경로
        file_path = Path(__file__).parent.parent / "prompts" / self.agent_type / "system_message.txt"
        
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                self.system_message = f.read()
                return self.system_message
        
        return None
    
    @abstractmethod
    def process_user_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        사용자 쿼리 처리 (하위 클래스에서 구현)
        
        Args:
            session_id: 세션 ID
            query: 사용자 쿼리
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        pass
    
    def get_state(self, session_id: str) -> Optional[State]:
        """
        세션 상태 조회
        
        Args:
            session_id: 세션 ID
        
        Returns:
            Optional[State]: 세션 상태
        """
        return self.state_manager.get_state(session_id)
    
    def update_state(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        세션 상태 업데이트
        
        Args:
            session_id: 세션 ID
            updates: 업데이트할 필드와 값
        
        Returns:
            bool: 성공 여부
        """
        state = self.state_manager.update_state(session_id, updates)
        return state is not None
    
    def add_to_conversation_history(self, session_id: str, message: str) -> bool:
        """
        대화 이력에 메시지 추가
        
        Args:
            session_id: 세션 ID
            message: 메시지 내용
        
        Returns:
            bool: 성공 여부
        """
        return self.state_manager.add_conversation_entry(
            session_id, 
            self.agent_id, 
            message
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        에이전트 정보를 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 에이전트 정보
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "task_handlers": list(self.task_handlers.keys())
        }
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"{self.agent_type} Agent ({self.agent_id})"
