#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 오류 처리 모듈
"""

import traceback
import logging
import functools
import time
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union, Type, cast
from datetime import datetime
from threading import Lock

from core.event_bus import get_event_bus, Event, publish_system_event

# 로깅 설정
logger = logging.getLogger(__name__)

# 타입 변수 정의
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class AgentError(Exception):
    """에이전트 오류 기본 클래스"""
    
    def __init__(self, message: str, agent_id: str, error_code: Optional[str] = None):
        """
        에이전트 오류 초기화
        
        Args:
            message: 오류 메시지
            agent_id: 오류가 발생한 에이전트 ID
            error_code: 오류 코드
        """
        super().__init__(message)
        self.message = message
        self.agent_id = agent_id
        self.error_code = error_code or "AGENT_ERROR"
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """오류를 딕셔너리로 변환"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "agent_id": self.agent_id,
            "error_code": self.error_code,
            "timestamp": self.timestamp
        }


class CommunicationError(AgentError):
    """에이전트 간 통신 오류"""
    
    def __init__(self, message: str, agent_id: str, target_agent_id: Optional[str] = None):
        """
        통신 오류 초기화
        
        Args:
            message: 오류 메시지
            agent_id: 오류가 발생한 에이전트 ID
            target_agent_id: 통신 대상 에이전트 ID
        """
        super().__init__(message, agent_id, "COMMUNICATION_ERROR")
        self.target_agent_id = target_agent_id


class LLMError(AgentError):
    """LLM API 오류"""
    
    def __init__(self, message: str, agent_id: str, llm_response: Optional[Dict[str, Any]] = None):
        """
        LLM 오류 초기화
        
        Args:
            message: 오류 메시지
            agent_id: 오류가 발생한 에이전트 ID
            llm_response: LLM 응답 정보
        """
        super().__init__(message, agent_id, "LLM_ERROR")
        self.llm_response = llm_response or {}


class ResourceError(AgentError):
    """자원 액세스 오류"""
    
    def __init__(self, message: str, agent_id: str, resource_type: str, resource_id: Optional[str] = None):
        """
        자원 오류 초기화
        
        Args:
            message: 오류 메시지
            agent_id: 오류가 발생한 에이전트 ID
            resource_type: 자원 유형
            resource_id: 자원 ID
        """
        super().__init__(message, agent_id, "RESOURCE_ERROR")
        self.resource_type = resource_type
        self.resource_id = resource_id


class TaskExecutionError(AgentError):
    """작업 실행 오류"""
    
    def __init__(
        self, 
        message: str, 
        agent_id: str, 
        task_id: Optional[str] = None, 
        task_type: Optional[str] = None
    ):
        """
        작업 실행 오류 초기화
        
        Args:
            message: 오류 메시지
            agent_id: 오류가 발생한 에이전트 ID
            task_id: 작업 ID
            task_type: 작업 유형
        """
        super().__init__(message, agent_id, "TASK_EXECUTION_ERROR")
        self.task_id = task_id
        self.task_type = task_type


class ErrorHandler:
    """오류 처리 관리자"""
    
    def __init__(self):
        """오류 처리 관리자 초기화"""
        self._error_history: List[Dict[str, Any]] = []
        self._error_handlers: Dict[str, List[Callable]] = {}
        self._default_handlers: List[Callable] = []
        self._lock = Lock()
    
    def register_handler(self, error_type: str, handler: Callable[[AgentError], None]) -> None:
        """
        특정 오류 유형에 대한 핸들러 등록
        
        Args:
            error_type: 오류 유형
            handler: 오류 핸들러 함수
        """
        with self._lock:
            if error_type not in self._error_handlers:
                self._error_handlers[error_type] = []
            
            if handler not in self._error_handlers[error_type]:
                self._error_handlers[error_type].append(handler)
                logger.debug(f"Registered handler for error type: {error_type}")
    
    def register_default_handler(self, handler: Callable[[AgentError], None]) -> None:
        """
        기본 오류 핸들러 등록
        
        Args:
            handler: 오류 핸들러 함수
        """
        with self._lock:
            if handler not in self._default_handlers:
                self._default_handlers.append(handler)
                logger.debug("Registered default error handler")
    
    def handle_error(self, error: AgentError) -> None:
        """
        오류 처리
        
        Args:
            error: 처리할 오류
        """
        with self._lock:
            # 오류 이력에 추가
            self._error_history.append(error.to_dict())
            
            # 오류 로깅
            logger.error(f"Agent error: {error.message} [Agent: {error.agent_id}, Code: {error.error_code}]")
            
            # 이벤트 발행
            publish_system_event(
                event_type="error.occurred",
                data=error.to_dict(),
                source=error.agent_id
            )
            
            # 특정 오류 유형 핸들러 호출
            error_type = error.__class__.__name__
            handlers = self._error_handlers.get(error_type, [])
            
            for handler in handlers:
                try:
                    handler(error)
                except Exception as e:
                    logger.error(f"Error in error handler: {e}")
            
            # 기본 핸들러 호출
            if not handlers or error_type not in self._error_handlers:
                for handler in self._default_handlers:
                    try:
                        handler(error)
                    except Exception as e:
                        logger.error(f"Error in default error handler: {e}")
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """
        오류 이력 조회
        
        Returns:
            List[Dict[str, Any]]: 오류 이력
        """
        with self._lock:
            return self._error_history.copy()
    
    def get_agent_errors(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        특정 에이전트의 오류 이력 조회
        
        Args:
            agent_id: 에이전트 ID
        
        Returns:
            List[Dict[str, Any]]: 오류 이력
        """
        with self._lock:
            return [error for error in self._error_history if error.get("agent_id") == agent_id]
    
    def clear_error_history(self) -> None:
        """오류 이력 삭제"""
        with self._lock:
            self._error_history.clear()
            logger.debug("Cleared error history")


# 전역 오류 처리 관리자 인스턴스
error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """오류 처리 관리자 인스턴스 가져오기"""
    return error_handler


def handle_agent_error(error_type: Optional[Type[AgentError]] = None) -> Callable[[F], F]:
    """
    에이전트 오류 처리 데코레이터
    
    Args:
        error_type: 처리할 오류 유형 (없으면 모든 AgentError 처리)
    
    Returns:
        Callable: 데코레이터 함수
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, AgentError) and (error_type is None or isinstance(e, error_type)):
                    error_handler.handle_error(e)
                    return None
                else:
                    # AgentError가 아닌 예외는 그대로 전파
                    raise
        
        return cast(F, wrapper)
    
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable[[F], F]:
    """
    재시도 데코레이터
    
    Args:
        max_attempts: 최대 시도 횟수
        delay: 초기 대기 시간 (초)
        backoff: 대기 시간 증가 배수
    
    Returns:
        Callable: 데코레이터 함수
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            current_delay = delay
            last_exception = None
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    last_exception = e
                    
                    if attempts < max_attempts:
                        logger.warning(
                            f"Attempt {attempts}/{max_attempts} failed: {str(e)}. "
                            f"Retrying in {current_delay:.2f} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed.")
            
            # 모든 시도 실패 시 마지막 예외 재발생
            if last_exception:
                raise last_exception
        
        return cast(F, wrapper)
    
    return decorator


def default_error_handler(error: AgentError) -> None:
    """
    기본 오류 처리 함수
    
    Args:
        error: 처리할 오류
    """
    error_msg = f"Agent error handled: {error.message}"
    if hasattr(error, 'task_id') and getattr(error, 'task_id'):
        error_msg += f" [Task: {getattr(error, 'task_id')}]"
    
    logger.error(error_msg)
    traceback.print_exc()

# 기본 오류 핸들러 등록
error_handler.register_default_handler(default_error_handler)
