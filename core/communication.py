#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 통신 모듈
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from threading import Lock
from queue import Queue

# 로깅 설정
logger = logging.getLogger(__name__)

class Message:
    """에이전트 간 메시지 클래스"""
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        content: Dict[str, Any],
        message_type: str = "request",
        priority: str = "normal",
        message_id: Optional[str] = None,
        references: Optional[List[str]] = None,
        status: str = "pending"
    ):
        """
        메시지 초기화
        
        Args:
            sender: 발신자 ID
            receiver: 수신자 ID 또는 'broadcast'
            content: 메시지 내용 (action, parameters, data, metadata 포함)
            message_type: 메시지 유형 (request, response, notification, error)
            priority: 우선순위 (high, normal, low)
            message_id: 메시지 ID (없으면 자동 생성)
            references: 관련 메시지 ID 목록
            status: 메시지 상태 (pending, processing, completed, failed)
        """
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.priority = priority
        self.content = content
        self.references = references or []
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """
        메시지를 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 메시지 딕셔너리
        """
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "priority": self.priority,
            "content": self.content,
            "references": self.references,
            "status": self.status
        }
    
    def to_json(self) -> str:
        """
        메시지를 JSON 문자열로 변환
        
        Returns:
            str: JSON 문자열
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        딕셔너리에서 메시지 생성
        
        Args:
            data: 메시지 딕셔너리
        
        Returns:
            Message: 생성된 메시지 객체
        """
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            message_type=data.get("message_type", "request"),
            priority=data.get("priority", "normal"),
            message_id=data.get("message_id"),
            references=data.get("references", []),
            status=data.get("status", "pending")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """
        JSON 문자열에서 메시지 생성
        
        Args:
            json_str: JSON 문자열
        
        Returns:
            Message: 생성된 메시지 객체
        """
        return cls.from_dict(json.loads(json_str))


class CommunicationManager:
    """에이전트 간 통신 관리자"""
    
    def __init__(self):
        """통신 관리자 초기화"""
        self._message_queues: Dict[str, Queue] = {}
        self._message_history: Dict[str, List[Message]] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = Lock()
    
    def register_agent(self, agent_id: str) -> None:
        """
        에이전트 등록
        
        Args:
            agent_id: 에이전트 ID
        """
        with self._lock:
            if agent_id not in self._message_queues:
                self._message_queues[agent_id] = Queue()
                self._message_history[agent_id] = []
                self._subscribers[agent_id] = []
                logger.info(f"Registered agent: {agent_id}")
    
    def subscribe(self, agent_id: str, callback: Callable[[Message], None]) -> None:
        """
        메시지 구독
        
        Args:
            agent_id: 에이전트 ID
            callback: 메시지 수신 시 호출할 콜백 함수
        """
        with self._lock:
            if agent_id not in self._subscribers:
                self._subscribers[agent_id] = []
            self._subscribers[agent_id].append(callback)
    
    def send_message(self, message: Message) -> bool:
        """
        메시지 전송
        
        Args:
            message: 전송할 메시지
        
        Returns:
            bool: 성공 여부
        """
        if message.receiver == "broadcast":
            return self._broadcast_message(message)
        
        with self._lock:
            if message.receiver not in self._message_queues:
                logger.warning(f"Unknown receiver: {message.receiver}")
                return False
            
            # 메시지 큐에 추가
            self._message_queues[message.receiver].put(message)
            
            # 발신자 이력에 추가
            if message.sender in self._message_history:
                self._message_history[message.sender].append(message)
            
            # 수신자 이력에 추가
            self._message_history[message.receiver].append(message)
            
            # 구독자에게 알림
            for callback in self._subscribers.get(message.receiver, []):
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
            
            logger.debug(f"Message sent: {message.message_id} from {message.sender} to {message.receiver}")
            return True
    
    def _broadcast_message(self, message: Message) -> bool:
        """
        브로드캐스트 메시지 전송
        
        Args:
            message: 브로드캐스트할 메시지
        
        Returns:
            bool: 성공 여부
        """
        with self._lock:
            success = True
            
            # 발신자 이력에 추가
            if message.sender in self._message_history:
                self._message_history[message.sender].append(message)
            
            # 모든 에이전트에게 전송 (발신자 제외)
            for agent_id, queue in self._message_queues.items():
                if agent_id != message.sender:
                    # 각 수신자에 대한 복사본 생성
                    copy_message = Message(
                        sender=message.sender,
                        receiver=agent_id,
                        content=message.content.copy(),
                        message_type=message.message_type,
                        priority=message.priority,
                        references=message.references.copy(),
                        status=message.status
                    )
                    
                    # 메시지 큐에 추가
                    queue.put(copy_message)
                    
                    # 수신자 이력에 추가
                    self._message_history[agent_id].append(copy_message)
                    
                    # 구독자에게 알림
                    for callback in self._subscribers.get(agent_id, []):
                        try:
                            callback(copy_message)
                        except Exception as e:
                            logger.error(f"Error in subscriber callback: {e}")
                            success = False
            
            logger.debug(f"Broadcast message sent: {message.message_id} from {message.sender}")
            return success
    
    def get_message(self, agent_id: str, block: bool = False, timeout: Optional[float] = None) -> Optional[Message]:
        """
        메시지 수신
        
        Args:
            agent_id: 에이전트 ID
            block: 메시지가 없을 때 대기 여부
            timeout: 대기 시간 (초)
        
        Returns:
            Optional[Message]: 수신한 메시지 (없으면 None)
        """
        if agent_id not in self._message_queues:
            logger.warning(f"Unknown agent: {agent_id}")
            return None
        
        try:
            return self._message_queues[agent_id].get(block=block, timeout=timeout)
        except Exception:
            return None
    
    def update_message_status(self, message_id: str, status: str) -> bool:
        """
        메시지 상태 업데이트
        
        Args:
            message_id: 메시지 ID
            status: 새 상태 (pending, processing, completed, failed)
        
        Returns:
            bool: 성공 여부
        """
        with self._lock:
            for agent_id, messages in self._message_history.items():
                for message in messages:
                    if message.message_id == message_id:
                        message.status = status
                        logger.debug(f"Updated message status: {message_id} -> {status}")
                        return True
            
            logger.warning(f"Message not found: {message_id}")
            return False
    
    def get_message_history(self, agent_id: str) -> List[Message]:
        """
        에이전트의 메시지 이력 조회
        
        Args:
            agent_id: 에이전트 ID
        
        Returns:
            List[Message]: 메시지 이력
        """
        return self._message_history.get(agent_id, []).copy()
    
    def clear_message_queue(self, agent_id: str) -> bool:
        """
        에이전트의 메시지 큐 비우기
        
        Args:
            agent_id: 에이전트 ID
        
        Returns:
            bool: 성공 여부
        """
        if agent_id not in self._message_queues:
            logger.warning(f"Unknown agent: {agent_id}")
            return False
        
        with self._lock:
            while not self._message_queues[agent_id].empty():
                self._message_queues[agent_id].get()
            
            logger.debug(f"Cleared message queue for agent: {agent_id}")
            return True

# 전역 통신 관리자 인스턴스
communication_manager = CommunicationManager()

def get_communication_manager() -> CommunicationManager:
    """통신 관리자 인스턴스 가져오기"""
    return communication_manager


def create_request_message(
    sender: str, 
    receiver: str, 
    action: str, 
    parameters: Dict[str, Any] = None, 
    data: Dict[str, Any] = None, 
    metadata: Dict[str, Any] = None,
    priority: str = "normal",
    references: List[str] = None
) -> Message:
    """
    요청 메시지 생성 유틸리티
    
    Args:
        sender: 발신자 ID
        receiver: 수신자 ID
        action: 요청 작업
        parameters: 작업 매개변수
        data: 작업 데이터
        metadata: 메타데이터
        priority: 우선순위
        references: 관련 메시지 ID 목록
    
    Returns:
        Message: 생성된 요청 메시지
    """
    content = {
        "action": action,
        "parameters": parameters or {},
        "data": data or {},
        "metadata": metadata or {}
    }
    
    return Message(
        sender=sender,
        receiver=receiver,
        content=content,
        message_type="request",
        priority=priority,
        references=references or []
    )


def create_response_message(
    sender: str, 
    receiver: str, 
    request_id: str, 
    result: Dict[str, Any] = None, 
    status: str = "completed", 
    error: Dict[str, Any] = None,
    priority: str = "normal"
) -> Message:
    """
    응답 메시지 생성 유틸리티
    
    Args:
        sender: 발신자 ID
        receiver: 수신자 ID
        request_id: 요청 메시지 ID
        result: 작업 결과
        status: 작업 상태
        error: 오류 정보
        priority: 우선순위
    
    Returns:
        Message: 생성된 응답 메시지
    """
    content = {
        "result": result or {},
        "status": status,
        "error": error
    }
    
    return Message(
        sender=sender,
        receiver=receiver,
        content=content,
        message_type="response",
        priority=priority,
        references=[request_id],
        status=status
    )


def create_error_message(
    sender: str, 
    receiver: str, 
    error_type: str, 
    description: str, 
    severity: str = "error", 
    related_message_id: Optional[str] = None,
    priority: str = "high"
) -> Message:
    """
    오류 메시지 생성 유틸리티
    
    Args:
        sender: 발신자 ID
        receiver: 수신자 ID
        error_type: 오류 유형
        description: 오류 설명
        severity: 심각도 (error, warning, info)
        related_message_id: 관련 메시지 ID
        priority: 우선순위
    
    Returns:
        Message: 생성된 오류 메시지
    """
    content = {
        "error_type": error_type,
        "description": description,
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    }
    
    references = [related_message_id] if related_message_id else []
    
    return Message(
        sender=sender,
        receiver=receiver,
        content=content,
        message_type="error",
        priority=priority,
        references=references,
        status="failed"
    )
