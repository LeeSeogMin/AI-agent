#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 이벤트 버스 모듈
"""

import uuid
from typing import Dict, List, Any, Optional, Callable, Set
import logging
from threading import Lock
from datetime import datetime

# 로깅 설정
logger = logging.getLogger(__name__)

class Event:
    """이벤트 클래스"""
    
    def __init__(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str,
        event_id: Optional[str] = None
    ):
        """
        이벤트 초기화
        
        Args:
            event_type: 이벤트 유형
            data: 이벤트 데이터
            source: 이벤트 발생 소스
            event_id: 이벤트 ID (없으면 자동 생성)
        """
        self.event_id = event_id or str(uuid.uuid4())
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """이벤트를 딕셔너리로 변환"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp
        }


class EventBus:
    """이벤트 버스 클래스"""
    
    def __init__(self):
        """이벤트 버스 초기화"""
        self._subscribers: Dict[str, Set[Callable[[Event], None]]] = {}
        self._all_subscribers: Set[Callable[[Event], None]] = set()
        self._lock = Lock()
        self._events: List[Event] = []
    
    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        특정 이벤트 유형 구독
        
        Args:
            event_type: 구독할 이벤트 유형
            callback: 이벤트 발생 시 호출할 콜백 함수
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = set()
            self._subscribers[event_type].add(callback)
            logger.debug(f"Subscribed to event type: {event_type}")
    
    def subscribe_all(self, callback: Callable[[Event], None]) -> None:
        """
        모든 이벤트 구독
        
        Args:
            callback: 이벤트 발생 시 호출할 콜백 함수
        """
        with self._lock:
            self._all_subscribers.add(callback)
            logger.debug("Subscribed to all events")
    
    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> bool:
        """
        특정 이벤트 유형 구독 취소
        
        Args:
            event_type: 구독 취소할 이벤트 유형
            callback: 구독 취소할 콜백 함수
        
        Returns:
            bool: 성공 여부
        """
        with self._lock:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from event type: {event_type}")
                return True
            return False
    
    def unsubscribe_all(self, callback: Callable[[Event], None]) -> bool:
        """
        모든 이벤트 구독 취소
        
        Args:
            callback: 구독 취소할 콜백 함수
        
        Returns:
            bool: 성공 여부
        """
        with self._lock:
            if callback in self._all_subscribers:
                self._all_subscribers.remove(callback)
                logger.debug("Unsubscribed from all events")
                return True
            
            # 특정 이벤트 유형 구독도 취소
            unsubscribed = False
            for event_type, subscribers in self._subscribers.items():
                if callback in subscribers:
                    subscribers.remove(callback)
                    unsubscribed = True
            
            if unsubscribed:
                logger.debug("Unsubscribed from specific event types")
            
            return unsubscribed
    
    def publish(self, event: Event) -> None:
        """
        이벤트 발행
        
        Args:
            event: 발행할 이벤트
        """
        with self._lock:
            # 이벤트 이력에 추가
            self._events.append(event)
            
            # 특정 이벤트 유형 구독자에게 알림
            subscribers = self._subscribers.get(event.event_type, set())
            for subscriber in subscribers:
                try:
                    subscriber(event)
                except Exception as e:
                    logger.error(f"Error in event subscriber: {e}")
            
            # 모든 이벤트 구독자에게 알림
            for subscriber in self._all_subscribers:
                try:
                    subscriber(event)
                except Exception as e:
                    logger.error(f"Error in all-events subscriber: {e}")
            
            logger.debug(f"Published event: {event.event_type} from {event.source}")
    
    def get_events_by_type(self, event_type: str) -> List[Event]:
        """
        특정 유형의 이벤트 조회
        
        Args:
            event_type: 조회할 이벤트 유형
        
        Returns:
            List[Event]: 이벤트 목록
        """
        with self._lock:
            return [event for event in self._events if event.event_type == event_type]
    
    def get_events_by_source(self, source: str) -> List[Event]:
        """
        특정 소스의 이벤트 조회
        
        Args:
            source: 조회할 이벤트 소스
        
        Returns:
            List[Event]: 이벤트 목록
        """
        with self._lock:
            return [event for event in self._events if event.source == source]
    
    def get_all_events(self) -> List[Event]:
        """
        모든 이벤트 조회
        
        Returns:
            List[Event]: 이벤트 목록
        """
        with self._lock:
            return self._events.copy()
    
    def clear_events(self) -> None:
        """모든 이벤트 이력 삭제"""
        with self._lock:
            self._events.clear()
            logger.debug("Cleared all event history")


# 전역 이벤트 버스 인스턴스
event_bus = EventBus()

def get_event_bus() -> EventBus:
    """이벤트 버스 인스턴스 가져오기"""
    return event_bus


def publish_agent_started_event(agent_id: str, agent_type: str) -> None:
    """
    에이전트 시작 이벤트 발행
    
    Args:
        agent_id: 에이전트 ID
        agent_type: 에이전트 유형
    """
    event = Event(
        event_type="agent.started",
        data={"agent_id": agent_id, "agent_type": agent_type},
        source=agent_id
    )
    event_bus.publish(event)


def publish_agent_stopped_event(agent_id: str, agent_type: str) -> None:
    """
    에이전트 정지 이벤트 발행
    
    Args:
        agent_id: 에이전트 ID
        agent_type: 에이전트 유형
    """
    event = Event(
        event_type="agent.stopped",
        data={"agent_id": agent_id, "agent_type": agent_type},
        source=agent_id
    )
    event_bus.publish(event)


def publish_task_started_event(agent_id: str, task_id: str, task_type: str) -> None:
    """
    작업 시작 이벤트 발행
    
    Args:
        agent_id: 에이전트 ID
        task_id: 작업 ID
        task_type: 작업 유형
    """
    event = Event(
        event_type="task.started",
        data={"agent_id": agent_id, "task_id": task_id, "task_type": task_type},
        source=agent_id
    )
    event_bus.publish(event)


def publish_task_completed_event(agent_id: str, task_id: str, task_type: str, result: Dict[str, Any]) -> None:
    """
    작업 완료 이벤트 발행
    
    Args:
        agent_id: 에이전트 ID
        task_id: 작업 ID
        task_type: 작업 유형
        result: 작업 결과
    """
    event = Event(
        event_type="task.completed",
        data={
            "agent_id": agent_id, 
            "task_id": task_id, 
            "task_type": task_type,
            "result": result
        },
        source=agent_id
    )
    event_bus.publish(event)


def publish_task_failed_event(agent_id: str, task_id: str, task_type: str, error: Dict[str, Any]) -> None:
    """
    작업 실패 이벤트 발행
    
    Args:
        agent_id: 에이전트 ID
        task_id: 작업 ID
        task_type: 작업 유형
        error: 오류 정보
    """
    event = Event(
        event_type="task.failed",
        data={
            "agent_id": agent_id, 
            "task_id": task_id, 
            "task_type": task_type,
            "error": error
        },
        source=agent_id
    )
    event_bus.publish(event)


def publish_system_event(event_type: str, data: Dict[str, Any], source: str = "system") -> None:
    """
    시스템 이벤트 발행
    
    Args:
        event_type: 이벤트 유형
        data: 이벤트 데이터
        source: 이벤트 소스
    """
    event = Event(
        event_type=event_type,
        data=data,
        source=source
    )
    event_bus.publish(event)
