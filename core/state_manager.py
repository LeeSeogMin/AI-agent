#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 상태 관리 모듈
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import logging
from threading import Lock

# 로깅 설정
logger = logging.getLogger(__name__)

class State:
    """시스템 상태 클래스"""
    
    def __init__(self, session_id: Optional[str] = None, user_query: Optional[str] = None):
        """
        상태 초기화
        
        Args:
            session_id: 세션 ID (없으면 자동 생성)
            user_query: 사용자 쿼리
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.user_query = user_query
        self.parsed_intent = None
        self.current_stage = "planning"  # planning, information_gathering, analysis, synthesis, reporting
        self.active_agents = []
        self.task_status = {}
        self.shared_knowledge = {
            "collected_information": [],
            "analysis_results": [],
            "decisions": []
        }
        self.conversation_history = []
        self._metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        상태를 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 상태 딕셔너리
        """
        return {
            "session_id": self.session_id,
            "user_query": self.user_query,
            "parsed_intent": self.parsed_intent,
            "current_stage": self.current_stage,
            "active_agents": self.active_agents,
            "task_status": self.task_status,
            "shared_knowledge": self.shared_knowledge,
            "conversation_history": self.conversation_history,
            "_metadata": self._metadata
        }
    
    def to_json(self) -> str:
        """
        상태를 JSON 문자열로 변환
        
        Returns:
            str: JSON 문자열
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """
        딕셔너리에서 상태 생성
        
        Args:
            data: 상태 딕셔너리
        
        Returns:
            State: 생성된 상태 객체
        """
        state = cls(session_id=data.get("session_id"), user_query=data.get("user_query"))
        state.parsed_intent = data.get("parsed_intent")
        state.current_stage = data.get("current_stage", "planning")
        state.active_agents = data.get("active_agents", [])
        state.task_status = data.get("task_status", {})
        state.shared_knowledge = data.get("shared_knowledge", {"collected_information": [], "analysis_results": [], "decisions": []})
        state.conversation_history = data.get("conversation_history", [])
        state._metadata = data.get("_metadata", {"created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat()})
        return state
    
    @classmethod
    def from_json(cls, json_str: str) -> 'State':
        """
        JSON 문자열에서 상태 생성
        
        Args:
            json_str: JSON 문자열
        
        Returns:
            State: 생성된 상태 객체
        """
        return cls.from_dict(json.loads(json_str))


class StateManager:
    """시스템 상태 관리자"""
    
    def __init__(self):
        """상태 관리자 초기화"""
        self._states: Dict[str, State] = {}
        self._lock = Lock()
    
    def create_state(self, user_query: Optional[str] = None, session_id: Optional[str] = None) -> State:
        """
        새 상태 생성
        
        Args:
            user_query: 사용자 쿼리
            session_id: 세션 ID (없으면 자동 생성)
        
        Returns:
            State: 생성된 상태 객체
        """
        with self._lock:
            state = State(session_id=session_id, user_query=user_query)
            self._states[state.session_id] = state
            logger.info(f"Created new state with session_id: {state.session_id}")
            return state
    
    def get_state(self, session_id: str) -> Optional[State]:
        """
        세션 ID로 상태 조회
        
        Args:
            session_id: 세션 ID
        
        Returns:
            Optional[State]: 상태 객체 (없으면 None)
        """
        return self._states.get(session_id)
    
    def update_state(self, session_id: str, updates: Dict[str, Any]) -> Optional[State]:
        """
        상태 업데이트
        
        Args:
            session_id: 세션 ID
            updates: 업데이트할 필드와 값
        
        Returns:
            Optional[State]: 업데이트된 상태 (없으면 None)
        """
        with self._lock:
            state = self.get_state(session_id)
            if not state:
                logger.warning(f"No state found with session_id: {session_id}")
                return None
            
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            
            state._metadata["updated_at"] = datetime.now().isoformat()
            return state
    
    def add_conversation_entry(self, session_id: str, agent: str, message: str) -> bool:
        """
        대화 이력에 메시지 추가
        
        Args:
            session_id: 세션 ID
            agent: 에이전트 ID 또는 "user"
            message: 메시지 내용
        
        Returns:
            bool: 성공 여부
        """
        with self._lock:
            state = self.get_state(session_id)
            if not state:
                logger.warning(f"No state found with session_id: {session_id}")
                return False
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": agent,
                "message": message
            }
            
            state.conversation_history.append(entry)
            state._metadata["updated_at"] = datetime.now().isoformat()
            return True
    
    def update_task_status(self, session_id: str, task_id: str, 
                          status: str, progress: float = 0.0, 
                          result_location: Optional[str] = None) -> bool:
        """
        작업 상태 업데이트
        
        Args:
            session_id: 세션 ID
            task_id: 작업 ID
            status: 작업 상태 (pending, in_progress, completed, failed)
            progress: 진행률 (0.0 ~ 1.0)
            result_location: 결과 위치 참조
        
        Returns:
            bool: 성공 여부
        """
        with self._lock:
            state = self.get_state(session_id)
            if not state:
                logger.warning(f"No state found with session_id: {session_id}")
                return False
            
            state.task_status[task_id] = {
                "status": status,
                "progress": progress,
                "result_location": result_location,
                "updated_at": datetime.now().isoformat()
            }
            
            state._metadata["updated_at"] = datetime.now().isoformat()
            return True
    
    def add_to_shared_knowledge(self, session_id: str, knowledge_type: str, data: Any) -> bool:
        """
        공유 지식에 데이터 추가
        
        Args:
            session_id: 세션 ID
            knowledge_type: 지식 유형 (collected_information, analysis_results, decisions)
            data: 추가할 데이터
        
        Returns:
            bool: 성공 여부
        """
        with self._lock:
            state = self.get_state(session_id)
            if not state:
                logger.warning(f"No state found with session_id: {session_id}")
                return False
            
            if knowledge_type in state.shared_knowledge:
                state.shared_knowledge[knowledge_type].append(data)
            else:
                state.shared_knowledge[knowledge_type] = [data]
            
            state._metadata["updated_at"] = datetime.now().isoformat()
            return True
    
    def delete_state(self, session_id: str) -> bool:
        """
        상태 삭제
        
        Args:
            session_id: 세션 ID
        
        Returns:
            bool: 성공 여부
        """
        with self._lock:
            if session_id in self._states:
                del self._states[session_id]
                logger.info(f"Deleted state with session_id: {session_id}")
                return True
            return False

# 전역 상태 관리자 인스턴스
state_manager = StateManager()

def get_state_manager() -> StateManager:
    """상태 관리자 인스턴스 가져오기"""
    return state_manager
