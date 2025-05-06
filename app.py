#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 메인 애플리케이션
"""

import os
import sys
import json
import uuid
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import time
import asyncio

from core.config import get_config
from core.state_manager import get_state_manager
from core.logging_manager import get_logger
from agents.supervisor_agent import get_supervisor_agent
from agents.rag_agent import get_rag_agent
from agents.web_search_agent import get_web_search_agent
from agents.data_analysis_agent import get_data_analysis_agent
from agents.psychological_agent import get_psychological_agent
from agents.report_writer_agent import get_report_writer_agent

# 설정 가져오기
config = get_config()

# 로거 설정
logger = get_logger("app")

class Application:
    """멀티 에이전트 애플리케이션 클래스"""
    
    def __init__(self):
        """애플리케이션 초기화"""
        self.supervisor = get_supervisor_agent()
        self.active_agents = {}
        
        # 사용 가능한 에이전트 등록
        if config.agents["rag"]["enabled"]:
            self.active_agents["rag"] = get_rag_agent()
        
        if config.agents["web_search"]["enabled"]:
            self.active_agents["web_search"] = get_web_search_agent()
        
        if config.agents["data_analysis"]["enabled"]:
            self.active_agents["data_analysis"] = get_data_analysis_agent()
        
        if config.agents["psychological"]["enabled"]:
            self.active_agents["psychological"] = get_psychological_agent()
        
        if config.agents["report_writer"]["enabled"]:
            self.active_agents["report_writer"] = get_report_writer_agent()
        
        logger.info(f"Application initialized with {len(self.active_agents)} active agents")
    
    def start(self):
        """애플리케이션 시작"""
        logger.info("Starting multi-agent system...")
        
        # 감독 에이전트 시작
        self.supervisor.start()
        
        # 각 에이전트 시작
        for agent_type, agent in self.active_agents.items():
            logger.info(f"Starting {agent_type} agent...")
            agent.start()
        
        logger.info("All agents started successfully")
    
    def stop(self):
        """애플리케이션 정지"""
        logger.info("Stopping multi-agent system...")
        
        # 각 에이전트 정지
        for agent_type, agent in self.active_agents.items():
            logger.info(f"Stopping {agent_type} agent...")
            agent.stop()
        
        # 감독 에이전트 정지
        self.supervisor.stop()
        
        logger.info("All agents stopped successfully")
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        사용자 쿼리 처리
        
        Args:
            query: 사용자 쿼리
            session_id: 세션 ID (없으면 자동 생성)
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        session_id = session_id or str(uuid.uuid4())
        
        start_time = time.time()
        logger.info(f"Processing query: {query[:100]}... [session: {session_id}]")
        
        # 감독 에이전트에 쿼리 처리 요청
        result = self.supervisor.process_user_query(session_id, query)
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        # 결과에 메타데이터 추가
        result["metadata"] = {
            "session_id": session_id,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        세션 대화 기록 조회
        
        Args:
            session_id: 세션 ID
        
        Returns:
            List[Dict[str, Any]]: 대화 기록
        """
        state_manager = get_state_manager()
        state = state_manager.get_state(session_id)
        
        if not state:
            return []
        
        return state.conversation_history


def main():
    """애플리케이션 실행"""
    parser = argparse.ArgumentParser(description="LangGraph 멀티 에이전트 시스템")
    parser.add_argument("--query", type=str, help="처리할 쿼리 (CLI 모드)")
    parser.add_argument("--session", type=str, help="세션 ID")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    
    args = parser.parse_args()
    
    # 디버그 모드 설정
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 애플리케이션 초기화
        app = Application()
        app.start()
        
        if args.query:
            # CLI 모드
            result = app.process_query(args.query, args.session)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            # 인터랙티브 모드
            print("\nLangGraph 멀티 에이전트 시스템")
            print("종료하려면 'exit' 또는 'quit'을 입력하세요.\n")
            
            session_id = args.session or str(uuid.uuid4())
            print(f"세션 ID: {session_id}\n")
            
            while True:
                query = input("\n질문을 입력하세요: ")
                
                if query.lower() in ["exit", "quit"]:
                    break
                
                if not query.strip():
                    continue
                
                try:
                    result = app.process_query(query, session_id)
                    print(f"\n답변: {result['answer']}\n")
                    
                    if "explanation" in result and result["explanation"]:
                        print(f"설명: {result['explanation']}\n")
                    
                    if "sources" in result and result["sources"]:
                        print("출처:")
                        for source in result["sources"]:
                            print(f"- {source.get('source', 'Unknown')}: {source.get('information', '')}")
                except Exception as e:
                    print(f"\n오류 발생: {str(e)}")
        
    except KeyboardInterrupt:
        print("\n프로그램 종료...")
    except Exception as e:
        logger.error(f"애플리케이션 오류: {str(e)}")
    finally:
        # 애플리케이션 종료
        if 'app' in locals():
            app.stop()


if __name__ == "__main__":
    main() 