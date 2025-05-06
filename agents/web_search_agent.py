#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 웹 검색 에이전트 모듈
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union

from agents.base import Agent
from core.config import get_config
from core.error_handler import handle_agent_error, TaskExecutionError
from services.web.search_service import search_web
from services.web.html_parser import extract_content

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

class WebSearchAgent(Agent):
    """웹 검색 에이전트 클래스"""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        system_message: Optional[str] = None
    ):
        """
        웹 검색 에이전트 초기화
        
        Args:
            agent_id: 에이전트 ID (없으면 자동 생성)
            system_message: 시스템 메시지
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="web_search",
            system_message=system_message
        )
        
        # 시스템 메시지 로드
        if not self.system_message:
            self.system_message = self.load_system_message()
        
        # 작업 핸들러 등록
        self._register_task_handlers()
        
        # 검색 설정
        self.search_limit = config.web_search.get("search_limit", 5) if hasattr(config, 'web_search') else 5
        self.extract_limit = config.web_search.get("extract_limit", 3) if hasattr(config, 'web_search') else 3
        
        logger.info(f"Web search agent initialized with ID: {self.agent_id}")
    
    def _register_task_handlers(self) -> None:
        """작업 핸들러 등록"""
        self.task_handlers = {
            "process_task": self._handle_process_task,
            "search": self._handle_search,
            "extract_content": self._handle_extract_content,
            "analyze_results": self._handle_analyze_results
        }
    
    @handle_agent_error()
    def _handle_process_task(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        작업 처리 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        task_description = message_content.get("data", {}).get("task_description", "")
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for web search",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 검색어 형식화
        search_query = self._format_search_query(query, task_description)
        
        # 웹 검색 수행
        search_results = search_web(search_query, limit=self.search_limit)
        
        # 콘텐츠 추출
        extracted_contents = []
        for result in search_results[:self.extract_limit]:
            try:
                url = result.get("url", "")
                if url:
                    content = extract_content(url)
                    if content:
                        extracted_contents.append({
                            "url": url,
                            "title": result.get("title", ""),
                            "content": content,
                            "summary": self._summarize_content(content, query)
                        })
            except Exception as e:
                logger.warning(f"Error extracting content from {result.get('url', '')}: {str(e)}")
        
        # 검색 결과 분석
        analysis = self._analyze_search_results(extracted_contents, query)
        
        return {
            "status": "success",
            "result": {
                "search_query": search_query,
                "sources": [
                    {
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "summary": item.get("summary", "")
                    } for item in extracted_contents
                ],
                "analysis": analysis
            }
        }
    
    @handle_agent_error()
    def _handle_search(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        웹 검색 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for web search",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 웹 검색 수행
        search_results = search_web(query, limit=self.search_limit)
        
        return {
            "status": "success",
            "result": {
                "search_results": search_results
            }
        }
    
    @handle_agent_error()
    def _handle_extract_content(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        콘텐츠 추출 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        urls = message_content.get("data", {}).get("urls", [])
        
        if not urls:
            raise TaskExecutionError(
                message="No URLs provided for content extraction",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 콘텐츠 추출
        extracted_contents = []
        for url in urls:
            try:
                content = extract_content(url)
                if content:
                    extracted_contents.append({
                        "url": url,
                        "content": content
                    })
            except Exception as e:
                logger.warning(f"Error extracting content from {url}: {str(e)}")
        
        return {
            "status": "success",
            "result": {
                "contents": extracted_contents
            }
        }
    
    @handle_agent_error()
    def _handle_analyze_results(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        검색 결과 분석 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        contents = message_content.get("data", {}).get("contents", [])
        query = message_content.get("data", {}).get("query", "")
        
        if not contents or not query:
            raise TaskExecutionError(
                message="No contents or query provided for analysis",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 검색 결과 분석
        analysis = self._analyze_search_results(contents, query)
        
        return {
            "status": "success",
            "result": {
                "analysis": analysis
            }
        }
    
    def _format_search_query(self, query: str, task_description: str) -> str:
        """
        효과적인 검색을 위한 쿼리 형식화
        
        Args:
            query: 사용자 쿼리
            task_description: 작업 설명
        
        Returns:
            str: 형식화된 검색 쿼리
        """
        # 간단한 형식화 구현
        return query
    
    def _summarize_content(self, content: str, query: str) -> str:
        """
        웹 콘텐츠 요약
        
        Args:
            content: 웹 콘텐츠
            query: 사용자 쿼리
        
        Returns:
            str: 요약된 콘텐츠
        """
        # 간단한 요약 (실제로는 LLM 사용 또는 더 정교한 알고리즘 적용)
        if len(content) > 1000:
            return content[:997] + "..."
        return content
    
    def _analyze_search_results(self, contents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        검색 결과 분석
        
        Args:
            contents: 추출된 콘텐츠 목록
            query: 사용자 쿼리
        
        Returns:
            Dict[str, Any]: 분석 결과
        """
        # 간단한 분석 결과 (실제로는 LLM 사용)
        if not contents:
            return {
                "answer": "검색 결과를 찾을 수 없습니다.",
                "confidence": 0.0
            }
        
        return {
            "answer": f"웹 검색 결과로 {len(contents)}개의 소스에서 정보를 찾았습니다.",
            "confidence": 0.7,
            "key_points": [f"소스 {i+1}의 정보" for i in range(len(contents))]
        }

    @handle_agent_error()
    def process_user_query(self, query: str, task_description: str = "") -> Dict[str, Any]:
        """
        사용자 쿼리 처리를 위한 추상 메소드 구현
        
        Args:
            query: 사용자 쿼리
            task_description: 작업 설명
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            logger.info(f"Processing user query: {query[:50]}...")
            
            # 검색어 형식화
            search_query = self._format_search_query(query, task_description)
            
            # 웹 검색 수행
            search_results = search_web(search_query, limit=self.search_limit)
            
            # 콘텐츠 추출
            extracted_contents = []
            for result in search_results[:self.extract_limit]:
                try:
                    url = result.get("url", "")
                    if url:
                        content = extract_content(url)
                        if content:
                            extracted_contents.append({
                                "url": url,
                                "title": result.get("title", ""),
                                "content": content,
                                "summary": self._summarize_content(content, query)
                            })
                except Exception as e:
                    logger.warning(f"Error extracting content from {result.get('url', '')}: {str(e)}")
            
            # 검색 결과 분석
            analysis = self._analyze_search_results(extracted_contents, query)
            
            return {
                "status": "success",
                "data": {
                    "search_query": search_query,
                    "sources": [
                        {
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "summary": item.get("summary", "")
                        } for item in extracted_contents
                    ],
                    "analysis": analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing user query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }


# 전역 웹 검색 에이전트 인스턴스
web_search_agent = WebSearchAgent()

def get_web_search_agent() -> WebSearchAgent:
    """웹 검색 에이전트 인스턴스 가져오기"""
    return web_search_agent
