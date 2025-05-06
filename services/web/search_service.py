#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 웹 검색 서비스 모듈
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import asyncio
import aiohttp
import requests
from duckduckgo_search import DDGS

from core.config import get_config
from core.error_handler import ResourceError, retry
from services.web.html_parser import extract_text_from_html

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

class SearchService:
    """웹 검색 서비스 클래스"""
    
    def __init__(self, provider: Optional[str] = None):
        """
        웹 검색 서비스 초기화
        
        Args:
            provider: 검색 제공자 (duckduckgo, serper)
        """
        self.provider = provider or config.web_search_provider
        self.max_results = config.web_search_max_results
        self.timeout = config.web_search_timeout
        
        # API 키 설정
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        
        logger.info(f"Search service initialized with provider: {self.provider}")
    
    @retry(max_attempts=3)
    def search(
        self, 
        query: str, 
        num_results: Optional[int] = None, 
        search_type: str = "web"
    ) -> List[Dict[str, Any]]:
        """
        웹 검색 실행
        
        Args:
            query: 검색 쿼리
            num_results: 결과 개수 (없으면 기본값 사용)
            search_type: 검색 유형 (web, news, images, videos)
        
        Returns:
            List[Dict[str, Any]]: 검색 결과
        
        Raises:
            ResourceError: 검색 오류
        """
        num_results = num_results or self.max_results
        
        if self.provider == "duckduckgo":
            return self._search_duckduckgo(query, num_results, search_type)
        elif self.provider == "serper":
            return self._search_serper(query, num_results, search_type)
        else:
            logger.warning(f"Unknown search provider: {self.provider}. Using DuckDuckGo.")
            return self._search_duckduckgo(query, num_results, search_type)
    
    def _search_duckduckgo(
        self, 
        query: str, 
        num_results: int, 
        search_type: str
    ) -> List[Dict[str, Any]]:
        """
        DuckDuckGo 검색 실행
        
        Args:
            query: 검색 쿼리
            num_results: 결과 개수
            search_type: 검색 유형
        
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        try:
            with DDGS() as ddgs:
                if search_type == "web":
                    results = list(ddgs.text(query, max_results=num_results))
                elif search_type == "news":
                    results = list(ddgs.news(query, max_results=num_results))
                elif search_type == "images":
                    results = list(ddgs.images(query, max_results=num_results))
                elif search_type == "videos":
                    results = list(ddgs.videos(query, max_results=num_results))
                else:
                    results = list(ddgs.text(query, max_results=num_results))
            
            # 검색 결과 정규화
            normalized = []
            for result in results:
                normalized.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", result.get("description", "")),
                    "url": result.get("href", ""),
                    "source": "duckduckgo",
                    "published": result.get("published", ""),
                    "position": len(normalized) + 1
                })
            
            return normalized
            
        except Exception as e:
            error_msg = f"DuckDuckGo search error: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="search_service",
                resource_type="web_search",
                resource_id="duckduckgo"
            )
    
    def _search_serper(
        self, 
        query: str, 
        num_results: int, 
        search_type: str
    ) -> List[Dict[str, Any]]:
        """
        Serper.dev 검색 실행
        
        Args:
            query: 검색 쿼리
            num_results: 결과 개수
            search_type: 검색 유형
        
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        if not self.serper_api_key:
            logger.warning("Serper API key not found. Using DuckDuckGo instead.")
            return self._search_duckduckgo(query, num_results, search_type)
        
        try:
            url = "https://google.serper.dev/search"
            
            payload = json.dumps({
                "q": query,
                "num": num_results
            })
            
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # 검색 결과 정규화
            normalized = []
            
            # 유기 검색 결과
            organic = data.get("organic", [])
            for idx, result in enumerate(organic):
                normalized.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "url": result.get("link", ""),
                    "source": "serper",
                    "published": "",
                    "position": idx + 1
                })
            
            return normalized
            
        except Exception as e:
            error_msg = f"Serper search error: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="search_service",
                resource_type="web_search",
                resource_id="serper"
            )
    
    @retry(max_attempts=2)
    async def fetch_url_content(
        self, 
        url: str, 
        timeout: Optional[int] = None,
        extract_text: bool = True
    ) -> Dict[str, Any]:
        """
        URL 콘텐츠 가져오기
        
        Args:
            url: 가져올 URL
            timeout: 타임아웃 (초)
            extract_text: HTML에서 텍스트 추출 여부
        
        Returns:
            Dict[str, Any]: 가져온 콘텐츠 정보
        """
        timeout = timeout or self.timeout
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    timeout=timeout,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                ) as response:
                    content_type = response.headers.get("Content-Type", "")
                    status = response.status
                    raw_content = await response.text()
                    
                    result = {
                        "url": url,
                        "status_code": status,
                        "content_type": content_type,
                        "raw_content": raw_content,
                        "text_content": "",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # HTML 텍스트 추출
                    if extract_text and "text/html" in content_type.lower():
                        text_content = extract_text_from_html(raw_content)
                        result["text_content"] = text_content
                    
                    return result
                    
        except Exception as e:
            error_msg = f"Error fetching URL {url}: {str(e)}"
            logger.error(error_msg)
            
            # 실패 정보 반환
            return {
                "url": url,
                "status_code": 0,
                "content_type": "",
                "raw_content": "",
                "text_content": "",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def fetch_multiple_urls(
        self, 
        urls: List[str], 
        timeout: Optional[int] = None,
        extract_text: bool = True
    ) -> List[Dict[str, Any]]:
        """
        여러 URL 콘텐츠 동시에 가져오기
        
        Args:
            urls: 가져올 URL 목록
            timeout: 타임아웃 (초)
            extract_text: HTML에서 텍스트 추출 여부
        
        Returns:
            List[Dict[str, Any]]: 가져온 콘텐츠 정보 목록
        """
        # 중복 URL 제거
        unique_urls = list(set(urls))
        
        # 동시에 실행
        tasks = [
            self.fetch_url_content(url, timeout, extract_text) 
            for url in unique_urls
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 원래 순서대로 결과 반환
        ordered_results = []
        for url in urls:
            for result in results:
                if result["url"] == url:
                    ordered_results.append(result)
                    break
        
        return ordered_results
    
    def search_and_extract(
        self, 
        query: str, 
        num_results: Optional[int] = None,
        search_type: str = "web",
        fetch_content: bool = True,
        summary_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        검색 및 콘텐츠 추출 (동기 버전)
        
        Args:
            query: 검색 쿼리
            num_results: 결과 개수
            search_type: 검색 유형
            fetch_content: 검색 결과 콘텐츠 가져오기 여부
            summary_length: 요약 길이 (단어 수)
        
        Returns:
            Dict[str, Any]: 검색 및 추출 결과
        """
        # 검색 실행
        search_results = self.search(query, num_results, search_type)
        
        # 콘텐츠 가져오기
        if fetch_content and search_results:
            # URL 목록 추출
            urls = [result["url"] for result in search_results]
            
            # 비동기 호출을 위한 이벤트 루프 생성
            loop = asyncio.get_event_loop()
            content_results = loop.run_until_complete(
                self.fetch_multiple_urls(urls)
            )
            
            # 검색 결과에 콘텐츠 추가
            for i, result in enumerate(search_results):
                if i < len(content_results):
                    result["content"] = content_results[i].get("text_content", "")
                    result["status_code"] = content_results[i].get("status_code", 0)
                    
                    # 요약 길이 제한
                    if summary_length and result["content"]:
                        words = result["content"].split()
                        if len(words) > summary_length:
                            result["content"] = " ".join(words[:summary_length]) + "..."
        
        return {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "num_results": len(search_results),
            "results": search_results
        }
    
    async def async_search_and_extract(
        self, 
        query: str, 
        num_results: Optional[int] = None,
        search_type: str = "web",
        fetch_content: bool = True,
        summary_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        검색 및 콘텐츠 추출 (비동기 버전)
        
        Args:
            query: 검색 쿼리
            num_results: 결과 개수
            search_type: 검색 유형
            fetch_content: 검색 결과 콘텐츠 가져오기 여부
            summary_length: 요약 길이 (단어 수)
        
        Returns:
            Dict[str, Any]: 검색 및 추출 결과
        """
        # 검색 실행 (동기 함수)
        search_results = self.search(query, num_results, search_type)
        
        # 콘텐츠 가져오기
        if fetch_content and search_results:
            # URL 목록 추출
            urls = [result["url"] for result in search_results]
            
            # 비동기 콘텐츠 가져오기
            content_results = await self.fetch_multiple_urls(urls)
            
            # 검색 결과에 콘텐츠 추가
            for i, result in enumerate(search_results):
                if i < len(content_results):
                    result["content"] = content_results[i].get("text_content", "")
                    result["status_code"] = content_results[i].get("status_code", 0)
                    
                    # 요약 길이 제한
                    if summary_length and result["content"]:
                        words = result["content"].split()
                        if len(words) > summary_length:
                            result["content"] = " ".join(words[:summary_length]) + "..."
        
        return {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "num_results": len(search_results),
            "results": search_results
        }


# 전역 검색 서비스 인스턴스
search_service = SearchService()

def get_search_service() -> SearchService:
    """검색 서비스 인스턴스 가져오기"""
    return search_service


def search(
    query: str, 
    num_results: Optional[int] = None, 
    search_type: str = "web"
) -> List[Dict[str, Any]]:
    """
    웹 검색 실행 유틸리티 함수
    
    Args:
        query: 검색 쿼리
        num_results: 결과 개수
        search_type: 검색 유형
    
    Returns:
        List[Dict[str, Any]]: 검색 결과
    """
    return search_service.search(query, num_results, search_type)


def search_and_extract(
    query: str, 
    num_results: Optional[int] = None,
    fetch_content: bool = True
) -> Dict[str, Any]:
    """
    검색 및 콘텐츠 추출 유틸리티 함수
    
    Args:
        query: 검색 쿼리
        num_results: 결과 개수
        fetch_content: 검색 결과 콘텐츠 가져오기 여부
    
    Returns:
        Dict[str, Any]: 검색 및 추출 결과
    """
    return search_service.search_and_extract(query, num_results, "web", fetch_content) 