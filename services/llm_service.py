#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 LLM 서비스 모듈
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from functools import lru_cache
import asyncio
from openai import OpenAI, AsyncOpenAI
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type
)

from core.config import get_config
from core.error_handler import LLMError

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

class LLMService:
    """LLM API 서비스 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        LLM 서비스 초기화
        
        Args:
            api_key: OpenAI API 키 (없으면 환경 변수에서 가져옴)
        """
        self.api_key = api_key or config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # 동기 클라이언트
        self.client = OpenAI(api_key=self.api_key)
        
        # 비동기 클라이언트
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # 기본 모델 설정
        self.default_model = config.default_llm_model
        self.default_embedding_model = config.default_embedding_model
        
        logger.info(f"LLM service initialized with default model: {self.default_model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            model: 사용할 모델
            temperature: 온도
            max_tokens: 최대 토큰 수
            system_message: 시스템 메시지
            stop: 중지 토큰 목록
            response_format: 응답 형식
            agent_id: 에이전트 ID (오류 시 사용)
        
        Returns:
            str: 생성된 텍스트
        
        Raises:
            LLMError: LLM API 오류
        """
        model = model or self.default_model
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                response_format=response_format
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 응답 로깅
            logger.debug(f"LLM response generated in {duration:.2f}s")
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"LLM API error: {str(e)}"
            logger.error(error_msg)
            
            # LLM 오류 발생
            raise LLMError(
                message=error_msg,
                agent_id=agent_id or "llm_service",
                llm_response={"error": str(e), "model": model}
            )
    
    async def generate_text_async(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ) -> str:
        """
        비동기 텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            model: 사용할 모델
            temperature: 온도
            max_tokens: 최대 토큰 수
            system_message: 시스템 메시지
            stop: 중지 토큰 목록
            response_format: 응답 형식
            agent_id: 에이전트 ID (오류 시 사용)
        
        Returns:
            str: 생성된 텍스트
        
        Raises:
            LLMError: LLM API 오류
        """
        model = model or self.default_model
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            start_time = time.time()
            
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                response_format=response_format
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 응답 로깅
            logger.debug(f"Async LLM response generated in {duration:.2f}s")
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Async LLM API error: {str(e)}"
            logger.error(error_msg)
            
            # LLM 오류 발생
            raise LLMError(
                message=error_msg,
                agent_id=agent_id or "llm_service",
                llm_response={"error": str(e), "model": model}
            )
    
    def generate_json(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        JSON 형식 텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            schema: JSON 스키마
            model: 사용할 모델
            temperature: 온도
            system_message: 시스템 메시지
            agent_id: 에이전트 ID (오류 시 사용)
        
        Returns:
            Dict[str, Any]: 생성된 JSON
        
        Raises:
            LLMError: LLM API 오류
        """
        response_format = {"type": "json_object"}
        
        model = model or self.default_model
        
        # 기본 시스템 메시지가 없으면 추가
        if not system_message:
            schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
            system_message = f"""You are a helpful AI assistant that generates valid JSON.
Your response should strictly follow this JSON schema:
```json
{schema_str}
```
Only respond with valid JSON that follows the schema exactly."""
        
        try:
            # JSON 형식 응답 요청
            json_str = self.generate_text(
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_message=system_message,
                response_format=response_format,
                agent_id=agent_id
            )
            
            # JSON 파싱
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            logger.error(f"{error_msg} - Response: {json_str if 'json_str' in locals() else 'None'}")
            
            # LLM 오류 발생
            raise LLMError(
                message=error_msg,
                agent_id=agent_id or "llm_service",
                llm_response={"error": str(e), "response": json_str if 'json_str' in locals() else None}
            )
    
    @lru_cache(maxsize=128)
    def create_embedding(
        self, 
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        텍스트 임베딩 생성
        
        Args:
            text: 입력 텍스트
            model: 사용할 모델
        
        Returns:
            List[float]: 임베딩 벡터
        """
        model = model or self.default_embedding_model
        
        try:
            response = self.client.embeddings.create(
                input=text.replace("\n", " "),
                model=model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            error_msg = f"Embedding API error: {str(e)}"
            logger.error(error_msg)
            raise
    
    async def create_embedding_async(
        self, 
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        비동기 텍스트 임베딩 생성
        
        Args:
            text: 입력 텍스트
            model: 사용할 모델
        
        Returns:
            List[float]: 임베딩 벡터
        """
        model = model or self.default_embedding_model
        
        try:
            response = await self.async_client.embeddings.create(
                input=text.replace("\n", " "),
                model=model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            error_msg = f"Async embedding API error: {str(e)}"
            logger.error(error_msg)
            raise
    
    async def create_embeddings_batch(
        self, 
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        배치 텍스트 임베딩 생성
        
        Args:
            texts: 입력 텍스트 목록
            model: 사용할 모델
        
        Returns:
            List[List[float]]: 임베딩 벡터 목록
        """
        model = model or self.default_embedding_model
        
        try:
            # 개행 문자 제거
            cleaned_texts = [text.replace("\n", " ") for text in texts]
            
            response = await self.async_client.embeddings.create(
                input=cleaned_texts,
                model=model
            )
            
            # 임베딩 순서대로 정렬
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [emb.embedding for emb in embeddings]
            
        except Exception as e:
            error_msg = f"Batch embedding API error: {str(e)}"
            logger.error(error_msg)
            raise


# 전역 LLM 서비스 인스턴스
llm_service = LLMService()

def get_llm_service() -> LLMService:
    """LLM 서비스 인스턴스 가져오기"""
    return llm_service


def generate_text(
    prompt: str, 
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    stop: Optional[List[str]] = None,
    agent_id: Optional[str] = None
) -> str:
    """
    텍스트 생성 유틸리티 함수
    
    Args:
        prompt: 입력 프롬프트
        model: 사용할 모델
        temperature: 온도
        max_tokens: 최대 토큰 수
        system_message: 시스템 메시지
        stop: 중지 토큰 목록
        agent_id: 에이전트 ID (오류 시 사용)
    
    Returns:
        str: 생성된 텍스트
    """
    return llm_service.generate_text(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_message=system_message,
        stop=stop,
        agent_id=agent_id
    )


def generate_json(
    prompt: str, 
    schema: Dict[str, Any],
    model: Optional[str] = None,
    temperature: float = 0.7,
    system_message: Optional[str] = None,
    agent_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    JSON 형식 텍스트 생성 유틸리티 함수
    
    Args:
        prompt: 입력 프롬프트
        schema: JSON 스키마
        model: 사용할 모델
        temperature: 온도
        system_message: 시스템 메시지
        agent_id: 에이전트 ID (오류 시 사용)
    
    Returns:
        Dict[str, Any]: 생성된 JSON
    """
    return llm_service.generate_json(
        prompt=prompt,
        schema=schema,
        model=model,
        temperature=temperature,
        system_message=system_message,
        agent_id=agent_id
    )


def create_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """
    텍스트 임베딩 생성 유틸리티 함수
    
    Args:
        text: 입력 텍스트
        model: 사용할 모델
    
    Returns:
        List[float]: 임베딩 벡터
    """
    return llm_service.create_embedding(text=text, model=model)


async def create_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None
) -> List[List[float]]:
    """
    배치 텍스트 임베딩 생성 유틸리티 함수
    
    Args:
        texts: 입력 텍스트 목록
        model: 사용할 모델
    
    Returns:
        List[List[float]]: 임베딩 벡터 목록
    """
    return await llm_service.create_embeddings_batch(texts=texts, model=model) 