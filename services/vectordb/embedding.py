#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 벡터 임베딩 모듈
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
import logging
from functools import lru_cache
import numpy as np

from services.llm_service import (
    create_embedding, 
    create_embeddings_batch
)
from core.error_handler import retry

# 로거 설정
logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """
    텍스트 임베딩 벡터 생성 (캐시 지원)
    
    Args:
        text: 입력 텍스트
        model: 임베딩 모델
    
    Returns:
        List[float]: 임베딩 벡터
    """
    return create_embedding(text, model)


async def get_embeddings_batch(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    배치 텍스트 임베딩 벡터 생성
    
    Args:
        texts: 입력 텍스트 목록
        model: 임베딩 모델
    
    Returns:
        List[List[float]]: 임베딩 벡터 목록
    """
    return await create_embeddings_batch(texts, model)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    두 벡터 간의 코사인 유사도 계산
    
    Args:
        vec_a: 첫 번째 벡터
        vec_b: 두 번째 벡터
    
    Returns:
        float: 코사인 유사도 (-1 ~ 1)
    """
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


def euclidean_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """
    두 벡터 간의 유클리드 거리 계산
    
    Args:
        vec_a: 첫 번째 벡터
        vec_b: 두 번째 벡터
    
    Returns:
        float: 유클리드 거리
    """
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    return np.linalg.norm(vec_a - vec_b)


def dot_product(vec_a: List[float], vec_b: List[float]) -> float:
    """
    두 벡터 간의 내적 계산
    
    Args:
        vec_a: 첫 번째 벡터
        vec_b: 두 번째 벡터
    
    Returns:
        float: 내적
    """
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    return np.dot(vec_a, vec_b)


async def compute_embeddings_for_chunks(chunks: List[str]) -> List[List[float]]:
    """
    청크 목록의 임베딩 계산
    
    Args:
        chunks: 텍스트 청크 목록
    
    Returns:
        List[List[float]]: 임베딩 목록
    """
    # 청크가 없으면 빈 목록 반환
    if not chunks:
        return []
    
    embeddings = await get_embeddings_batch(chunks)
    
    return embeddings


def chunk_text(
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50,
    separator: str = " "
) -> List[str]:
    """
    텍스트를 오버랩 청크로 분할
    
    Args:
        text: 분할할 텍스트
        chunk_size: 청크 크기 (단어 수)
        chunk_overlap: 청크 간 오버랩 크기
        separator: 단어 구분자
    
    Returns:
        List[str]: 청크 목록
    """
    # 텍스트를 단어로 분할
    words = text.split(separator)
    
    # 텍스트가 청크 크기보다 작으면 그대로 반환
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    i = 0
    
    while i < len(words):
        # 청크 범위 계산
        chunk_end = min(i + chunk_size, len(words))
        
        # 청크 추출
        chunk = separator.join(words[i:chunk_end])
        chunks.append(chunk)
        
        # 다음 시작 위치 계산 (오버랩 고려)
        i += (chunk_size - chunk_overlap)
    
    return chunks


def calculate_optimal_chunk_size(
    text: str, 
    target_chunk_count: int = 5,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000,
    separator: str = " "
) -> int:
    """
    목표 청크 수에 맞는 최적의 청크 크기 계산
    
    Args:
        text: 분할할 텍스트
        target_chunk_count: 목표 청크 수
        min_chunk_size: 최소 청크 크기
        max_chunk_size: 최대 청크 크기
        separator: 단어 구분자
    
    Returns:
        int: 최적의 청크 크기
    """
    # 단어 수 계산
    word_count = len(text.split(separator))
    
    # 초기 청크 크기 계산
    chunk_size = word_count // target_chunk_count
    
    # 범위 내로 조정
    chunk_size = max(min_chunk_size, min(max_chunk_size, chunk_size))
    
    return chunk_size


def semantic_chunking(
    text: str, 
    embedding_fn: Callable[[str], List[float]] = get_embedding,
    threshold: float = 0.7,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000,
    window_size: int = 50,
    separator: str = "\n"
) -> List[str]:
    """
    의미적 유사성에 기반한 텍스트 청킹
    
    Args:
        text: 분할할 텍스트
        embedding_fn: 임베딩 함수
        threshold: 유사성 임계값
        min_chunk_size: 최소 청크 크기
        max_chunk_size: 최대 청크 크기
        window_size: 윈도우 크기
        separator: 구분자
    
    Returns:
        List[str]: 청크 목록
    """
    # 기본 분할
    paragraphs = text.split(separator)
    
    # 기본 분할이 너무 적으면 그냥 반환
    if len(paragraphs) <= 3:
        return [text]
    
    # 너무 긴 텍스트는 단순 청킹으로 처리
    if len(text) > 30000:
        logger.warning("Text too long for semantic chunking, using simple chunking")
        return chunk_text(
            text, 
            chunk_size=max_chunk_size, 
            chunk_overlap=window_size
        )
    
    # 실험적 기능으로 미구현 (실제 구현 시 고려사항)
    # 1. 각 패러그래프 임베딩 계산
    # 2. 유사성이 높은 인접 패러그래프 병합
    # 3. 최대 청크 크기 제한 준수
    
    # 현재는 간단한 청킹 사용
    return chunk_text(
        text, 
        chunk_size=max_chunk_size // 5, 
        chunk_overlap=window_size
    ) 