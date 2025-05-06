#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 RAG 모델 모듈
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
from datetime import datetime

from core.config import get_config
from core.error_handler import ResourceError, retry
from services.vectordb.chroma_client import get_chroma_client
from services.vectordb.embedding import (
    get_embedding, compute_embeddings_for_chunks, 
    chunk_text, semantic_chunking, cosine_similarity
)
from services.llm_service import generate_text

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

class RAGModel:
    """RAG (Retrieval Augmented Generation) 모델"""
    
    def __init__(
        self,
        vectordb_client = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.7,
        max_context_docs: int = 5
    ):
        """
        RAG 모델 초기화
        
        Args:
            vectordb_client: 벡터 DB 클라이언트 (없으면 기본값 사용)
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            similarity_threshold: 유사도 임계값
            max_context_docs: 최대 컨텍스트 문서 수
        """
        self.vectordb_client = vectordb_client or get_chroma_client()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.max_context_docs = max_context_docs
        
        logger.info(f"RAG model initialized with chunk_size={chunk_size}, max_context_docs={max_context_docs}")
    
    def add_document(
        self,
        document: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
        chunking_method: str = "simple"
    ) -> List[str]:
        """
        문서 추가
        
        Args:
            document: 문서 내용
            metadata: 메타데이터
            doc_id: 문서 ID (없으면 자동 생성)
            chunking_method: 청킹 방법 (simple, semantic)
        
        Returns:
            List[str]: 생성된 청크 ID 목록
        """
        # 문서 청킹
        if chunking_method == "semantic":
            chunks = semantic_chunking(document, min_chunk_size=100, max_chunk_size=self.chunk_size)
        else:
            chunks = chunk_text(document, self.chunk_size, self.chunk_overlap)
        
        # 청크 ID 목록
        chunk_ids = []
        
        # 각 청크에 공통 메타데이터 추가
        common_metadata = {
            "doc_id": doc_id,
            "timestamp": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            **metadata
        }
        
        # 청크별 메타데이터 및 벡터 DB 추가
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_index": i,
                "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                **common_metadata
            }
            
            # 벡터 DB에 추가
            chunk_id = self.vectordb_client.add_document(
                document=chunk,
                metadata=chunk_metadata
            )
            
            chunk_ids.append(chunk_id)
        
        logger.info(f"Added document with {len(chunks)} chunks, doc_id: {doc_id}")
        return chunk_ids
    
    async def add_document_async(
        self,
        document: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
        chunking_method: str = "simple"
    ) -> List[str]:
        """
        문서 비동기 추가
        
        Args:
            document: 문서 내용
            metadata: 메타데이터
            doc_id: 문서 ID (없으면 자동 생성)
            chunking_method: 청킹 방법 (simple, semantic)
        
        Returns:
            List[str]: 생성된 청크 ID 목록
        """
        # 문서 청킹
        if chunking_method == "semantic":
            chunks = semantic_chunking(document, min_chunk_size=100, max_chunk_size=self.chunk_size)
        else:
            chunks = chunk_text(document, self.chunk_size, self.chunk_overlap)
        
        # 청크 임베딩 계산
        embeddings = await compute_embeddings_for_chunks(chunks)
        
        # 각 청크에 공통 메타데이터 추가
        common_metadata = {
            "doc_id": doc_id,
            "timestamp": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            **metadata
        }
        
        # 청크별 메타데이터 생성
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_index": i,
                "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                **common_metadata
            }
            metadatas.append(chunk_metadata)
        
        # 벡터 DB에 일괄 추가
        chunk_ids = self.vectordb_client.add_documents(
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        logger.info(f"Asynchronously added document with {len(chunks)} chunks, doc_id: {doc_id}")
        return chunk_ids
    
    def remove_document(self, doc_id: str) -> bool:
        """
        문서 삭제
        
        Args:
            doc_id: 문서 ID
        
        Returns:
            bool: 성공 여부
        """
        try:
            # 문서 ID로 청크 검색
            filter_dict = {"doc_id": doc_id}
            results = self.vectordb_client.query(
                query_text="",
                n_results=100,
                filter_dict=filter_dict
            )
            
            # 청크 ID 목록 추출
            chunk_ids = [result["id"] for result in results]
            
            # 각 청크 삭제
            success = True
            for chunk_id in chunk_ids:
                success = success and self.vectordb_client.delete_document(chunk_id)
            
            logger.info(f"Removed document with {len(chunk_ids)} chunks, doc_id: {doc_id}")
            return success
        except Exception as e:
            logger.error(f"Error removing document {doc_id}: {str(e)}")
            return False
    
    def query(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        n_results: int = 5,
        rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        쿼리 실행
        
        Args:
            query: 검색 쿼리
            filter_dict: 필터 조건
            n_results: 결과 개수
            rerank: 결과 재정렬 여부
        
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        # 벡터 DB 검색
        results = self.vectordb_client.query(
            query_text=query,
            n_results=n_results * 2 if rerank else n_results,  # 재정렬 시 더 많은 결과 검색
            filter_dict=filter_dict
        )
        
        # 재정렬
        if rerank and results:
            # 쿼리 임베딩 계산
            query_embedding = get_embedding(query)
            
            # 재점수 계산
            for result in results:
                # 기존 거리 점수
                distance_score = 1.0 - result.get("distance", 0)
                
                # BM25 스타일 키워드 점수 계산 (단순화)
                text = result.get("document", "")
                keywords = set(query.lower().split())
                text_words = set(text.lower().split())
                keyword_overlap = len(keywords.intersection(text_words))
                keyword_score = keyword_overlap / max(1, len(keywords))
                
                # 최종 점수 = 벡터 유사도 * 0.7 + 키워드 점수 * 0.3
                final_score = distance_score * 0.7 + keyword_score * 0.3
                
                # 결과에 점수 추가
                result["score"] = final_score
            
            # 점수로 정렬하고 상위 n_results 선택
            results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:n_results]
        
        # 메타데이터에서 필요한 정보만 추출
        for result in results:
            metadata = result.get("metadata", {})
            result["source"] = metadata.get("source", "unknown")
            result["doc_id"] = metadata.get("doc_id", "")
            result["title"] = metadata.get("title", "")
            result["url"] = metadata.get("url", "")
            
            # 대용량 메타데이터 정리
            if "chunk_text" in metadata:
                del metadata["chunk_text"]
        
        return results
    
    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = 500
    ) -> str:
        """
        검색 결과 기반 답변 생성
        
        Args:
            query: 사용자 쿼리
            context_docs: 컨텍스트 문서 목록
            system_message: 시스템 메시지
            max_tokens: 최대 토큰 수
        
        Returns:
            str: 생성된 답변
        """
        # 컨텍스트 구성
        context = ""
        for i, doc in enumerate(context_docs):
            text = doc.get("document", "")
            source = doc.get("source", "unknown")
            url = doc.get("url", "")
            
            if url:
                source_info = f"[{source}]({url})"
            else:
                source_info = f"[{source}]"
            
            context += f"\n\n출처 {i+1} {source_info}:\n{text}\n"
        
        # 기본 시스템 메시지
        if not system_message:
            system_message = """주어진 컨텍스트 정보를 기반으로 사용자 질문에 정확하게 답변하세요.
답변은 제공된 컨텍스트 내용에만 근거해야 합니다.
컨텍스트에 관련 정보가 없다면, "제공된 정보에서 답을 찾을 수 없습니다"라고 답변하세요.
답변은 논리적이고 잘 구성되어야 하며, 필요한 경우 여러 단락으로 나누어 설명하세요."""
        
        # 프롬프트 구성
        prompt = f"""다음은 사용자의 질문입니다:
질문: {query}

다음은 질문에 답변하는 데 사용할 수 있는 컨텍스트 정보입니다:
{context}

위 컨텍스트만 사용하여 사용자 질문에 답변하세요. 컨텍스트 정보를 기반으로 명확하고 정확한 답변을 제공하세요."""
        
        # LLM으로 답변 생성
        answer = generate_text(
            prompt=prompt,
            system_message=system_message,
            max_tokens=max_tokens
        )
        
        return answer
    
    def retrieve_and_generate(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        system_message: Optional[str] = None,
        n_results: int = 5,
        rerank: bool = True,
        max_tokens: Optional[int] = 500
    ) -> Dict[str, Any]:
        """
        검색 및 생성 (완전한 RAG 파이프라인)
        
        Args:
            query: 사용자 쿼리
            filter_dict: 필터 조건
            system_message: 시스템 메시지
            n_results: 검색 결과 개수
            rerank: 결과 재정렬 여부
            max_tokens: 최대 토큰 수
        
        Returns:
            Dict[str, Any]: RAG 실행 결과
        """
        # 검색 실행
        search_results = self.query(
            query=query,
            filter_dict=filter_dict,
            n_results=n_results,
            rerank=rerank
        )
        
        # 답변 생성
        answer = self.generate_answer(
            query=query,
            context_docs=search_results,
            system_message=system_message,
            max_tokens=max_tokens
        )
        
        # 결과 구성
        result = {
            "query": query,
            "answer": answer,
            "search_results": search_results,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    async def retrieve_and_generate_async(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        system_message: Optional[str] = None,
        n_results: int = 5,
        rerank: bool = True,
        max_tokens: Optional[int] = 500
    ) -> Dict[str, Any]:
        """
        검색 및 생성 비동기 버전 (완전한 RAG 파이프라인)
        
        Args:
            query: 사용자 쿼리
            filter_dict: 필터 조건
            system_message: 시스템 메시지
            n_results: 검색 결과 개수
            rerank: 결과 재정렬 여부
            max_tokens: 최대 토큰 수
        
        Returns:
            Dict[str, Any]: RAG 실행 결과
        """
        # 검색 실행 (동기 함수)
        search_results = self.query(
            query=query,
            filter_dict=filter_dict,
            n_results=n_results,
            rerank=rerank
        )
        
        # 답변 생성 (동기 함수)
        answer = self.generate_answer(
            query=query,
            context_docs=search_results,
            system_message=system_message,
            max_tokens=max_tokens
        )
        
        # 결과 구성
        result = {
            "query": query,
            "answer": answer,
            "search_results": search_results,
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# 전역 RAG 모델 인스턴스
rag_model = RAGModel()

def get_rag_model() -> RAGModel:
    """RAG 모델 인스턴스 가져오기"""
    return rag_model 