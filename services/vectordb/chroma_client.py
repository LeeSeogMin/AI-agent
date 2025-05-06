#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 Chroma 벡터 데이터베이스 클라이언트 모듈
"""

import os
import uuid
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

from core.config import get_config
from core.error_handler import ResourceError, retry
from services.vectordb.embedding import get_embedding

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

class ChromaClient:
    """Chroma 벡터 데이터베이스 클라이언트"""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Chroma 클라이언트 초기화
        
        Args:
            persist_directory: 영구 저장 디렉토리 (없으면 설정에서 가져옴)
            collection_name: 컬렉션 이름 (없으면 설정에서 가져옴)
        """
        self.persist_directory = persist_directory or config.vector_db_directory
        self.collection_name = collection_name or config.vector_db_collection
        
        # 디렉토리 생성
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 컬렉션 가져오기
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # 컬렉션이 없으면 생성
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "LangGraph 멀티 에이전트 시스템 지식 베이스"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def get_count(self) -> int:
        """
        컬렉션 문서 개수 조회
        
        Returns:
            int: 문서 개수
        """
        return self.collection.count()
    
    @retry(max_attempts=3)
    def add_document(
        self,
        document: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """
        문서 추가
        
        Args:
            document: 문서 내용
            metadata: 메타데이터
            document_id: 문서 ID (없으면 자동 생성)
            embedding: 임베딩 벡터 (없으면 자동 생성)
        
        Returns:
            str: 문서 ID
        """
        # 문서 ID 생성
        doc_id = document_id or str(uuid.uuid4())
        
        # 임베딩 생성
        if embedding is None:
            embedding = get_embedding(document)
        
        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document]
            )
            logger.debug(f"Added document with ID: {doc_id}")
            return doc_id
        except Exception as e:
            error_msg = f"Error adding document to vector DB: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="chroma_client",
                resource_type="vector_db",
                resource_id=self.collection_name
            )
    
    @retry(max_attempts=3)
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        document_ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        여러 문서 추가
        
        Args:
            documents: 문서 내용 목록
            metadatas: 메타데이터 목록
            document_ids: 문서 ID 목록 (없으면 자동 생성)
            embeddings: 임베딩 벡터 목록 (없으면 자동 생성)
        
        Returns:
            List[str]: 문서 ID 목록
        """
        # 문서 수 검증
        n_docs = len(documents)
        if len(metadatas) != n_docs:
            raise ValueError(f"Number of documents ({n_docs}) and metadatas ({len(metadatas)}) do not match")
        
        # 문서 ID 생성
        if document_ids is None:
            document_ids = [str(uuid.uuid4()) for _ in range(n_docs)]
        elif len(document_ids) != n_docs:
            raise ValueError(f"Number of documents ({n_docs}) and document_ids ({len(document_ids)}) do not match")
        
        # 임베딩 생성은 추가 시 자동으로 처리 (비동기 처리 용이하게)
        
        try:
            self.collection.add(
                ids=document_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.debug(f"Added {n_docs} documents")
            return document_ids
        except Exception as e:
            error_msg = f"Error adding documents to vector DB: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="chroma_client",
                resource_type="vector_db",
                resource_id=self.collection_name
            )
    
    @retry(max_attempts=3)
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        문서 조회
        
        Args:
            document_id: 문서 ID
        
        Returns:
            Optional[Dict[str, Any]]: 문서 정보 (없으면 None)
        """
        try:
            result = self.collection.get(ids=[document_id], include=["documents", "metadatas", "embeddings"])
            
            if not result["ids"]:
                return None
            
            return {
                "id": result["ids"][0],
                "document": result["documents"][0],
                "metadata": result["metadatas"][0],
                "embedding": result["embeddings"][0] if result["embeddings"] else None
            }
        except Exception as e:
            error_msg = f"Error getting document from vector DB: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="chroma_client",
                resource_type="vector_db",
                resource_id=self.collection_name
            )
    
    @retry(max_attempts=3)
    def update_document(
        self,
        document_id: str,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        문서 업데이트
        
        Args:
            document_id: 문서 ID
            document: 새 문서 내용 (None이면 변경 없음)
            metadata: 새 메타데이터 (None이면 변경 없음)
            embedding: 새 임베딩 벡터 (None이면 변경 없음)
        
        Returns:
            bool: 성공 여부
        """
        try:
            # 기존 문서 조회
            existing = self.get_document(document_id)
            if not existing:
                logger.warning(f"Document not found for update: {document_id}")
                return False
            
            # 필요한 필드만 업데이트
            documents = [document] if document is not None else None
            metadatas = [metadata] if metadata is not None else None
            embeddings = [embedding] if embedding is not None else None
            
            # 내용만 변경시 임베딩도 자동 업데이트
            if document is not None and embedding is None:
                embeddings = [get_embedding(document)]
            
            self.collection.update(
                ids=[document_id],
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.debug(f"Updated document: {document_id}")
            return True
        except Exception as e:
            error_msg = f"Error updating document in vector DB: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="chroma_client",
                resource_type="vector_db",
                resource_id=self.collection_name
            )
    
    @retry(max_attempts=3)
    def delete_document(self, document_id: str) -> bool:
        """
        문서 삭제
        
        Args:
            document_id: 문서 ID
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.collection.delete(ids=[document_id])
            logger.debug(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            error_msg = f"Error deleting document from vector DB: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="chroma_client",
                resource_type="vector_db",
                resource_id=self.collection_name
            )
    
    @retry(max_attempts=3)
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        벡터 검색 실행
        
        Args:
            query_text: 검색 쿼리
            n_results: 결과 개수
            filter_dict: 필터 조건
            embedding: 쿼리 임베딩 (없으면 자동 생성)
        
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        try:
            # 임베딩 생성
            if embedding is None:
                embedding = get_embedding(query_text)
            
            # 검색 실행
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 변환
            docs = []
            for i in range(len(results["ids"][0])):
                docs.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            
            return docs
        except Exception as e:
            error_msg = f"Error querying vector DB: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="chroma_client",
                resource_type="vector_db",
                resource_id=self.collection_name
            )
    
    @retry(max_attempts=3)
    def hybrid_search(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        k1: float = 1.5,
        b: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 실행 (벡터 + 키워드)
        
        Args:
            query_text: 검색 쿼리
            n_results: 결과 개수
            filter_dict: 필터 조건
            k1: BM25 k1 파라미터
            b: BM25 b 파라미터
        
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        try:
            # Chroma는 하이브리드 검색 지원 안함
            # 키워드 검색 대신 벡터 검색만 수행
            logger.warning("Hybrid search not directly supported, using vector search only")
            return self.query(query_text, n_results, filter_dict)
        except Exception as e:
            error_msg = f"Error performing hybrid search: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="chroma_client",
                resource_type="vector_db",
                resource_id=self.collection_name
            )
    
    def get_collection_metadata(self) -> Dict[str, Any]:
        """
        컬렉션 메타데이터 조회
        
        Returns:
            Dict[str, Any]: 메타데이터
        """
        try:
            collection_info = {
                "name": self.collection_name,
                "count": self.get_count(),
                "metadata": self.collection.metadata
            }
            return collection_info
        except Exception as e:
            error_msg = f"Error getting collection metadata: {str(e)}"
            logger.error(error_msg)
            raise ResourceError(
                message=error_msg,
                agent_id="chroma_client",
                resource_type="vector_db",
                resource_id=self.collection_name
            )


# 전역 Chroma 클라이언트 인스턴스
chroma_client = ChromaClient()

def get_chroma_client() -> ChromaClient:
    """Chroma 클라이언트 인스턴스 가져오기"""
    return chroma_client 