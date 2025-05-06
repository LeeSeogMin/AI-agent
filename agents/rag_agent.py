#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 RAG 에이전트 모듈
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union

from agents.base import Agent
from core.config import get_config
from core.error_handler import handle_agent_error, TaskExecutionError
from models.rag_model import RAGModel
from services.vectordb.chroma_client import get_chroma_client
from services.vectordb.embedding import get_embeddings_model

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

class RAGAgent(Agent):
    """RAG(Retrieval-Augmented Generation) 에이전트 클래스"""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        system_message: Optional[str] = None
    ):
        """
        RAG 에이전트 초기화
        
        Args:
            agent_id: 에이전트 ID (없으면 자동 생성)
            system_message: 시스템 메시지
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="rag",
            system_message=system_message
        )
        
        # 시스템 메시지 로드
        if not self.system_message:
            self.system_message = self.load_system_message()
        
        # RAG 모델 초기화
        self.rag_model = RAGModel(
            embedding_model=get_embeddings_model(),
            vector_store=get_chroma_client(),
            top_k=getattr(config, 'rag', {}).get("top_k", 3) if hasattr(config, 'rag') else 3
        )
        
        # 작업 핸들러 등록
        self._register_task_handlers()
        
        logger.info(f"RAG agent initialized with ID: {self.agent_id}")
    
    def _register_task_handlers(self) -> None:
        """작업 핸들러 등록"""
        self.task_handlers = {
            "process_task": self._handle_process_task,
            "retrieve_documents": self._handle_retrieve_documents,
            "generate_answer": self._handle_generate_answer
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
            
            # 문서 검색
            documents = self.rag_model.retrieve(query)
            
            # 답변 생성
            answer = self.rag_model.generate(query, documents, task_description)
            
            result = {
                "answer": answer.get("answer", ""),
                "reasoning": answer.get("reasoning", ""),
                "documents": [
                    {
                        "content": doc.get("document", ""),
                        "metadata": doc.get("metadata", {}),
                        "score": doc.get("score", 0.0)
                    } for doc in documents
                ]
            }
            
            return {
                "status": "success",
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Error processing user query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
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
                message="No query provided for RAG processing",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 문서 검색
        documents = self.rag_model.retrieve(query)
        
        # 답변 생성
        answer = self.rag_model.generate(query, documents, task_description)
        
        return {
            "status": "success",
            "result": {
                "answer": answer.get("answer", ""),
                "documents": [
                    {
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "score": doc.get("score", 0.0)
                    } for doc in documents
                ],
                "reasoning": answer.get("reasoning", "")
            }
        }
    
    @handle_agent_error()
    def _handle_retrieve_documents(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        문서 검색 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for document retrieval",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 문서 검색
        documents = self.rag_model.retrieve(query)
        
        return {
            "status": "success",
            "result": {
                "documents": [
                    {
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "score": doc.get("score", 0.0)
                    } for doc in documents
                ]
            }
        }
    
    @handle_agent_error()
    def _handle_generate_answer(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        답변 생성 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        documents = message_content.get("data", {}).get("documents", [])
        task_description = message_content.get("data", {}).get("task_description", "")
        
        if not query or not documents:
            raise TaskExecutionError(
                message="No query or documents provided for answer generation",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 답변 생성
        answer = self.rag_model.generate(query, documents, task_description)
        
        return {
            "status": "success",
            "result": {
                "answer": answer.get("answer", ""),
                "reasoning": answer.get("reasoning", "")
            }
        }


# 전역 RAG 에이전트 인스턴스
rag_agent = RAGAgent()

def get_rag_agent() -> RAGAgent:
    """RAG 에이전트 인스턴스 가져오기"""
    return rag_agent
