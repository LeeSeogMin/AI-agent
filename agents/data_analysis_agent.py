#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 데이터 분석 에이전트 모듈
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from datetime import datetime, timedelta

from agents.base import Agent
from core.config import get_config, DATA_DIR
from core.error_handler import handle_agent_error, TaskExecutionError
from services.llm_service import generate_text, generate_json
from services.vectordb.chroma_client import get_chroma_client

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

class DataAnalysisAgent(Agent):
    """데이터 분석 에이전트 클래스"""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        system_message: Optional[str] = None
    ):
        """
        데이터 분석 에이전트 초기화
        
        Args:
            agent_id: 에이전트 ID (없으면 자동 생성)
            system_message: 시스템 메시지
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="data_analysis",
            system_message=system_message
        )
        
        # 시스템 메시지 로드
        if not self.system_message:
            self.system_message = self.load_system_message()
        
        # 작업 핸들러 등록
        self._register_task_handlers()
        
        # 데이터 디렉토리 설정
        self.data_dir = os.path.join(DATA_DIR, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 분석 결과 저장 디렉토리
        self.output_dir = os.path.join(DATA_DIR, "analysis_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 벡터 DB 클라이언트
        self.vector_db = get_chroma_client()
        
        logger.info(f"Data analysis agent initialized with ID: {self.agent_id}")
    
    def _register_task_handlers(self) -> None:
        """작업 핸들러 등록"""
        self.task_handlers = {
            "process_task": self._handle_process_task,
            "load_data": self._handle_load_data,
            "analyze_data": self._handle_analyze_data,
            "generate_visualization": self._handle_generate_visualization,
            "analyze_time_series": self._handle_analyze_time_series,
            "analyze_conversation_trends": self._handle_analyze_conversation_trends
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
        data_path = message_content.get("data", {}).get("data_path", "")
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for data analysis",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 데이터 로드 (경로가 제공된 경우)
        data = None
        if data_path:
            data = self._load_data(data_path)
        else:
            # 쿼리에서 데이터 요구사항 추출
            data_requirements = self._extract_data_requirements(query, task_description)
            
            # 가장 적합한 데이터 소스 결정
            data_path = self._find_best_data_source(data_requirements)
            if data_path:
                data = self._load_data(data_path)
        
        if data is None:
            return {
                "status": "error",
                "message": "No suitable data found for analysis"
            }
        
        # 데이터 분석
        analysis_result = self._analyze_data(data, query, task_description)
        
        # 시각화 생성 (필요한 경우)
        visualizations = []
        if analysis_result.get("visualization_required", False):
            visualizations = self._generate_visualizations(data, analysis_result.get("visualization_specs", []))
        
        # 결과 취합
        return {
            "status": "success",
            "result": {
                "data_source": data_path,
                "data_summary": {
                    "rows": len(data),
                    "columns": list(data.columns),
                    "data_types": {col: str(data[col].dtype) for col in data.columns}
                },
                "analysis": analysis_result.get("analysis", {}),
                "visualizations": visualizations,
                "insights": analysis_result.get("insights", [])
            }
        }
    
    @handle_agent_error()
    def _handle_load_data(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        데이터 로드 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        data_path = message_content.get("data", {}).get("data_path", "")
        
        if not data_path:
            raise TaskExecutionError(
                message="No data path provided for loading",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 데이터 로드
        data = self._load_data(data_path)
        
        if data is None:
            return {
                "status": "error",
                "message": f"Failed to load data from {data_path}"
            }
        
        # 데이터 요약
        summary = {
            "rows": len(data),
            "columns": list(data.columns),
            "data_types": {col: str(data[col].dtype) for col in data.columns},
            "missing_values": {col: int(data[col].isna().sum()) for col in data.columns},
            "sample": data.head(5).to_dict(orient="records")
        }
        
        return {
            "status": "success",
            "result": {
                "data_source": data_path,
                "summary": summary
            }
        }
    
    @handle_agent_error()
    def _handle_analyze_data(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        데이터 분석 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        data_path = message_content.get("data", {}).get("data_path", "")
        query = message_content.get("data", {}).get("query", "")
        
        if not data_path or not query:
            raise TaskExecutionError(
                message="Data path or query missing for analysis",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 데이터 로드
        data = self._load_data(data_path)
        
        if data is None:
            return {
                "status": "error",
                "message": f"Failed to load data from {data_path}"
            }
        
        # 데이터 분석
        analysis_result = self._analyze_data(data, query)
        
        return {
            "status": "success",
            "result": analysis_result
        }
    
    @handle_agent_error()
    def _handle_generate_visualization(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        시각화 생성 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        data_path = message_content.get("data", {}).get("data_path", "")
        specs = message_content.get("data", {}).get("visualization_specs", [])
        
        if not data_path or not specs:
            raise TaskExecutionError(
                message="Data path or visualization specs missing",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 데이터 로드
        data = self._load_data(data_path)
        
        if data is None:
            return {
                "status": "error",
                "message": f"Failed to load data from {data_path}"
            }
        
        # 시각화 생성
        visualizations = self._generate_visualizations(data, specs)
        
        return {
            "status": "success",
            "result": {
                "visualizations": visualizations
            }
        }
    
    def _load_data(self, data_path: str) -> Optional[pd.DataFrame]:
        """
        데이터 로드
        
        Args:
            data_path: 데이터 파일 경로 또는 특수 URI (vector_db://)
        
        Returns:
            Optional[pd.DataFrame]: 로드된 데이터프레임
        """
        try:
            # 벡터 DB에서 데이터 로드
            if data_path.startswith("vector_db://"):
                return self._load_data_from_vector_db(data_path)
            
            # 절대 경로가 아니면 데이터 디렉토리에서 찾기
            if not os.path.isabs(data_path):
                full_path = os.path.join(self.data_dir, data_path)
            else:
                full_path = data_path
            
            # 파일 확장자에 따라 로드 방식 결정
            if full_path.endswith('.csv'):
                return pd.read_csv(full_path)
            elif full_path.endswith('.xlsx') or full_path.endswith('.xls'):
                return pd.read_excel(full_path)
            elif full_path.endswith('.json'):
                return pd.read_json(full_path)
            elif full_path.endswith('.parquet'):
                return pd.read_parquet(full_path)
            else:
                logger.warning(f"Unsupported file format: {full_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            return None
            
    def _load_data_from_vector_db(self, data_uri: str) -> pd.DataFrame:
        """
        벡터 DB에서 대화 데이터 로드 및 데이터프레임 변환
        
        Args:
            data_uri: 벡터 DB URI (예: vector_db://conversation_data)
            
        Returns:
            pd.DataFrame: 대화 데이터 데이터프레임
        """
        # 모든 문서 가져오기 (실제로는 필터링 로직 추가 필요)
        # Chroma에서는 컬렉션의 모든 문서를 쉽게 가져올 수 없으므로 임베딩 없이 쿼리로 대체
        try:
            # 더미 쿼리로 최대한 많은 문서 검색
            results = self.vector_db.collection.query(
                query_texts=[""],
                n_results=10000,  # 최대치로 설정
                include=["documents", "metadatas"]
            )
            
            # 결과 변환
            records = []
            for i in range(len(results["ids"][0])):
                document = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                
                # 기본 필드 설정
                record = {
                    "id": results["ids"][0][i],
                    "text": document,
                    "timestamp": metadata.get("timestamp", ""),
                }
                
                # 메타데이터의 모든 필드 추가
                for key, value in metadata.items():
                    if key != "timestamp":  # 이미 추가함
                        record[key] = value
                
                records.append(record)
            
            # 데이터프레임 생성
            df = pd.DataFrame(records)
            
            # 타임스탬프 변환
            if "timestamp" in df.columns and not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from vector DB: {str(e)}")
            # 빈 데이터프레임 반환
            return pd.DataFrame()
    
    def _extract_data_requirements(self, query: str, task_description: str) -> Dict[str, Any]:
        """
        쿼리에서 데이터 요구사항 추출
        
        Args:
            query: 사용자 쿼리
            task_description: 작업 설명
        
        Returns:
            Dict[str, Any]: 데이터 요구사항
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "data_type": {"type": "string"},
                "fields": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "time_period": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"}
                    }
                },
                "subject_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["general", "time_series", "conversation_trends", "mental_health_tracking"]
                }
            },
            "required": ["data_type", "analysis_type"]
        }
        
        # 프롬프트 준비
        prompt = f"""사용자의 쿼리를 분석하여 필요한 데이터의 요구사항을 JSON 형식으로 추출하세요:

사용자 쿼리: "{query}"
작업 설명: "{task_description}"

다음 정보를 포함하세요:
1. 필요한 데이터 유형 (예: 노인 대화 텍스트, 심리 분석 데이터, 사용자 상호작용 등)
2. 분석 유형: 다음 중 하나 선택
   - general: 일반적인 데이터 분석
   - time_series: 시간에 따른 변화 추세 분석
   - conversation_trends: 대화 내용의 주제/감정 변화 분석
   - mental_health_tracking: 정신건강 상태 변화 추적
3. 필요한 필드/컬럼 목록
4. 관련 시간 범위 (시작 및 종료 시간, ISO 형식: YYYY-MM-DDTHH:MM:SS)
5. 특정 사용자/화자 ID 목록 (있는 경우)
6. 관련 키워드 목록

JSON 스키마에 맞춰 응답하세요."""
        
        # LLM으로 데이터 요구사항 추출
        try:
            requirements = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info(f"Extracted data requirements: {json.dumps(requirements, ensure_ascii=False)[:100]}...")
            return requirements
            
        except Exception as e:
            logger.error(f"Error extracting data requirements: {str(e)}")
            # 기본 요구사항 반환
            return {
                "data_type": "conversation_text",
                "analysis_type": "general",
                "fields": [],
                "time_period": {},
                "subject_ids": [],
                "keywords": []
            }
    
    def _find_best_data_source(self, requirements: Dict[str, Any]) -> Optional[str]:
        """
        요구사항에 가장 적합한 데이터 소스 찾기
        
        Args:
            requirements: 데이터 요구사항
        
        Returns:
            Optional[str]: 데이터 소스 경로
        """
        # 분석 유형에 따라 데이터 소스 결정
        analysis_type = requirements.get("analysis_type", "general")
        
        # 시계열 분석이나 대화 트렌드 분석의 경우 벡터 DB 사용
        if analysis_type in ["time_series", "conversation_trends", "mental_health_tracking"]:
            # 벡터 DB를 사용한다는 것을 표시하는 특별 경로 반환
            return "vector_db://conversation_data"
        
        # 일반 분석의 경우 파일 기반 데이터 소스 검색
        available_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.csv', '.xlsx', '.xls', '.json', '.parquet')):
                    available_files.append(os.path.join(root, file))
        
        # 적합한 파일이 없으면 None 반환
        if not available_files:
            return None
        
        # 간단한 구현: 첫 번째 파일 반환 (실제로는 파일 메타데이터와 요구사항 매칭)
        return available_files[0]
    
    def _analyze_data(self, data: pd.DataFrame, query: str, task_description: str = "") -> Dict[str, Any]:
        """
        데이터 분석 수행
        
        Args:
            data: 분석할 데이터프레임
            query: 사용자 쿼리
            task_description: 작업 설명
        
        Returns:
            Dict[str, Any]: 분석 결과
        """
        # 쿼리에서 요구사항 추출
        requirements = self._extract_data_requirements(query, task_description)
        analysis_type = requirements.get("analysis_type", "general")
        
        # 분석 유형에 따라 다른 처리
        if analysis_type == "time_series":
            return self._analyze_time_series_data(data, requirements)
        elif analysis_type == "conversation_trends":
            return self._analyze_conversation_trends(data, requirements)
        elif analysis_type == "mental_health_tracking":
            return self._analyze_mental_health_trends(data, requirements)
        
        # 일반적인 데이터 분석 (기본)
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 기술 통계
        stats = {}
        if numeric_columns:
            stats["numeric"] = data[numeric_columns].describe().to_dict()
        
        if categorical_columns:
            cat_stats = {}
            for col in categorical_columns:
                cat_stats[col] = data[col].value_counts().head(10).to_dict()
            stats["categorical"] = cat_stats
        
        # 상관관계 분석 (숫자형 데이터에 대해)
        correlation = None
        if len(numeric_columns) > 1:
            correlation = data[numeric_columns].corr().to_dict()
        
        # LLM 기반 분석 인사이트 생성
        insights = self._generate_insights(data, query, stats)
        
        # 시각화 필요 여부 및 명세 결정
        visualization_required = len(numeric_columns) > 0
        visualization_specs = []
        
        if visualization_required:
            # 간단한 시각화 명세 생성
            if len(numeric_columns) == 1:
                visualization_specs.append({
                    "type": "histogram",
                    "title": f"Distribution of {numeric_columns[0]}",
                    "x": numeric_columns[0]
                })
            elif len(numeric_columns) >= 2:
                visualization_specs.append({
                    "type": "scatter",
                    "title": f"Relationship between {numeric_columns[0]} and {numeric_columns[1]}",
                    "x": numeric_columns[0],
                    "y": numeric_columns[1]
                })
                
                visualization_specs.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "data": "correlation"
                })
        
        return {
            "analysis": {
                "summary_statistics": stats,
                "correlation": correlation
            },
            "visualization_required": visualization_required,
            "visualization_specs": visualization_specs,
            "insights": insights
        }
    
    def _generate_insights(self, data: pd.DataFrame, query: str, stats: Dict[str, Any]) -> List[str]:
        """
        데이터 분석 인사이트 생성
        
        Args:
            data: 데이터프레임
            query: 사용자 쿼리
            stats: 계산된 통계
        
        Returns:
            List[str]: 인사이트 목록
        """
        # 간단한 데이터 설명 생성
        data_description = f"데이터는 {len(data)}행, {len(data.columns)}열로 구성되어 있습니다."
        
        # 숫자형 열에 대한 설명
        numeric_insights = []
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns[:3]:  # 상위 3개만 처리
            mean = data[col].mean()
            median = data[col].median()
            min_val = data[col].min()
            max_val = data[col].max()
            
            insight = f"{col}의 평균은 {mean:.2f}, 중앙값은 {median:.2f}이며, 범위는 {min_val:.2f}에서 {max_val:.2f}입니다."
            numeric_insights.append(insight)
        
        # 범주형 열에 대한 설명
        categorical_insights = []
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_columns[:3]:  # 상위 3개만 처리
            if data[col].nunique() > 0:
                top_category = data[col].value_counts().index[0]
                top_count = data[col].value_counts().iloc[0]
                top_pct = (top_count / len(data)) * 100
                
                insight = f"{col}에서 가장 많은 값은 '{top_category}'로, 전체의 {top_pct:.1f}%({top_count}개)를 차지합니다."
                categorical_insights.append(insight)
        
        # 시간 데이터 여부 확인 및 인사이트 추가
        time_insights = []
        if 'timestamp' in data.columns and pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            time_range = f"데이터의 시간 범위는 {data['timestamp'].min()} 부터 {data['timestamp'].max()} 까지입니다."
            time_span = (data['timestamp'].max() - data['timestamp'].min()).days
            time_insights.append(time_range)
            time_insights.append(f"데이터는 총 {time_span}일 동안의 정보를 포함하고 있습니다.")
        
        # 모든 인사이트 결합
        all_insights = [data_description] + time_insights + numeric_insights + categorical_insights
        
        return all_insights
        
    def _handle_analyze_time_series(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        시계열 분석 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        data_path = message_content.get("data", {}).get("data_path", "")
        time_field = message_content.get("data", {}).get("time_field", "timestamp")
        value_field = message_content.get("data", {}).get("value_field", "")
        group_by = message_content.get("data", {}).get("group_by", "")
        time_range = message_content.get("data", {}).get("time_range", {})
        
        if not data_path:
            raise TaskExecutionError(
                message="Data path missing for time series analysis",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 데이터 로드
        data = self._load_data(data_path)
        
        if data is None or data.empty:
            return {
                "status": "error",
                "message": f"Failed to load data from {data_path}"
            }
        
        # 요구사항 생성
        requirements = {
            "time_field": time_field,
            "value_field": value_field,
            "group_by": group_by,
            "time_range": time_range
        }
        
        # 시계열 분석 수행
        result = self._analyze_time_series_data(data, requirements)
        
        return {
            "status": "success",
            "result": result
        }
        
    def _analyze_time_series_data(self, data: pd.DataFrame, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        시계열 데이터 분석
        
        Args:
            data: 분석할 데이터프레임
            requirements: 분석 요구사항
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        # 필요한 필드 추출
        time_field = requirements.get("time_field", "timestamp")
        value_field = requirements.get("value_field", "")
        group_by = requirements.get("group_by", "")
        time_range = requirements.get("time_period", {})
        
        # 시간 필드가 없으면 오류
        if time_field not in data.columns:
            return {
                "error": f"Time field '{time_field}' not found in data",
                "available_fields": list(data.columns)
            }
        
        # 시간 형식 변환
        if not pd.api.types.is_datetime64_any_dtype(data[time_field]):
            try:
                data[time_field] = pd.to_datetime(data[time_field])
            except:
                return {"error": f"Could not convert '{time_field}' to datetime"}
        
        # 시간 범위 필터링
        if time_range:
            if 'start' in time_range:
                start_time = pd.to_datetime(time_range['start'])
                data = data[data[time_field] >= start_time]
            if 'end' in time_range:
                end_time = pd.to_datetime(time_range['end'])
                data = data[data[time_field] <= end_time]
        
        # 데이터가 충분한지 확인
        if len(data) < 2:
            return {"error": "Not enough data points for time series analysis"}
        
        # 시계열 집계 (일/시간/분/초 단위)
        ts_analysis = {}
        
        # 시간 간격 결정
        time_span = data[time_field].max() - data[time_field].min()
        
        if time_span.days > 365:
            freq = 'M'  # 월별
            freq_name = "월별"
        elif time_span.days > 30:
            freq = 'W'  # 주별
            freq_name = "주별"
        elif time_span.days > 2:
            freq = 'D'  # 일별
            freq_name = "일별"
        elif time_span.days > 0 or time_span.seconds > 3600:
            freq = 'H'  # 시간별
            freq_name = "시간별"
        else:
            freq = 'min'  # 분별
            freq_name = "분별"
        
        # 기본 시계열 분석 - 전체 데이터 대상
        ts_analysis['time_range'] = {
            'start': data[time_field].min().isoformat(),
            'end': data[time_field].max().isoformat(),
            'span_days': time_span.days
        }
        
        # 값 필드가 제공되었는지 확인
        if value_field and value_field in data.columns:
            # 집계 분석
            time_series = data.set_index(time_field)
            resampled = time_series[value_field].resample(freq).agg(['mean', 'sum', 'count'])
            
            # 결과 JSON 변환 가능하도록 처리
            ts_analysis[f'{freq_name}_집계'] = resampled.reset_index().to_dict(orient='records')
            
            # 기본 통계
            ts_analysis['statistics'] = {
                'mean': data[value_field].mean(),
                'std': data[value_field].std(),
                'min': data[value_field].min(),
                'max': data[value_field].max(),
                'median': data[value_field].median()
            }
            
            # 트렌드 분석
            try:
                from scipy import stats
                x = np.arange(len(resampled))
                y = resampled['mean'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                trend_direction = "증가" if slope > 0 else "감소" if slope < 0 else "유지"
                trend_strength = abs(r_value)
                
                ts_analysis['trend'] = {
                    'direction': trend_direction,
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'strength': "강함" if trend_strength > 0.7 else "중간" if trend_strength > 0.3 else "약함",
                    'p_value': p_value
                }
            except:
                # scipy가 없거나 계산 실패 시 건너뜀
                pass
        
        # 그룹별 분석 (group_by가 제공된 경우)
        if group_by and group_by in data.columns and value_field and value_field in data.columns:
            group_analysis = {}
            
            # 각 그룹별 시계열 분석
            for group_val, group_data in data.groupby(group_by):
                # 해당 그룹에 데이터가 충분한지 확인
                if len(group_data) < 2:
                    continue
                    
                group_ts = group_data.set_index(time_field)
                group_resampled = group_ts[value_field].resample(freq).agg(['mean', 'sum', 'count'])
                
                group_analysis[str(group_val)] = {
                    'count': len(group_data),
                    'mean': group_data[value_field].mean(),
                    'min': group_data[value_field].min(),
                    'max': group_data[value_field].max()
                }
            
            ts_analysis[f'{group_by}_그룹별_분석'] = group_analysis
        
        # 시각화 명세 생성
        visualization_specs = []
        
        # 기본 시계열 차트
        if value_field and value_field in data.columns:
            visualization_specs.append({
                "type": "line",
                "title": f"{freq_name} {value_field} 추이",
                "x": time_field,
                "y": value_field
            })
            
            # 그룹별 차트 (그룹이 있는 경우)
            if group_by and group_by in data.columns:
                visualization_specs.append({
                    "type": "grouped_line",
                    "title": f"{group_by}별 {value_field} 추이",
                    "x": time_field,
                    "y": value_field,
                    "group": group_by
                })
        
        # 인사이트 생성
        insights = self._generate_time_series_insights(data, time_field, value_field, group_by, ts_analysis)
        
        return {
            "analysis": ts_analysis,
            "visualization_required": True,
            "visualization_specs": visualization_specs,
            "insights": insights
        }
        
    def _generate_time_series_insights(self, data: pd.DataFrame, time_field: str, value_field: str, 
                                      group_by: str, analysis: Dict[str, Any]) -> List[str]:
        """
        시계열 데이터 분석 인사이트 생성
        
        Args:
            data: 데이터프레임
            time_field: 시간 필드
            value_field: 값 필드
            group_by: 그룹화 필드
            analysis: 분석 결과
            
        Returns:
            List[str]: 인사이트 목록
        """
        insights = []
        
        # 기본 정보
        time_range = analysis.get('time_range', {})
        if time_range:
            insights.append(f"분석 기간: {time_range.get('start')} 부터 {time_range.get('end')} ({time_range.get('span_days')}일)")
        
        # 값 필드 트렌드 정보
        trend = analysis.get('trend', {})
        if trend:
            direction = trend.get('direction', '')
            strength = trend.get('strength', '')
            if direction and strength:
                insights.append(f"{value_field}의 전체적인 추세는 {direction}하는 경향을 보이며, 이 추세의 강도는 {strength}입니다.")
        
        # 통계 정보
        stats = analysis.get('statistics', {})
        if stats:
            insights.append(f"{value_field}의 평균값은 {stats.get('mean', 0):.2f}이며, 최솟값은 {stats.get('min', 0):.2f}, 최댓값은 {stats.get('max', 0):.2f}입니다.")
        
        # 그룹별 분석 정보
        group_analysis_key = f'{group_by}_그룹별_분석'
        if group_analysis_key in analysis:
            group_data = analysis[group_analysis_key]
            if group_data:
                # 가장 높은 평균값을 가진 그룹 찾기
                top_group = max(group_data.items(), key=lambda x: x[1].get('mean', 0))[0]
                insights.append(f"{group_by} 중에서는 '{top_group}'가 가장 높은 {value_field} 평균값을 보입니다.")
        
        return insights
        
    def _handle_analyze_conversation_trends(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        대화 트렌드 분석 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        data_path = message_content.get("data", {}).get("data_path", "")
        time_range = message_content.get("data", {}).get("time_range", {})
        subject_ids = message_content.get("data", {}).get("subject_ids", [])
        
        if not data_path:
            raise TaskExecutionError(
                message="Data path missing for conversation trend analysis",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 데이터 로드
        data = self._load_data(data_path)
        
        if data is None or data.empty:
            return {
                "status": "error",
                "message": f"Failed to load data from {data_path}"
            }
        
        # 요구사항 생성
        requirements = {
            "time_period": time_range,
            "subject_ids": subject_ids
        }
        
        # 대화 트렌드 분석 수행
        result = self._analyze_conversation_trends(data, requirements)
        
        return {
            "status": "success",
            "result": result
        }
        
    def _analyze_conversation_trends(self, data: pd.DataFrame, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        대화 텍스트 시계열 트렌드 분석
        
        Args:
            data: 분석할 대화 데이터프레임
            requirements: 분석 요구사항
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        # 필요한 필드가 있는지 확인
        if 'text' not in data.columns or 'timestamp' not in data.columns:
            return {
                "error": "Required fields 'text' and 'timestamp' not found in data",
                "available_fields": list(data.columns)
            }
        
        # 시간 형식 변환
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except:
                return {"error": "Could not convert 'timestamp' to datetime"}
        
        # 시간 범위 필터링
        time_range = requirements.get('time_period', {})
        if time_range:
            if 'start' in time_range:
                start_time = pd.to_datetime(time_range['start'])
                data = data[data['timestamp'] >= start_time]
            if 'end' in time_range:
                end_time = pd.to_datetime(time_range['end'])
                data = data[data['timestamp'] <= end_time]
        
        # 특정 대상 필터링
        subject_ids = requirements.get('subject_ids', [])
        if subject_ids and 'subject_id' in data.columns:
            data = data[data['subject_id'].isin(subject_ids)]
        
        # 데이터가 충분한지 확인
        if len(data) < 5:
            return {"error": "Not enough conversation data for analysis"}
        
        # 시간 간격 결정
        time_span = data['timestamp'].max() - data['timestamp'].min()
        
        if time_span.days > 365:
            freq = 'M'  # 월별
            freq_name = "월별"
        elif time_span.days > 30:
            freq = 'W'  # 주별
            freq_name = "주별"
        elif time_span.days > 2:
            freq = 'D'  # 일별
            freq_name = "일별"
        else:
            freq = 'H'  # 시간별
            freq_name = "시간별"
        
        # 분석 결과
        trend_analysis = {}
        
        # 시간 범위 정보
        trend_analysis['time_range'] = {
            'start': data['timestamp'].min().isoformat(),
            'end': data['timestamp'].max().isoformat(),
            'span_days': time_span.days
        }
        
        # 대화량 분석
        data_by_time = data.set_index('timestamp')
        message_count = data_by_time.resample(freq).size()
        
        trend_analysis[f'{freq_name}_대화량'] = message_count.reset_index().rename(columns={0: 'message_count'}).to_dict(orient='records')
        
        # 평균 대화 길이 분석
        data['text_length'] = data['text'].str.len()
        length_by_time = data.set_index('timestamp')
        avg_length = length_by_time['text_length'].resample(freq).mean()
        
        trend_analysis[f'{freq_name}_평균_대화_길이'] = avg_length.reset_index().to_dict(orient='records')
        
        # 대화 주제 및 감정 분석 (여기서는 실제 감정 분석은 LLM을 통해 수행해야 하지만 간소화)
        # 실제 구현에서는 심리분석 에이전트와 협업하여 감정 분석 수행
        
        # 시각화 명세 생성
        visualization_specs = []
        
        # 대화량 추이 차트
        visualization_specs.append({
            "type": "line",
            "title": f"{freq_name} 대화량 추이",
            "x": "timestamp",
            "y": "message_count",
            "data": trend_analysis[f'{freq_name}_대화량']
        })
        
        # 대화 길이 추이 차트
        visualization_specs.append({
            "type": "line",
            "title": f"{freq_name} 평균 대화 길이 추이",
            "x": "timestamp",
            "y": "text_length",
            "data": trend_analysis[f'{freq_name}_평균_대화_길이']
        })
        
        # 인사이트 생성
        insights = [
            f"분석 기간: {trend_analysis['time_range']['start']} 부터 {trend_analysis['time_range']['end']} ({trend_analysis['time_range']['span_days']}일)",
            f"총 대화 메시지 수: {len(data)}개",
            f"평균 대화 길이: {data['text_length'].mean():.1f} 글자"
        ]
        
        # 대화량 증감 분석
        if len(message_count) > 1:
            first_count = message_count.iloc[0]
            last_count = message_count.iloc[-1]
            change_pct = ((last_count - first_count) / first_count * 100) if first_count > 0 else 0
            
            if change_pct > 10:
                insights.append(f"분석 기간 동안 대화량이 {change_pct:.1f}% 증가했습니다.")
            elif change_pct < -10:
                insights.append(f"분석 기간 동안 대화량이 {abs(change_pct):.1f}% 감소했습니다.")
            else:
                insights.append(f"분석 기간 동안 대화량이 비교적 일정하게 유지되었습니다.")
        
        return {
            "analysis": trend_analysis,
            "visualization_required": True,
            "visualization_specs": visualization_specs,
            "insights": insights
        }
        
    def _analyze_mental_health_trends(self, data: pd.DataFrame, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        정신건강 지표의 시계열 트렌드 분석
        
        Args:
            data: 분석할 대화 데이터프레임
            requirements: 분석 요구사항
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        # 필요한 필드가 있는지 확인
        required_fields = ['text', 'timestamp']
        missing_fields = [field for field in required_fields if field not in data.columns]
        
        if missing_fields:
            return {
                "error": f"Required fields {missing_fields} not found in data",
                "available_fields": list(data.columns)
            }
        
        # 시간 형식 변환
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except:
                return {"error": "Could not convert 'timestamp' to datetime"}
        
        # 시간 범위 필터링
        time_range = requirements.get('time_period', {})
        if time_range:
            if 'start' in time_range:
                start_time = pd.to_datetime(time_range['start'])
                data = data[data['timestamp'] >= start_time]
            if 'end' in time_range:
                end_time = pd.to_datetime(time_range['end'])
                data = data[data['timestamp'] <= end_time]
        
        # 특정 대상 필터링
        subject_ids = requirements.get('subject_ids', [])
        if subject_ids and 'subject_id' in data.columns:
            data = data[data['subject_id'].isin(subject_ids)]
        
        # 데이터가 충분한지 확인
        if len(data) < 5:
            return {"error": "Not enough data for mental health trend analysis"}
        
        # 시간 간격 결정
        time_span = data['timestamp'].max() - data['timestamp'].min()
        
        if time_span.days > 365:
            freq = 'M'  # 월별
            freq_name = "월별"
        elif time_span.days > 30:
            freq = 'W'  # 주별
            freq_name = "주별"
        elif time_span.days > 2:
            freq = 'D'  # 일별
            freq_name = "일별"
        else:
            freq = 'H'  # 시간별
            freq_name = "시간별"
        
        # 분석 결과
        mental_health_analysis = {}
        
        # 시간 범위 정보
        mental_health_analysis['time_range'] = {
            'start': data['timestamp'].min().isoformat(),
            'end': data['timestamp'].max().isoformat(),
            'span_days': time_span.days
        }
        
        # LLM을 활용한 텍스트 감정 분석 수행 (실제로는 심리분석 에이전트 활용)
        # 여기서는 간단한 키워드 기반 분석으로 대체
        
        # 우울 관련 키워드
        depression_keywords = ['우울', '슬프', '힘들', '외롭', '공허', '의욕', '무기력', '피곤', '잠', '식욕']
        # 불안 관련 키워드
        anxiety_keywords = ['불안', '걱정', '초조', '두려움', '공포', '긴장', '스트레스', '떨림', '불면']
        # 인지기능 관련 키워드
        cognitive_keywords = ['기억', '잊어', '혼란', '집중', '생각', '판단', '결정', '머리']
        
        # 키워드 검색 함수
        def count_keywords(text, keywords):
            count = 0
            for keyword in keywords:
                if keyword in str(text):
                    count += 1
            return count
        
        # 각 메시지에 대한 키워드 카운트 추가
        data['depression_score'] = data['text'].apply(lambda x: count_keywords(x, depression_keywords))
        data['anxiety_score'] = data['text'].apply(lambda x: count_keywords(x, anxiety_keywords))
        data['cognitive_score'] = data['text'].apply(lambda x: count_keywords(x, cognitive_keywords))
        
        # 시간별 집계
        mental_data = data.set_index('timestamp')
        
        depression_trend = mental_data['depression_score'].resample(freq).mean()
        anxiety_trend = mental_data['anxiety_score'].resample(freq).mean()
        cognitive_trend = mental_data['cognitive_score'].resample(freq).mean()
        
        # 결과 저장
        mental_health_analysis[f'{freq_name}_우울_지표'] = depression_trend.reset_index().to_dict(orient='records')
        mental_health_analysis[f'{freq_name}_불안_지표'] = anxiety_trend.reset_index().to_dict(orient='records')
        mental_health_analysis[f'{freq_name}_인지_지표'] = cognitive_trend.reset_index().to_dict(orient='records')
        
        # 시각화 명세 생성
        visualization_specs = []
        
        # 통합 정신건강 지표 차트
        visualization_specs.append({
            "type": "multi_line",
            "title": f"{freq_name} 정신건강 지표 추이",
            "x": "timestamp",
            "y_series": ["depression_score", "anxiety_score", "cognitive_score"],
            "y_labels": ["우울 지표", "불안 지표", "인지기능 지표"]
        })
        
        # 인사이트 생성
        insights = [
            f"분석 기간: {mental_health_analysis['time_range']['start']} 부터 {mental_health_analysis['time_range']['end']} ({mental_health_analysis['time_range']['span_days']}일)",
        ]
        
        # 우울 지표 트렌드 분석
        if len(depression_trend) > 1:
            first_score = depression_trend.iloc[0]
            last_score = depression_trend.iloc[-1]
            change = last_score - first_score
            
            if change > 0.5:
                insights.append(f"우울 관련 표현이 증가 추세를 보입니다. (변화량: +{change:.2f})")
            elif change < -0.5:
                insights.append(f"우울 관련 표현이 감소 추세를 보입니다. (변화량: {change:.2f})")
        
        # 불안 지표 트렌드 분석
        if len(anxiety_trend) > 1:
            first_score = anxiety_trend.iloc[0]
            last_score = anxiety_trend.iloc[-1]
            change = last_score - first_score
            
            if change > 0.5:
                insights.append(f"불안 관련 표현이 증가 추세를 보입니다. (변화량: +{change:.2f})")
            elif change < -0.5:
                insights.append(f"불안 관련 표현이 감소 추세를 보입니다. (변화량: {change:.2f})")
        
        # 인지기능 지표 트렌드 분석
        if len(cognitive_trend) > 1:
            first_score = cognitive_trend.iloc[0]
            last_score = cognitive_trend.iloc[-1]
            change = last_score - first_score
            
            if change > 0.5:
                insights.append(f"인지기능 관련 표현이 증가 추세를 보입니다. (변화량: +{change:.2f})")
            elif change < -0.5:
                insights.append(f"인지기능 관련 표현이 감소 추세를 보입니다. (변화량: {change:.2f})")
        
        # 추가 통찰
        insights.append("※ 참고: 이 분석은 단순 키워드 기반으로 수행되었으며, 정확한 정신건강 평가는 전문가의 판단이 필요합니다.")
        
        return {
            "analysis": mental_health_analysis,
            "visualization_required": True,
            "visualization_specs": visualization_specs,
            "insights": insights
        }
    
    def _generate_visualizations(self, data: pd.DataFrame, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        데이터 시각화 생성
        
        Args:
            data: 데이터프레임
            specs: 시각화 명세 목록
        
        Returns:
            List[Dict[str, Any]]: 생성된 시각화 정보
        """
        visualizations = []
        
        for i, spec in enumerate(specs):
            vis_type = spec.get("type", "")
            title = spec.get("title", f"Visualization {i+1}")
            
            try:
                # 시각화 생성
                plt.figure(figsize=(10, 6))
                
                if vis_type == "histogram":
                    x = spec.get("x")
                    if x in data.columns:
                        data[x].hist()
                        plt.xlabel(x)
                        plt.ylabel("Frequency")
                
                elif vis_type == "scatter":
                    x = spec.get("x")
                    y = spec.get("y")
                    if x in data.columns and y in data.columns:
                        plt.scatter(data[x], data[y])
                        plt.xlabel(x)
                        plt.ylabel(y)
                
                elif vis_type == "heatmap":
                    if spec.get("data") == "correlation":
                        numeric_data = data.select_dtypes(include=[np.number])
                        if not numeric_data.empty:
                            corr = numeric_data.corr()
                            plt.imshow(corr, cmap='coolwarm')
                            plt.colorbar()
                            plt.xticks(range(len(corr)), corr.columns, rotation=90)
                            plt.yticks(range(len(corr)), corr.columns)
                
                elif vis_type == "bar":
                    x = spec.get("x")
                    y = spec.get("y")
                    if x in data.columns:
                        if y in data.columns:
                            data.groupby(x)[y].mean().plot(kind='bar')
                        else:
                            data[x].value_counts().head(10).plot(kind='bar')
                        plt.xlabel(x)
                        plt.xticks(rotation=45)
                
                elif vis_type == "line":
                    x = spec.get("x")
                    y = spec.get("y")
                    if x in data.columns and y in data.columns:
                        data.sort_values(x).plot(x=x, y=y, kind='line')
                        plt.xlabel(x)
                        plt.ylabel(y)
                
                plt.title(title)
                plt.tight_layout()
                
                # 이미지를 base64로 인코딩
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                
                visualizations.append({
                    "title": title,
                    "type": vis_type,
                    "image_data": img_str,
                    "spec": spec
                })
                
            except Exception as e:
                logger.error(f"Error generating visualization: {str(e)}")
                plt.close()
        
        return visualizations

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
            
            # 쿼리에서 데이터 요구사항 추출
            data_requirements = self._extract_data_requirements(query, task_description)
            
            # 가장 적합한 데이터 소스 결정
            data_path = self._find_best_data_source(data_requirements)
            
            if not data_path:
                return {
                    "status": "error",
                    "message": "No suitable data found for analysis"
                }
            
            # 데이터 로드
            data = self._load_data(data_path)
            
            if data is None:
                return {
                    "status": "error",
                    "message": f"Failed to load data from {data_path}"
                }
            
            # 데이터 분석
            analysis_result = self._analyze_data(data, query, task_description)
            
            # 시각화 생성 (필요한 경우)
            visualizations = []
            if analysis_result.get("visualization_required", False):
                visualizations = self._generate_visualizations(data, analysis_result.get("visualization_specs", []))
            
            # 결과 취합
            return {
                "status": "success",
                "data": {
                    "data_source": data_path,
                    "data_summary": {
                        "rows": len(data),
                        "columns": list(data.columns),
                        "data_types": {col: str(data[col].dtype) for col in data.columns}
                    },
                    "analysis": analysis_result.get("analysis", {}),
                    "visualizations": visualizations,
                    "insights": analysis_result.get("insights", [])
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing user query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }


# 전역 데이터 분석 에이전트 인스턴스
data_analysis_agent = DataAnalysisAgent()

def get_data_analysis_agent() -> DataAnalysisAgent:
    """데이터 분석 에이전트 인스턴스 가져오기"""
    return data_analysis_agent
