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
import base64
from io import BytesIO

from agents.base import Agent
from core.config import get_config, DATA_DIR
from core.error_handler import handle_agent_error, TaskExecutionError
from services.llm_service import generate_text, generate_json

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
        
        logger.info(f"Data analysis agent initialized with ID: {self.agent_id}")
    
    def _register_task_handlers(self) -> None:
        """작업 핸들러 등록"""
        self.task_handlers = {
            "process_task": self._handle_process_task,
            "load_data": self._handle_load_data,
            "analyze_data": self._handle_analyze_data,
            "generate_visualization": self._handle_generate_visualization
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
            data_path: 데이터 파일 경로
        
        Returns:
            Optional[pd.DataFrame]: 로드된 데이터프레임
        """
        try:
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
                "time_period": {"type": "string"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["data_type", "fields", "keywords"]
        }
        
        # 프롬프트 준비
        prompt = f"""사용자의 쿼리를 분석하여 필요한 데이터의 요구사항을 JSON 형식으로 추출하세요:

사용자 쿼리: "{query}"
작업 설명: "{task_description}"

다음 정보를 포함하세요:
1. 필요한 데이터 유형 (예: 금융 데이터, 인구통계 데이터, 텍스트 데이터 등)
2. 필요한 필드/컬럼 목록
3. 관련 시간 범위 (해당되는 경우)
4. 관련 키워드 목록

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
                "data_type": "unknown",
                "fields": [],
                "time_period": "",
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
        # 데이터 디렉토리의 모든 파일 스캔 (실제로는 메타데이터 DB 사용 권장)
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
        # 기본적인 데이터 분석 수행
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
            top_category = data[col].value_counts().index[0]
            top_count = data[col].value_counts().iloc[0]
            top_pct = (top_count / len(data)) * 100
            
            insight = f"{col}에서 가장 많은 값은 '{top_category}'로, 전체의 {top_pct:.1f}%({top_count}개)를 차지합니다."
            categorical_insights.append(insight)
        
        # 모든 인사이트 결합
        all_insights = [data_description] + numeric_insights + categorical_insights
        
        return all_insights
    
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
