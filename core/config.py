#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 설정 모듈
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

# 기본 경로 설정
ROOT_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
USER_UPLOADS_DIR = DATA_DIR / "user_uploads"
OUTPUT_DIR = DATA_DIR / "output"
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIRECTORY", str(ROOT_DIR / "vector_db"))

# 필요한 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(USER_UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(Path(VECTOR_DB_DIR), exist_ok=True)

class Config:
    """시스템 설정 클래스"""
    
    def __init__(self, env: str = "default"):
        """
        설정 초기화
        
        Args:
            env: 환경 설정 (default, development, production)
        """
        self.env = env
        self.config_data = self._load_config()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # 시스템 설정
        self.debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
        
        # 벡터 DB 설정
        self.vector_db_provider = os.getenv("VECTOR_DB_PROVIDER", "chroma")
        self.vector_db_collection = os.getenv("VECTOR_DB_COLLECTION", "knowledge_base")
        self.vector_db_directory = VECTOR_DB_DIR
        
        # 웹 검색 설정
        self.web_search_provider = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")
        self.web_search_max_results = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
        self.web_search_timeout = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
        
        # LLM 설정
        self.default_llm_model = "gpt-4o"
        self.default_embedding_model = "text-embedding-3-large"
        
        # 에이전트 설정
        self.agents = {
            "supervisor": {"enabled": True},
            "rag": {"enabled": True},
            "web_search": {"enabled": True},
            "data_analysis": {"enabled": True},
            "psychological": {"enabled": True},
            "report_writer": {"enabled": True},
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """
        YAML 설정 파일 로드
        
        Returns:
            Dict[str, Any]: 로드된 설정 데이터
        """
        config_files = [
            CONFIG_DIR / "default.yaml",
            CONFIG_DIR / f"{self.env}.yaml"
        ]
        
        config_data = {}
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if data:
                            config_data.update(data)
                except Exception as e:
                    logger.error(f"설정 파일 로드 오류 {config_file}: {e}")
        
        return config_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값 가져오기
        
        Args:
            key: 설정 키
            default: 기본값
        
        Returns:
            Any: 설정 값
        """
        keys = key.split(".")
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """
        설정 값 업데이트
        
        Args:
            key: 설정 키
            value: 설정 값
        """
        keys = key.split(".")
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

# 기본 설정 인스턴스
system_config = Config(env=os.getenv("ENV", "default"))

def get_config() -> Config:
    """전역 설정 인스턴스 가져오기"""
    return system_config
