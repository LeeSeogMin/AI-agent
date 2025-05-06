#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 로깅 관리 모듈
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from logging.handlers import RotatingFileHandler
import datetime
from pathlib import Path

from core.config import get_config

# 로그 파일 경로 설정
config = get_config()
LOG_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 로그 레벨 매핑
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

class JsonFormatter(logging.Formatter):
    """JSON 형식 로그 포맷터"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        로그 레코드를 JSON 형식으로 포맷
        
        Args:
            record: 로그 레코드
        
        Returns:
            str: JSON 형식 문자열
        """
        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 예외 정보 추가
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # 추가 속성 처리
        for key, value in record.__dict__.items():
            if key not in [
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread", "threadName"
            ]:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


class LoggingManager:
    """로깅 관리자 클래스"""
    
    def __init__(self):
        """로깅 관리자 초기화"""
        self.loggers: Dict[str, logging.Logger] = {}
        self.default_level = LOG_LEVELS.get(
            os.getenv("LOG_LEVEL", "info").lower(), 
            logging.INFO
        )
        
        # 기본 콘솔 핸들러 설정
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        
        # 기본 파일 핸들러 설정
        self.file_handler = RotatingFileHandler(
            LOG_DIR / "system.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        self.file_handler.setFormatter(JsonFormatter())
        
        # JSON 파일 핸들러 설정
        self.json_handler = RotatingFileHandler(
            LOG_DIR / "system.json",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        self.json_handler.setFormatter(JsonFormatter())
        
        # 루트 로거 설정
        self.setup_root_logger()
    
    def setup_root_logger(self) -> None:
        """루트 로거 설정"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 핸들러 추가
        root_logger.addHandler(self.console_handler)
        root_logger.addHandler(self.file_handler)
        
        # 디버그 모드에서만 JSON 핸들러 추가
        if config.debug:
            root_logger.addHandler(self.json_handler)
    
    def get_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        로거 가져오기
        
        Args:
            name: 로거 이름
            level: 로그 레벨 (없으면 기본 레벨 사용)
        
        Returns:
            logging.Logger: 로거 인스턴스
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        
        # 로그 레벨 설정
        if level:
            logger.setLevel(LOG_LEVELS.get(level.lower(), self.default_level))
        else:
            logger.setLevel(self.default_level)
        
        # 핸들러 추가하지 않음 (루트 로거에서 상속)
        
        # 캐시에 저장
        self.loggers[name] = logger
        
        return logger
    
    def set_level(self, name: str, level: str) -> bool:
        """
        로거 레벨 설정
        
        Args:
            name: 로거 이름
            level: 로그 레벨
        
        Returns:
            bool: 성공 여부
        """
        log_level = LOG_LEVELS.get(level.lower())
        if not log_level:
            return False
        
        if name in self.loggers:
            self.loggers[name].setLevel(log_level)
            return True
        
        # 로거가 없으면 생성하고 레벨 설정
        logger = self.get_logger(name)
        logger.setLevel(log_level)
        return True
    
    def add_file_handler(self, name: str, filename: str, json_format: bool = False) -> bool:
        """
        로거에 파일 핸들러 추가
        
        Args:
            name: 로거 이름
            filename: 로그 파일 이름
            json_format: JSON 형식 사용 여부
        
        Returns:
            bool: 성공 여부
        """
        if name not in self.loggers:
            return False
        
        logger = self.loggers[name]
        
        # 로그 파일 경로 설정
        if not os.path.isabs(filename):
            filename = os.path.join(LOG_DIR, filename)
        
        # 파일 핸들러 생성
        file_handler = RotatingFileHandler(
            filename,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8"
        )
        
        # 포맷터
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
        
        # 로거에 핸들러 추가
        logger.addHandler(file_handler)
        return True
    
    def configure_agent_logger(
        self, 
        agent_id: str, 
        level: Optional[str] = None, 
        log_to_file: bool = True
    ) -> logging.Logger:
        """
        에이전트 로거 설정
        
        Args:
            agent_id: 에이전트 ID
            level: 로그 레벨
            log_to_file: 파일 로깅 활성화 여부
        
        Returns:
            logging.Logger: 로거 인스턴스
        """
        logger_name = f"agent.{agent_id}"
        logger = self.get_logger(logger_name, level)
        
        if log_to_file:
            filename = f"agent_{agent_id}.log"
            self.add_file_handler(logger_name, filename)
            
            # JSON 로그 파일도 추가
            json_filename = f"agent_{agent_id}.json"
            self.add_file_handler(logger_name, json_filename, json_format=True)
        
        return logger
    
    def add_context_filter(self, name: str, context: Dict[str, Any]) -> None:
        """
        로거에 컨텍스트 필터 추가
        
        Args:
            name: 로거 이름
            context: 컨텍스트 데이터
        """
        if name not in self.loggers:
            return
        
        logger = self.loggers[name]
        
        class ContextFilter(logging.Filter):
            def filter(self, record):
                for key, value in context.items():
                    setattr(record, key, value)
                return True
        
        logger.addFilter(ContextFilter())


# 전역 로깅 관리자 인스턴스
logging_manager = LoggingManager()

def get_logging_manager() -> LoggingManager:
    """로깅 관리자 인스턴스 가져오기"""
    return logging_manager


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    로거 가져오기 유틸리티 함수
    
    Args:
        name: 로거 이름
        level: 로그 레벨
    
    Returns:
        logging.Logger: 로거 인스턴스
    """
    return logging_manager.get_logger(name, level)


def get_agent_logger(agent_id: str, level: Optional[str] = None) -> logging.Logger:
    """
    에이전트 로거 가져오기 유틸리티 함수
    
    Args:
        agent_id: 에이전트 ID
        level: 로그 레벨
    
    Returns:
        logging.Logger: 로거 인스턴스
    """
    return logging_manager.configure_agent_logger(agent_id, level)
