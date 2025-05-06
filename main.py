#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 진입점
"""

import argparse
import logging
import os
from dotenv import load_dotenv

# 내부 모듈 임포트
from core.config import load_config
from core.logging_manager import setup_logging
from core.state_manager import StateManager
from workflow.graph_definitions import create_agent_workflow
from agents.supervisor import SupervisorAgent
from agents.rag_agent import RAGAgent
from agents.web_search_agent import WebSearchAgent
from agents.data_analysis_agent import DataAnalysisAgent
from agents.psychological_agent import PsychologicalAgent
from agents.report_writer_agent import ReportWriterAgent

# 환경 변수 로드
load_dotenv()

def parse_arguments():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='LangGraph 기반 멀티 에이전트 시스템')
    parser.add_argument('--query', type=str, help='처리할 쿼리 또는 질문')
    parser.add_argument('--config', type=str, default='./config/default.yaml', 
                        help='설정 파일 경로')
    parser.add_argument('--output-format', type=str, default='text', 
                        choices=['text', 'json', 'markdown'], 
                        help='출력 형식')
    parser.add_argument('--verbose', action='store_true', 
                        help='상세 로그 출력')
    return parser.parse_args()

def initialize_system(config_path, verbose=False):
    """시스템을 초기화합니다."""
    # 로깅 설정
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_level)
    
    # 설정 로드
    config = load_config(config_path)
    
    # 상태 관리자 초기화
    state_manager = StateManager()
    
    # 에이전트 초기화
    supervisor = SupervisorAgent(config=config.get('agents', {}).get('supervisor', {}))
    rag_agent = RAGAgent(config=config.get('agents', {}).get('rag', {}))
    web_search_agent = WebSearchAgent(config=config.get('agents', {}).get('web_search', {}))
    data_analysis_agent = DataAnalysisAgent(config=config.get('agents', {}).get('data_analysis', {}))
    psychological_agent = PsychologicalAgent(config=config.get('agents', {}).get('psychological', {}))
    report_writer_agent = ReportWriterAgent(config=config.get('agents', {}).get('report_writer', {}))
    
    # 워크플로 생성
    workflow = create_agent_workflow(
        supervisor=supervisor,
        rag_agent=rag_agent,
        web_search_agent=web_search_agent,
        data_analysis_agent=data_analysis_agent,
        psychological_agent=psychological_agent,
        report_writer_agent=report_writer_agent,
        state_manager=state_manager
    )
    
    return {
        'config': config,
        'state_manager': state_manager,
        'workflow': workflow,
        'supervisor': supervisor
    }

def process_query(query, system, output_format='text'):
    """쿼리를 처리하고 결과를 반환합니다."""
    logging.info(f"쿼리 처리 시작: {query}")
    
    # 초기 상태 설정
    initial_state = {
        "user_query": query,
        "current_stage": "planning",
        "active_agents": ["supervisor"],
        "conversation_history": [],
        "shared_knowledge": {}
    }
    
    # 워크플로 실행
    try:
        result_state = system['workflow'].invoke(initial_state)
        
        # 결과 포맷팅
        if output_format == 'json':
            import json
            return json.dumps(result_state, ensure_ascii=False, indent=2)
        elif output_format == 'markdown':
            # 마크다운 형식으로 결과 변환
            if 'final_response' in result_state:
                return result_state['final_response']
            else:
                return "# 처리 결과\n\n" + str(result_state)
        else:
            # 기본 텍스트 형식
            if 'final_response' in result_state:
                return result_state['final_response']
            else:
                return str(result_state)
    except Exception as e:
        logging.error(f"쿼리 처리 중 오류 발생: {str(e)}", exc_info=True)
        return f"오류: {str(e)}"

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 시스템 초기화
    system = initialize_system(args.config, args.verbose)
    
    # 쿼리 처리
    if args.query:
        result = process_query(args.query, system, args.output_format)
        print(result)
    else:
        # 대화형 모드
        print("LangGraph 기반 멀티 에이전트 시스템에 오신 것을 환영합니다.")
        print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
        
        while True:
            query = input("\n질문이나 쿼리를 입력하세요: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            result = process_query(query, system, args.output_format)
            print("\n결과:")
            print(result)

if __name__ == "__main__":
    main()
