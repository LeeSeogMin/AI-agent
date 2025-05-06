#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 보고서 작성 에이전트 모듈
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from agents.base import Agent
from core.config import get_config, DATA_DIR
from core.error_handler import handle_agent_error, TaskExecutionError
from services.llm_service import generate_text, generate_json

# 설정 가져오기
config = get_config()

# 로거 설정
logger = logging.getLogger(__name__)

class ReportWriterAgent(Agent):
    """보고서 작성 에이전트 클래스"""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        system_message: Optional[str] = None
    ):
        """
        보고서 작성 에이전트 초기화
        
        Args:
            agent_id: 에이전트 ID (없으면 자동 생성)
            system_message: 시스템 메시지
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="report_writer",
            system_message=system_message
        )
        
        # 시스템 메시지 로드
        if not self.system_message:
            self.system_message = self.load_system_message()
        
        # 작업 핸들러 등록
        self._register_task_handlers()
        
        # 보고서 저장 디렉토리
        self.reports_dir = os.path.join(DATA_DIR, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        logger.info(f"Report writer agent initialized with ID: {self.agent_id}")
    
    def _register_task_handlers(self) -> None:
        """작업 핸들러 등록"""
        self.task_handlers = {
            "process_task": self._handle_process_task,
            "generate_report": self._handle_generate_report,
            "summarize_content": self._handle_summarize_content,
            "format_report": self._handle_format_report
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
        sources = message_content.get("data", {}).get("sources", [])
        insights = message_content.get("data", {}).get("insights", [])
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for report generation",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 보고서 유형 결정
        report_type = self._determine_report_type(query, task_description)
        
        # 보고서 생성
        report = self._generate_report(
            query=query,
            report_type=report_type,
            sources=sources,
            insights=insights
        )
        
        # 보고서 저장 (옵션)
        report_id = self._save_report(report, query)
        
        return {
            "status": "success",
            "result": {
                "report": report,
                "report_id": report_id,
                "report_type": report_type
            }
        }
    
    @handle_agent_error()
    def _handle_generate_report(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        보고서 생성 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        query = message_content.get("data", {}).get("query", "")
        report_type = message_content.get("data", {}).get("report_type", "general")
        sources = message_content.get("data", {}).get("sources", [])
        insights = message_content.get("data", {}).get("insights", [])
        
        if not query:
            raise TaskExecutionError(
                message="No query provided for report generation",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 보고서 생성
        report = self._generate_report(
            query=query,
            report_type=report_type,
            sources=sources,
            insights=insights
        )
        
        return {
            "status": "success",
            "result": {
                "report": report
            }
        }
    
    @handle_agent_error()
    def _handle_summarize_content(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        콘텐츠 요약 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        content = message_content.get("data", {}).get("content", "")
        max_length = message_content.get("data", {}).get("max_length", 500)
        
        if not content:
            raise TaskExecutionError(
                message="No content provided for summarization",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 콘텐츠 요약
        summary = self._summarize_content(content, max_length)
        
        return {
            "status": "success",
            "result": {
                "summary": summary
            }
        }
    
    @handle_agent_error()
    def _handle_format_report(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        보고서 형식 지정 핸들러
        
        Args:
            message_content: 메시지 내용
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        report = message_content.get("data", {}).get("report", {})
        format_type = message_content.get("data", {}).get("format", "markdown")
        
        if not report:
            raise TaskExecutionError(
                message="No report provided for formatting",
                agent_id=self.agent_id,
                task_id=message_content.get("metadata", {}).get("task_id", "unknown")
            )
        
        # 보고서 형식 지정
        formatted_report = self._format_report(report, format_type)
        
        return {
            "status": "success",
            "result": {
                "formatted_report": formatted_report
            }
        }
    
    def _determine_report_type(self, query: str, task_description: str) -> str:
        """
        쿼리 분석을 통한 보고서 유형 결정
        
        Args:
            query: 사용자 쿼리
            task_description: 작업 설명
        
        Returns:
            str: 보고서 유형 (summary, analysis, research, comparison, guide, general)
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "report_type": {
                    "type": "string",
                    "enum": ["summary", "analysis", "research", "comparison", "guide", "general"]
                },
                "reasoning": {"type": "string"}
            },
            "required": ["report_type", "reasoning"]
        }
        
        # 프롬프트 준비
        prompt = f"""사용자의 쿼리를 분석하여 가장 적합한 보고서 유형을 결정하세요:

사용자 쿼리: "{query}"
작업 설명: "{task_description}"

다음 보고서 유형 중에서 선택하세요:
1. summary: 정보를 간결하게 요약하는 보고서
2. analysis: 깊이 있는 분석을 제공하는 보고서
3. research: 조사 결과를 정리한 보고서
4. comparison: 여러 옵션이나 관점을 비교하는 보고서
5. guide: 단계별 안내나 방법론을 제공하는 보고서
6. general: 일반적인 정보를 제공하는 보고서

JSON 스키마에 맞춰 응답하세요."""
        
        # LLM으로 보고서 유형 결정
        try:
            report_type_result = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            report_type = report_type_result.get("report_type", "general")
            logger.info(f"Determined report type: {report_type}")
            return report_type
            
        except Exception as e:
            logger.error(f"Error determining report type: {str(e)}")
            return "general"
    
    def _generate_report(
        self, 
        query: str, 
        report_type: str,
        sources: List[Dict[str, Any]] = [],
        insights: List[Any] = []
    ) -> Dict[str, Any]:
        """
        보고서 생성
        
        Args:
            query: 사용자 쿼리
            report_type: 보고서 유형
            sources: 정보 소스 목록
            insights: 인사이트 목록
        
        Returns:
            Dict[str, Any]: 생성된 보고서
        """
        # JSON 스키마 정의
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "executive_summary": {"type": "string"},
                "introduction": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                },
                "conclusion": {"type": "string"},
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "citation": {"type": "string"}
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created_at": {"type": "string"},
                        "report_type": {"type": "string"},
                        "query": {"type": "string"}
                    }
                }
            },
            "required": ["title", "executive_summary", "introduction", "sections", "conclusion", "metadata"]
        }
        
        # 소스 정보 형식화
        sources_text = ""
        if sources:
            sources_text = "제공된 정보 소스:\n"
            for i, source in enumerate(sources):
                source_info = f"{i+1}. "
                if "title" in source:
                    source_info += f"제목: {source.get('title', '')}\n"
                if "url" in source:
                    source_info += f"URL: {source.get('url', '')}\n"
                if "summary" in source:
                    source_info += f"요약: {source.get('summary', '')}\n"
                sources_text += source_info + "\n"
        
        # 인사이트 정보 형식화
        insights_text = ""
        if insights:
            insights_text = "제공된 인사이트:\n"
            for i, insight in enumerate(insights):
                if isinstance(insight, str):
                    insights_text += f"{i+1}. {insight}\n"
                elif isinstance(insight, dict):
                    insights_text += f"{i+1}. {json.dumps(insight, ensure_ascii=False)}\n"
        
        # 보고서 유형별 지침
        type_instruction = {
            "summary": "간결하고 핵심적인 요약 보고서를 작성하세요. 중요한 정보만 포함하고, 불필요한 세부 사항은 제외하세요.",
            "analysis": "깊이 있는 분석 보고서를 작성하세요. 다양한 측면에서 주제를 검토하고, 통찰력 있는 분석을 제공하세요.",
            "research": "철저한 연구 보고서를 작성하세요. 소스를 적절히 인용하고, 방법론과 발견 사항을 명확히 설명하세요.",
            "comparison": "비교 보고서를 작성하세요. 주요 유사점과 차이점을 강조하고, 균형 잡힌 관점을 제공하세요.",
            "guide": "단계별 가이드를 작성하세요. 실용적인 정보와 구체적인 지침을 제공하세요.",
            "general": "일반적인 정보 제공 보고서를 작성하세요. 명확하고 이해하기 쉬운 언어를 사용하세요."
        }
        
        # 프롬프트 준비
        prompt = f"""다음 정보를 기반으로 "{report_type}" 유형의 보고서를 생성하세요:

쿼리: "{query}"

{sources_text}

{insights_text}

보고서 지침: {type_instruction.get(report_type, "일반적인 보고서를 작성하세요.")}

보고서에는 다음을 포함하세요:
1. 명확하고 설명적인 제목
2. 핵심 내용을 요약한 경영 요약
3. 주제와 보고서의 목적을 소개하는 서론
4. 주요 섹션 (적절한 제목과 내용으로 구성)
5. 주요 포인트를 요약하는 결론
6. 적절한 경우 실행 가능한 권장사항
7. 사용된 소스에 대한 참조 (제공된 경우)

각 섹션은
- 명확하고 간결한 언어를 사용하세요
- 관련성 있고 사실에 기반한 정보를 제공하세요
- 논리적인 구조로 정보를 구성하세요
- 중요한 포인트를 강조하세요
- 전문 용어를 사용할 경우 설명을 추가하세요

JSON 스키마에 맞춰 응답하세요."""
        
        # LLM으로 보고서 생성
        try:
            timestamp = datetime.now().isoformat()
            
            report = generate_json(
                prompt=prompt,
                schema=schema,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            # 메타데이터 추가
            if "metadata" not in report:
                report["metadata"] = {}
            
            report["metadata"].update({
                "created_at": timestamp,
                "report_type": report_type,
                "query": query
            })
            
            logger.info(f"Generated report: {report.get('title', 'Untitled')}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            # 기본 보고서 반환
            return {
                "title": "보고서 생성 실패",
                "executive_summary": "보고서를 생성하는 중 오류가 발생했습니다.",
                "introduction": "죄송합니다, 요청하신 보고서를 생성할 수 없습니다.",
                "sections": [
                    {
                        "heading": "오류 발생",
                        "content": f"보고서 생성 중 다음 오류가 발생했습니다: {str(e)}"
                    }
                ],
                "conclusion": "다시 시도해주세요.",
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "report_type": report_type,
                    "query": query
                }
            }
    
    def _summarize_content(self, content: str, max_length: int = 500) -> str:
        """
        콘텐츠 요약
        
        Args:
            content: 요약할 콘텐츠
            max_length: 최대 요약 길이 (문자 수)
        
        Returns:
            str: 요약된 콘텐츠
        """
        # 콘텐츠가 이미 충분히 짧으면 그대로 반환
        if len(content) <= max_length:
            return content
        
        # 프롬프트 준비
        prompt = f"""다음 콘텐츠를 {max_length}자 이내로 요약하세요. 원본의 핵심 정보를 보존하면서 가능한 간결하게 요약하세요:

{content}

요약본에는 가장 중요한 정보, 주요 포인트, 결론을 포함해야 합니다. 불필요한 세부 사항이나 반복적인 내용은 제외하세요."""
        
        # LLM으로 요약 생성
        try:
            summary = generate_text(
                prompt=prompt,
                system_message=self.system_message,
                agent_id=self.agent_id
            )
            
            logger.info(f"Content summarized: {len(content)} -> {len(summary)} chars")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}")
            # 단순 잘라내기로 대체
            return content[:max_length - 3] + "..."
    
    def _format_report(self, report: Dict[str, Any], format_type: str = "markdown") -> str:
        """
        보고서 형식 지정
        
        Args:
            report: 보고서 데이터
            format_type: 형식 유형 (markdown, html, text)
        
        Returns:
            str: 형식이 지정된 보고서
        """
        if format_type == "markdown":
            return self._format_markdown(report)
        elif format_type == "html":
            return self._format_html(report)
        else:  # 기본 텍스트
            return self._format_text(report)
    
    def _format_markdown(self, report: Dict[str, Any]) -> str:
        """마크다운 형식으로 보고서 변환"""
        md = f"# {report.get('title', '제목 없음')}\n\n"
        
        # 경영 요약
        if "executive_summary" in report:
            md += f"## 요약\n\n{report['executive_summary']}\n\n"
        
        # 서론
        if "introduction" in report:
            md += f"## 서론\n\n{report['introduction']}\n\n"
        
        # 섹션
        if "sections" in report:
            for section in report["sections"]:
                md += f"## {section.get('heading', '섹션')}\n\n{section.get('content', '')}\n\n"
        
        # 결론
        if "conclusion" in report:
            md += f"## 결론\n\n{report['conclusion']}\n\n"
        
        # 권장사항
        if "recommendations" in report and report["recommendations"]:
            md += "## 권장사항\n\n"
            for rec in report["recommendations"]:
                md += f"- {rec}\n"
            md += "\n"
        
        # 참조
        if "references" in report and report["references"]:
            md += "## 참조\n\n"
            for ref in report["references"]:
                md += f"- {ref.get('citation', ref.get('source', ''))}\n"
            md += "\n"
        
        # 메타데이터
        if "metadata" in report:
            md += "---\n"
            md += f"생성일: {report['metadata'].get('created_at', '')}\n"
            md += f"보고서 유형: {report['metadata'].get('report_type', '')}\n"
            md += f"쿼리: {report['metadata'].get('query', '')}\n"
        
        return md
    
    def _format_html(self, report: Dict[str, Any]) -> str:
        """HTML 형식으로 보고서 변환"""
        html = f"<!DOCTYPE html>\n<html>\n<head>\n<title>{report.get('title', '제목 없음')}</title>\n"
        html += "<style>\nbody { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }\n"
        html += "h1 { color: #2c3e50; }\nh2 { color: #3498db; }\n"
        html += ".summary { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; }\n"
        html += ".recommendations li { margin-bottom: 10px; }\n"
        html += ".metadata { font-size: 0.8em; color: #7f8c8d; border-top: 1px solid #ddd; margin-top: 30px; padding-top: 10px; }\n"
        html += "</style>\n</head>\n<body>\n"
        
        # 제목
        html += f"<h1>{report.get('title', '제목 없음')}</h1>\n"
        
        # 경영 요약
        if "executive_summary" in report:
            html += f"<div class='summary'>\n<h2>요약</h2>\n<p>{report['executive_summary']}</p>\n</div>\n"
        
        # 서론
        if "introduction" in report:
            html += f"<h2>서론</h2>\n<p>{report['introduction']}</p>\n"
        
        # 섹션
        if "sections" in report:
            for section in report["sections"]:
                html += f"<h2>{section.get('heading', '섹션')}</h2>\n<p>{section.get('content', '')}</p>\n"
        
        # 결론
        if "conclusion" in report:
            html += f"<h2>결론</h2>\n<p>{report['conclusion']}</p>\n"
        
        # 권장사항
        if "recommendations" in report and report["recommendations"]:
            html += "<h2>권장사항</h2>\n<ul class='recommendations'>\n"
            for rec in report["recommendations"]:
                html += f"<li>{rec}</li>\n"
            html += "</ul>\n"
        
        # 참조
        if "references" in report and report["references"]:
            html += "<h2>참조</h2>\n<ul>\n"
            for ref in report["references"]:
                html += f"<li>{ref.get('citation', ref.get('source', ''))}</li>\n"
            html += "</ul>\n"
        
        # 메타데이터
        if "metadata" in report:
            html += "<div class='metadata'>\n"
            html += f"생성일: {report['metadata'].get('created_at', '')}<br>\n"
            html += f"보고서 유형: {report['metadata'].get('report_type', '')}<br>\n"
            html += f"쿼리: {report['metadata'].get('query', '')}\n"
            html += "</div>\n"
        
        html += "</body>\n</html>"
        return html
    
    def _format_text(self, report: Dict[str, Any]) -> str:
        """일반 텍스트 형식으로 보고서 변환"""
        text = f"{report.get('title', '제목 없음').upper()}\n\n"
        
        # 경영 요약
        if "executive_summary" in report:
            text += "요약\n" + "="*50 + "\n"
            text += f"{report['executive_summary']}\n\n"
        
        # 서론
        if "introduction" in report:
            text += "서론\n" + "="*50 + "\n"
            text += f"{report['introduction']}\n\n"
        
        # 섹션
        if "sections" in report:
            for section in report["sections"]:
                text += f"{section.get('heading', '섹션')}\n" + "-"*50 + "\n"
                text += f"{section.get('content', '')}\n\n"
        
        # 결론
        if "conclusion" in report:
            text += "결론\n" + "="*50 + "\n"
            text += f"{report['conclusion']}\n\n"
        
        # 권장사항
        if "recommendations" in report and report["recommendations"]:
            text += "권장사항\n" + "="*50 + "\n"
            for i, rec in enumerate(report["recommendations"], 1):
                text += f"{i}. {rec}\n"
            text += "\n"
        
        # 참조
        if "references" in report and report["references"]:
            text += "참조\n" + "="*50 + "\n"
            for i, ref in enumerate(report["references"], 1):
                text += f"{i}. {ref.get('citation', ref.get('source', ''))}\n"
            text += "\n"
        
        # 메타데이터
        if "metadata" in report:
            text += "="*50 + "\n"
            text += f"생성일: {report['metadata'].get('created_at', '')}\n"
            text += f"보고서 유형: {report['metadata'].get('report_type', '')}\n"
            text += f"쿼리: {report['metadata'].get('query', '')}\n"
        
        return text
    
    def _save_report(self, report: Dict[str, Any], query: str) -> str:
        """
        보고서 저장
        
        Args:
            report: 보고서 데이터
            query: 사용자 쿼리
        
        Returns:
            str: 보고서 ID
        """
        try:
            # 보고서 ID 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_id = f"report_{timestamp}"
            
            # 보고서 파일 이름 생성
            safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
            filename = f"{report_id}_{safe_query}.json"
            filepath = os.path.join(self.reports_dir, filename)
            
            # JSON으로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Report saved: {filepath}")
            
            # 마크다운 버전도 저장
            md_filename = f"{report_id}_{safe_query}.md"
            md_filepath = os.path.join(self.reports_dir, md_filename)
            
            with open(md_filepath, 'w', encoding='utf-8') as f:
                f.write(self._format_markdown(report))
            
            logger.info(f"Markdown report saved: {md_filepath}")
            
            return report_id
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return f"unsaved_{timestamp}"

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
            
            # 보고서 유형 결정
            report_type = self._determine_report_type(query, task_description)
            
            # 보고서 생성
            report = self._generate_report(
                query=query,
                report_type=report_type,
                sources=[],  # 실제로는 다른 에이전트로부터 데이터를 받아야 함
                insights=[]  # 실제로는 다른 에이전트로부터 데이터를 받아야 함
            )
            
            # 보고서 저장 (옵션)
            report_id = self._save_report(report, query)
            
            # 보고서를 마크다운 형식으로 변환
            formatted_report = self._format_report(report, "markdown")
            
            return {
                "status": "success",
                "data": {
                    "report": report,
                    "report_id": report_id,
                    "report_type": report_type,
                    "formatted_report": formatted_report
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing user query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }


# 전역 보고서 작성 에이전트 인스턴스
report_writer_agent = ReportWriterAgent()

def get_report_writer_agent() -> ReportWriterAgent:
    """보고서 작성 에이전트 인스턴스 가져오기"""
    return report_writer_agent
