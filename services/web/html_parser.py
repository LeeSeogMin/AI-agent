#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 기반 멀티 에이전트 시스템의 HTML 파서 모듈
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
from bs4 import BeautifulSoup
import justext
import trafilatura

# 로거 설정
logger = logging.getLogger(__name__)

def extract_text_from_html(
    html_content: str,
    method: str = "trafilatura",
    min_text_length: int = 20,
    include_links: bool = True,
    include_images: bool = False,
    include_tables: bool = True
) -> str:
    """
    HTML에서 텍스트 추출
    
    Args:
        html_content: HTML 내용
        method: 추출 방법 (trafilatura, justext, beautifulsoup)
        min_text_length: 최소 텍스트 길이
        include_links: 링크 포함 여부
        include_images: 이미지 설명 포함 여부
        include_tables: 테이블 내용 포함 여부
    
    Returns:
        str: 추출된 텍스트
    """
    if not html_content:
        return ""
    
    try:
        if method == "trafilatura":
            return extract_with_trafilatura(html_content, include_links, min_text_length)
        elif method == "justext":
            return extract_with_justext(html_content, min_text_length)
        elif method == "beautifulsoup":
            return extract_with_beautifulsoup(
                html_content, min_text_length, include_links, include_images, include_tables
            )
        else:
            # 기본값으로 trafilatura 사용
            text = extract_with_trafilatura(html_content, include_links, min_text_length)
            
            # 실패하면 BeautifulSoup으로 백업
            if not text:
                text = extract_with_beautifulsoup(
                    html_content, min_text_length, include_links, include_images, include_tables
                )
            
            return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        # 오류 시 BeautifulSoup으로 백업
        try:
            return extract_with_beautifulsoup(
                html_content, min_text_length, include_links, include_images, include_tables
            )
        except Exception as e2:
            logger.error(f"Backup extraction also failed: {str(e2)}")
            return ""


def extract_with_trafilatura(
    html_content: str,
    include_links: bool = True,
    min_text_length: int = 20
) -> str:
    """
    Trafilatura 라이브러리를 사용하여 HTML에서 텍스트 추출
    
    Args:
        html_content: HTML 내용
        include_links: 링크 포함 여부
        min_text_length: 최소 텍스트 길이
    
    Returns:
        str: 추출된 텍스트
    """
    extracted_text = trafilatura.extract(
        html_content,
        include_links=include_links,
        include_images=False,
        include_tables=True,
        min_extracted_size=min_text_length
    )
    
    return extracted_text or ""


def extract_with_justext(
    html_content: str,
    min_text_length: int = 20
) -> str:
    """
    Justext 라이브러리를 사용하여 HTML에서 텍스트 추출
    
    Args:
        html_content: HTML 내용
        min_text_length: 최소 텍스트 길이
    
    Returns:
        str: 추출된 텍스트
    """
    try:
        paragraphs = justext.justext(html_content, justext.get_stoplist("English"))
        
        text_parts = []
        for paragraph in paragraphs:
            if not paragraph.is_boilerplate:
                if len(paragraph.text) >= min_text_length:
                    text_parts.append(paragraph.text)
        
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error in justext extraction: {str(e)}")
        return ""


def extract_with_beautifulsoup(
    html_content: str,
    min_text_length: int = 20,
    include_links: bool = True,
    include_images: bool = False,
    include_tables: bool = True
) -> str:
    """
    BeautifulSoup을 사용하여 HTML에서 텍스트 추출
    
    Args:
        html_content: HTML 내용
        min_text_length: 최소 텍스트 길이
        include_links: 링크 포함 여부
        include_images: 이미지 설명 포함 여부
        include_tables: 테이블 내용 포함 여부
    
    Returns:
        str: 추출된 텍스트
    """
    # HTML 파싱
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 불필요한 요소 제거
    for script in soup(["script", "style", "nav", "footer", "iframe"]):
        script.extract()
    
    # 링크 처리
    if not include_links:
        for a in soup.find_all('a'):
            a.replace_with(a.get_text())
    
    # 이미지 처리
    if include_images:
        for img in soup.find_all('img'):
            alt_text = img.get('alt')
            if alt_text:
                img.replace_with(f"[Image: {alt_text}]")
            else:
                img.extract()
    else:
        for img in soup.find_all('img'):
            img.extract()
    
    # 테이블 처리
    if not include_tables:
        for table in soup.find_all('table'):
            table.extract()
    
    # 텍스트 추출
    text = soup.get_text(separator=' ')
    
    # 공백 정리
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # 연속적인 여러 줄바꿈을 하나로 치환
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def extract_metadata_from_html(html_content: str) -> Dict[str, str]:
    """
    HTML에서 메타데이터 추출
    
    Args:
        html_content: HTML 내용
    
    Returns:
        Dict[str, str]: 메타데이터 (제목, 설명, 이미지 URL 등)
    """
    metadata = {
        "title": "",
        "description": "",
        "image": "",
        "author": "",
        "published_date": "",
        "language": ""
    }
    
    if not html_content:
        return metadata
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 제목 추출
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        
        # Open Graph 메타데이터
        og_title = soup.find('meta', property='og:title')
        if og_title:
            metadata["title"] = og_title.get('content', '').strip()
        
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            metadata["description"] = og_desc.get('content', '').strip()
        
        og_image = soup.find('meta', property='og:image')
        if og_image:
            metadata["image"] = og_image.get('content', '').strip()
        
        # 기본 메타 설명
        if not metadata["description"]:
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata["description"] = meta_desc.get('content', '').strip()
        
        # 저자 추출
        author = soup.find('meta', attrs={'name': 'author'})
        if author:
            metadata["author"] = author.get('content', '').strip()
        
        # 언어 추출
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata["language"] = html_tag.get('lang').strip()
        
        # 발행일 추출
        published = soup.find('meta', property='article:published_time')
        if published:
            metadata["published_date"] = published.get('content', '').strip()
        
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from HTML: {str(e)}")
        return metadata


def extract_links_from_html(html_content: str, base_url: Optional[str] = None) -> List[Dict[str, str]]:
    """
    HTML에서 링크 추출
    
    Args:
        html_content: HTML 내용
        base_url: 기본 URL (상대 경로 변환용)
    
    Returns:
        List[Dict[str, str]]: 추출된 링크 목록
    """
    links = []
    
    if not html_content:
        return links
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for a in soup.find_all('a', href=True):
            link = a['href'].strip()
            text = a.get_text().strip()
            
            # 상대 경로 처리
            if base_url and not link.startswith(('http://', 'https://')):
                if not base_url.endswith('/'):
                    base_url += '/'
                link = base_url + link.lstrip('/')
            
            if link and text:
                links.append({
                    "url": link,
                    "text": text
                })
        
        return links
    except Exception as e:
        logger.error(f"Error extracting links from HTML: {str(e)}")
        return links 