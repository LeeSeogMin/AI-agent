# LangGraph 기반 멀티 에이전트 시스템 아키텍처

## 1. 시스템 아키텍처 개요

LangGraph 기반 멀티 에이전트 시스템은 6개의 특화된 에이전트를 통합하여 복잡하고 다면적인 사용자 쿼리를 효율적으로 처리하는 아키텍처입니다. 시스템은 계층적 구조로 설계되어 감독 계층에서 정보 검색, 처리, 출력 계층으로 이어지는 워크플로를 제공합니다.

### 시스템 계층 구조

```
+------------------------+
|    감독 계층 (Supervision Layer)    |
|    - 감독 에이전트    |
|    - 작업 관리자      |
|    - 상태 관리자      |
+------------------------+
            |
+------------------------+
| 정보 검색 계층 (Information Retrieval) |
|    - RAG 에이전트     |
|    - 웹 검색 에이전트   |
|    - 벡터 저장소      |
+------------------------+
            |
+------------------------+
|    처리 계층 (Processing Layer)    |
|    - 데이터 분석 에이전트 |
|    - 심리분석 에이전트  |
|    - 분석 파이프라인   |
+------------------------+
            |
+------------------------+
|    출력 계층 (Output Layer)    |
|    - 보고서 작성 에이전트 |
|    - 포맷 변환기      |
|    - 응답 검증기      |
+------------------------+
```

## 2. 에이전트 역할 및 인터페이스

### 에이전트별 역할과 책임

| 에이전트 | 역할 | 주요 책임 | 접근법 |
|----------|------|-----------|--------|
| **감독 에이전트** | 전체 시스템 조정자 | 쿼리 분석, 작업 분해, 에이전트 할당, 결과 검증 | 프롬프트 중심 + 최소 코드 통합 |
| **RAG 에이전트** | 지식 검색 전문가 | 벡터 DB 검색, 관련성 평가, 정보 통합 | 코드 중심 + 보조적 프롬프트 |
| **웹 검색 에이전트** | 웹 정보 탐색 전문가 | 검색 쿼리 생성, 웹 크롤링, 자료 다운로드 | 코드 중심 + 보조적 프롬프트 |
| **데이터 분석 에이전트** | 데이터 처리 전문가 | 통계 분석, 시각화, 인사이트 도출 | 하이브리드 (코드 + 프롬프트) |
| **심리분석 에이전트** | 심리 분석 전문가 | 감정 분석, 심리 패턴 파악, 공감적 응답 | 프롬프트 중심 |
| **보고서 작성 에이전트** | 정보 통합 전문가 | 결과 종합, 보고서 구성, 포맷팅 | 프롬프트 중심 |

### 에이전트 인터페이스 규격

| 인터페이스 유형 | 형식 | 주요 구성 요소 | 사용 에이전트 |
|----------------|------|---------------|-------------|
| **작업 요청** | JSON | message_id, timestamp, sender, receiver, action, parameters | 감독 → 전문 에이전트 |
| **작업 응답** | JSON | message_id, timestamp, sender, receiver, status, result, metadata | 전문 → 감독 에이전트 |
| **정보 요청** | JSON | message_id, timestamp, sender, receiver, query_type, query | 에이전트 간 |
| **상태 업데이트** | JSON | message_id, timestamp, sender, status, progress, next_steps | 모든 에이전트 |
| **오류 알림** | JSON | message_id, timestamp, sender, error_type, description, severity | 모든 에이전트 |
| **작업 완료** | JSON | message_id, timestamp, sender, task_id, output, confidence | 모든 에이전트 |

### 모듈별 기능 및 책임

| 모듈 | 주요 기능 | 종속성 | 관련 에이전트 |
|------|----------|--------|-------------|
| **GraphManager** | LangGraph 워크플로 관리, 에이전트 전이 | LangGraph | 감독 에이전트 |
| **StateManager** | 중앙 상태 저장소, 컨텍스트 관리 | - | 모든 에이전트 |
| **Communication** | 메시지 교환, 프로토콜 관리 | - | 모든 에이전트 |
| **VectorDBService** | 벡터 DB 연동, 임베딩 관리 | Chroma, OpenAI | RAG 에이전트 |
| **WebService** | 웹 검색 API 연동, 크롤링 | Google CSE, arXiv | 웹 검색 에이전트 |
| **LLMService** | LLM API 호출 관리, 최적화 | OpenAI | 모든 에이전트 |
| **DataProcessing** | 데이터 처리, 통계 분석 | pandas, numpy | 데이터 분석 에이전트 |
| **Visualization** | 차트 및 그래프 생성 | matplotlib, plotly | 데이터 분석 에이전트 |
| **DocumentProcessor** | 문서 청킹, 파싱, 메타데이터 | - | RAG, 웹 검색 에이전트 |

## 3. 데이터 흐름

### 표준 작업 흐름

1. 사용자 쿼리 → 감독 에이전트(쿼리 분석 및 작업 계획)
2. 감독 에이전트 → 필요한 전문 에이전트(작업 할당)
3. 전문 에이전트 → 관련 서비스/모듈(기능 실행)
4. 전문 에이전트 → 상태 관리자(중간 결과 저장)
5. 상태 관리자 → 다른 에이전트(필요시 중간 결과 공유)
6. 전문 에이전트 → 감독 에이전트(작업 완료 보고)
7. 감독 에이전트 → 보고서 작성 에이전트(최종 응답 생성)
8. 보고서 작성 에이전트 → 사용자(최종 결과 전달)

### 주요 데이터 구조

#### 공유 컨텍스트 구조
```json
{
  "session_id": "unique_session_identifier",
  "user_query": "original user query",
  "parsed_intent": "interpreted user intent",
  "current_stage": "planning|information_gathering|analysis|synthesis|reporting",
  "active_agents": ["list of currently active agents"],
  "task_status": {
    "task_id": {
      "agent": "assigned_agent",
      "status": "pending|in_progress|completed|failed",
      "progress": 0.75,
      "result_location": "reference to result"
    }
  },
  "shared_knowledge": {
    "key": "value",
    "collected_information": [],
    "analysis_results": [],
    "decisions": []
  },
  "conversation_history": [
    {
      "timestamp": "ISO-8601 timestamp",
      "agent": "agent_id or user",
      "message": "message content"
    }
  ]
}
```

#### 메시지 구조
```json
{
  "message_id": "unique_id",
  "timestamp": "ISO-8601 timestamp",
  "sender": "agent_id",
  "receiver": "agent_id or 'broadcast'",
  "message_type": "request|response|notification|error",
  "priority": "high|normal|low",
  "content": {
    "action": "action_type",
    "parameters": {},
    "data": {},
    "metadata": {}
  },
  "references": ["related_message_ids"],
  "status": "pending|processing|completed|failed"
}
```

## 4. 프로젝트 파일 시스템 구조

```
multi-agent-system/
├── agents/                     # 각 전문 에이전트 구현
│   ├── base.py                 # 기본 에이전트 클래스
│   ├── supervisor.py           # 감독 에이전트
│   ├── rag_agent.py            # RAG 에이전트
│   ├── web_search_agent.py     # 웹 검색 에이전트
│   ├── data_analysis_agent.py  # 데이터 분석 에이전트
│   ├── psychological_agent.py  # 심리분석 에이전트
│   ├── report_writer_agent.py  # 보고서 작성 에이전트
│   └── prompts/                # 에이전트별 프롬프트 템플릿
│       ├── supervisor/
│       │   ├── system_message.txt
│       │   └── templates.py
│       ├── rag/
│       │   ├── system_message.txt
│       │   └── templates.py
│       ├── web_search/
│       │   ├── system_message.txt
│       │   └── templates.py
│       ├── data_analysis/
│       │   ├── system_message.txt
│       │   └── templates.py
│       ├── psychological/
│       │   ├── system_message.txt
│       │   └── templates.py
│       └── report_writer/
│           ├── system_message.txt
│           └── templates.py
├── core/                       # 코어 시스템 구성요소
│   ├── config.py               # 시스템 구성 및 환경 설정
│   ├── logging_manager.py      # 로깅 시스템
│   ├── state_manager.py        # 공유 상태 관리
│   ├── communication.py        # 에이전트 간 통신 프로토콜
│   ├── event_bus.py            # 이벤트 기반 통신 시스템
│   └── error_handler.py        # 예외 처리 및 복구 메커니즘
├── services/                   # 외부 서비스 연동
│   ├── llm_service.py          # LLM API 서비스
│   ├── vectordb/              # 벡터 데이터베이스 관련
│   │   ├── chroma_client.py    # Chroma DB 클라이언트
│   │   ├── embedding.py        # 임베딩 생성 및 관리
│   │   └── index_manager.py    # 인덱스 관리
│   └── web/                   # 웹 서비스 관련
│       ├── google_search.py    # Google 검색 API 래퍼
│       ├── arxiv_search.py     # arXiv API 래퍼
│       └── search_router.py    # 검색 라우팅 및 통합
├── knowledge/                  # 지식 처리 관련
│   └── document_processor/    # 문서 처리 관련
│       ├── chunker.py          # 문서 청킹 시스템
│       ├── parser.py           # 다양한 형식 파서
│       └── metadata_extractor.py # 메타데이터 추출기
├── data_processing/            # 데이터 처리 모듈
│   ├── analysis/              # 분석 관련
│   │   ├── statistics.py       # 통계 분석 도구
│   │   ├── timeseries.py       # 시계열 분석
│   │   └── correlation.py      # 상관관계 분석
│   ├── visualization/         # 시각화 관련
│   │   ├── chart_generator.py  # 차트 생성기
│   │   └── plot_templates.py   # 시각화 템플릿
│   └── formatters/            # 출력 포맷 관련
│       ├── report_formatter.py # 보고서 포맷팅
│       └── data_exporter.py    # 데이터 내보내기
├── workflow/                   # 워크플로 및 작업 관리
│   ├── graph_definitions.py    # LangGraph 워크플로 정의
│   ├── router.py               # 쿼리-작업 라우팅 시스템
│   └── executor.py             # 작업 실행 관리자
├── api/                        # API 인터페이스
│   ├── endpoints.py            # API 엔드포인트
│   ├── schemas.py              # 요청/응답 스키마
│   └── middleware.py           # API 미들웨어
├── tests/                      # 테스트 코드
│   ├── unit/                  # 단위 테스트
│   │   ├── test_agents.py      # 에이전트 테스트
│   │   ├── test_services.py    # 서비스 테스트
│   │   └── test_core.py        # 코어 모듈 테스트
│   ├── integration/           # 통합 테스트
│   │   ├── test_workflows.py   # 워크플로 테스트
│   │   └── test_end_to_end.py  # 엔드투엔드 테스트
│   └── fixtures/              # 테스트 픽스처
│       ├── sample_queries.json # 샘플 쿼리 데이터
│       └── mock_responses.json # 목업 응답 데이터
├── config/                     # 설정 파일
│   ├── default.yaml            # 기본 설정
│   ├── development.yaml        # 개발 환경 설정
│   └── production.yaml         # 프로덕션 환경 설정
├── data/                       # 데이터 저장소
│   ├── knowledge_base/        # 지식 베이스 데이터
│   ├── user_uploads/          # 사용자 업로드 파일
│   └── output/                # 생성된 결과물
├── scripts/                    # 유틸리티 스크립트
│   ├── setup.py                # 초기 설정 스크립트
│   ├── update_kb.py            # 지식 베이스 업데이트
│   └── deploy.py               # 배포 스크립트
├── docs/                       # 프로젝트 문서
│   ├── architecture.md         # 아키텍처 문서
│   ├── api.md                  # API 문서
│   └── user_guide.md           # 사용자 가이드
├── .env.example                # 환경 변수 예시
├── .gitignore                  # Git 제외 파일
├── main.py                     # 시스템 진입점
├── README.md                   # 프로젝트 개요
└── requirements.txt            # 의존성 패키지
```

## 5. 기술 스택

- **언어 및 코어 프레임워크**: Python, LangGraph
- **LLM 서비스**: OpenAI GPT-4o
- **벡터 데이터베이스**: Chroma (대안: FAISS, Pinecone)
- **임베딩 모델**: OpenAI text-embedding-3-large
- **웹 검색 API**: Google Custom Search API, arXiv API
- **데이터 처리**: pandas, numpy, matplotlib, plotly
- **웹 크롤링**: requests, beautifulsoup4, playwright
- **API 프레임워크**: FastAPI
- **데이터 검증**: Pydantic
- **테스트 프레임워크**: pytest

## 6. 구현 우선순위 및 로드맵

1. **단계 1: 코어 시스템 구축** (2주)
   - 기본 아키텍처 구현
   - 상태 관리 및 통신 프로토콜 개발
   - LangGraph 기반 워크플로 설정

2. **단계 2: 정보 검색 계층 구현** (3주)
   - RAG 에이전트 및 벡터 DB 연동
   - 웹 검색 에이전트 개발
   - 문서 처리 시스템 구현

3. **단계 3: 처리 및 출력 계층 구현** (3주)
   - 데이터 분석 에이전트 개발
   - 심리분석 에이전트 개발
   - 보고서 작성 에이전트 개발

4. **단계 4: 통합 및 최적화** (2주)
   - 전체 시스템 통합
   - 성능 최적화
   - 오류 처리 강화

5. **단계 5: API 및 문서화** (2주)
   - REST API 개발
   - 문서화 및 사용자 가이드 작성
   - 배포 준비 