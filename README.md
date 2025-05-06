# LangGraph 기반 멀티 에이전트 시스템

LangGraph를 활용한 고급 멀티 에이전트 시스템으로, 복잡한 쿼리를 여러 전문 에이전트의 협업을 통해 효율적으로 처리합니다.

## 프로젝트 개요

본 시스템은 6개의 특화된 에이전트로 구성되어 있습니다:

- **감독 에이전트**: 쿼리 분석, 작업 할당, 전체 프로세스 조정
- **RAG 에이전트**: 벡터 DB 기반 지식 검색 및 통합
- **웹 검색 에이전트**: 실시간 웹 정보 검색 및 수집
- **데이터 분석 에이전트**: 통계 분석, 시각화, 인사이트 도출
- **심리분석 에이전트**: 감정 분석, 심리 패턴 파악, 공감적 응답
- **보고서 작성 에이전트**: 결과 종합, 일관성 있는 보고서 생성

이 시스템은 계층적 구조로 설계되어 감독 계층에서 정보 검색, 처리, 출력 계층으로 이어지는 워크플로를 제공합니다.

## 주요 기능

- 복잡한 쿼리의 자동 분해와 최적 에이전트 조합 활용
- 벡터 데이터베이스와 웹 검색을 통합한 지식 접근
- 데이터 기반 분석 및 시각화
- 감정 및 심리적 측면 인식
- 다양한 소스의 정보를 통합한 일관성 있는 보고서 생성
- JSON 기반 표준화된 에이전트 간 통신

## 설치 방법

### 요구사항

- Python 3.9+
- OpenAI API 키 (GPT-4o 접근 필요)
- Google Custom Search API 키 (웹 검색 기능용)

### 설치 과정

1. 저장소 클론:
```bash
git clone https://github.com/your-username/multi-agent-system.git
cd multi-agent-system
```

2. 가상 환경 생성 및 활성화:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 의존성 패키지 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 API 키와 설정 추가
```

## 사용법

### 기본 실행

```bash
python main.py --query "분석하려는 쿼리나 질문"
```

### API 서버 실행

```bash
uvicorn api.endpoints:app --reload
```

그 후 `http://localhost:8000/docs`에서 API 문서를 확인할 수 있습니다.

### 고급 옵션

```bash
python main.py --query "복잡한 분석 쿼리" --output-format markdown --verbose
```

## 프로젝트 구조

```
multi-agent-system/
├── agents/                     # 각 전문 에이전트 구현
├── core/                       # 코어 시스템 구성요소
├── services/                   # 외부 서비스 연동
├── knowledge/                  # 지식 처리 관련
├── data_processing/            # 데이터 처리 모듈
├── workflow/                   # 워크플로 및 작업 관리
├── api/                        # API 인터페이스
└── ...
```

## 라이센스

MIT

## 기여하기

기여는 언제나 환영합니다! 자세한 내용은 `CONTRIBUTING.md`를 참조하세요.
