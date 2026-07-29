# Pension RAG data preprocessing

PDF와 DOCX 원문을 RAG용 JSONL 청크로 변환하는 전처리 스크립트입니다.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

`pandoc`는 DOCX 입력을 처리할 때만 필요합니다.

## Run

원문 PDF/DOCX를 이 저장소 아래에 둔 뒤 실행합니다.

```bash
python pdf_test.py
```

출력은 `processed_chunks.jsonl`입니다. 각 청크는 본문과 함께 `source_file`, `page_number`, `doc_type`, `category`, `작성기준일`, 헤더 계층, `block_type` 메타데이터를 가집니다.

기존 출력 파일이 있으면 이미 처리된 `source_file`은 건너뜁니다. 스키마를 변경한 경우에는 기존 JSONL을 삭제하거나 다른 출력 경로를 사용한 뒤 전체 재생성해야 합니다.

기존 결과를 보존하며 새 스키마를 검증하려면 환경 변수로 출력 경로를 지정할 수 있습니다.

```powershell
$env:RAG_OUTPUT_PATH = "processed_chunks_v2.jsonl"
$env:RAG_MANIFEST_PATH = "processing_manifest_v2.jsonl"
python pdf_test.py
```

## Product fact extraction

투자설명서 표에서 클래스별 총보수와 기간 수익률을 정형 SQLite로 추출합니다.

```bash
python extract_product_facts.py --input processed_chunks.jsonl --output product_facts.sqlite
```

`product_facts` 테이블은 원본 표와 파일·페이지 근거를 함께 저장합니다. `confidence=needs_review` 행은 숫자가 아닌 값이거나 후속 규칙 보완이 필요한 후보입니다.

## Quality validation

전처리 후에는 JSONL 형식, 필수 메타데이터, 장문 청크 수와 문서별 처리 상태를 확인합니다.

```bash
python validate_chunks.py --input processed_chunks.jsonl --manifest processing_manifest.jsonl
```

파이프라인은 `processing_manifest.jsonl`에 문서별 성공·OCR 필요·오류 상태와 생성 청크 수를 기록합니다. 투자설명서의 반복 산문 청크는 문서 간 정확 일치 시 한 번만 보존합니다. 표는 정형 데이터 근거이므로 이 전역 중복 제거 대상에서 제외합니다.
