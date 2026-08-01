import glob
import hashlib
import json
import os
import re
import subprocess
from collections import Counter

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

OUTPUT_PATH = os.environ.get("RAG_OUTPUT_PATH", "processed_chunks.jsonl")
MANIFEST_PATH = os.environ.get("RAG_MANIFEST_PATH", "processing_manifest.jsonl")


# ── 파일 타입별 텍스트 추출 ──────────────────────────────────

def get_markdown_pages(file_path: str) -> list[tuple[int | None, str]]:
    """문서를 페이지 단위 Markdown으로 추출한다.

    PDF는 페이지 번호를 유지해야 답변 근거를 원문 페이지까지 연결할 수 있다.
    DOCX는 페이지 경계가 보장되지 않으므로 ``None``으로 기록한다.

    PDF는 페이지마다 pymupdf4llm.to_markdown()을 개별 호출하는 대신
    page_chunks=True로 한 번에 처리한다. 페이지 수만큼 파일을 반복해서
    새로 열고 파싱하던 구조를 없애 대형 투자설명서에서 체감 속도를 줄인다.
    """
    ext = file_path.lower().rsplit('.', 1)[-1]
    if ext == 'pdf':
        pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        return [
            (page.get("metadata", {}).get("page", index) + 1, page.get("text", ""))
            for index, page in enumerate(pages)
        ]
    elif ext == 'docx':
        result = subprocess.run(
            ['pandoc', '-t', 'markdown', file_path],
            capture_output=True, text=True, check=True
        )
        return [(None, result.stdout)]
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")


# ── 클리닝 함수들 ──────────────────────────────────────────

def strip_html_tags(text: str) -> str:
    """pymupdf4llm이 밑줄/하이라이트를 <u><mark> 태그로 변환하는 문제 해결"""
    text = re.sub(r'</?(?:u|mark|b|i|strong|em)>', '', text)
    # 태그 벗긴 후 홀로 남는 페이지 번호 줄 제거 (예: "36" 한 줄만 있는 경우)
    text = re.sub(r'(?m)^\s*\d{1,4}\s*$', '', text)
    return text


def remove_picture_text_blocks(text: str) -> str:
    """그래프/차트 이미지의 OCR 잔재(축 눈금 텍스트 등) 제거"""
    return re.sub(
        r'<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->',
        '',
        text,
        flags=re.DOTALL
    )


def remove_ocr_artifact_lines(text: str) -> str:
    """이미지/화살표 등 OCR 잔재 라인 제거 (숫자만 반복되거나 기호만 있는 줄)"""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r'\d{5,}', stripped):
            continue
        if stripped and re.fullmatch(r'[➔▶※\'"“”‘’\s]+', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def remove_repeated_boilerplate_lines(full_text: str, min_occurrences: int = 3, max_len: int = 60) -> str:
    """문서 내 반복되는 러닝헤더(펀드명 등) 제거 - 빈도 기반, 특정 펀드명 하드코딩 없음"""
    lines = full_text.split('\n')
    counts = Counter(l.strip() for l in lines if l.strip())
    boilerplate = {l for l, c in counts.items() if c >= min_occurrences and len(l) <= max_len}
    cleaned = [l for l in lines if l.strip() not in boilerplate]
    return '\n'.join(cleaned)


def repeated_boilerplate_lines(full_text: str, min_occurrences: int = 3, max_len: int = 60) -> set[str]:
    """문서 전체에서 반복되는 짧은 러닝헤더 후보를 반환한다."""
    counts = Counter(line.strip() for line in full_text.split('\n') if line.strip())
    return {
        line for line, count in counts.items()
        if count >= min_occurrences and len(line) <= max_len
    }


def remove_lines(text: str, lines_to_remove: set[str]) -> str:
    return '\n'.join(
        line for line in text.split('\n')
        if line.strip() not in lines_to_remove
    )


REFERENCE_DATE_RE = re.compile(
    r'(?:작성기준일|기준일|작성일|시행일)\s*[:：]?\s*'
    r'((?:20)?\d{2}[.년/-]\s*\d{1,2}[.월/-]\s*\d{1,2}(?:일)?)'
)

FUND_CODE_RE = re.compile(r'\bKR\d{10}\b')

def extract_fund_code(file_path: str) -> str | None:
    """source_file 경로에서 표준 펀드코드(예: KR5118201004)를 추출한다.
    폴더/파일명에 코드가 없으면 None. API 서버 쪽 product_kb 연결 시
    이 fund_code를 product_facts.sqlite의 source_file과 매칭하는 키로 쓴다."""
    match = FUND_CODE_RE.search(file_path)
    return match.group(0) if match else None

def extract_reference_date(md_text: str) -> str | None:
    """문서에서 작성·기준일 후보를 추출한다. 확정 파싱은 후속 단계에서 보완한다."""
    match = REFERENCE_DATE_RE.search(md_text[:10000])
    return match.group(1).strip() if match else None


# '#'이 최소 1개 이상 있어야 매칭 (목차 안의 순수 텍스트는 헤더로 오인하지 않도록)
PART_HEADER_RE = re.compile(r'^#+\s*(\[?제\s*\d+\s*부[.\s별첨].*)$')

def normalize_part_headers(md_text: str) -> str:
    """"제N부" 패턴을 강제로 H1(#)으로 정규화 (운용사마다 헤더 레벨이 다른 문제 해결)"""
    lines = md_text.split('\n')
    out = []
    for line in lines:
        m = PART_HEADER_RE.match(line)
        if m:
            out.append(f"# {m.group(1).strip()}")
        else:
            out.append(line)
    return '\n'.join(out)


def strip_markdown_emphasis(text: str) -> str:
    """헤더 텍스트에 남는 **볼드** 마크업 제거"""
    return re.sub(r'\*\*(.+?)\*\*', r'\1', text).strip()


TABLE_LINE = re.compile(r'^\s*\|.*\|\s*$')

def split_tables_and_prose(text: str) -> list:
    """표 블록과 산문 블록 분리 (표는 캐릭터 스플리터로 자르지 않고 통째로 보존)"""
    lines = text.split('\n')
    blocks = []
    buffer, buffer_type = [], None
    for line in lines:
        line_type = "table" if TABLE_LINE.match(line) else "prose"
        if buffer_type is None:
            buffer_type = line_type
        if line_type != buffer_type:
            blocks.append({"type": buffer_type, "content": "\n".join(buffer)})
            buffer = []
            buffer_type = line_type
        buffer.append(line)
    if buffer:
        blocks.append({"type": buffer_type, "content": "\n".join(buffer)})
    return blocks


def looks_like_fake_table(text: str) -> bool:
    """구분선(---) 없이 데이터 행이 1~2개뿐인 경우 표가 아니라 리스트일 가능성이 높음"""
    lines = [l for l in text.split('\n') if l.strip()]
    return len(lines) <= 2


def is_meaningless_chunk(text: str, min_len: int = 15) -> bool:
    """날짜/기호만 있는 등 실질적 정보가 없는 짧은 청크 판별"""
    stripped = re.sub(r'[\d\.\-/\s]+', '', text)
    return len(stripped) < 3 or len(text.strip()) < min_len


def normalized_content_hash(text: str) -> str:
    """서식 차이를 제거한 본문 해시. 문서 간 보일러플레이트 비교에 사용한다."""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_global_boilerplate_candidate(chunk: dict) -> bool:
    """표·짧은 답변은 보존하고, 반복되는 투자설명서 산문만 전역 중복 제거한다."""
    metadata = chunk["metadata"]
    return (
        metadata.get("doc_type") == "product_prospectus"
        and metadata.get("block_type") == "prose"
        and len(chunk["page_content"]) >= 80
    )


# ── 문서 타입 분류 ──────────────────────────────────────────

PROSPECTUS_KEYWORDS = ["투자설명서", "집합투자기구", "위험등급", "집합투자증권", "판매수수료"]

def classify_doc(md_text: str) -> str:
    head = md_text[:3000]
    hit_count = sum(1 for kw in PROSPECTUS_KEYWORDS if kw in head)
    return "product_prospectus" if hit_count >= 2 else "institutional_guide"


# ── 파이프라인 본체 ──────────────────────────────────────────

pdf_files = glob.glob("**/*.[pP][dD][fF]", recursive=True)
docx_files = glob.glob("**/*.docx", recursive=True)
all_files = sorted(pdf_files + docx_files)

done_files = set()
seen_boilerplate_hashes = {}
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            existing_chunk = json.loads(line)
            done_files.add(existing_chunk["metadata"]["source_file"])
            if is_global_boilerplate_candidate(existing_chunk):
                seen_boilerplate_hashes.setdefault(
                    normalized_content_hash(existing_chunk["page_content"]),
                    existing_chunk["metadata"]["source_file"],
                )

all_files = [p for p in all_files if p not in done_files]
print(f"총 {len(all_files)}개 파일 처리 예정 (이미 완료: {len(done_files)}개 스킵)\n")

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "대분류"), ("##", "중분류"), ("###", "소분류")]
)
char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f, open(
    MANIFEST_PATH, "a", encoding="utf-8"
) as manifest_f:
    for file_path in all_files:
        print(f"처리 중인 파일: {file_path}")
        try:
            raw_pages = get_markdown_pages(file_path)
            raw_document_text = "\n".join(text for _, text in raw_pages)
            if not raw_document_text or not raw_document_text.strip():
                print("   경고: 텍스트 추출 실패, 건너뜀")
                manifest_f.write(json.dumps({
                    "source_file": file_path, "status": "skipped_empty",
                    "pages_total": len(raw_pages), "chunks_generated": 0,
                }, ensure_ascii=False) + "\n")
                continue

            if is_junk_extraction(raw_document_text):
                print("   경고: 추출된 내용이 거의 없음 (스캔본/이미지 PDF 의심, OCR 필요) — 건너뜀")
                manifest_f.write(json.dumps({
                    "source_file": file_path, "status": "skipped_needs_ocr",
                    "pages_total": len(raw_pages), "chunks_generated": 0,
                }, ensure_ascii=False) + "\n")
                continue

document_text = normalize_part_headers(
                remove_ocr_artifact_lines(
                    remove_picture_text_blocks(strip_html_tags(raw_document_text))
                )
            )
            doc_type = classify_doc(document_text)
            fund_code = extract_fund_code(file_path)
            # 문서 앞부분(첫 10000자)에서 찾은 기준일을 기본값으로 삼되,
            # 페이지를 넘어가며 새 기준일이 나오면 그 시점부터 최신 값으로 갱신한다.
            # 멀티펀드 통합 투자설명서에서 섹션(펀드/클래스)마다 작성기준일이
            # 다시 명시되는 경우를 반영하기 위함이다.
            current_reference_date = extract_reference_date(document_text)
            boilerplate = repeated_boilerplate_lines(document_text)
            file_chunks = []
            for page_number, raw_page_text in raw_pages:
                page_text = normalize_part_headers(
                    remove_lines(
                        remove_ocr_artifact_lines(
                            remove_picture_text_blocks(strip_html_tags(raw_page_text))
                        ),
                        boilerplate,
                    )
                )
                if not page_text.strip() or is_junk_extraction(page_text):
                    continue

                page_reference_date = extract_reference_date(page_text)
                if page_reference_date:
                    current_reference_date = page_reference_date

                for header_chunk in markdown_splitter.split_text(page_text):
                    blocks = split_tables_and_prose(header_chunk.page_content)
                    prose_buffer = []
                    for block in blocks:
                        if block["type"] == "table" and not looks_like_fake_table(block["content"]):
                            if block["content"].strip():
                                file_chunks.append({
                                    "page_content": block["content"].strip(),
                                    "metadata": {
                                        **header_chunk.metadata,
                                        "block_type": "table",
                                        "page_number": page_number,
                                        "작성기준일": current_reference_date,
                                    },
                                })
                        else:
                            prose_buffer.append(block["content"])

                    prose_text = "\n".join(prose_buffer).strip()
                    if prose_text:
                        for sub in char_splitter.split_text(prose_text):
                            if sub.strip():
                                file_chunks.append({
                                    "page_content": sub.strip(),
                                    "metadata": {
                                        **header_chunk.metadata,
                                        "block_type": "prose",
                                        "page_number": page_number,
                                        "작성기준일": current_reference_date,
                                    },
                                })

            for chunk in file_chunks:
                chunk["metadata"]["source_file"] = file_path
                chunk["metadata"]["doc_type"] = doc_type
                chunk["metadata"]["category"] = (
                    "상품" if doc_type == "product_prospectus" else "제도"
                )
                chunk["metadata"]["fund_code"] = fund_code

                # 메타데이터의 헤더 값에서 ** 마크업 제거
                for k in ("대분류", "중분류", "소분류"):
                    if k in chunk["metadata"]:
                        chunk["metadata"][k] = strip_markdown_emphasis(chunk["metadata"][k])

                section_path = " > ".join(
                    v for k in ("대분류", "중분류", "소분류") if (v := chunk["metadata"].get(k))
                )
                if section_path and chunk["metadata"]["block_type"] == "prose":
                    chunk["page_content"] = f"[{section_path}]\n{chunk['page_content']}"

            # 빈 청크 / 의미 없는 청크(날짜만 있는 줄 등) 필터링
            file_chunks = [
                c for c in file_chunks
                if c["page_content"].strip() and not is_meaningless_chunk(c["page_content"])
            ]

            kept_chunks = []
            duplicate_chunks_removed = 0
            for chunk in file_chunks:
                if is_global_boilerplate_candidate(chunk):
                    content_hash = normalized_content_hash(chunk["page_content"])
                    first_source = seen_boilerplate_hashes.get(content_hash)
                    if first_source and first_source != file_path:
                        duplicate_chunks_removed += 1
                        continue
                    seen_boilerplate_hashes.setdefault(content_hash, file_path)
                kept_chunks.append(chunk)

            for chunk in kept_chunks:
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            out_f.flush()
            manifest_f.write(json.dumps({
                "source_file": file_path,
                "status": "success",
                "doc_type": doc_type,
                "pages_total": len(raw_pages),
                "pages_with_text": len({c["metadata"]["page_number"] for c in kept_chunks}),
                "chunks_generated": len(kept_chunks),
                "duplicate_chunks_removed": duplicate_chunks_removed,
            }, ensure_ascii=False) + "\n")
            manifest_f.flush()

            n_table = sum(1 for c in kept_chunks if c['metadata']['block_type'] == 'table')
            n_prose = sum(1 for c in kept_chunks if c['metadata']['block_type'] == 'prose')
            print(f"   성공: {len(kept_chunks)}개 덩어리 생성됨 ({doc_type} / 표 {n_table}개 / 산문 {n_prose}개 / 전역 중복 제거 {duplicate_chunks_removed}개)")

        except Exception as e:
            print(f"   오류: 파싱 에러로 해당 파일을 스킵합니다 ({file_path}): {e}")
            manifest_f.write(json.dumps({
                "source_file": file_path, "status": "error", "error": str(e),
            }, ensure_ascii=False) + "\n")
            manifest_f.flush()
            continue

print(f"\n완료: '{OUTPUT_PATH}'에 저장되었습니다.")
