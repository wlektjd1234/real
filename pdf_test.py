import glob
import json
import os
import re
import subprocess
from collections import Counter

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

OUTPUT_PATH = "processed_chunks.jsonl"


# ── 파일 타입별 텍스트 추출 ──────────────────────────────────

def get_markdown_text(file_path: str) -> str:
    """파일 확장자에 따라 적절한 방법으로 마크다운 텍스트 추출"""
    ext = file_path.lower().rsplit('.', 1)[-1]
    if ext == 'pdf':
        return pymupdf4llm.to_markdown(file_path)
    elif ext == 'docx':
        result = subprocess.run(
            ['pandoc', '-t', 'markdown', file_path],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")


def is_junk_extraction(md_text: str, min_meaningful_chars: int = 50) -> bool:
    """텍스트 추출이 사실상 실패했는지 판별 (스캔본/이미지 PDF 의심 - 특수문자/공백만 있는 경우)"""
    meaningful = re.sub(r'[\s\|\-∙·:*#\[\]<>br/]+', '', md_text)
    return len(meaningful) < min_meaningful_chars


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
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            done_files.add(json.loads(line)["metadata"]["source_file"])

all_files = [p for p in all_files if p not in done_files]
print(f"총 {len(all_files)}개 파일 처리 예정 (이미 완료: {len(done_files)}개 스킵)\n")

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "대분류"), ("##", "중분류"), ("###", "소분류")]
)
char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
    for file_path in all_files:
        print(f"📄 처리 중인 파일: {file_path}")
        try:
            md_text = get_markdown_text(file_path)
            if not md_text or not md_text.strip():
                print(f"   ⚠️ 텍스트 추출 실패, 건너뜀")
                continue

            if is_junk_extraction(md_text):
                print(f"   ⚠️ 추출된 내용이 거의 없음 (스캔본/이미지 PDF 의심, OCR 필요) — 건너뜀")
                continue

            # 클리닝 단계 (분할 전, 문서 전체 단위로 수행)
            md_text = strip_html_tags(md_text)
            md_text = remove_picture_text_blocks(md_text)
            md_text = remove_ocr_artifact_lines(md_text)
            md_text = remove_repeated_boilerplate_lines(md_text)
            md_text = normalize_part_headers(md_text)

            doc_type = classify_doc(md_text)

            header_chunks = markdown_splitter.split_text(md_text)

            file_chunks = []
            for header_chunk in header_chunks:
                blocks = split_tables_and_prose(header_chunk.page_content)

                prose_buffer = []
                for block in blocks:
                    if block["type"] == "table" and not looks_like_fake_table(block["content"]):
                        if block["content"].strip():
                            file_chunks.append({
                                "page_content": block["content"].strip(),
                                "metadata": {**header_chunk.metadata, "block_type": "table"},
                            })
                    else:
                        # 표가 아니거나(fake table) 산문이면 산문 버퍼로
                        prose_buffer.append(block["content"])

                prose_text = "\n".join(prose_buffer).strip()
                if prose_text:
                    sub_chunks = char_splitter.split_text(prose_text)
                    for sub in sub_chunks:
                        if sub.strip():
                            file_chunks.append({
                                "page_content": sub.strip(),
                                "metadata": {**header_chunk.metadata, "block_type": "prose"},
                            })

            for chunk in file_chunks:
                chunk["metadata"]["source_file"] = file_path
                chunk["metadata"]["doc_type"] = doc_type

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

            for chunk in file_chunks:
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            out_f.flush()

            n_table = sum(1 for c in file_chunks if c['metadata']['block_type'] == 'table')
            n_prose = sum(1 for c in file_chunks if c['metadata']['block_type'] == 'prose')
            print(f"   └ 성공! {len(file_chunks)}개 덩어리 생성됨 ({doc_type} / 표 {n_table}개 / 산문 {n_prose}개)")

        except Exception as e:
            print(f"   ❌ 파싱 에러 발생으로 해당 파일은 스킵합니다 ({file_path}): {e}")
            continue

print(f"\n✨ 완료! '{OUTPUT_PATH}'에 저장되었습니다.")