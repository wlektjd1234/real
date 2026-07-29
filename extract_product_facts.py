"""투자설명서 표 청크에서 클래스별 정형 지표를 SQLite로 추출한다.

이 스크립트는 원문 표(`processed_chunks.jsonl`)를 보존한 채, 숫자 조회가 필요한
총보수·기간별 수익률을 별도 DB로 만든다. 추출값에는 원문 표와 페이지 근거를 같이
저장하므로, 후속 검증 단계에서 사람이 쉽게 확인할 수 있다.
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path


CLASS_RE = re.compile(r"Class\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*(?:\([^)]*\))?)")
CLASS_TRAILING_PAREN_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\)\s*$")


def find_class_name(text: str) -> str | None:
    """'ClassA-G' 형식과 '오프라인(A-E)'처럼 괄호로 끝나는 형식을 모두 지원한다."""
    match = CLASS_RE.search(text)
    if match:
        return match.group(1)
    match = CLASS_TRAILING_PAREN_RE.search(text)
    return match.group(1) if match else None
NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
SEPARATOR_RE = re.compile(r"^\s*:?-{3,}:?\s*$")


def clean_cell(value: str) -> str:
    value = re.sub(r"<br>\s*", " ", value)
    value = re.sub(r"~~", "", value)
    return re.sub(r"\s+", " ", value).strip()


def markdown_rows(table_text: str) -> list[list[str]]:
    rows = []
    for line in table_text.splitlines():
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [clean_cell(cell) for cell in line[1:-1].split("|")]
        if cells and all(SEPARATOR_RE.fullmatch(cell) for cell in cells):
            continue
        rows.append(cells)
    return rows


def metric_columns(headers: list[str]) -> dict[int, str]:
    metrics = {}
    for index, header in enumerate(headers):
        compact = header.replace(" ", "")
        if "동종유형" in compact and "총보수" in compact:
            metrics[index] = "peer_group_total_fee_pct"
        elif "총보수ㆍ비용" in compact:
            metrics[index] = "total_cost_pct"
        elif "총보수" in compact:
            metrics[index] = "total_fee_pct"
        elif "최근1년" in compact:
            metrics[index] = "return_1y_pct"
        elif "최근2년" in compact:
            metrics[index] = "return_2y_pct"
        elif "최근3년" in compact:
            metrics[index] = "return_3y_pct"
        elif "최근5년" in compact:
            metrics[index] = "return_5y_pct"
        elif "설정일이후" in compact:
            metrics[index] = "return_since_inception_pct"
        elif "순자산" in compact or "시장잔고" in compact or "AUM" in compact.upper():
            metrics[index] = "aum"
        elif "위험등급" in compact:
            metrics[index] = "risk_grade"
    return metrics


def extract_rows(table_text: str) -> list[tuple[str, str, str, float | None, str]]:
    """표 한 개에서 (class_name, metric, raw_value, numeric_value, confidence)를 만든다."""
    rows = markdown_rows(table_text)
    if len(rows) < 2:
        return []

    # PDF 표는 다단 헤더가 많고 깊이도 표마다 다르다.
    # 고정 행수 대신, 클래스명이 처음 등장하는 행을 데이터 시작점으로 삼는다.
    header_rows = []
    data_start = None
    for index, row in enumerate(rows):
        if row and find_class_name(row[0]):
            data_start = index
            break
        header_rows.append(row)

    if data_start is None or not header_rows:
        return []

    width = max(map(len, header_rows))
    headers = [" ".join(row[index] for row in header_rows if index < len(row)) for index in range(width)]
    metrics = metric_columns(headers)
    if not metrics:
        return []

    facts = []
    for row in rows[data_start:]:
        if not row:
            continue
        class_name = find_class_name(row[0])
        if not class_name:
            continue
        for index, metric in metrics.items():
            if index >= len(row):
                continue
            raw_value = row[index].strip()
            if not raw_value or raw_value == "-":
                continue
            parsed_value = raw_value.replace(",", "").replace("%", "").strip()
            numeric_value = float(parsed_value) if NUMBER_RE.fullmatch(parsed_value) else None
            confidence = "high" if numeric_value is not None else "needs_review"
            facts.append((class_name, metric, raw_value, numeric_value, confidence))
    return facts
def initialize_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS product_facts (
            id INTEGER PRIMARY KEY,
            source_file TEXT NOT NULL,
            page_number INTEGER,
            class_name TEXT NOT NULL,
            metric TEXT NOT NULL,
            raw_value TEXT NOT NULL,
            numeric_value REAL,
            confidence TEXT NOT NULL CHECK(confidence IN ('high', 'needs_review')),
            source_table TEXT NOT NULL,
            UNIQUE(source_file, page_number, class_name, metric, raw_value)
        );
        CREATE INDEX IF NOT EXISTS idx_product_facts_lookup
            ON product_facts(source_file, class_name, metric);
        """
    )


def run(input_path: Path, db_path: Path) -> int:
    connection = sqlite3.connect(db_path)
    initialize_db(connection)
    inserted = 0
    with input_path.open(encoding="utf-8") as input_file:
        for line in input_file:
            chunk = json.loads(line)
            metadata = chunk["metadata"]
            if metadata.get("doc_type") != "product_prospectus" or metadata.get("block_type") != "table":
                continue
            for class_name, metric, raw_value, numeric_value, confidence in extract_rows(chunk["page_content"]):
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO product_facts
                    (source_file, page_number, class_name, metric, raw_value, numeric_value, confidence, source_table)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (metadata["source_file"], metadata.get("page_number"), class_name,
                     metric, raw_value, numeric_value, confidence, chunk["page_content"]),
                )
                inserted += cursor.rowcount
    connection.commit()
    connection.close()
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed_chunks.jsonl")
    parser.add_argument("--output", default="product_facts.sqlite")
    args = parser.parse_args()
    count = run(Path(args.input), Path(args.output))
    print(f"{count}개 정형 지표를 {args.output}에 저장했습니다.")
