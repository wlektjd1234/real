"""RAG 청크 JSONL과 처리 manifest의 품질을 점검한다."""

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_METADATA = {
    "source_file", "page_number", "doc_type", "category", "작성기준일", "block_type",
}


def validate(chunks_path: Path, manifest_path: Path | None) -> dict:
    summary = Counter()
    missing_metadata = Counter()
    sources = set()

    with chunks_path.open(encoding="utf-8") as chunks_file:
        for line_number, line in enumerate(chunks_file, start=1):
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                summary["invalid_jsonl_rows"] += 1
                continue
            summary["chunks"] += 1
            text = chunk.get("page_content", "")
            metadata = chunk.get("metadata", {})
            if not text.strip():
                summary["empty_chunks"] += 1
            sources.add(metadata.get("source_file"))
            for key in REQUIRED_METADATA:
                if key not in metadata:
                    missing_metadata[key] += 1
            if len(text) > 2000:
                summary["chunks_over_2000_chars"] += 1
            summary[f"doc_type:{metadata.get('doc_type', 'missing')}"] += 1
            summary[f"block_type:{metadata.get('block_type', 'missing')}"] += 1

    summary["source_files"] = len(sources - {None})
    result = {
        "summary": dict(sorted(summary.items())),
        "missing_metadata": dict(sorted(missing_metadata.items())),
    }

    if manifest_path and manifest_path.exists():
        statuses = Counter()
        with manifest_path.open(encoding="utf-8") as manifest_file:
            for line in manifest_file:
                statuses[json.loads(line).get("status", "missing")] += 1
        result["manifest_status"] = dict(sorted(statuses.items()))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed_chunks.jsonl")
    parser.add_argument("--manifest", default="processing_manifest.jsonl")
    args = parser.parse_args()
    print(json.dumps(validate(Path(args.input), Path(args.manifest)), ensure_ascii=False, indent=2))
