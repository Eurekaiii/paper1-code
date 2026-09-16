"""Collect public text workloads for real MoE trace extraction.

The output is deliberately simple:

- prompts.jsonl keeps provenance and task type.
- prompts.txt contains one prompt per line for trace_extractor.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _write_jsonl(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_prompts(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = str(row["prompt"]).replace("\r", " ").replace("\n", " ")
            handle.write(prompt + "\n")


def collect_workload(
    output_dir: str | Path = "data/real_world/tasks",
    ag_news_count: int = 20,
    squad_count: int = 20,
) -> List[Dict[str, object]]:
    """Download small public text workloads and write local task files."""

    from datasets import load_dataset

    rows: List[Dict[str, object]] = []
    task_id = 0

    if ag_news_count > 0:
        ag_news = load_dataset("fancyzhx/ag_news", split=f"train[:{ag_news_count}]")
        for item in ag_news:
            text = str(item["text"]).replace("\\", " ")
            rows.append(
                {
                    "id": task_id,
                    "source_dataset": "fancyzhx/ag_news",
                    "task_type": "news_classification",
                    "label": int(item["label"]),
                    "text": text,
                    "prompt": (
                        "Classify the topic of this news item and explain the "
                        f"key evidence briefly: {text}"
                    ),
                }
            )
            task_id += 1

    if squad_count > 0:
        squad = load_dataset("rajpurkar/squad", split=f"train[:{squad_count}]")
        for item in squad:
            question = str(item["question"])
            context = str(item["context"])
            answer_texts = item.get("answers", {}).get("text", [])
            rows.append(
                {
                    "id": task_id,
                    "source_dataset": "rajpurkar/squad",
                    "task_type": "question_answering",
                    "title": str(item.get("title", "")),
                    "question": question,
                    "context": context,
                    "answer": answer_texts[0] if answer_texts else "",
                    "prompt": (
                        "Answer the question using the provided context. "
                        f"Question: {question} Context: {context}"
                    ),
                }
            )
            task_id += 1

    output = Path(output_dir)
    _write_jsonl(rows, output / "prompts.jsonl")
    _write_prompts(rows, output / "prompts.txt")

    manifest = {
        "num_tasks": len(rows),
        "sources": [
            {
                "dataset": "fancyzhx/ag_news",
                "count": ag_news_count,
                "task_type": "news_classification",
                "url": "https://huggingface.co/datasets/fancyzhx/ag_news",
            },
            {
                "dataset": "rajpurkar/squad",
                "count": squad_count,
                "task_type": "question_answering",
                "url": "https://huggingface.co/datasets/rajpurkar/squad",
            },
        ],
    }
    with (output / "task_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect real text workload samples.")
    parser.add_argument("--output-dir", default="data/real_world/tasks")
    parser.add_argument("--ag-news-count", type=int, default=20)
    parser.add_argument("--squad-count", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = collect_workload(
        output_dir=args.output_dir,
        ag_news_count=args.ag_news_count,
        squad_count=args.squad_count,
    )
    print(f"Wrote {len(rows)} real text tasks to {args.output_dir}")


if __name__ == "__main__":
    main()

