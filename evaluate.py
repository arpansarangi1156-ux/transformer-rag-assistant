"""
Evaluation script for the Transformer Research Paper RAG Assistant.
Evaluates the RAG pipeline against the 5 assessment questions,
recording actual outputs, source citations, latency, and comparisons.
Outputs results to outputs/assessment_results.json and outputs/assessment_results.md.
"""

import os
import time
import json
from typing import Dict, Any, List

import rag

# Ground truth benchmarks (used ONLY for evaluation, never in RAG pipeline)
EVALUATION_BENCHMARKS = [
    {
        "id": "Q1",
        "question": "What exact learning rate, number of warmup steps, and Adam optimizer beta values were used in the primary pre-training run described in Section 4?",
        "expected_answer": "Peak learning rate 3e-4 (0.0003), 4,000 linear warmup steps, AdamW beta1=0.9 and beta2=0.95. Source: Page 16.",
        "expected_sources": [16],
        "category": "In-Corpus Factual"
    },
    {
        "id": "Q2",
        "question": "Who is the current Chief Executive Officer of OpenAI?",
        "expected_answer": "[not found]",
        "expected_sources": [],
        "category": "Out-of-Corpus (Strict Grounding Test)"
    },
    {
        "id": "Q3",
        "question": "By what percentage did the proposed hybrid retrieval architecture improve Exact Match (EM) accuracy on the benchmark dataset compared with the dense-only baseline?",
        "expected_answer": "14.6% improvement; EM increased from 61.2% to 75.8%. Source: Page 31.",
        "expected_sources": [31],
        "category": "In-Corpus Factual"
    },
    {
        "id": "Q4",
        "question": "What GPU cluster configuration (number and model of GPUs) and total training duration, in days, were required to train the 13-billion-parameter model?",
        "expected_answer": "128 NVIDIA H100 (80GB SXM5) GPUs, 400 Gbps InfiniBand, 18.5 days. Source: Page 45.",
        "expected_sources": [45],
        "category": "In-Corpus Factual"
    },
    {
        "id": "Q5",
        "question": "What was the worldwide box-office revenue of the film Oppenheimer in 2023?",
        "expected_answer": "[not found]",
        "expected_sources": [],
        "category": "Out-of-Corpus (Strict Grounding Test)"
    }
]


def run_evaluation() -> Dict[str, Any]:
    """Runs evaluation across all five benchmark questions."""
    os.makedirs("outputs", exist_ok=True)

    print("=" * 80)
    print("TRANSFORMER RESEARCH PAPER RAG PIPELINE EVALUATION")
    print("=" * 80)

    # 1. Verify index state
    print("[1/3] Checking vector store & ingestion state...")
    col, stats = rag.initialize_or_update_index()
    print(f"Index status: {stats['status']} (Chunks: {stats['chunks_count']}, Fingerprint: {stats['fingerprint'][:16]}...)")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    has_api_key = bool(api_key and api_key != "your_groq_api_key_here")

    if not has_api_key:
        print("\n[NOTE] GROQ_API_KEY is not set in environment or .env file.")
        print("Running full vector retrieval, chunk similarity, and ranking evaluation.")
        print("To run LLM generation tests, add your GROQ_API_KEY to .env and re-run evaluate.py.\n")

    results: List[Dict[str, Any]] = []

    print("[2/3] Executing benchmark evaluation...")
    for item in EVALUATION_BENCHMARKS:
        qid = item["id"]
        q = item["question"]
        print(f"\nEvaluating {qid}: {q[:70]}...")

        # Measure retrieval
        t_start = time.perf_counter()
        retrieved_chunks = rag.retrieve_chunks(q, top_k=5)
        retrieval_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        top_page = retrieved_chunks[0]["page"] if retrieved_chunks else None
        top_sim = retrieved_chunks[0]["similarity"] if retrieved_chunks else 0.0
        retrieved_pages = [c["page"] for c in retrieved_chunks]

        # Execute generation if API key is present, otherwise test retrieval pipeline
        gen_latency_ms = 0.0
        actual_answer = ""
        actual_sources = []
        status = "PENDING_API_KEY"

        if has_api_key:
            t_gen_start = time.perf_counter()
            resp = rag.answer(q, return_debug=True)
            gen_latency_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
            actual_answer = resp.get("answer", "")
            actual_sources = resp.get("sources", [])

            # Check correctness
            if item["expected_sources"] == []:
                # Expecting [not found]
                if actual_answer.strip() == "[not found]" and actual_sources == []:
                    status = "PASS"
                else:
                    status = "FAIL"
            else:
                # Expecting grounded answer with expected page in sources
                expected_p = item["expected_sources"][0]
                if expected_p in actual_sources and actual_answer != "[not found]":
                    status = "PASS"
                else:
                    status = "FAIL"
        else:
            # Evaluate retrieval correctness without API key
            if item["expected_sources"] == []:
                # Out-of-corpus: top similarity should be low (< 0.20)
                retrieval_correct = top_sim < 0.20
                actual_answer = "[LLM Generation requires GROQ_API_KEY in .env]"
                actual_sources = []
                status = "PASS (Retrieval Grounding)" if retrieval_correct else "NEEDS_REVIEW"
            else:
                expected_p = item["expected_sources"][0]
                retrieval_correct = (top_page == expected_p)
                actual_answer = f"[LLM Generation requires GROQ_API_KEY in .env | Top chunk retrieved from Page {top_page} (cos sim: {top_sim})]"
                actual_sources = [top_page] if retrieval_correct else retrieved_pages[:1]
                status = "PASS (Retrieval Rank 1)" if retrieval_correct else "FAIL"

        print(f"  Status: {status}")
        print(f"  Retrieval latency: {retrieval_latency_ms} ms | Top page: {top_page} | Max sim: {top_sim}")
        if has_api_key:
            print(f"  Generation latency: {gen_latency_ms} ms")
            print(f"  Answer: {actual_answer[:100]}...")
            print(f"  Sources: {actual_sources}")

        results.append({
            "id": qid,
            "category": item["category"],
            "question": q,
            "expected_answer": item["expected_answer"],
            "expected_sources": item["expected_sources"],
            "actual_answer": actual_answer,
            "actual_sources": actual_sources,
            "status": status,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": gen_latency_ms,
            "top_retrieved_page": top_page,
            "top_cosine_similarity": top_sim,
            "retrieved_pages": retrieved_pages,
            "retrieved_chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "page": c["page"],
                    "similarity": c["similarity"],
                    "text_preview": c["text"][:140] + "..."
                }
                for c in retrieved_chunks[:3]
            ]
        })

    # Summary report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "has_groq_api_key": has_api_key,
        "embedding_model": rag.EMBEDDING_MODEL_NAME,
        "vector_store": "ChromaDB Persistent",
        "total_questions": len(EVALUATION_BENCHMARKS),
        "passed_tests": sum(1 for r in results if "PASS" in r["status"]),
        "results": results
    }

    print("\n[3/3] Saving assessment reports...")

    # Write JSON report
    json_path = "outputs/assessment_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    # Write Markdown report
    md_path = "outputs/assessment_results.md"
    write_markdown_report(summary, md_path)
    print(f"Saved: {md_path}")

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Score: {summary['passed_tests']} / {summary['total_questions']} Passed")
    print("=" * 80)
    return summary


def write_markdown_report(summary: Dict[str, Any], file_path: str):
    """Formats the evaluation summary into clean GitHub-flavored markdown."""
    lines = [
        "# RAG Assessment Evaluation Report",
        "",
        f"- **Execution Timestamp:** `{summary['timestamp']}`",
        f"- **Embedding Model:** `{summary['embedding_model']}` (Local)",
        f"- **Vector Storage:** `{summary['vector_store']}`",
        f"- **Groq API Connected:** `{'Yes' if summary['has_groq_api_key'] else 'No (Vector retrieval evaluated)'}`",
        f"- **Overall Evaluation:** `{summary['passed_tests']} / {summary['total_questions']} Passed`",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| ID | Question Type | Expected Sources | Actual Sources | Top Sim | Retrieval Status | Result |",
        "|---|---|---|---|---|---|---|"
    ]

    for r in summary["results"]:
        exp_src = str(r["expected_sources"]) if r["expected_sources"] else "[] ([not found])"
        act_src = str(r["actual_sources"]) if r["actual_sources"] else "[]"
        top_sim = f"{r['top_cosine_similarity']:.3f}"
        lines.append(
            f"| **{r['id']}** | {r['category']} | `{exp_src}` | `{act_src}` | `{top_sim}` | Top P.{r['top_retrieved_page']} | **{r['status']}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Question Analysis",
        ""
    ])

    for r in summary["results"]:
        lines.extend([
            f"### {r['id']}: {r['question']}",
            f"- **Category:** {r['category']}",
            f"- **Expected Answer:** {r['expected_answer']}",
            f"- **Expected Sources:** `{r['expected_sources']}`",
            f"- **Actual Answer:** `{r['actual_answer']}`",
            f"- **Actual Sources:** `{r['actual_sources']}`",
            f"- **Retrieval Latency:** `{r['retrieval_latency_ms']} ms`",
            f"- **Top Cosine Similarity:** `{r['top_cosine_similarity']}`",
            f"- **Top Retrieved Pages:** `{r['retrieved_pages']}`",
            "",
            "**Retrieved Context Excerpt:**",
            "```text",
            r["retrieved_chunks"][0]["text_preview"] if r["retrieved_chunks"] else "None",
            "```",
            "",
            f"**Evaluation Status:** `{r['status']}`",
            "",
            "---",
            ""
        ])

    with open(file_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()
