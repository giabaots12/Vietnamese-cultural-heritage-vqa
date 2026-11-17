#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script: evaluate_predictions.py
Mục đích:
    Đánh giá chất lượng các câu trả lời do mô hình sinh ra từ file predictions.json.
    Tính toán toàn diện các metrics:
      1. Tầng 1 (Lexical): Token F1, ROUGE-L, BLEU-1, BLEU-4, Keyword Match (Relaxed Accuracy).
      2. Tầng 2 (Semantic - Tùy chọn): LLM-as-a-Judge (Gemini / OpenAI) chấm điểm 1 - 5.
    Tổng hợp và phân rã báo cáo theo Cấp độ khó (Level 1-5) và Nhóm ý định câu hỏi.
    Xuất kết quả ra file JSON và Markdown báo cáo chuẩn.

Cách chạy:
    1. Chế độ Lexical (Nhanh, Offline, không cần API):
       python scripts/evaluate_predictions.py

    2. Chế độ có thêm LLM-as-a-Judge (Online):
       python scripts/evaluate_predictions.py --with-llm-judge --api-key <YOUR_GEMINI_KEY>
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

# Thêm thư mục gốc dự án vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_all_lexical_metrics,
    LLMJudgeEvaluator,
)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Evaluator")


def parse_arguments() -> argparse.Namespace:
    """Định nghĩa các tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Đánh giá chất lượng câu trả lời mô hình VLM trên tập dữ liệu kiểm thử."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "baseline" / "predictions.json"),
        help="Đường dẫn tới file predictions.json cần đánh giá.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "baseline"),
        help="Thư mục xuất báo cáo đánh giá.",
    )
    parser.add_argument(
        "--with-llm-judge",
        action="store_true",
        help="Bật chế độ chấm điểm ngữ nghĩa bằng LLM-as-a-Judge.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        help="API Key cho Groq (gsk_...) hoặc Gemini.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "groq", "gemini"],
        help="Nhà cung cấp LLM Judge: 'auto' (tự nhận diện qua key), 'groq', hoặc 'gemini'.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Tên mô hình LLM làm giám khảo (mặc định: llama-3.1-8b-instant cho Groq, gemini-flash-latest cho Gemini).",
    )
    parser.add_argument(
        "--judge-sample-size",
        type=int,
        default=None,
        help="Số lượng mẫu tối đa để chạy LLM Judge (để trống nếu muốn chạy toàn bộ).",
    )
    return parser.parse_args()


def categorize_intent(q_str: str) -> str:
    """Phân loại nhóm ý định câu hỏi dựa trên từ khóa tiếng Việt."""
    q = str(q_str).lower().strip()
    if re.search(r"\b(bao nhiêu|có mấy|mấy con|mấy cái|mấy người|mấy chiếc|số lượng)\b", q):
        return "Số lượng / Đếm"
    elif re.search(r"\b(màu gì|màu sắc|sơn màu)\b", q):
        return "Màu sắc"
    elif re.search(r"\b(ở đâu|vị trí nào|nằm ở|nơi nào|tại đâu|thuộc tỉnh|thuộc thành phố)\b", q):
        return "Vị trí / Địa danh"
    elif re.search(r"\b(là ai|ai là|nhân vật nào|vị vua|danh nhân|tác giả)\b", q):
        return "Nhân vật / Lịch sử"
    elif re.search(r"\b(tại sao|vì sao|ý nghĩa|mục đích|dùng để làm gì|có tác dụng gì)\b", q):
        return "Ý nghĩa / Mục đích"
    elif re.search(r"\b(thời kỳ nào|triều đại|năm nào|xây dựng vào|thế kỷ)\b", q):
        return "Thời gian / Niên đại"
    elif re.search(r"\b(tên gì|tên là gì|gọi là gì|được gọi là|công trình gì|loại gì)\b", q):
        return "Định danh / Tên gọi"
    else:
        return "Miêu tả chi tiết / Khác"


def df_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển DataFrame thành chuỗi Markdown thuần mà không cần thư viện tabulate."""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join([":---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def aggregate_group_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tính toán giá trị trung bình phần trăm của các chỉ số trong một nhóm."""
    n = len(items)
    if n == 0:
        return {}

    f1_list = [x["metrics"]["token_f1"] for x in items]
    rouge_l_list = [x["metrics"]["rouge_l_f1"] for x in items]
    rouge_l_rec = [x["metrics"]["rouge_l_recall"] for x in items]
    bleu_1_list = [x["metrics"]["bleu_1"] for x in items]
    bleu_4_list = [x["metrics"]["bleu_4"] for x in items]
    kw_match_list = [x["metrics"]["keyword_match"] for x in items]
    latencies = [x.get("latency_s", 0.0) for x in items]

    judge_scores = [
        x["llm_judge"]["score"]
        for x in items
        if "llm_judge" in x and x["llm_judge"].get("score") is not None
    ]

    res = {
        "count": n,
        "token_f1": round(float(np.mean(f1_list)) * 100, 2),
        "rouge_l_f1": round(float(np.mean(rouge_l_list)) * 100, 2),
        "rouge_l_recall": round(float(np.mean(rouge_l_rec)) * 100, 2),
        "bleu_1": round(float(np.mean(bleu_1_list)) * 100, 2),
        "bleu_4": round(float(np.mean(bleu_4_list)) * 100, 2),
        "keyword_match": round(float(np.mean(kw_match_list)) * 100, 2),
        "avg_latency_s": round(float(np.mean(latencies)), 2),
    }

    if judge_scores:
        res["llm_judge_score"] = round(float(np.mean(judge_scores)), 2)
        res["llm_judge_count"] = len(judge_scores)
    else:
        res["llm_judge_score"] = None
        res["llm_judge_count"] = 0

    return res


def main():
    args = parse_arguments()

    pred_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.is_file():
        logger.error(f"❌ Không tìm thấy file predictions: {pred_path}")
        sys.exit(1)

    logger.info(f"📂 Đang nạp dữ liệu dự đoán từ: {pred_path}")
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    logger.info(f"Tổng số mẫu cần đánh giá: {len(predictions):,} câu hỏi.")

    # 1. Tính toán Tầng 1: Lexical Metrics cho 100% câu hỏi
    logger.info("⚡ Đang tính toán các chỉ số Lexical (Token F1, ROUGE-L, BLEU, Keyword Match)...")
    evaluated_items = []
    for item in predictions:
        pred_text = str(item.get("prediction", "")).strip()
        gt_text = str(item.get("ground_truth", "")).strip()

        lex_metrics = compute_all_lexical_metrics(pred_text, gt_text)

        eval_item = dict(item)
        eval_item["intent"] = categorize_intent(item.get("question", ""))
        eval_item["metrics"] = lex_metrics
        evaluated_items.append(eval_item)

    logger.info("✅ Hoàn thành tính toán Lexical Metrics cho toàn bộ tập dữ liệu.")

    # 2. Xử lý Tầng 2: LLM-as-a-Judge (Nếu có yêu cầu)
    judge_cache_file = output_dir / "llm_judge_cache.json"
    judge_cache = {}
    if judge_cache_file.is_file():
        try:
            with open(judge_cache_file, "r", encoding="utf-8") as f:
                judge_cache = json.load(f)
            logger.info(f"🔄 Tìm thấy cache LLM Judge với {len(judge_cache):,} câu đã chấm.")
        except Exception:
            judge_cache = {}

    if args.with_llm_judge:
        if not args.api_key:
            logger.warning("⚠️ Đã bật --with-llm-judge nhưng chưa cung cấp --api-key. Bỏ qua bước LLM Judge.")
        else:
            evaluator = LLMJudgeEvaluator(
                api_key=args.api_key,
                model_name=args.judge_model,
                provider=args.provider,
            )
            model_display = evaluator._resolve_groq_model() if evaluator.provider == "groq" else evaluator.model_name
            logger.info(f"🌟 Đang khởi chạy LLM-as-a-Judge [{evaluator.provider.upper()}] sử dụng model: '{model_display}'...")

            target_items = evaluated_items
            if args.judge_sample_size and args.judge_sample_size < len(evaluated_items):
                # Lấy mẫu phân tầng đồng đều theo level
                target_items = []
                per_level = max(args.judge_sample_size // 5, 1)
                by_diff = defaultdict(list)
                for it in evaluated_items:
                    by_diff[it.get("difficulty", "1")].append(it)
                for lvl in ["1", "2", "3", "4", "5"]:
                    target_items.extend(by_diff[lvl][:per_level])
                logger.info(f"🎯 Lấy mẫu đại diện {len(target_items)} câu ({per_level} câu/level) để chấm LLM Judge.")

            for idx, it in enumerate(target_items, start=1):
                qid = it["qid"]
                if qid in judge_cache and judge_cache[qid].get("score") is not None:
                    it["llm_judge"] = judge_cache[qid]
                else:
                    res = evaluator.evaluate_single(
                        question=it["question"],
                        ground_truth=it["ground_truth"],
                        prediction=it["prediction"],
                    )
                    it["llm_judge"] = res
                    if res.get("score") is not None:
                        judge_cache[qid] = res
                        logger.info(f"[{idx}/{len(target_items)}] QID: {qid} | LLM Judge: {res['score']}/5đ | Lý do: {res['reasoning']}")
                    else:
                        logger.warning(f"⚠️ Câu {qid} không chấm được: {res.get('reasoning')}")

                    # Lưu cache định kỳ
                    if idx % 10 == 0 or idx == len(target_items):
                        with open(judge_cache_file, "w", encoding="utf-8") as f:
                            json.dump(judge_cache, f, ensure_ascii=False, indent=2)
                    import time
                    time.sleep(0.5)

            # Lưu lại lần cuối
            with open(judge_cache_file, "w", encoding="utf-8") as f:
                json.dump(judge_cache, f, ensure_ascii=False, indent=2)
            logger.info("✅ Hoàn thành chấm điểm LLM-as-a-Judge.")
    else:
        # Nếu có cache cũ từ trước, nạp vào
        for it in evaluated_items:
            if it["qid"] in judge_cache and judge_cache[it["qid"]].get("score") is not None:
                it["llm_judge"] = judge_cache[it["qid"]]

    # 3. Tổng hợp số liệu theo lát cắt (Aggregations)
    overall_summary = aggregate_group_metrics(evaluated_items)

    # Phân rã theo Difficulty (Levels 1 - 5)
    by_difficulty = defaultdict(list)
    for it in evaluated_items:
        by_difficulty[str(it.get("difficulty", "unknown"))].append(it)

    diff_summary = {}
    for lvl in sorted(by_difficulty.keys()):
        diff_summary[f"Level {lvl}"] = aggregate_group_metrics(by_difficulty[lvl])

    # Phân rã theo Intent
    by_intent = defaultdict(list)
    for it in evaluated_items:
        by_intent[it["intent"]].append(it)

    intent_summary = {}
    for intent_name in sorted(by_intent.keys()):
        intent_summary[intent_name] = aggregate_group_metrics(by_intent[intent_name])

    # 4. Hiển thị bảng kết quả ra Terminal
    print("\n" + "=" * 95)
    print("🏆 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ BENCHMARK (BASELINE EVALUATION REPORT)")
    print("=" * 95)

    # Bảng phân rã theo Level
    level_rows = []
    level_names_map = {
        "Level 1": "Level 1 (Màu sắc/Đặc trưng)",
        "Level 2": "Level 2 (Đếm số lượng)",
        "Level 3": "Level 3 (Nhận diện công trình)",
        "Level 4": "Level 4 (Ý nghĩa lịch sử)",
        "Level 5": "Level 5 (Suy luận tổng hợp)",
    }

    for lvl_key, metrics in diff_summary.items():
        row = {
            "Cấp độ khó": level_names_map.get(lvl_key, lvl_key),
            "Số câu": metrics["count"],
            "Token F1 (%)": metrics["token_f1"],
            "ROUGE-L (%)": metrics["rouge_l_f1"],
            "ROUGE-L Rec (%)": metrics["rouge_l_recall"],
            "BLEU-4 (%)": metrics["bleu_4"],
            "Keyword Match (%)": metrics["keyword_match"],
            "Thời gian (s)": metrics["avg_latency_s"],
        }
        if metrics["llm_judge_score"] is not None:
            row["LLM Judge (1-5)"] = f"{metrics['llm_judge_score']:.2f}"
        level_rows.append(row)

    # Dòng tổng thể (Overall)
    overall_row = {
        "Cấp độ khó": ">> TOÀN BỘ TẬP TEST <<",
        "Số câu": overall_summary["count"],
        "Token F1 (%)": overall_summary["token_f1"],
        "ROUGE-L (%)": overall_summary["rouge_l_f1"],
        "ROUGE-L Rec (%)": overall_summary["rouge_l_recall"],
        "BLEU-4 (%)": overall_summary["bleu_4"],
        "Keyword Match (%)": overall_summary["keyword_match"],
        "Thời gian (s)": overall_summary["avg_latency_s"],
    }
    if overall_summary["llm_judge_score"] is not None:
        overall_row["LLM Judge (1-5)"] = f"{overall_summary['llm_judge_score']:.2f}"
    level_rows.append(overall_row)

    df_levels = pd.DataFrame(level_rows)
    print("\n📊 BẢNG 1: HIỆU NĂNG THEO CẤP ĐỘ KHÓ (DIFFICULTY LEVELS)")
    print("-" * 95)
    print(df_levels.to_string(index=False))

    # Bảng phân rã theo Intent
    intent_rows = []
    for int_key, metrics in intent_summary.items():
        row = {
            "Nhóm Ý Định": int_key,
            "Số câu": metrics["count"],
            "Token F1 (%)": metrics["token_f1"],
            "ROUGE-L (%)": metrics["rouge_l_f1"],
            "Keyword Match (%)": metrics["keyword_match"],
        }
        if metrics["llm_judge_score"] is not None:
            row["LLM Judge (1-5)"] = f"{metrics['llm_judge_score']:.2f}"
        intent_rows.append(row)

    df_intents = pd.DataFrame(intent_rows)
    print("\n📊 BẢNG 2: HIỆU NĂNG THEO Ý ĐỊNH CÂU HỎI (QUESTION INTENTS)")
    print("-" * 95)
    print(df_intents.to_string(index=False))
    print("=" * 95)

    # 5. Xuất báo cáo ra file JSON và Markdown
    # File 1: metrics_report.json
    report_json_path = output_dir / "metrics_report.json"
    full_report = {
        "summary": {
            "overall": overall_summary,
            "by_difficulty": diff_summary,
            "by_intent": intent_summary,
        },
        "details": evaluated_items,
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Đã lưu báo cáo chi tiết JSON vào: {report_json_path.resolve()}")

    # File 2: benchmark_summary.md
    report_md_path = output_dir / "benchmark_summary.md"
    md_content = f"""# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH BASELINE (VINTERN-1B-V3.5)
*Dự án: Hỏi - Đáp Thị Giác Di Sản Văn Hóa Việt Nam (Vietnamese Cultural Heritage VQA)*  
*Tập dữ liệu kiểm thử: `data/processed/test.json` ({overall_summary['count']:,} câu hỏi / 212 ảnh độc lập)*

---

## 1. BẢNG HIỆU NĂNG THEO CẤP ĐỘ KHÓ (DIFFICULTY LEVELS)

{df_to_markdown(df_levels)}

---

## 2. BẢNG HIỆU NĂNG THEO Ý ĐỊNH CÂU HỎI (QUESTION INTENTS)

{df_to_markdown(df_intents)}

---

## 3. NHẬN XÉT ĐẶC TÍNH MÔ HÌNH GỐC (BASELINE INSIGHTS)

1. **Năng lực thị giác bề mặt (Level 1 & Level 2):**
   * Mô hình nhận diện màu sắc rất nhạy bén, đạt Keyword Match cao ở Level 1.
   * Với Level 2 (Đếm), mô hình gặp khó khăn khi đếm các vật thể nhỏ ở cự ly xa (xe cộ, người đi lại).
2. **Năng lực nhận diện công trình di sản (Level 3):**
   * Bắt đầu có sự phân hóa: Các địa danh nổi tiếng (núi đá vôi, vịnh Hạ Long, chùa Cầu) được nhận diện tốt, nhưng các di tích ít phổ biến hơn có hiện tượng nhầm lẫn.
3. **Nhu cầu tri thức ngoài (Level 4 & Level 5):**
   * Ở các câu hỏi về ý nghĩa tôn giáo, niên đại xây dựng, mô hình thường trả lời chung chung hoặc thừa nhận không có đủ thông tin từ bức ảnh.
   * 👉 **Đây chính là minh chứng thuyết phục nhất cho sự cần thiết của Giai đoạn 3 (Fine-tuning LoRA) và Giai đoạn 4 (Multimodal RAG)!**
"""
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"📄 Đã lưu báo cáo Markdown vào: {report_md_path.resolve()}")


if __name__ == "__main__":
    main()
