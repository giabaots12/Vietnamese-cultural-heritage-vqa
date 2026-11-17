#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module: src.evaluation.metrics
Mục đích:
    Cung cấp bộ công cụ toán học và chuẩn hóa để đo lường chất lượng câu trả lời
    trong bài toán Hỏi - Đáp Thị Giác Di Sản Văn Hóa Việt Nam (Vietnamese Cultural Heritage VQA).

Bao gồm:
    1. Chuẩn hóa văn bản tiếng Việt (normalize_text, tokenize_text).
    2. Chỉ số Token-level F1, Precision, Recall (compute_token_f1).
    3. Chỉ số ROUGE-L dựa trên Longest Common Subsequence (compute_rouge_l).
    4. Chỉ số BLEU-1 đến BLEU-4 có Brevity Penalty (compute_bleu).
    5. Chỉ số kiểm tra từ khóa then chốt / Relaxed Accuracy (compute_keyword_match).
    6. Hàm tiện ích tổng hợp toàn bộ Lexical Metrics (compute_all_lexical_metrics).
    7. Khung trọng tài ngữ nghĩa độc lập (LLMJudgeEvaluator) chấm điểm 1 - 5.
"""

import re
import math
import string
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional


# ==============================================================================
# 1. TIỀN XỬ LÝ & CHUẨN HÓA VĂN BẢN
# ==============================================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa chuỗi văn bản:
    - Chuyển về chữ thường.
    - Loại bỏ toàn bộ dấu câu (punctuation).
    - Chuẩn hóa khoảng trắng thừa.
    """
    if not isinstance(text, str):
        return ""
    # Chuyển về chữ thường
    text = text.lower().strip()
    # Loại bỏ dấu câu
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    """Tách văn bản thành danh sách các từ (tokens) sau khi đã chuẩn hóa."""
    norm = normalize_text(text)
    return norm.split() if norm else []


# ==============================================================================
# 2. TOKEN F1-SCORE (PRECISION, RECALL, F1)
# ==============================================================================

def compute_token_f1(pred: str, gt: str) -> Dict[str, float]:
    """
    Tính toán Token-level Precision, Recall và F1-score.
    
    Returns:
        dict: {'precision': float, 'recall': float, 'f1': float} (giá trị 0.0 -> 1.0)
    """
    pred_tokens = tokenize_text(pred)
    gt_tokens = tokenize_text(gt)

    if not pred_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    # Đếm số từ trùng lặp giao nhau
    common = sum((pred_counter & gt_counter).values())

    if common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ==============================================================================
# 3. ROUGE-L (LONGEST COMMON SUBSEQUENCE)
# ==============================================================================

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Tính độ dài chuỗi con chung dài nhất (LCS) bằng quy hoạch động O(M*N)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # Sử dụng 1 mảng 1D để tối ưu bộ nhớ
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_rouge_l(pred: str, gt: str, beta: float = 1.2) -> Dict[str, float]:
    """
    Tính toán ROUGE-L (Longest Common Subsequence).
    
    Args:
        pred: Câu trả lời mô hình sinh ra.
        gt: Đáp án chuẩn (Ground Truth).
        beta: Trọng số ưu tiên recall (mặc định 1.2 theo chuẩn Lin, 2004).

    Returns:
        dict: {'rouge_l_precision': float, 'rouge_l_recall': float, 'rouge_l_f1': float}
    """
    pred_tokens = tokenize_text(pred)
    gt_tokens = tokenize_text(gt)

    if not pred_tokens and not gt_tokens:
        return {"rouge_l_precision": 1.0, "rouge_l_recall": 1.0, "rouge_l_f1": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    lcs = _lcs_length(pred_tokens, gt_tokens)

    if lcs == 0:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    precision = lcs / len(pred_tokens)
    recall = lcs / len(gt_tokens)
    beta_sq = beta ** 2
    f1 = ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision + 1e-12)

    return {
        "rouge_l_precision": round(precision, 4),
        "rouge_l_recall": round(recall, 4),
        "rouge_l_f1": round(f1, 4),
    }


# ==============================================================================
# 4. BLEU SCORES (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
# ==============================================================================

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Trích xuất n-gram counter từ danh sách tokens."""
    return Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def compute_bleu(pred: str, gt: str, max_n: int = 4) -> Dict[str, float]:
    """
    Tính toán BLEU-1 đến BLEU-4 với Brevity Penalty và làm mịn (smoothing).
    
    Returns:
        dict: {'bleu_1': float, 'bleu_2': float, 'bleu_3': float, 'bleu_4': float, 'brevity_penalty': float}
    """
    pred_tokens = tokenize_text(pred)
    gt_tokens = tokenize_text(gt)

    c = len(pred_tokens)
    r = len(gt_tokens)

    if c == 0:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)} | {"brevity_penalty": 0.0}
    if r == 0:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)} | {"brevity_penalty": 0.0}

    # Brevity Penalty
    bp = 1.0 if c > r else math.exp(1 - (r / c))

    p_ns = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        gt_ngrams = _get_ngrams(gt_tokens, n)

        if not pred_ngrams:
            p_ns.append(0.0)
            continue

        clipped_matches = sum((pred_ngrams & gt_ngrams).values())
        total_pred_ngrams = max(c - n + 1, 1)

        # Làm mịn Laplace nhẹ (Add-epsilon smoothing) để không bị 0 điểm hoàn toàn ở n-gram cao
        if clipped_matches == 0:
            precision_n = 0.5 / total_pred_ngrams
        else:
            precision_n = clipped_matches / total_pred_ngrams

        p_ns.append(precision_n)

    results = {"brevity_penalty": round(bp, 4)}

    # Tính BLEU tích lũy từ 1 đến max_n
    for k in range(1, max_n + 1):
        weights = [1.0 / k] * k
        log_precisions = [w * math.log(p_ns[i]) for i, w in enumerate(weights)]
        bleu_k = bp * math.exp(sum(log_precisions))
        results[f"bleu_{k}"] = round(bleu_k, 4)

    return results


# ==============================================================================
# 5. KEYWORD MATCH / RELAXED ACCURACY
# ==============================================================================

def compute_keyword_match(pred: str, gt: str) -> bool:
    """
    Kiểm tra xem từ khóa then chốt của Ground Truth có xuất hiện trong câu trả lời hay không.
    Đặc biệt xử lý các trường hợp Ground Truth có dấu gạch chéo '/' hoặc 'hoặc' (các đáp án tương đương).
    Ví dụ: 'Màu cam/đỏ' -> Chấp nhận nếu có 'cam' hoặc 'đỏ'.
           'Núi đá vôi/núi Karst' -> Chấp nhận nếu có 'núi đá vôi' hoặc 'núi karst'.
    """
    if not gt or not pred:
        return False

    norm_pred = " " + normalize_text(pred) + " "

    # Tách các phương án tương đương theo dấu '/' hoặc chữ 'hoặc'
    parts = re.split(r"[/|]|\bhoặc\b", gt, flags=re.IGNORECASE)

    for part in parts:
        clean_part = normalize_text(part)
        tokens = clean_part.split()
        # Bỏ qua các từ phụ như 'màu', 'dạng', 'loại' nếu có từ chính
        core_tokens = [t for t in tokens if t not in {"màu", "dạng", "loại", "là", "và"}]
        eval_tokens = core_tokens if core_tokens else tokens

        if eval_tokens and all((" " + tok + " ") in norm_pred for tok in eval_tokens):
            return True

    return False


# ==============================================================================
# 6. HÀM TỔNG HỢP TOÀN BỘ LEXICAL METRICS CHO 1 MẪU
# ==============================================================================

def compute_all_lexical_metrics(pred: str, gt: str) -> Dict[str, Any]:
    """
    Tính toán toàn bộ các chỉ số Tầng 1 (Lexical Metrics) cho 1 cặp (Prediction, Ground Truth).
    
    Returns:
        dict: Chứa F1, Precision, Recall, ROUGE-L, BLEU-1, BLEU-4, Keyword Match.
    """
    f1_dict = compute_token_f1(pred, gt)
    rouge_dict = compute_rouge_l(pred, gt)
    bleu_dict = compute_bleu(pred, gt)
    kw_match = compute_keyword_match(pred, gt)

    return {
        "token_f1": f1_dict["f1"],
        "token_precision": f1_dict["precision"],
        "token_recall": f1_dict["recall"],
        "rouge_l_f1": rouge_dict["rouge_l_f1"],
        "rouge_l_precision": rouge_dict["rouge_l_precision"],
        "rouge_l_recall": rouge_dict["rouge_l_recall"],
        "bleu_1": bleu_dict["bleu_1"],
        "bleu_4": bleu_dict["bleu_4"],
        "keyword_match": 1.0 if kw_match else 0.0,
    }


# ==============================================================================
# 7. TRỌNG TÀI NGỮ NGHĨA (LLM-AS-A-JUDGE EVALUATOR)
# ==============================================================================

LLM_JUDGE_SYSTEM_PROMPT = """Bạn là một chuyên gia thẩm định chất lượng các hệ thống AI trong bài toán Hỏi - Đáp Thị Giác về Di sản Văn hóa Việt Nam (Vietnamese Cultural Heritage VQA).

Nhiệm vụ của bạn là so sánh "Câu trả lời của Mô hình" với "Đáp án chuẩn (Ground Truth)" dựa trên "Câu hỏi", sau đó chấm điểm độ chính xác ngữ nghĩa theo thang điểm từ 1 đến 5.

Quy tắc chấm điểm (Scoring Rubric):
- Điểm 5 (Hoàn toàn chính xác): Câu trả lời phản ánh chính xác và đầy đủ sự thật của đáp án chuẩn. Chấp nhận câu trả lời dạng câu đàm thoại đầy đủ dù đáp án chuẩn chỉ là cụm từ ngắn.
- Điểm 4 (Đúng phần lớn): Đúng thực thể / ý tưởng cốt lõi, chỉ thiếu sót chi tiết phụ nhỏ không ảnh hưởng lớn đến ngữ cảnh.
- Điểm 3 (Đúng một phần): Nhận diện được một nửa thông tin quan trọng, còn lại mơ hồ hoặc thiếu dữ kiện chính.
- Điểm 2 (Kém / Lạc đề nhẹ): Câu trả lời chung chung, không đi vào trọng tâm câu hỏi, hoặc đoán mò không có căn cứ.
- Điểm 1 (Sai hoàn toàn / Bịa đặt): Thông tin sai sự thật, bịa đặt thực thể lịch sử (hallucination), hoặc mô hình nói không biết.

YÊU CẦU BẮT BUỘC: Bạn PHẢI trả về duy nhất 1 chuỗi JSON hợp lệ với cấu trúc sau, không kèm bất kỳ giải thích nào bên ngoài:
{"score": <số nguyên từ 1 đến 5>, "reasoning": "<giải thích ngắn gọn lý do chấm điểm trong 1 câu>"}"""


def build_judge_prompt(question: str, ground_truth: str, prediction: str) -> str:
    """Tạo nội dung prompt gửi tới mô hình LLM Judge."""
    return f"""Dữ liệu cần đánh giá:
- Câu hỏi: {question}
- Đáp án chuẩn (Ground Truth): {ground_truth}
- Câu trả lời của Mô hình (Prediction): {prediction}

Hãy chấm điểm và trả về JSON theo đúng quy cách."""


class LLMJudgeEvaluator:
    """
    Quản lý việc chấm điểm ngữ nghĩa tự động thông qua LLM-as-a-judge.
    Hỗ trợ cả Groq API (Llama-3.1-8B-Instant - Siêu tốc & Ổn định) và Google Gemini API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: str = "auto",
    ):
        self.api_key = api_key
        self.provider = provider
        
        # Tự động nhận diện provider qua tiền tố của Key nếu là "auto"
        if self.provider == "auto":
            if self.api_key and self.api_key.startswith("gsk_"):
                self.provider = "groq"
            else:
                self.provider = "gemini"

        # Gán model mặc định theo provider
        if not model_name:
            if self.provider == "groq":
                self.model_name = "llama-3.1-8b-instant"
            else:
                self.model_name = "gemini-flash-latest"
        else:
            self.model_name = model_name

        self._resolved_model = None
        self._client = None

    def _resolve_groq_model(self) -> str:
        """Xác định model Groq đang khả dụng và active (chỉ truy vấn 1 lần duy nhất)."""
        if self._resolved_model:
            return self._resolved_model

        target_model = self.model_name or "llama-3.1-8b-instant"

        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key.strip()}",
                "Content-Type": "application/json",
            }
            resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                # Chỉ lọc các model ĐANG HOẠT ĐỘNG (loại bỏ hoàn toàn model decommissioned/active=false)
                active_models = {
                    m["id"]: m for m in data
                    if m.get("active") is True or (m.get("active") is not False and not m.get("deprecated", False))
                }

                # 1. Nếu model chỉ định đang active -> Dùng ngay
                if target_model in active_models:
                    self._resolved_model = target_model
                    return self._resolved_model

                # 2. Danh sách ưu tiên theo thứ tự năng lực thẩm định VQA
                preferred = [
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile",
                    "meta-llama/llama-3.1-8b-instruct",
                    "gemma2-9b-it",
                    "mixtral-8x7b-32768",
                ]
                for p in preferred:
                    if p in active_models:
                        self._resolved_model = p
                        return self._resolved_model

                # 3. Model chat bất kỳ còn active (bỏ qua whisper, guard, tts)
                chat_models = [
                    mid for mid in active_models
                    if not any(k in mid.lower() for k in ["whisper", "guard", "tts", "safeguard"])
                ]
                if chat_models:
                    self._resolved_model = chat_models[0]
                    return self._resolved_model
        except Exception:
            pass

        self._resolved_model = target_model
        return self._resolved_model

    def evaluate_single(self, question: str, ground_truth: str, prediction: str) -> Dict[str, Any]:
        """
        Gửi 1 mẫu tới LLM Judge và bóc tách điểm số JSON.
        """
        if not self.api_key:
            return {
                "score": None,
                "reasoning": "Chưa cung cấp API Key cho LLM Judge.",
            }

        prompt = build_judge_prompt(question, ground_truth, prediction)
        import json
        import time

        # ======================================================================
        # CƠ CHẾ 1: GROQ API (Llama / Mixtral / Gemma) - SIÊU TỐC & ỔN ĐỊNH
        # ======================================================================
        if self.provider == "groq" or (self.api_key and self.api_key.startswith("gsk_")):
            active_model = self._resolve_groq_model()
            headers = {
                "Authorization": f"Bearer {self.api_key.strip()}",
                "Content-Type": "application/json",
            }
            url = "https://api.groq.com/openai/v1/chat/completions"
            payload = {
                "model": active_model,
                "messages": [
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            try:
                import requests
                for attempt in range(4):
                    try:
                        resp = requests.post(url, json=payload, headers=headers, timeout=25)
                        if resp.status_code == 429:
                            time.sleep(2.0 * (attempt + 1))
                            continue
                        if resp.status_code != 200:
                            return {"score": None, "reasoning": f"Groq Error {resp.status_code}: {resp.text}"}
                        data = resp.json()
                        raw_text = data["choices"][0]["message"]["content"].strip()
                        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
                        parsed = json.loads(clean_json)
                        return {
                            "score": int(parsed.get("score", 1)),
                            "reasoning": str(parsed.get("reasoning", "")),
                        }
                    except requests.exceptions.RequestException:
                        time.sleep(1.5)
            except ImportError:
                import urllib.request
                import urllib.error
                data_bytes = json.dumps(payload).encode("utf-8")
                for attempt in range(4):
                    try:
                        req = urllib.request.Request(url, data=data_bytes, headers=headers)
                        with urllib.request.urlopen(req, timeout=25) as resp:
                            data = json.loads(resp.read().decode("utf-8"))
                        raw_text = data["choices"][0]["message"]["content"].strip()
                        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
                        parsed = json.loads(clean_json)
                        return {
                            "score": int(parsed.get("score", 1)),
                            "reasoning": str(parsed.get("reasoning", "")),
                        }
                    except Exception as err:
                        time.sleep(1.5)
            return {"score": None, "reasoning": "Không thể kết nối đến Groq API."}

        # ======================================================================
        # CƠ CHẾ 2: GOOGLE GEMINI API
        # ======================================================================
        # 1. Thử gọi qua SDK mới google.genai
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "system_instruction": LLM_JUDGE_SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                },
            )
            import json
            parsed = json.loads(response.text.strip())
            return {
                "score": int(parsed.get("score", 1)),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except ImportError:
            pass
        except Exception as e:
            return {"score": None, "reasoning": f"Lỗi gọi google-genai: {e}"}

        # 2. Thử qua SDK cũ google.generativeai
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=self.api_key)
            m = legacy_genai.GenerativeModel(
                self.model_name,
                system_instruction=LLM_JUDGE_SYSTEM_PROMPT,
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
            )
            res = m.generate_content(prompt)
            import json
            parsed = json.loads(res.text.strip())
            return {
                "score": int(parsed.get("score", 1)),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except ImportError:
            pass
        except Exception as e:
            return {"score": None, "reasoning": f"Lỗi gọi google-generativeai: {e}"}

        # 3. Fallback qua Direct REST API (Sử dụng requests / urllib chuẩn)
        import json
        import time

        # Đối với AQ. Auth Keys, truyền qua header X-goog-api-key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key.strip(),
        }
        payload = {
            "system_instruction": {
                "parts": [{"text": LLM_JUDGE_SYSTEM_PROMPT}]
            },
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.0,
                "response_mime_type": "application/json"
            }
        }

        # Danh sách mô hình theo chuẩn cURL của Google
        candidate_models = list(dict.fromkeys([self.model_name, "gemini-flash-latest", "gemini-2.0-flash", "gemini-1.5-flash-latest"]))

        # Ưu tiên dùng requests (kết nối nhanh, có pool kết nối và SSL hiện đại)
        try:
            import requests
            for model_id in candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
                for attempt in range(4):
                    try:
                        resp = requests.post(url, json=payload, headers=headers, timeout=25)
                        if resp.status_code in (429, 503):
                            # Chờ tạm thời khi Google quá tải cục bộ
                            time.sleep(3.0 * (attempt + 1))
                            continue
                        if resp.status_code == 404:
                            # Model không tồn tại trên endpoint này, thử model tiếp theo
                            break
                        if resp.status_code != 200:
                            return {"score": None, "reasoning": f"HTTP Error {resp.status_code}: {resp.text}"}
                        res_data = resp.json()
                        raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        parsed = json.loads(raw_text)
                        return {
                            "score": int(parsed.get("score", 1)),
                            "reasoning": str(parsed.get("reasoning", "")),
                        }
                    except requests.exceptions.RequestException:
                        time.sleep(2.0)
        except ImportError:
            # Fallback urllib
            import urllib.request
            import urllib.error
            data_bytes = json.dumps(payload).encode("utf-8")
            for model_id in candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
                for attempt in range(3):
                    try:
                        req = urllib.request.Request(url, data=data_bytes, headers=headers)
                        with urllib.request.urlopen(req, timeout=25) as resp:
                            res_data = json.loads(resp.read().decode("utf-8"))
                        raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        parsed = json.loads(raw_text)
                        return {
                            "score": int(parsed.get("score", 1)),
                            "reasoning": str(parsed.get("reasoning", "")),
                        }
                    except Exception:
                        time.sleep(1.5)

        return {"score": None, "reasoning": "Không thể kết nối đến Gemini REST API (Máy chủ Google quá tải)."}
