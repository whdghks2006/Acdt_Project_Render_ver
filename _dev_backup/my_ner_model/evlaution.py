import spacy
import json
import os
import warnings
from spacy.training.example import Example
from spacy.scorer import Scorer

# ==============================================================================
# Configuration / 설정
# ==============================================================================

# Path to the saved model
# 저장된 모델 경로
MODEL_DIR = r"C:/Users/82105/PycharmProjects/Acdt Project 2/my_ner_model"

# Path to the test data file
# 테스트 데이터 파일 경로
TEST_FILE = r"C:/Users/82105/PycharmProjects/Acdt Project 2/test.json"


# ==============================================================================
# Helper Function / 헬퍼 함수 (Training 코드와 동일)
# ==============================================================================

def clean_and_fix_data(nlp, data):
    """
    Aligns entities with token boundaries using 'expand' mode.
    테스트 데이터도 학습 때와 똑같이 엔티티 범위를 토큰에 맞춰 수정합니다.
    """
    clean_data_list = []
    fixed_count = 0

    for text, annotations in data:
        doc = nlp.make_doc(text)
        valid_ents = []

        if "entities" not in annotations:
            continue

        for start, end, label in annotations["entities"]:
            # 1. Strict alignment check
            span = doc.char_span(start, end, label=label)

            # 2. If failed, try expand alignment
            if span is None:
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if span is not None:
                    fixed_count += 1

            if span is not None:
                valid_ents.append(span)

        # Convert spans back to (start, end, label) tuples
        final_ents = [(e.start_char, e.end_char, e.label_) for e in valid_ents]

        # Add only if valid entities exist
        if final_ents:
            # Remove duplicates just in case
            final_ents = list(set(final_ents))
            clean_data_list.append((text, {"entities": final_ents}))

    print(f"🧹 Test Data cleaned.")
    print(f"   - Fixed/Aligned {fixed_count} entities.")
    print(f"   - Valid examples ready for evaluation: {len(clean_data_list)}")
    return clean_data_list


# ==============================================================================
# Main Execution / 메인 실행
# ==============================================================================

if __name__ == "__main__":
    # Ignore alignment warnings
    warnings.filterwarnings("ignore")

    print("--------------------------------------------------")
    print("🔄 Step 1: Loading Model and Test Data...")

    # 1. Load Model
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Error: Model not found at {MODEL_DIR}")
        exit()

    try:
        nlp = spacy.load(MODEL_DIR)
        print(f"✅ Model loaded from: {MODEL_DIR}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit()

    # 2. Load Test Data
    if not os.path.exists(TEST_FILE):
        print(f"❌ Error: Test file not found at {TEST_FILE}")
        exit()

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        RAW_TEST_DATA = json.load(f)
    print(f"✅ Raw Test data loaded: {len(RAW_TEST_DATA)} examples")

    print("\n🔄 Step 2: Preparing Data...")
    # Clean the test data using the loaded model's tokenizer
    # 불러온 모델의 토크나이저를 사용해 테스트 데이터도 '스마트 보정' 수행
    TEST_DATA = clean_and_fix_data(nlp, RAW_TEST_DATA)

    print("\n🔄 Step 3: Evaluating Performance...")

    examples = []
    for text, annotations in TEST_DATA:
        doc = nlp.make_doc(text)
        try:
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        except Exception:
            continue

    # Scoring
    scores = nlp.evaluate(examples)

    print("\n--------------------------------------------------")
    print("📊 EVALUATION RESULTS (KPIs)")
    print("--------------------------------------------------")

    precision = scores.get('ents_p', 0.0)
    recall = scores.get('ents_r', 0.0)
    f1_score = scores.get('ents_f', 0.0)

    print(f"🏆 Overall Precision : {precision:.2%}")
    print(f"🏆 Overall Recall    : {recall:.2%}")
    print(f"🏆 Overall F1-Score  : {f1_score:.2%}  <-- This is your main KPI")

    print("\n--------------------------------------------------")
    print("🔍 Breakdown by Entity Type:")

    per_ents = scores.get('ents_per_type', {})

    # Sort labels for cleaner output
    sorted_labels = sorted(per_ents.keys())

    if not sorted_labels:
        print("   (No entities detected)")

    for label in sorted_labels:
        metrics = per_ents[label]
        p = metrics.get('p', 0.0)
        r = metrics.get('r', 0.0)
        f = metrics.get('f', 0.0)
        print(f"   - {label:<6} | F1-Score: {f:.2%} (P: {p:.2f}, R: {r:.2f})")

    print("--------------------------------------------------")