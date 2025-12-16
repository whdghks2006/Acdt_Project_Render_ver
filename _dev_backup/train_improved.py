# -*- coding: utf-8 -*-
"""
NER Model Training Script (Improved Version)
- 6 Entity Labels: START_DATE, START_TIME, END_DATE, END_TIME, LOC, EVENT_TITLE
- Uses en_core_web_lg as base model
- Includes Train/Dev split and evaluation
"""

import spacy
import json
import random
import os
import warnings
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer

# ==============================================================================
# Configuration
# ==============================================================================

# Training data file path
TRAIN_FILE = "train_data.json"
DEV_FILE = "dev_data.json"

# Output directory for trained model
OUTPUT_DIR = "./output/new_ner_model"

# Number of training iterations
N_ITER = 30

# Note: Train/Dev split is done externally
# Train: train_data.json, Dev: dev_data.json, Test: test_data.json

# Dropout rate
DROPOUT = 0.35

# 6 Entity labels
LABELS = ["START_DATE", "START_TIME", "END_DATE", "END_TIME", "LOC", "EVENT_TITLE"]

# Base model (en_core_web_lg recommended)
BASE_MODEL = "en_core_web_lg"


# ==============================================================================
# Data Conversion Function (JSON dict -> spaCy format)
# ==============================================================================

def load_and_convert_data(filepath):
    """
    Convert final_training_data.json format to spaCy training format
    
    Input format:
    {
        "Text": "Meeting from 2 PM to 4 PM tomorrow.",
        "Date_Entity": "tomorrow",
        "Time_Entity": "2 PM",
        "End_Date_Entity": "",
        "End_Time_Entity": "4 PM",
        "Location_Entity": "",
        "Event_Entity": "Meeting"
    }
    
    Output format:
    ("Meeting from 2 PM to 4 PM tomorrow.", {"entities": [(0, 7, "EVENT_TITLE"), ...]})
    """
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"[INFO] Raw data loaded: {len(raw_data)} items")
    
    converted_data = []
    skipped_count = 0
    
    # Field to label mapping
    field_to_label = {
        "Date_Entity": "START_DATE",
        "Time_Entity": "START_TIME",
        "End_Date_Entity": "END_DATE",
        "End_Time_Entity": "END_TIME",
        "Location_Entity": "LOC",
        "Event_Entity": "EVENT_TITLE"
    }
    
    for item in raw_data:
        text = item.get("Text", "")
        if not text or not text.strip():
            skipped_count += 1
            continue
        
        entities = []
        
        for field, label in field_to_label.items():
            entity_value = item.get(field, "")
            
            # Handle non-string types (e.g., int, float)
            if entity_value is not None and entity_value != "":
                entity_value = str(entity_value).strip()
                
                if entity_value:
                    # Find entity position in text
                    start_idx = text.find(entity_value)
                    
                    if start_idx != -1:
                        end_idx = start_idx + len(entity_value)
                        entities.append((start_idx, end_idx, label))
        
        # Add only if at least one entity exists
        if entities:
            # Remove duplicates and overlaps
            entities = remove_overlapping_entities(entities)
            converted_data.append((text, {"entities": entities}))
        else:
            skipped_count += 1
    
    print(f"[INFO] Conversion complete: {len(converted_data)} items")
    print(f"[INFO] Skipped: {skipped_count} items (no text or no entities)")
    
    return converted_data


def remove_overlapping_entities(entities):
    """Remove overlapping entities (longer ones take priority)"""
    if not entities:
        return entities
    
    # Sort by start position, then by length (longer first)
    sorted_ents = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    
    result = [sorted_ents[0]]
    for start, end, label in sorted_ents[1:]:
        last_end = result[-1][1]
        if start >= last_end:  # No overlap, add it
            result.append((start, end, label))
    
    return result


# ==============================================================================
# Data Alignment Function (Align to token boundaries)
# ==============================================================================

def align_entities_to_tokens(nlp, data):
    """
    Align entity spans to token boundaries (using expand mode)
    """
    aligned_data = []
    fixed_count = 0
    skipped_count = 0
    
    for text, annotations in data:
        doc = nlp.make_doc(text)
        valid_ents = []
        
        if "entities" not in annotations:
            continue
        
        for start, end, label in annotations["entities"]:
            # Try strict alignment first
            span = doc.char_span(start, end, label=label)
            
            # If failed, try expand mode
            if span is None:
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if span is not None:
                    fixed_count += 1
            
            if span is not None:
                valid_ents.append((span.start_char, span.end_char, span.label_))
            else:
                skipped_count += 1
        
        if valid_ents:
            # Remove duplicates
            valid_ents = list(set(valid_ents))
            aligned_data.append((text, {"entities": valid_ents}))
    
    print(f"[INFO] Token alignment complete")
    print(f"   - Auto-fixed: {fixed_count}")
    print(f"   - Skipped: {skipped_count}")
    print(f"   - Final training data: {len(aligned_data)}")
    
    return aligned_data


# ==============================================================================
# Evaluation Function
# ==============================================================================

def evaluate_model(nlp, dev_data):
    """Evaluate model performance on dev data"""
    examples = []
    
    for text, annotations in dev_data:
        doc = nlp.make_doc(text)
        try:
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        except:
            continue
    
    if not examples:
        return {"p": 0, "r": 0, "f": 0}
    
    scorer = Scorer()
    scores = scorer.score(examples)
    
    # Extract NER scores
    ents_p = scores.get("ents_p", 0)
    ents_r = scores.get("ents_r", 0)
    ents_f = scores.get("ents_f", 0)
    
    return {"p": ents_p, "r": ents_r, "f": ents_f}


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    print("=" * 60)
    print("NER Model Training (Improved Version)")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Load and convert data
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading and converting data...")
    
    if not os.path.exists(TRAIN_FILE):
        print(f"[ERROR] File not found: {TRAIN_FILE}")
        exit()
    if not os.path.exists(DEV_FILE):
        print(f"[ERROR] File not found: {DEV_FILE}")
        exit()
    
    raw_train_data = load_and_convert_data(TRAIN_FILE)
    raw_dev_data = load_and_convert_data(DEV_FILE)
    
    # -------------------------------------------------------------------------
    # Step 2: Load base model
    # -------------------------------------------------------------------------
    print(f"\n[Step 2] Loading base model ({BASE_MODEL})...")
    
    try:
        nlp = spacy.load(BASE_MODEL)
        print(f"[INFO] '{BASE_MODEL}' loaded successfully")
    except OSError:
        print(f"[WARNING] '{BASE_MODEL}' not found. Download required:")
        print(f"   python -m spacy download {BASE_MODEL}")
        print(f"\n[TIP] Or modify BASE_MODEL variable to use 'en_core_web_sm'")
        exit()
    
    # -------------------------------------------------------------------------
    # Step 3: Align to token boundaries
    # -------------------------------------------------------------------------
    print("\n[Step 3] Aligning to token boundaries...")
    train_data = align_entities_to_tokens(nlp, raw_train_data)
    dev_data = align_entities_to_tokens(nlp, raw_dev_data)
    
    # -------------------------------------------------------------------------
    # Step 4: Data summary
    # -------------------------------------------------------------------------
    print("\n[Step 4] Data summary...")
    print(f"   - Train data: {len(train_data)}")
    print(f"   - Dev data: {len(dev_data)}")
    
    # -------------------------------------------------------------------------
    # Step 5: Configure NER pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 5] Configuring NER pipeline...")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Add 6 labels
    for label in LABELS:
        ner.add_label(label)
    
    print(f"   - Labels: {LABELS}")
    
    # -------------------------------------------------------------------------
    # Step 6: Start training
    # -------------------------------------------------------------------------
    print("\n[Step 6] Starting training...")
    print(f"   - Epochs: {N_ITER}")
    print(f"   - Dropout: {DROPOUT}")
    print("-" * 60)
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    best_f1 = 0
    best_epoch = 0
    
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        
        for epoch in range(N_ITER):
            random.shuffle(train_data)
            losses = {}
            
            # Batch training
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    try:
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    except:
                        continue
                
                if examples:
                    nlp.update(examples, drop=DROPOUT, losses=losses)
            
            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == N_ITER - 1:
                scores = evaluate_model(nlp, dev_data)
                f1 = scores["f"]
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_epoch = epoch + 1
                
                print(f"[Epoch {epoch + 1:2d}/{N_ITER}] Loss: {losses.get('ner', 0):8.2f} | "
                      f"P: {scores['p']:.1f}% R: {scores['r']:.1f}% F1: {scores['f']:.1f}%")
            else:
                print(f"[Epoch {epoch + 1:2d}/{N_ITER}] Loss: {losses.get('ner', 0):8.2f}")
    
    print("-" * 60)
    print(f"[BEST] F1: {best_f1:.1f}% (Epoch {best_epoch})")
    
    # -------------------------------------------------------------------------
    # Step 7: Save model
    # -------------------------------------------------------------------------
    print(f"\n[Step 7] Saving model...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    nlp.to_disk(OUTPUT_DIR)
    
    print(f"[INFO] Model saved to: {OUTPUT_DIR}")
    
    # -------------------------------------------------------------------------
    # Step 8: Test
    # -------------------------------------------------------------------------
    print("\n[Step 8] Testing...")
    
    test_sentences = [
        "Team meeting from 2 PM to 4 PM tomorrow at Conference Room A.",
        "The workshop runs from January 15th through January 20th.",
        "Dinner party at Sarah's house this Saturday at 7 PM.",
        "I'll be available between 10 AM and noon for calls.",
    ]
    
    print("-" * 60)
    for sentence in test_sentences:
        doc = nlp(sentence)
        print(f"\nInput: \"{sentence}\"")
        if doc.ents:
            for ent in doc.ents:
                print(f"   - {ent.label_}: \"{ent.text}\"")
        else:
            print("   - (No entities found)")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
