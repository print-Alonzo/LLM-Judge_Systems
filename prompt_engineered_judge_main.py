from sklearn.metrics import mean_squared_error
import json
import os
import csv
from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import pandas as pd
import numpy as np
from openai import OpenAI
import time
from itertools import islice
import json


"""The Prompt Template used for Evaluation"""

PROMPT_TEMPLATE = """
You are a rigorous, impartial English→Filipino translation judge with deep expertise in Filipino grammar, style, and cultural nuance. Evaluate ONLY the given source/translation using the criteria below. Favor idiomatic Filipino that preserves meaning. Penalize omissions/additions, mistranslations (polarity/negation, tense/aspect, quantities, named entities), awkward calques, unjustified Taglish, and register mismatches. Do NOT rewrite the translation—only judge it. When uncertain, choose the lower score and justify briefly with evidence. Assume formal register unless stated otherwise. If no domain/style guide is provided, use general editorial norms as the guideline.

SCORED REFERENCE EXAMPLES (for patterning; do NOT output these):

- Example A — Excellent
  Source: "The meeting was postponed because of the storm."
  Translation: "Naantala ang pagpupulong dahil sa bagyo."
  Expected JSON:
  {{"criteria": {{"accuracy": 1, "fluency": 1, "coherence": 1, "cultural_appropriateness": 1, "guideline_adherence": 1, "completeness": 1}},
    "total_points": 6, "overall_score": 5, "label": "excellent",
    "explanation": "Idiomatic and precise; preserves cause and entities; no omissions/additions."}}

- Example B — Very good (minor style issue)
  Source: "Please submit the report by Friday."
  Translation: "Pakiusap na isumite ang ulat pagsapit ng Biyernes."
  Expected JSON:
  {{"criteria": {{"accuracy": 1, "fluency": 1, "coherence": 1, "cultural_appropriateness": 1, "guideline_adherence": 1, "completeness": 0}},
    "total_points": 5, "overall_score": 4, "label": "very_good",
    "explanation": "Meaning preserved; minor completeness/style nuance (tone/softener not fully mirrored)."}}

- Example C — Good (loss of specificity)
  Source: "Do not turn off the main power switch."
  Translation: "Huwag patayin ang switch."
  Expected JSON:
  {{"criteria": {{"accuracy": 0, "fluency": 1, "coherence": 1, "cultural_appropriateness": 1, "guideline_adherence": 1, "completeness": 0}},
    "total_points": 4, "overall_score": 3, "label": "good",
    "explanation": "Omits 'main power' → specificity lost (accuracy/completeness↓); grammar/flow are fine."}}

- Example D — Fair (noticeable errors, mostly understandable)
  Source: "Store the medicine in a cool, dry place."
  Translation: "Itago ang gamot sa malamig na lugar."
  Expected JSON:
  {{"criteria": {{"accuracy": 1, "fluency": 1, "coherence": 1, "cultural_appropriateness": 1, "guideline_adherence": 0, "completeness": 0}},
    "total_points": 4, "overall_score": 3, "label": "good",
    "explanation": "Misses 'dry' and guidance nuance; otherwise natural. (If policy requires both conditions, consider Completeness=0 and Guideline=0.)"}}

- Example E — Poor (wrong meaning)
  Source: "Keep out of reach of children."
  Translation: "Maganda ang bata."
  Expected JSON:
  {{"criteria": {{"accuracy": 0, "fluency": 1, "coherence": 0, "cultural_appropriateness": 0, "guideline_adherence": 0, "completeness": 0}},
    "total_points": 1, "overall_score": 1, "label": "poor",
    "explanation": "Unrelated meaning; safety directive lost; incoherent to instruction context."}}

Additionally, you will also be given a few examples of English→Filipino pairs—each with a correct translation, a flawed translation, and a short remark explaining the flaw. Study those examples, then evaluate a new pair according to six binary criteria.

Examples (from CSV columns “source”, “correct”, “flawed”, “remarks”):
{examples}

Now evaluate this pair ONLY:

Source: {source}
Translation: {translation}

Scoring rubric (binary 0/1 for each):
1) Accuracy — Meaning preserved (entities, polarity, tense/aspect, quantities, conditions).
2) Fluency — Natural, grammatical Filipino (orthography, morphology, agreement).
3) Coherence — Logical flow; clear referents/connectors; consistent register.
4) Cultural Appropriateness — Idiomatic usage; avoids unjustified Taglish/calques; suitable register.
5) Guideline Adherence — Follows stated domain/style rules (or general editorial norms if none provided).
6) Completeness — No omissions/additions; all content rendered faithfully.

Hard rules:
- Critical meaning error (e.g., negation flip, wrong entity) → Accuracy=0.
- Major omission/addition → Completeness=0 (and Accuracy=0 if meaning affected).
- Pervasive unjustified Taglish/calques in formal context → Fluency=0 (and possibly Cultural=0).

Scoring aggregation:
- Compute total_points = sum of the six criteria (0–6).
- Map to overall_score (integer 1–5):
  0–1 → 1 (“poor”)
  2   → 2 (“fair”)
  3–4 → 3 (“good”)
  5   → 4 (“very_good”)
  6   → 5 (“excellent”)
- Label must match overall_score exactly:
  1→"poor", 2→"fair", 3→"good", 4→"very_good", 5→"excellent".

VALIDATION CHECKS (must hold):
- total_points == accuracy+fluency+coherence+cultural_appropriateness+guideline_adherence+completeness
- overall_score and label match the mapping above.
- Use integers only (0/1 for criteria; 1–5 for overall_score). No extra keys.

OUTPUT FORMAT — return JSON ONLY (no prose/backticks). Exactly this schema:
{{"criteria": {{"accuracy": 0 or 1, "fluency": 0 or 1, "coherence": 0 or 1, "cultural_appropriateness": 0 or 1, "guideline_adherence": 0 or 1, "completeness": 0 or 1}},
  "total_points": integer 0-6,
  "overall_score": integer 1-5,
  "label": "poor"|"fair"|"good"|"very_good"|"excellent",
  "explanation": "≤120 words; brief evidence for each criterion"}}
"""



"""Load the model"""

client = OpenAI(
    api_key="AIzaSyAzrz2ZORUKReonxTJ0tEIHR_IXDWRBqPY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def load_model():
    """Return a callable that mimics HF pipeline behavior using Groq chat API."""
    model_name = "gemini-2.5-flash-lite"

    def gen(inputs, max_new_tokens=512, do_sample=True, temperature=0.3, top_p=0.9, **kwargs):
        # Accept str or list[str]
        prompts = [inputs] if isinstance(inputs, str) else list(inputs)
        outputs = []
        for prompt in prompts:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=(temperature if do_sample else 0.0),
                top_p=(top_p if do_sample else 1.0),
                n=1,
                stream=False
            )
            text = resp.choices[0].message.content or ""
            outputs.append({"generated_text": text})
        return outputs

    return gen



"""Populating `FEW_SHOT_EXAMPLES` with the entries from the training dataset"""

def load_few_shot_examples(path: str):
    FEW_SHOT_EXAMPLES = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in islice(reader, 5):
            FEW_SHOT_EXAMPLES.append({
                'source': row.get('English'),
                'correct_translation': row.get('Filipino-Correct'),
                'flawed_translation': row.get('Filipino-Flawed'),
                'remarks': row.get('Remarks') or None
            })
    
    return FEW_SHOT_EXAMPLES



"""Defining the necessary functions for building the prompts and generating the evaluation from the Prompt-Engineered LLM Judge"""

def build_prompt(source: str, translation: str, examples: list) -> str:
    # Insert few-shot examples
    example_strs = []
    for ex in examples:
        example_strs.append(
            f"Source: {ex['source']}\nCorrect Translation: {ex['correct_translation']}\nFlawed Translation: {ex['flawed_translation']}\nRemarks: {ex['remarks']}"
        )
    examples_block = "\n\n".join(example_strs)
    return PROMPT_TEMPLATE.format(
        examples=examples_block,
        source=source,
        translation=translation
    )

def evaluate_translation(source: str, translation: str, examples: list, retries: int = 2, retry_delay: float = 0.0) -> dict:
    prompt = build_prompt(source, translation, examples)

    def _clean(out: str) -> str:
        # Prefer fenced JSON
        low = out.lower()
        if "```json" in low:
            out = out.split("```json", 1)[1]
            out = out.split("```", 1)[0]
        elif "```" in out:
            out = out.split("```", 1)[1]
            out = out.split("```", 1)[0]
        # Fallback: trim to first {...} block if present
        if "{" in out and "}" in out:
            s, e = out.find("{"), out.rfind("}")
            if s < e:
                out = out[s:e+1]
        # Keep your original cleanup
        out = out.replace('\\n', '').replace('\\', '')
        return out.strip()

    last_raw = None
    for attempt in range(retries + 1):
        # Generate evaluation
        raw = generator(prompt)[0]["generated_text"]
        last_raw = raw
        content = _clean(raw)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if attempt < retries:
                if retry_delay > 0:
                    time.sleep(retry_delay)
                continue
            # Final fallback after retries
            return {"error": "Invalid JSON response after retries", "raw": content, "raw_full": last_raw}



"""Testing the LLM on the Validation Set"""

def load_validation(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["Source Text (English)", "Target Text (Filipino)", "Final Score"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Validation CSV missing column: {c}")
    return df


def evaluate_in_validation_set(validation_set: pd.DataFrame, examples: list):
    record = []
    for i, row in validation_set.iterrows():
        source = row["Source Text (English)"]
        translation = row["Target Text (Filipino)"]
        evaluation = evaluate_translation(source, translation, examples)
        record.append({"source": source, "translation": translation, "evaluation": evaluation})
        time.sleep(5)

    res = pd.json_normalize(record)  # flattens nested dicts
    # res columns include: 'source', 'translation', 'evaluation.overall_score', ...
    overall_scores = pd.to_numeric(res['evaluation.overall_score'], errors='coerce')
    human_scores   = pd.to_numeric(validation_set['Final Score'], errors='coerce')

    # Pairwise valid rows
    mask = overall_scores.notna() & human_scores.notna()

    m = overall_scores[mask]
    h = human_scores[mask]

    mse = mean_squared_error(h, m)

    print(f"MSE:  {mse:.4f}")


def consistency_check(validation_set: pd.DataFrame, examples: list):
    # Consistency Check
    rows_to_be_checked = validation_set.head(3)
    consistency_check = []

    for i, row in rows_to_be_checked.iterrows():
        temp = []
        for j in range(5):
            source = row["Source Text (English)"]
            translation = row["Target Text (Filipino)"]
            evaluation = evaluate_translation(source, translation, examples)
            temp.append(evaluation['overall_score'])
            time.sleep(5)

        s = pd.to_numeric(pd.Series(temp), errors="coerce")
        stats = {
            "source": row["Source Text (English)"],
            "translation": row["Target Text (Filipino)"],
            "run1": s.iloc[0] if len(s) > 0 else np.nan,
            "run2": s.iloc[1] if len(s) > 1 else np.nan,
            "run3": s.iloc[2] if len(s) > 2 else np.nan,
            "run4": s.iloc[3] if len(s) > 3 else np.nan,
            "run5": s.iloc[4] if len(s) > 4 else np.nan,
            "mean": s.mean(),
            "std": s.std(),
            "consistency": 1-s.std()
        }
        consistency_check.append(stats)

    consistency_df = pd.DataFrame(consistency_check)
    print(consistency_df)





    
if __name__ == "__main__":
    # Initialize the generator
    generator = load_model()
    
    # Initialize the output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Intialize the few shot examples
    examples = load_few_shot_examples("./dataset/knowledge.csv")
    
    # Filename index
    index = 0
    
    print("Enter 'exit' as the source text to quit.")
    while True:
        source_text = input("\nEnter the source text (English): ").strip()
        if source_text.lower() == "exit":
            print("Exiting...")
            break

        translated_text = input("Enter the translation text (Filipino): ").strip()
        if translated_text.lower() == "exit":
            print("Exiting...")
            break

        print("\nEvaluating translation... Please wait.\n")
        try:
            response = evaluate_translation(source_text, translated_text, examples)
            edited_response = {
                "source": source_text,
                "translation": translated_text,
                "evaluation": response
            }
            
            filename = f"prompt_engineered_evaluation_{index}.json"
            out_path = f"./{output_dir}/{filename}"
            
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(edited_response, f, ensure_ascii=False, indent=2)
            
            print("--- EVALUATION RESULT ---")
            print(json.dumps(edited_response, ensure_ascii=False, indent=2))
            print(f"\n✅ Saved JSON to: /{output_dir}/{filename}")
            
            index = index + 1
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
    
    
    # Evaluating the LLM Judge
    # validation_set = load_validation("./dataset/validation.csv")
    
    # evaluate_in_validation_set(validation_set, examples)
    
    # consistency_check(validation_set, examples)