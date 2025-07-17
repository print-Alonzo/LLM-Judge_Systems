# NOTE: I ran it in Google Collab, cause hindi kaya ng device ko yung SEA-LION HAHAHAHAHAH

import json
import os
import csv
from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model selection: use SEA-LION 7B
MODEL_NAME = "aisingapore/Llama-SEA-LION-v3-8B-IT"

def load_model():
    """Load the SEA-LION 8B model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )
    # Create a deterministic text-generation pipeline
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False
    )
    return gen

# Initialize the generator
generator = load_model()

# Few-shot example list to guide the LLM judge 
FEW_SHOT_EXAMPLES = []
with open("examples.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        FEW_SHOT_EXAMPLES.append({
            'source': row.get('English'),
            'correct_translation': row.get('Filipino-Correct'),
            'flawed_translation': row.get('Filipino-Flawed'),
            'remarks': row.get('Remarks') or None
        })


PROMPT_TEMPLATE = """
You are an expert translation judge. You will be given a few examples of English→Filipino pairs—each with a correct translation, a flawed translation, and a short remark explaining the flaw. Study those examples, then evaluate a new pair according to six binary criteria.

Examples (from CSV columns “source”, “correct”, “flawed”, “remarks”):
{examples}

Now evaluate this pair:

Source: {source}  
Translation: {translation}

1. **Accuracy** (0–1): Does the Filipino version fully convey the English meaning?  
2. **Fluency** (0–1): Is the Filipino natural and grammatically correct?  
3. **Coherence** (0–1): Is the flow logical and clear?  
4. **Cultural Appropriateness** (0–1): Does it respect Filipino norms and idioms?  
5. **Guideline Adherence** (0–1): Does it follow any domain or style rules?  
6. **Completeness** (0–1): Are all parts of the source translated, with no additions?

Compute **total_points** = sum of the six criteria.  
Normalize to **overall_score** on a 1–5 scale:  
- 5–6 points → 5 (“excellent”)  
- 3–4 points → 3 (“good”)  
- 0–2 points → 1 (“poor”)

**Output only** a JSON object with these keys:
{{
  "criteria": {{
    "accuracy": 0 or 1,
    "fluency": 0 or 1,
    "coherence": 0 or 1,
    "cultural_appropriateness": 0 or 1,
    "guideline_adherence": 0 or 1,
    "completeness": 0 or 1
  }},
  "total_points": 0–6,
  "overall_score": 1–5,
  "label": "excellent" | "good" | "poor",
  "explanation": "A concise justification of each score in the criteria and the overall judgment."
}}
"""

def build_prompt(source: str, translation: str) -> str:
    # Insert few-shot examples
    example_strs = []
    for ex in FEW_SHOT_EXAMPLES:
        example_strs.append(
            f"Source: {ex['source']}\nCorrect Translation: {ex['correct_translation']}\nFlawed Translation: {ex['flawed_translation']}\nRemarks: {ex['remarks']}"
        )
    examples_block = "\n\n".join(example_strs)
    return PROMPT_TEMPLATE.format(
        examples=examples_block,
        source=source,
        translation=translation
    )


def evaluate_translation(source: str, translation: str) -> dict:
    prompt = build_prompt(source, translation)
    # Generate evaluation from SEA-LION
    output = generator(prompt)[0]["generated_text"]
    start = output.find("{")
    end = output.rfind("}") + 1
    output = output[start:end]
    content = output.strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"error": "Invalid JSON response", "raw": content}
    return result


output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Test a Prompt
source = "The cat sat on the mat."
translation = "Umupo ang pusa sa banig."
evaluation = evaluate_translation(source, translation)
record = {"source": source, "translation": translation, "evaluation": evaluation}
with open(os.path.join(output_dir, f"eval_1.json"), 'w', encoding='utf-8') as out_f:
        json.dump(record, out_f, ensure_ascii=False, indent=2)