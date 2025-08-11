from typing import Literal, Optional
from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
import re
import time

load_dotenv()

class TranslationEvaluation(BaseModel):
    """The final evaluation of the English-to-Filipino translation."""
    score: int = Field(..., description="Numerical score from 1 (poor) to 5 (perfect).")
    label: Literal["Incomprehensible", "Poor", "Good", "Excellent", "Perfect"] = Field(
        ..., description="Categorical label for the translation quality."
    )
    reasoning: str = Field(
        ..., description="Detailed, point-by-point reasoning for the score, citing specific examples from the text."
    )

search_tool = TavilySearchResults(k=1, tavily_api_key="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")
search_tool.description = (
    "Use this to search for definitions, synonyms, or cultural context of specific English or Filipino words and phrases. Mention the word TRANSLATE so that tavily knows that your looking for the counterpart of that word like \"Translate 'food' in Filino\"."
)

@tool
def opinion_pooling_tool(source_text: str, translated_text: str, reference_text: Optional[str] = None) -> str:
    """
    Use this ONLY as a last resort if the search tool did not clarify your uncertainty. 
    This tool consults other expert AI models (Gemini and GPT-4) for their evaluations. It is very expensive.
    """
    print("\n--- CONSULTING EXPENSIVE OPINION POOLING TOOL ---")
    
    # judge models
    gemini_judge = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2, api_key="AIzaSyAzrz2ZORUKReonxTJ0tEIHR_IXDWRBqPY")
    chatgpt_judge = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key="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") # Using a different GPT model

    judge_prompt = ChatPromptTemplate.from_template(
        """You are a rigorous, impartial English→Filipino translation judge with deep expertise in Filipino grammar, style, and cultural nuance. Evaluate ONLY the given source/translation using the criteria below. Favor idiomatic Filipino that preserves meaning. Penalize omissions/additions, mistranslations (polarity/negation, tense/aspect, quantities, named entities), awkward calques, unjustified Taglish, and register mismatches. Do NOT rewrite the translation—only judge it. When uncertain, choose the lower score and justify briefly with evidence. Assume formal register unless stated otherwise. If no domain/style guide is provided, use general editorial norms as the guideline.

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

Examples (from CSV columns “source”, “correct”, “flawed”, separated by |):
	1.	The Philippines is an archipelago made up of over 7,640 islands, though only about 2,000 are inhabited. | Ang Pilipinas ay isang kapulaang binubuo ng 7,640 na isla, ngunit 2,000 lamang ang tinitirahan | Ang Pilipinas ay isang puno na binubuo ng mahigit 7,640 manok, bagaman halos 2,000 lamang ang tumira.
	2.	Philippines was also a U.S. territory from 1898 to 1946. | Ang Pilipinas ay naging isang teritoryo rin ng Estados Unidos mula 1898 hanggang 1946 | Ang Estados Unidos ay naging isang teritoryo ng Pilipinas mula 1946 hanggang 1898
	3.	The national hero of the Philippines is Dr. Jose Rizal. | Si Dr. Jose Rizal ang pambansang bayani ng Pilipinas | Ang Pilipinas ang bansang bayani ni Dr. Jose Rizal
	4.	The national animal of the Philippines is the Carabao. | Ang pambansang hayop ng Pilipinas ay ang kalabaw | Ang pambansang hayop ng Pilipinas ay ang aso
	5.	The national bird is the Philippine Eagle, one of the largest and most powerful eagles in the world. | Ang pambansang ibon ay ang Philippine Eagle, isa sa pinakamalaki at pinakamalakas na agila sa mundo | Ang karaniwang ibon na Philippine Eagle ay isang maliit na Agila
    

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
    )
    
    parser = StrOutputParser()
    gemini_chain = judge_prompt | gemini_judge | parser
    chatgpt_chain = judge_prompt | chatgpt_judge | parser
    
    input_data = {
        "source": source_text,
        "translation": translated_text,
        "reference": reference_text or "N/A"
    }

    gemini_opinion = ""
    chatgpt_opinion = ""

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_gemini = executor.submit(gemini_chain.invoke, input_data)
        future_chatgpt = executor.submit(chatgpt_chain.invoke, input_data)
        
        try:
            print("...getting opinion from Gemini...")
            gemini_opinion = future_gemini.result()
            print("...getting opinion from GPT-4...")
            chatgpt_opinion = future_chatgpt.result()
        except Exception as e:
            return f"An error occurred while consulting models: {e}"

    return f"""Consultation results:
- Opinion from Gemini-2.5-Pro:
{gemini_opinion}

- Opinion from GPT-4:
{chatgpt_opinion}
"""

tools = [search_tool, opinion_pooling_tool]

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a rigorous, impartial English→Filipino translation judge with deep expertise in Filipino grammar, style, and cultural nuance. Evaluate ONLY the given source/translation using the criteria below. Favor idiomatic Filipino that preserves meaning. Penalize omissions/additions, mistranslations (polarity/negation, tense/aspect, quantities, named entities), awkward calques, unjustified Taglish, and register mismatches. Do NOT rewrite the translation—only judge it. When uncertain, choose the lower score and justify briefly with evidence. Assume formal register unless stated otherwise. If no domain/style guide is provided, use general editorial norms as the guideline.

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

Examples (from CSV columns “source”, “correct”, “flawed”, separated by |):
	1.	The Philippines is an archipelago made up of over 7,640 islands, though only about 2,000 are inhabited. | Ang Pilipinas ay isang kapulaang binubuo ng 7,640 na isla, ngunit 2,000 lamang ang tinitirahan | Ang Pilipinas ay isang puno na binubuo ng mahigit 7,640 manok, bagaman halos 2,000 lamang ang tumira.
	2.	Philippines was also a U.S. territory from 1898 to 1946. | Ang Pilipinas ay naging isang teritoryo rin ng Estados Unidos mula 1898 hanggang 1946 | Ang Estados Unidos ay naging isang teritoryo ng Pilipinas mula 1946 hanggang 1898
	3.	The national hero of the Philippines is Dr. Jose Rizal. | Si Dr. Jose Rizal ang pambansang bayani ng Pilipinas | Ang Pilipinas ang bansang bayani ni Dr. Jose Rizal
	4.	The national animal of the Philippines is the Carabao. | Ang pambansang hayop ng Pilipinas ay ang kalabaw | Ang pambansang hayop ng Pilipinas ay ang aso
	5.	The national bird is the Philippine Eagle, one of the largest and most powerful eagles in the world. | Ang pambansang ibon ay ang Philippine Eagle, isa sa pinakamalaki at pinakamalakas na agila sa mundo | Ang karaniwang ibon na Philippine Eagle ay isang maliit na Agila
    

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

You can use tools like Tavily Search to clarify uncertainties about specific words or phrases, but do not use them for general translation help. If you are still uncertain after using the search tool, you can use the opinion pooling tool to consult other AI models for their evaluations, Youre final answer must be the average of the two models' scores if ever you use the opinion pooling tool.

Please reason before answering like why thats your score for the criteria. After your done type your final answer by typing 'FINAL:' followed by your answer in the following JSON format schema:{{"criteria": {{"accuracy": 0 or 1, "fluency": 0 or 1, "coherence": 0 or 1, "cultural_appropriateness": 0 or 1, "guideline_adherence": 0 or 1, "completeness": 0 or 1}},
  "total_points": integer 0-6,
  "overall_score": integer 1-5,
  "label": "poor"|"fair"|"good"|"very_good"|"excellent",
  "explanation": "≤120 words; brief evidence for each criterion"}}
""",
        ),
        (
            "human",
            """NOW Please evaluate the following translation.

**Source:**
{source_text}

**Translation (Filipino):**
{translated_text}
""",
        ),
        ("ai", "{agent_scratchpad}"), # Where the agent keeps its intermediate work (thoughts, tool calls)
    ]
)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1, api_key="AIzaSyAzrz2ZORUKReonxTJ0tEIHR_IXDWRBqPY")

agent = create_openai_tools_agent(llm, tools, prompt_template)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


def extract_json_from_text(text: str):
    # Regex: match a { ... } block, including nested braces
    match = re.search(r'FINAL:\s*({.*?})', text, re.DOTALL)
    if not match:
        match = re.search(r'({.*})', text, re.DOTALL)
        if not match:
            raise ValueError("No 'FINAL:' marker or JSON object found in the input string.")
    
    json_str = match.group(1)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON found. Error: {e}. String was: {json_str}")


if __name__ == "__main__":
    # Initialize the output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
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
            response = agent_executor.invoke({
                "source_text": source_text,
                "translated_text": translated_text,
            })
            
            print("--- EVALUATION RESULT ---")
            
            extracted_json = extract_json_from_text(response.get("output", response))
            
            json_out = {
              "source": source_text,
              "translation": translated_text,
              "evaluation": extracted_json
            }
            
            filename = f"agentic_ai_evaluation_{index}.json"
            out_path = f"./{output_dir}/{filename}"
            
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(json_out, f, ensure_ascii=False, indent=2)
            
            print(json.dumps(json_out, ensure_ascii=False, indent=2))
            print(f"\n✅ Saved JSON to: /{output_dir}/{filename}")
            
            index = index + 1
            
        except ValueError as e:
          print(f"An error occurred in parsing the JSON")
          print("Printing Output Instead")
          print(response.get("output", response))
                