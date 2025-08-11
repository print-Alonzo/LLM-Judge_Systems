# Designing and Comparing LLM-as-a-Judge Systems for English-to-Filipino Translation

This project investigates how Large Language Models (LLMs) can serve as automated translation judges, specifically evaluating English-to-Filipino translations. We design and compare two evaluation systems:

* **Prompt-Engineered LLM-Judge**: A single-prompt LLM guided by structured instructions.
* **Agentic LLM-Judge**: An advanced system with modular memory, reasoning, and tool use, inspired by agentic architectures.

The goal is to explore the strengths and limitations of each approach, particularly in reasoning, explainability, and alignment with human judgment.

---

## Features

* **Translation Quality Assessment**: Scores translations across multiple custom-defined criteria (e.g., accuracy, fluency).
* **Structured Outputs**: Each judge provides a JSON with:
  * Overall score
  * Per-criterion scores
  * Qualitative justifications
  * Internal "thought process" (for Agentic Judge)
  
* **Agentic Capabilities**:
  * Thought → Action → Observation loops
  * Custom tool integration (e.g., similarity checkers, syntax analyzers)
  * Reflection and planning steps for thorough evaluation

---

## 📁 Repository Structure

```
📦 llm-judge_systems
├── agentic_judge_main.py                     # Agentic LLM-Judge script
├── prompt_engineered_judge_main.py           # Prompt-engineered LLM-Judge script
├── agentic_judge_main_experiments.ipynb      # Jupyter Notebook for the main experiments with the Agentic LLM
├── dataset/                                  # Contains the dataset used for few-shot prompting and evaluation
    ├── knowledge.csv                         # Dataset used for few-shot prompting
    ├── validation.csv                        # Dataset used for model evaluation
├── outputs/                                  # Contains the output JSON files from demonstration
    ├── agentic_ai_evaluation_i.json          # ith JSON Output file from the agentic LLM judge
    ├── prompt_engineered_evaluation_i.json   # ith JSON Output file from the prompt-engineered LLM judge
├── requirements.txt                          # Contains the dependencies
├── README.md                                 # Project documentation
```

---

## 🧪 How to Run

1. **Install Requirements**

```bash
pip install -r requirements.txt
```

2. **Run Prompt-Engineered Judge**

```bash
python prompt_engineered_judge_main.py
```

3. **Run Agentic Judge**

```bash
python agentic_judge_main.py
```

---

## 📊 Evaluation Criteria

The LLM Judges evaluate translations on the following dimensions:

* **Accuracy**: Does the Filipino translation correctly convey the English source text’s meaning, intent, and details? (1 point)
* **Fluency**: Is the translation grammatically correct, natural, and idiomatic in Filipino? (1 point)
* **Coherence**: Does the translation maintain logical flow, context, and structure from the source? (1 point)
* **Cultural Appropriateness**: Does the translation respect Filipino cultural norms, idioms, and sensitivities (e.g., use of "po" and "opo" for respect, regional expressions)? (1 point)
* **Guideline Adherence**: Does the translation follow domain-specific style, terminology, or guidelines (e.g., legal terms, medical precision in Filipino)? (1 point)
* **Completeness**: Are all elements of the English source text translated into Filipino without omissions or additions? (1 point)
* **Scoring**: For a final score, individual criteria points are summed up and normalize to a 1-5 scale.

Each is scored from 1 (poor) to 5 (excellent), with supporting justifications.

---

## ✍️ Authors

* Alonzo Andrei G. Rimando
* Paul Ivan Enclonar
  (De La Salle University)

---

## 📝 License

This project is for academic purposes only. Not intended for production use.
