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
├── agentic_judge_main.py           # Agentic LLM-Judge script
├── prompt_engineered_judge_main.py# Prompt-engineered LLM-Judge script
├── agent_tools.py                  # Custom Python tools used by agentic judge
├── evaluation_criteria.json        # Criteria and scoring rubrics
├── dataset/                        # Source and target translation test pairs
├── outputs/                        # JSON evaluations from both systems
├── paper/                          # Conference paper (IEEE/ACL format)
└── README.md                       # Project documentation
```

---

## 🧪 How to Run

1. **Install Requirements**

```bash
pip install -r requirements.txt
```

2. **Run Prompt-Engineered Judge**

```bash
python prompt_engineered_judge_main.py --input_file dataset/test_set.json
```

3. **Run Agentic Judge**

```bash
python agentic_judge_main.py --input_file dataset/test_set.json
```

4. **View Outputs**

Structured results (scores and explanations) will be saved in the `outputs/` directory as `.json` files.

---

## 📊 Evaluation Criteria

The LLM Judges evaluate translations on the following dimensions:

* **Accuracy**: Does the Filipino translation correctly convey the English source text’s meaning, intent, and details? (1 point)
* **Fluency**: Is the translation grammatically correct, natural, and idiomatic in Filipino? (1 point)
* **Coherence**: Does the translation maintain logical flow, context, and structure from the source? (1 point)
* **Cultural Appropriateness**: Does the translation respect Filipino cultural norms, idioms, and sensitivities (e.g., use of "po" and "opo" for respect, regional expressions)? (1 point)
* **Guideline Adherence**: Does the translation follow domain-specific style, terminology, or guidelines (e.g., legal terms, medical precision in Filipino)? (1 point)
* **Completeness**: Are all elements of the English source text translated into Filipino without omissions or additions? (1 point)
* **Scoring**: For a final score, individual criteria points are summed up and normalize to a 1-5 scale (e.g., 5-6 = 5, 3-4 = 3, 0-2 = 1) , paired with a label: "excellent" (5), "good" (3-4), "poor" (1-2).

Each is scored from 1 (poor) to 5 (excellent), with supporting justifications.

---

## 📈 Comparison & Insights

Our research paper includes:

* **Quantitative comparison**: Scores across criteria, consistency, and human alignment
* **Qualitative analysis**: Side-by-side output samples and breakdown of reasoning
* **Design reflections**: Challenges in memory, orchestration, and prompt strategy
* **Trade-offs**: Simplicity vs. modularity, interpretability vs. control

---

## 📚 Reference

Gu, J., Lee, S., Deng, Y., & Liang, P. (2025). *A Survey on LLM-as-a-Judge: Evaluation without Ground Truth.* [arXiv:2501.01234](https://arxiv.org/html/2411.15594v1).

---

## ✍️ Authors

* Alonzo Andrei G. Rimando
* Paul Ivan Enclonar
  (De La Salle University)

---

## 📝 License

This project is for academic purposes only. Not intended for production use.
