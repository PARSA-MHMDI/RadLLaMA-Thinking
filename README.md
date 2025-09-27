# Clinical Reasoning Enhancement for Chest X-ray Reporting

### Chain-of-Thought tuning of Large Vision-Language Models (RadLLaMA & RadLLaMA-Thinking)

> **TL;DR**: We fine-tune the LLaMA-3.2 Vision model for full chest X-ray report generation (Findings + Impression) in two stages: (1) large-scale domain tuning on MIMIC-CXR 2.1.0 + CheXpert Plus; (2) teacher–student **Chain-of-Thought (CoT)** distillation on a curated, 10-step reasoning dataset. The final **RadLLaMA-Thinking** model improves lexical, semantic, and clinical metrics (BLEU-4, ROUGE-L, METEOR, BERTScore, CheXbert-F1) under memory-efficient training (QLoRA, Unsloth).

---

## 📷 Cover Image

> *Add your hero/architecture/teaser image here.*

```
![RadLLaMA-Thinking Cover](PATH/TO/YOUR/COVER.png)
```

---

## 📌 Outline

* [Overview & Highlights](#overview--highlights)
* [Model Zoo](#model-zoo)
* [Quickstart: Inference Guide](#quickstart-inference-guide)

  * [GPU (4-bit, Transformers/bitsandbytes)](#gpu-4-bit-transformersbitsandbytes)
  * [CPU options](#cpu-options)
  * [Generate a full RSNA-style report](#generate-a-full-rsna-style-report)
  * [Generate with explicit 10-step CoT](#generate-with-explicit-10-step-cot)
* [Results (brief)](#results-brief)
* [Project Structure](#project-structure)
* [Citation](#citation)
* [Licenses & Data Access](#licenses--data-access)
* [Disclaimer](#disclaimer)

---

## Overview & Highlights

**RadLLaMA-Thinking** is a radiology-tuned multimodal LLM that generates complete chest X-ray clinical reports. It is trained in two stages:

1. **Large-scale domain tuning** on ~452k image–report pairs (MIMIC-CXR 2.1.0 + CheXpert Plus).
2. **CoT distillation** using a curated 300-sample dataset with **exactly 10 reasoning steps** prepended to the reference report, distilled from strong reasoning teachers.

**Headline numbers (MIMIC-CXR, report generation):**
BLEU-4 **0.143**, ROUGE-L **0.314**, METEOR **0.300**, BERTScore-F1 **0.443**, CheXbert-F1 **0.361**.

**Efficiency**: 4-bit QLoRA adapters (bitsandbytes) + Unsloth; 8-bit AdamW; trained on a single A100 40 GB.

---

## Model Zoo

> *Add your final Hugging Face links in the **Weights** column. Keep names consistent with your HF repos.*

| Model                           | Params | Quantization | Weights (HF) | Processor ID (HF)                               | Notes                             |
| ------------------------------- | :----: | :----------: | ------------ | ----------------------------------------------- | --------------------------------- |
| **LLaMA-3.2-Vision (Base)**     |   11B  |   FP16/BF16  | `[ADD LINK]` | e.g. `meta-llama/Llama-3.2-11B-Vision-Instruct` | Base vision-language model        |
| **RadLLaMA (Stage-1)**          |   11B  |  4-bit/FP16  | `[ADD LINK]` | `[ADD PROCESSOR ID]`                            | Domain-tuned on CXR               |
| **RadLLaMA-Thinking (Stage-2)** |   11B  |  4-bit/FP16  | `[ADD LINK]` | `[ADD PROCESSOR ID]`                            | CoT-distilled (10-step reasoning) |

> If you publish merged FP16 **and** 4-bit adapter-merged variants, list both links.

---

## Quickstart: Inference Guide

### Requirements

```bash
# Python 3.10+
pip install --upgrade "transformers>=4.44" accelerate safetensors pillow
# GPU path (recommended for 4-bit):
pip install bitsandbytes
# Optional: faster tokenization
pip install tiktoken
```

> **Note**: 4-bit/8-bit **bitsandbytes** quantization requires NVIDIA CUDA. On CPU-only environments, see [CPU options](#cpu-options).

### GPU (4-bit, Transformers/bitsandbytes)

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "YOUR_HF/RadLLaMA-Thinking"      # <-- replace with your HF repo
PROC_ID  = "YOUR_HF/RadLLaMA-Thinking"      # <-- often the same as model id

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

processor = AutoProcessor.from_pretrained(PROC_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

image = Image.open("path/to/cxr.png").convert("RGB")

# Minimal, no-CoT prompt (report only)
user_text = (
    "You are a board-certified radiologist. "
    "Generate a concise, RSNA/ACR-style report for this CXR with EXACTLY these "
    "headers: Technique, View, Findings, Impression."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": user_text},
        ],
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

# Strip the prompt part
gen = out[:, inputs["input_ids"].shape[-1]:]
text = processor.batch_decode(gen, skip_special_tokens=True)[0]
print(text)
```

### CPU options

* **Transformers on CPU (FP16 → float32)**: works but slower; omit `BitsAndBytesConfig` and load with `torch_dtype=torch.float32`, `device_map={"": "cpu"}`.
* **Highly recommended for CPU**: provide a **GGUF** export and run with [llama.cpp] or use **optimum-intel** for INT8 CPU inference. Add those links/instructions if you publish such artifacts.

Example (Transformers CPU fallback):

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch

MODEL_ID = "YOUR_HF/RadLLaMA-Thinking"
PROC_ID  = "YOUR_HF/RadLLaMA-Thinking"

processor = AutoProcessor.from_pretrained(PROC_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    trust_remote_code=True,
)

# ... (use the same messaging / generate code as above)
```

### Generate a full RSNA-style report

Use this when you **don’t** want the model to print internal reasoning, only the final report:

```
"You are a board-certified radiologist specialized in adult chest radiography.
Base all statements on the provided image only. Use EXACTLY these headers:
Technique, View, Findings, Impression. Handle uncertainty explicitly and avoid
hallucinations."
```

### Generate with explicit 10-step CoT

If you want the **reasoning steps** first (10 numbered steps, then the report), prompt like this:

```
"First, produce EXACTLY TEN (10) concise numbered reasoning steps grounded in the image.
Then generate the final clinical report with EXACTLY these headers:
Technique, View, Findings, Impression. No extra sections."
```

> ⚠️ Use CoT output responsibly. If you ship a product, consider hiding intermediate reasoning and only surfacing the final report.

---

## Results (brief)

* **CheXpert Plus (report generation)**: after CoT distillation, RadLLaMA-Thinking outperforms the base and stage-1 models across BLEU-4, ROUGE-L, METEOR, BERTScore, and CheXbert-F1.
* **MIMIC-CXR (report generation)**: RadLLaMA-Thinking achieves **BLEU-4 0.143**, **ROUGE-L 0.314**, **METEOR 0.300**, **BERTScore-F1 0.443**, **CheXbert-F1 0.361**.

> Full tables/plots are in the paper and the `results/` folder (add if you plan to commit them).

---

## Project Structure

```
.
├─ README.md
├─ env/                  # (optional) environment files
├─ scripts/
│  ├─ infer_transformers.py    # CLI inference (GPU/CPU)
│  └─ utils.py
├─ examples/
│  ├─ sample_cxr.png
│  └─ prompts/
│     ├─ rsna_report.txt
│     └─ cot_plus_report.txt
├─ weights/              # (optional) local weights or HF pointer
└─ results/              # (optional) metrics, figures
```

---

## Citation

If you use this repository, models, or datasets in your research, please cite:

```bibtex
@article{MohammadiSharifian2025RadLLaMAThinking,
  title   = {Clinical Reasoning Enhancement for Chest X-ray Reporting using Chain-of-Thought tuning of Large Vision Language Models},
  author  = {Parsa Mohammadi and Saeed Sharifian},
  journal = {Computers in Biology and Medicine},
  year    = {2025},
  note    = {Code and weights: https://github.com/PARSA-MHMDI/RadLLaMA-Thinking},
}
```

**Datasets** (please also cite when applicable):

```bibtex
@dataset{MIMIC_CXR_2_1_0,
  title   = {MIMIC-CXR (version 2.1.0)},
  author  = {Johnson, A. et al.},
  year    = {2019},
  url     = {https://physionet.org/content/mimic-cxr/2.1.0/}
}

@dataset{CheXpertPlus2024,
  title   = {CheXpert Plus},
  author  = {Chambon, P. et al.},
  year    = {2024},
  url     = {https://stanfordaimi.azurewebsites.net/datasets}
}
```

> Replace with your preferred canonical citations/DOIs once finalized.

---

## Licenses & Data Access

* **Models & Code**: add your chosen license (e.g., Apache-2.0).
* **MIMIC-CXR / CheXpert Plus**: require credentialed access and adherence to their data use agreements. **We do not redistribute images or PHI.**
* **Intended Use**: research and education.

---

## Disclaimer

This software and its models **do not** provide medical advice and **must not** be used for autonomous clinical decision making. Always involve a licensed radiologist for interpretation and reporting.

---

### Maintainers

* **Parsa Mohammadi** — Amirkabir University of Technology

For questions, please open a GitHub issue or discussion in this repo.
