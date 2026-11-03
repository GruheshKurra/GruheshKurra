# Gruhesh Sri Sai Karthik Kurra

MSc Computing (Artificial Intelligence and Machine Learning), Imperial College London · London

[Website](https://gruheshkurra.com) · [CV](https://gruheshkurra.com/assets/Resume.pdf) · [Google Scholar](https://scholar.google.com/citations?user=dwY966oAAAAJ) · [ORCID](https://orcid.org/0009-0002-0558-2882) · [LinkedIn](https://www.linkedin.com/in/gruheshkurra/) · [Email](mailto:gk326@ic.ac.uk)

I'm an MSc student at Imperial, working on large language models and NLP. My papers so far ask three questions. How should a model represent a word it has never seen? How can one prompt give several different answers instead of one? How can I store a network in much less memory without losing quality?

Before Imperial I studied Computer Science and Engineering at KL University, Hyderabad. I was also a research intern at IIIT Hyderabad.

I'm looking for funded PhD positions in language models and NLP, starting October 2027. Email me at [gk326@ic.ac.uk](mailto:gk326@ic.ac.uk).

## News

- **Oct 2026:** Version 2 of [BIE](https://doi.org/10.5281/zenodo.17217217) is on Zenodo, with a new storage format and a full measured evaluation.
- **Sep 2026:** Started the MSc at Imperial.
- **Jul 2026:** My team won first place at Imperial's Global Challenge Lab, a 14-day sprint with over 1,000 students. I led the AI work on OrbitOps.
- **2026:** Dual-Stream and Prism Tuning accepted at IEEE NEPCON 2026.
- **2025:** MANCE published at IEEE ICCCMLA 2025.

## Papers

**[Morphology-Aware Nested Character Embeddings for Word Representation](https://ieeexplore.ieee.org/document/11580466)**
Gruhesh Sri Sai Karthik Kurra, Chandanasree Moparthi, Prathipati Manish Chowdary, Pavan Kumar Pagadala
*IEEE ICCCMLA 2025, published in IEEE Xplore*

A small Transformer builds each word vector from the word's characters. So *play* and *plays* share most of their computation, and a word the model has never seen still gets a vector. In my re-implementation (3 seeds) it reaches 63.0 F1 on GermEval 2014 German NER, against 48.2 for word embeddings with the same encoder. It matches them on AG News and DBpedia.
[Paper](https://ieeexplore.ieee.org/document/11580466) · [Code](https://github.com/GruheshKurra/MANCE-NLP) · [Results](https://github.com/GruheshKurra/MANCE-NLP/blob/main/experiments/RESULTS.md)

**[Dual-Stream Artifact Detection with Iterative Evidence Refinement for Frame-Level Deepfake Recognition](https://github.com/GruheshKurra/radar_deepfake)**
Gruhesh Sri Sai Karthik Kurra, Pavan Kumar Pagadala, Malathy Batumalay, Veda Boddapati
*Accepted, IEEE NEPCON 2026 (presentation December 2026)*

A ViT-Small detector with two streams. One looks at edges for blending boundaries, the other at frequencies for generator fingerprints. Iterative cross-attention combines them. An audit showed that 97.3% of WildDeepfake test frames came from videos also used in training. So the paper reports a video-disjoint subset of 5,680 frames: 96.3% AUC overall and 87.1% on Celeb-DF, from a single training run.
[Code](https://github.com/GruheshKurra/radar_deepfake) · [Dataset](https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames)

**Prism Tuning: Repulsion-Trained Seed Embeddings in a Frozen Transformer for Non-Redundant Generation**
Gruhesh Sri Sai Karthik Kurra, Pavan Kumar Pagadala, Malathy Batumalay, Sushma Reddy Koduru
*Accepted, IEEE NEPCON 2026 (presentation December 2026)*

I add K seed vectors to the input embeddings of a frozen transformer and train only those, with a repulsion loss. One prompt then gives K different outputs in one batched pass, with no sampling and filtering at inference. The paper gives the method and its formulation.
Follow-up (Oct 2026): I redesigned it and tested it in pre-registered GPU experiments on Qwen2.5. It did not significantly beat tuned min-p sampling. The main suspected cause is the training data.

**[BIE: Bit-Index Encoding of Sparse Neural Network Weights](https://doi.org/10.5281/zenodo.17217217)**
Gruhesh Sri Sai Karthik Kurra
*Preprint on Zenodo, version 2, October 2026*

A storage format for pruned weight matrices. The gaps between non-zero weights are Golomb-Rice coded, and the values are 8-bit codes stored as bitplanes. A row-parallel kernel multiplies directly on the compressed form. At 95% sparsity it is 44.5x smaller than float32 at 0.6% error, against 25.7x for CSR with the same values. Below about 50% sparsity it is no smaller than dense int8.
[Paper](https://doi.org/10.5281/zenodo.17217217) · [Code](https://github.com/GruheshKurra/bit-index-encoding-research-) · [Results](https://github.com/GruheshKurra/bit-index-encoding-research-/blob/main/experiments/RESULTS.md)

The MANCE and BIE repositories include the configs, seeds and per-run result files behind these numbers, so you can re-run each one.

## Experience

**AI Integration Engineer**, InfoAjax Consulting · Oct to Nov 2025, remote
I built a proof-of-concept browser agent that acts on screenshots, using models on Azure AI Foundry. I also built a React and FastAPI console to watch agents live.

**Research Intern**, IIIT Hyderabad · Dec 2024 to Jun 2025
I built a document translation system for English, Hindi, Telugu and German that keeps the original page layout. It uses Vision Transformer layout detection, font detection, colour extraction and Tesseract OCR. I demonstrated it at the institute's research expo in May 2025 ([demo](https://youtu.be/rcvSuBcBjyg)).

**Software Engineer (freelance)**, InfoAjax Consulting · Oct 2024 to Jun 2025, remote
I built REST APIs in WSO2 Micro Integrator that connect the service-desk, billing and inventory systems of a Philippine telecoms company. I supported them in production.

## Education and awards

- **Imperial College London**, MSc Computing (AI and Machine Learning), 2026 to 2027
- **KL University, Hyderabad**, B.Tech Computer Science and Engineering, 2021 to 2025, CGPA 9.72 / 10
- **First place**, Global Challenge Lab 2026, Imperial College London. I led the AI and ML work on OrbitOps, a reliability layer for AI running on satellites.
- **First place**, University Webathon, KL University, 2022
- **Technical Lead**, Student Activity Council, KL University, 2023 to 2025

## Writing

I write [AI from Scratch](https://blogs.gruheshkurra.com/series/ai/), a blog series. Each post takes one model from its derivation to a working implementation. Recent posts:

<!-- BLOG-POST-LIST:START --><a href="https://blogs.gruheshkurra.com/blog/colibri-llm-engine-explained/">Colibri Explained: How a 744B LLM Runs From Disk</a><br/><a href="https://blogs.gruheshkurra.com/blog/convolutional-neural-networks-explained/">Convolutional Neural Networks: CNN Math Explained</a><br/><a href="https://blogs.gruheshkurra.com/blog/deepseek-v4-inside-one-token/">DeepSeek V4 Inside: One Token Through Every Block</a><br/><a href="https://blogs.gruheshkurra.com/blog/looped-transformers-explained/">Looped Transformers Explained: Recurrent Depth and Astra</a><br/><a href="https://blogs.gruheshkurra.com/blog/natural-language-inference-explained/">Natural Language Inference Explained: Entailment in NLP</a><br/><!-- BLOG-POST-LIST:END -->

[All posts](https://blogs.gruheshkurra.com/ai-explanations/) · [RSS](https://blogs.gruheshkurra.com/feed.xml)
