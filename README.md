# Gruhesh Sri Sai Karthik Kurra

MSc Computing (Artificial Intelligence and Machine Learning) · Imperial College London

[Website](https://gruheshkurra.com) · [Google Scholar](https://scholar.google.com/citations?user=dwY966oAAAAJ) · [ORCID](https://orcid.org/0009-0002-0558-2882) · [LinkedIn](https://www.linkedin.com/in/gruheshkurra/) · [Email](mailto:gk326@ic.ac.uk)

I work on large language models and natural language processing. My papers so far ask how a model should represent words it has never seen, how one prompt can give several different answers instead of one, and how to store a network in much less memory without losing quality.

Before Imperial I studied Computer Science and Engineering at KL University, Hyderabad, and was a research intern at IIIT Hyderabad.

## News

- Oct 2026 · Version 2 of BIE is on Zenodo, with a new storage format and a full measured evaluation.
- Sep 2026 · Started the MSc in Computing (AI and Machine Learning) at Imperial College London.
- Jul 2026 · My team won first place at Imperial's Global Challenge Lab; I led the AI work on OrbitOps.
- 2026 · Dual-Stream and Prism Tuning accepted at IEEE NEPCON 2026.
- 2025 · MANCE published at IEEE ICCCMLA 2025.

## Publications

<table>
<tr>
<td width="200" valign="top"><img src="https://gruheshkurra.com/assets/projects/thumb-mance.svg" width="200" alt="Characters p, l, a, y, s combined into one word vector" /></td>
<td valign="top">
<b><a href="https://ieeexplore.ieee.org/document/11580466">Morphology-Aware Nested Character Embeddings for Word Representation</a></b><br/>
<b>Gruhesh Sri Sai Karthik Kurra</b>, Chandanasree Moparthi, Prathipati Manish Chowdary, Pavan Kumar Pagadala<br/>
<i>IEEE ICCCMLA 2025 · Published in IEEE Xplore</i>
<p>Each word vector is built from the word's characters by a small Transformer, so <i>play</i> and <i>plays</i> share most of their computation and an unseen word still gets a vector. In my seeded re-implementation (3 seeds) it reaches 63.0 F1 on GermEval 2014 German NER against 48.2 for word embeddings with the same encoder, and matches them on AG News and DBpedia.</p>
<a href="https://ieeexplore.ieee.org/document/11580466">Paper</a> · <a href="https://github.com/GruheshKurra/MANCE-NLP">Code</a> · <a href="https://github.com/GruheshKurra/MANCE-NLP/blob/main/experiments/RESULTS.md">Results</a>
</td>
</tr>
<tr>
<td width="200" valign="top"><img src="https://gruheshkurra.com/assets/projects/thumb-dualstream.svg" width="200" alt="Edge and frequency streams merged by cross-attention" /></td>
<td valign="top">
<b><a href="https://github.com/GruheshKurra/radar_deepfake">Dual-Stream Artifact Detection with Iterative Evidence Refinement for Frame-Level Deepfake Recognition</a></b><br/>
<b>Gruhesh Sri Sai Karthik Kurra</b>, Pavan Kumar Pagadala, Malathy Batumalay, Veda Boddapati<br/>
<i>IEEE NEPCON 2026 · Accepted, presentation December 2026</i>
<p>A ViT-Small detector with an edge stream for blending boundaries and a frequency stream for generator fingerprints, combined by iterative cross-attention. An audit found that 97.3% of WildDeepfake test frames came from videos also used in training, so the paper reports a video-disjoint subset of 5,680 frames: 96.3% AUC overall and 87.1% on Celeb-DF, from a single training run.</p>
<a href="https://github.com/GruheshKurra/radar_deepfake">Code</a> · <a href="https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames">Frame dataset</a>
</td>
</tr>
<tr>
<td width="200" valign="top"><img src="https://gruheshkurra.com/assets/projects/thumb-prism.svg" width="200" alt="One prompt split into K different outputs" /></td>
<td valign="top">
<b>Prism Tuning: Repulsion-Trained Seed Embeddings in a Frozen Transformer for Non-Redundant Generation</b><br/>
<b>Gruhesh Sri Sai Karthik Kurra</b>, Pavan Kumar Pagadala, Malathy Batumalay, Sushma Reddy Koduru<br/>
<i>IEEE NEPCON 2026 · Accepted, presentation December 2026</i>
<p>K seed vectors are added to the input embeddings of a frozen transformer and trained only by a repulsion loss, so one prompt gives K different outputs in one batched pass, with no sampling and filtering at inference. The paper gives the method and its formulation; running the experiments is my next step.</p>
</td>
</tr>
<tr>
<td width="200" valign="top"><img src="https://gruheshkurra.com/assets/projects/thumb-bie.svg" width="200" alt="Sparse matrix stored as coded gaps and bitplanes" /></td>
<td valign="top">
<b><a href="https://doi.org/10.5281/zenodo.17217217">BIE: Bit-Index Encoding of Sparse Neural Network Weights</a></b><br/>
<b>Gruhesh Sri Sai Karthik Kurra</b><br/>
<i>Preprint, Zenodo · Version 2, October 2026</i>
<p>A storage format for pruned weight matrices. The gaps between non-zero weights are Golomb–Rice coded, the values are 8-bit codes stored as bitplanes, and a row-parallel kernel multiplies on the compressed form. At 95% sparsity it is 44.5× smaller than float32 at 0.6% error, against 25.7× for CSR with the same values. Below about 50% sparsity it is no smaller than dense int8.</p>
<a href="https://doi.org/10.5281/zenodo.17217217">Paper</a> · <a href="https://github.com/GruheshKurra/bit-index-encoding-research-">Code</a> · <a href="https://github.com/GruheshKurra/bit-index-encoding-research-/blob/main/experiments/RESULTS.md">Results</a>
</td>
</tr>
</table>

The MANCE and BIE repositories include the configs, seeds and per-run result files behind the numbers above, so each one can be re-run.

## Experience

**AI Integration Engineer**, InfoAjax Consulting · Oct–Nov 2025<br/>
Built a proof-of-concept browser agent that acts on screenshots using models on Azure AI Foundry, and a React and FastAPI console for watching agents live.

**Research Intern**, IIIT Hyderabad · Dec 2024 – Jun 2025<br/>
Built a document translation system for English, Hindi, Telugu and German that keeps the original page layout, using Vision Transformer layout detection, font detection, colour extraction and Tesseract OCR. Demonstrated it at the institute's research expo in May 2025 ([demo](https://youtu.be/rcvSuBcBjyg)).

**Software Engineer (freelance)**, InfoAjax Consulting · Oct 2024 – Jun 2025<br/>
Built REST APIs in WSO2 Micro Integrator connecting the service-desk, billing and inventory systems of a Philippine telecoms company, and supported them in production.

## Education

**Imperial College London** · MSc Computing (Artificial Intelligence and Machine Learning) · 2026–2027<br/>
**KL University, Hyderabad** · B.Tech Computer Science and Engineering · 2021–2025 · CGPA 9.72/10

## Awards

- First place, Global Challenge Lab 2026, Imperial College London. A 14-day sprint with over 1,000 students; I led the AI and ML work for our team's project, OrbitOps, a reliability layer for AI running on satellites.
- First place, University Webathon, KL University, 2022.
- Technical Lead, Student Activity Council, KL University, 2023–2025.

## Writing

I write [AI from Scratch](https://blogs.gruheshkurra.com/series/ai/), a blog series that takes one model at a time from its derivation to a working implementation. Recent posts:

<!-- BLOG-POST-LIST:START --><a href="https://blogs.gruheshkurra.com/blog/convolutional-neural-networks-explained/">Convolutional Neural Networks: CNN Math Explained</a><br/><a href="https://blogs.gruheshkurra.com/blog/deepseek-v4-inside-one-token/">DeepSeek V4 Inside: One Token Through Every Block</a><br/><a href="https://blogs.gruheshkurra.com/blog/looped-transformers-explained/">Looped Transformers Explained: Recurrent Depth and Astra</a><br/><a href="https://blogs.gruheshkurra.com/blog/natural-language-inference-explained/">Natural Language Inference Explained: Entailment in NLP</a><br/><a href="https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/">Build a Mini LLM from Scratch in NumPy: RoPE, GQA, SwiGLU</a><br/><!-- BLOG-POST-LIST:END -->

[All posts](https://blogs.gruheshkurra.com/ai-explanations/) · [RSS](https://blogs.gruheshkurra.com/feed.xml)

## Contact

I'm looking for funded PhD positions in language models and NLP starting from October 2027. Email me at [gk326@ic.ac.uk](mailto:gk326@ic.ac.uk).
