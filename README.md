<p align="center">
  <img src="assets/profile-header.svg" width="100%" alt="Karthik Kurra. Machine learning, from the maths up. Model training, evaluation, and efficient inference.">
</p>

<h1 align="center">Gruhesh Sri Sai Karthik Kurra</h1>

<p align="center">
  MSc Computing (AI &amp; ML) · Imperial College London · 2026–27<br/>
  Language models from scratch · Deepfake forensics · On-device ML
</p>

<p align="center">
  <a href="https://gruheshkurra.com">Portfolio</a> &nbsp;·&nbsp;
  <a href="https://blogs.gruheshkurra.com">Blog</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/karthik-2905">Hugging Face</a> &nbsp;·&nbsp;
  <a href="https://orcid.org/0009-0002-0558-2882">ORCID</a> &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/gruheshkurra/">LinkedIn</a> &nbsp;·&nbsp;
  <a href="mailto:gruheshkurra2@gmail.com">Email</a>
</p>

---

I derive the equations, write the model, then train and test it. I have built a language model in pure NumPy with my own autograd engine, fine-tuned Qwen3 to call tools, and written papers on word embeddings, weight compression, and deepfake detection.

**Now**

- Starting the MSc Computing (Artificial Intelligence and Machine Learning) at Imperial College London, September 2026.
- Two first-author papers under review at IEEE NEPCON 2026.
- Writing [AI from Scratch](https://blogs.gruheshkurra.com/series/ai/), a 19-part series from linear algebra to working language models.
- Looking for research engineering internships in model training, evaluation, or efficient inference.

## Projects

<table>
<tr>
<td width="50%" valign="top">
  <sub>LANGUAGE MODELS</sub>
  <h3><a href="https://github.com/GruheshKurra/core-language-model">Language model in pure NumPy</a></h3>
  <p>A 3.87M-parameter decoder with my own autograd engine. RoPE, grouped-query attention, SwiGLU, QK-Norm, and a KV-cache. Pretrained on DailyDialog, then fine-tuned on EmpatheticDialogues.</p>
  <p><sub>78 unit tests across 12 files</sub></p>
  <p><a href="https://github.com/GruheshKurra/core-language-model">Code</a> · <a href="https://huggingface.co/karthik-2905/model-a-scratch">Model</a></p>
</td>
<td width="50%" valign="top">
  <sub>POST-TRAINING</sub>
  <h3><a href="https://github.com/GruheshKurra/AL1-model-B">Tool calling for Qwen3-0.6B</a></h3>
  <p>LoRA fine-tune on 1,423 tool and chat examples with TRL, assistant-only loss, and five file-system and shell tools in Hermes format.</p>
  <p><sub>Full-call exact match 6/12 → 11/12 on a 12-case greedy evaluation</sub></p>
  <p><a href="https://github.com/GruheshKurra/AL1-model-B">Code</a> · <a href="https://huggingface.co/karthik-2905/AL1-model-B">Model</a></p>
</td>
</tr>
<tr>
<td width="50%" valign="top">
  <sub>DEEPFAKE FORENSICS</sub>
  <h3><a href="https://github.com/GruheshKurra/radar_deepfake">Dual-Stream deepfake detection</a></h3>
  <p>A Sobel boundary stream and a multi-band FFT stream, fused by iterative cross-attention. Trained on 227,504 frames from FaceForensics++, Celeb-DF v2, and WildDeepfake.</p>
  <p><sub>96.3% AUC on a video-disjoint split · 87.1% on Celeb-DF</sub></p>
  <p><a href="https://github.com/GruheshKurra/radar_deepfake">Code</a> · <a href="https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames">Dataset</a></p>
</td>
<td width="50%" valign="top">
  <sub>NLP RESEARCH</sub>
  <h3><a href="https://github.com/GruheshKurra/MANCE-NLP">MANCE: morphology-aware embeddings</a></h3>
  <p>Words built from nested character sequences, so related word forms share information. Encoders: Transformer, CNN, and BiLSTM.</p>
  <p><sub>IEEE ICCCMLA 2025 · 99.0% DBpedia · 93.5% AG News · 81.3% F1 GermEval NER</sub></p>
  <p><a href="https://github.com/GruheshKurra/MANCE-NLP">Code</a> · <a href="https://ieeexplore.ieee.org/document/11580466">Paper</a></p>
</td>
</tr>
<tr>
<td width="50%" valign="top">
  <sub>ON-DEVICE ML</sub>
  <h3><a href="https://github.com/GruheshKurra/Deepguard">DeepGuard</a></h3>
  <p>EfficientNet-B1 deepfake detector trained on 25k images on an Apple M4, then converted to a 26 MB Core ML model for iOS.</p>
  <p><sub>98.62% accuracy · under 500 ms per image · trained and converted, not a shipped product</sub></p>
  <p><a href="https://github.com/GruheshKurra/Deepguard">Code</a> · <a href="https://www.youtube.com/watch?v=4MmHJNjLRy4">Demo</a></p>
</td>
<td width="50%" valign="top">
  <sub>MODEL COMPRESSION</sub>
  <h3><a href="https://github.com/GruheshKurra/bit-index-encoding-research-">BIE: bit-index encoding</a></h3>
  <p>Compresses sparse weight matrices by storing the positions of nonzero bits. Includes Numba-compiled sparse matrix multiplication on the compressed form.</p>
  <p><sub>40× compression at 95% sparsity · bitplane variants reach MSE below 10⁻⁶</sub></p>
  <p><a href="https://github.com/GruheshKurra/bit-index-encoding-research-">Code</a> · <a href="https://zenodo.org/records/17217218">Preprint</a></p>
</td>
</tr>
</table>

<details>
<summary>More projects</summary>

| Project | What it is | Links |
|:---|:---|:---|
| GPT-2 from scratch | 124M-parameter GPT-2 in PyTorch with a BPE tokeniser and full training loop, trained locally | [Code](https://github.com/GruheshKurra/FirstGPTFromScratch) |
| Transformers from scratch | The encoder-decoder from *Attention Is All You Need*, replicated with BLEU benchmarks | [Code](https://github.com/GruheshKurra/TransformersFromScratch) |
| LLaMA-style decoder | RMSNorm, RoPE, and SwiGLU with a PyTorch training recipe | [Code](https://github.com/GruheshKurra/LLamaModel) |
| NL2SQL | Plain-English questions to SQL with an attention-based model; perplexity 1.42 | [Code](https://github.com/GruheshKurra/nl2sql-pretrained) |
| Clinical trial similarity | PubMedBERT embeddings with FAISS retrieval over trial records | [Code](https://github.com/GruheshKurra/Clinical-Trial-Similarity-Analysis) |
| GNN suite | GCN, GraphSAGE, and GAT for node classification | [Code](https://github.com/GruheshKurra/GraphNeuralNetworks-GNN-) |
| Diffusion from scratch | DDPM with a ~130K-parameter UNet | [Code](https://github.com/GruheshKurra/DiffusionModelFromScratch) |
| OrbitOps | Reliability layer for satellite onboard AI. Won Global Challenge Lab 2026 | No public repository |

**Also:** [AI Voice Assistant for Windows](https://github.com/GruheshKurra/AI-Voice-Assistant-for-Windows) · [WSO2 on Kubernetes](https://github.com/GruheshKurra/WSO2-Kubernetes-Support) · [FarmCare](https://github.com/GruheshKurra/Farmcare) · [healthcareAI](https://github.com/GruheshKurra/healthcareAI) · [AI learning roadmaps](https://github.com/GruheshKurra/awesome-ai-roadmaps)

</details>

<details>
<summary>From-scratch collection: algorithms rebuilt from the maths</summary>

| Area | Repositories |
|:---|:---|
| Deep learning | [AttentionMechanisms](https://github.com/GruheshKurra/AttentionMechanisms) · [SequenceModeling](https://github.com/GruheshKurra/SequenceModeling) (RNN, LSTM, GRU) · [GAN_Implementation](https://github.com/GruheshKurra/GAN_Implementation) · [VariationalAutoencoders](https://github.com/GruheshKurra/VariationalAutoencoders) |
| Reinforcement learning | [TemporalDifferenceLearning](https://github.com/GruheshKurra/TemporalDifferenceLearning) · [MonteCarloMethods](https://github.com/GruheshKurra/MonteCarloMethods) · [MarkovsDecisionPlane](https://github.com/GruheshKurra/MarkovsDecisionPlane) |
| Supervised learning | [SVM](https://github.com/GruheshKurra/SVM-Implementation-From-Scratch) · [Random forest](https://github.com/GruheshKurra/random-forest-from-scratch) · [Decision trees](https://github.com/GruheshKurra/decision-trees-from-scratch) · [Logistic regression](https://github.com/GruheshKurra/logistic-regression-impl) · [Linear regression](https://github.com/GruheshKurra/linear-regression-impl) · [Naive Bayes](https://github.com/GruheshKurra/naive-bayes-implementation) · [kNN](https://github.com/GruheshKurra/knn-implementation) |
| Probabilistic models | [Bayesian networks](https://github.com/GruheshKurra/Bayesian-Networks) · [Anomaly detection](https://github.com/GruheshKurra/AnomalyDetection) |
| Clustering | [k-means](https://github.com/GruheshKurra/k-means-clustering) · [DBSCAN](https://github.com/GruheshKurra/dbscan-clustering) · [Hierarchical](https://github.com/GruheshKurra/hierarchical-clustering) |
| Dimensionality reduction | [t-SNE](https://github.com/GruheshKurra/tsne-from-scratch) · [UMAP](https://github.com/GruheshKurra/umap-dimensionality-reduction) · [PCA and more](https://github.com/GruheshKurra/dimensionality-reduction) |

</details>

## Research

| Paper | Status |
|:---|:---|
| [Morphology-Aware Nested Character Embeddings for Word Representation (MANCE)](https://ieeexplore.ieee.org/document/11580466)<br/><sub>**G. S. S. K. Kurra**, C. Moparthi, P. M. Chowdary, P. K. Pagadala</sub> | Accepted, IEEE ICCCMLA 2025 |
| [Dual-Stream Artifact Detection with Iterative Evidence Refinement for Frame-Level Deepfake Recognition](https://github.com/GruheshKurra/radar_deepfake)<br/><sub>**G. S. S. K. Kurra**, P. K. Pagadala, M. Batumalay, V. Boddapati</sub> | Under review, IEEE NEPCON 2026 |
| Prism Tuning: Repulsion-Trained Seed Embeddings in a Frozen Transformer for Non-Redundant Generation<br/><sub>**G. S. S. K. Kurra**, P. K. Pagadala, M. Batumalay, S. R. Koduru</sub> | Under review, IEEE NEPCON 2026 |
| [BIE: Bit-Index Encoding for Efficient Neural Network Weight Compression](https://zenodo.org/records/17217218)<br/><sub>**G. S. S. K. Kurra**</sub> | Preprint, Zenodo, 2025 |
| [Hybrid RAG-Enhanced Deepfake Detection: A Novel Approach Combining Retrieval-Augmented Generation with Visual Inconsistency Analysis](https://zenodo.org/records/16732053)<br/><sub>**G. S. S. K. Kurra**</sub> | Preprint, Zenodo, 2025 |
| RADAR: Reasoning-Augmented Deepfake Artifact Recognition via Multi-Branch Evidence Aggregation<br/><sub>**G. S. S. K. Kurra**</sub> | Preprint, 2026 |

<sub>Prism Tuning, RADAR, and Hybrid RAG set out a method and evaluation plan; they do not report measured results. RADAR's measured results are in the Dual-Stream paper. Preprints are not peer-reviewed.</sub>

## Experience

<table>
<tr>
<td width="44"><img src="assets/org-logos/iiith.png" width="40" height="40" alt="IIIT Hyderabad"></td>
<td><b>Research Intern</b> · IIIT Hyderabad · Dec 2024 – Jun 2025<br/>Built layout-preserving document translation for English, Hindi, Telugu, and German: Vision Transformer layout detection, font detection from character metrics, and K-means colour palettes, served through FastAPI and Tesseract OCR. Demonstrated live at the IIIT Hyderabad research expo, May 2025. <a href="https://youtu.be/rcvSuBcBjyg">Demo</a></td>
</tr>
<tr>
<td><img src="assets/org-logos/infoajax.png" width="40" height="40" alt="InfoAjax"></td>
<td><b>AI Integration Engineer</b> · InfoAjax Consulting · Oct – Nov 2025<br/>Integrated enterprise apps with cybersecurity services. Used Azure OpenAI agents for schema mapping, connected Android and Apple enterprise apps to Azure, built Azure-to-Salesforce proofs of concept, and a React + FastAPI + WebSocket console for live agent monitoring.</td>
</tr>
<tr>
<td><img src="assets/org-logos/wso2.png" width="40" height="40" alt="WSO2"></td>
<td><b>WSO2 API Developer</b> · InfoAjax Consulting · Oct 2024 – Jun 2025<br/>Shipped REST ticketing APIs for PLDT, a Philippine telecom, in WSO2 Integration Studio. Deployed on WSO2 Choreo with GitLab vault secrets, under client SLAs.</td>
</tr>
<tr>
<td><img src="assets/org-logos/zynthetix.png" width="40" height="40" alt="Zynthetix"></td>
<td><b>Founder</b> · Zynthetix · Mar 2024 – Jan 2025<br/>Designed a privacy-preserving synthetic-data architecture in which a parent model coordinates hundreds of specialised child models to generate tabular, image, and text data without duplication. The same idea led to Prism Tuning.</td>
</tr>
<tr>
<td><img src="assets/org-logos/ihub.png" width="40" height="40" alt="iHub-Data"></td>
<td><b>AI &amp; ML Research Trainee</b> · iHub-Data, IIIT Hyderabad · May – Oct 2024<br/>Six-month faculty-mentored programme: model architecture, language models, fine-tuning, quantisation, and deployment.</td>
</tr>
</table>

**Education**

- **Imperial College London** · MSc Computing (Artificial Intelligence and Machine Learning) · Sep 2026 – Sep 2027
- **KL University, Hyderabad** · B.Tech Computer Science and Engineering · 2021–2025 · CGPA 9.72 / 10

## Awards

- **1st place, Global Challenge Lab 2026**, Imperial College London (July 2026). A 14-day global sprint with 1,000+ students. I led the AI/ML work and technical strategy for Team Corio's OrbitOps, a reliability layer for satellite onboard AI: fault injection before launch, known-answer probes in orbit to catch silent degradation, and rollback to a clean copy. £1,500 team prize.
- **1st place, University Webathon**, KL University (2022). A diet-management platform built in 4 hours, against 50+ teams.
- **2nd place, Design Expo**, KL University (2022–23). An Arduino smart switchboard with energy monitoring and app control.
- **SAC Momentum Award**, KL University (2023–25). As Technical Lead, I mentored 20+ juniors, ran workshops for 100+ students, and coordinated 15+ events.

<details>
<summary>Certificates (20)</summary>

| Certificate | Issuer · year |
|:---|:---|
| [Global Challenge Lab 2026, 1st place](certificates/global-challenge-lab-2026.jpg) | Imperial College London · 2026 |
| [TensorFlow Developer](certificates/tensorflow-developer.jpg) | Google · 2024 |
| [Solutions Architect – Associate](certificates/aws-solutions-architect.jpg) | AWS · 2023 |
| [Cloud Practitioner](certificates/aws-cloud-practitioner.jpg) | AWS · 2023 |
| [OCI Generative AI Professional](certificates/oracle-genai.jpg) | Oracle · 2024 |
| [OCI Architect Associate](certificates/oracle-architect.jpg) | Oracle · 2023 |
| [Oracle Database](certificates/oracle-database.jpg) | Oracle · 2023 |
| [Enterprise Application Developer](certificates/redhat-developer.jpg) | Red Hat · 2024 |
| [Micro Integrator Developer V4](certificates/wso2-mi-developer.jpg) | WSO2 · 2025 |
| [Micro Integrator Practitioner V4](certificates/wso2-mi-practitioner.jpg) | WSO2 · 2025 |
| [Advanced Automation Professional](certificates/automation-anywhere.jpg) | Automation Anywhere · 2024 |
| [Six-month AI/ML training](certificates/iiit-training.jpg) | iHub-Data, IIIT Hyderabad · 2024 |
| [Build LLMs From Scratch](certificates/visuara-llms.jpg) | Visuara · 2024 |
| [MCP Course, Unit 1](certificates/hf-mcp-unit1.jpg) | Hugging Face · 2024 |
| [MCP Course, Unit 3](certificates/hf-mcp-unit3.jpg) | Hugging Face · 2024 |
| [Linear Algebra Master](certificates/udemy-linear-algebra.jpg) | Udemy · 2026 |
| [A Deep Understanding of Deep Learning](certificates/udemy-deep-learning.jpg) | Udemy · 2026 |
| [Google Developer Groups](certificates/google-gdg.jpg) | GDG · 2025 |
| [Certificate of Appreciation](certificates/sac-appreciation.jpg) | KL University · 2022–23 |
| [Volunteer](certificates/streetcause.jpg) | Street Cause · 2023–24 |

IELTS Academic: 7.5 overall (CEFR C1).

</details>

## Writing

[AI from Scratch](https://blogs.gruheshkurra.com/series/ai/) is my 19-part series. Each post derives the equations, works a small numeric example by hand, then implements it.

| Foundations | Building blocks | Models |
|:---|:---|:---|
| [Linear algebra, visually](https://blogs.gruheshkurra.com/blog/essence-of-linear-algebra/) | [Build an autograd engine](https://blogs.gruheshkurra.com/blog/build-autograd-from-scratch/) | [GPT-2 in PyTorch](https://blogs.gruheshkurra.com/blog/build-gpt2-from-scratch/) |
| [Cross-entropy loss](https://blogs.gruheshkurra.com/blog/cross-entropy-loss-explained/) | [BPE tokenisers](https://blogs.gruheshkurra.com/blog/byte-pair-encoding-from-scratch/) | [A mini LLM in NumPy](https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/) |
| [Adam and AdamW](https://blogs.gruheshkurra.com/blog/adam-optimizer-explained/) | [Attention: queries, keys, values](https://blogs.gruheshkurra.com/blog/attention-in-transformers-explained/) | [DeepSeek V4, one token at a time](https://blogs.gruheshkurra.com/blog/deepseek-v4-inside-one-token/) |

<details>
<summary>Latest posts (updated automatically)</summary>

<!-- BLOG-POST-LIST:START --><a href="https://blogs.gruheshkurra.com/blog/convolutional-neural-networks-explained/">Convolutional Neural Networks: CNN Math Explained</a><br/><a href="https://blogs.gruheshkurra.com/blog/deepseek-v4-inside-one-token/">DeepSeek V4 Inside: One Token Through Every Block</a><br/><a href="https://blogs.gruheshkurra.com/blog/looped-transformers-explained/">Looped Transformers Explained: Recurrent Depth and Astra</a><br/><a href="https://blogs.gruheshkurra.com/blog/natural-language-inference-explained/">Natural Language Inference Explained: Entailment in NLP</a><br/><a href="https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/">Build a Mini LLM from Scratch in NumPy: RoPE, GQA, SwiGLU</a><br/><!-- BLOG-POST-LIST:END -->

[All posts](https://blogs.gruheshkurra.com/ai-explanations/) · [RSS](https://blogs.gruheshkurra.com/feed.xml) · [DEV](https://dev.to/gruhesh_kurra_6eb933146da)

</details>

## Toolkit

| Area | Tools |
|:---|:---|
| Daily | Python · PyTorch · NumPy · Hugging Face Transformers · TRL · PEFT / LoRA · FastAPI · Docker · Git |
| Models | Transformers (GQA, RoPE, SwiGLU, KV-cache) · BPE tokenisation · GANs · VAEs · diffusion · GNNs · RAG · FAISS |
| Vision and on-device | EfficientNet · Vision Transformers · CLIP · OpenCV · Tesseract OCR · Core ML · Swift / SwiftUI |
| Efficiency | LoRA · quantisation · pruning · distillation · Numba kernels |
| Cloud and backend | AWS · Azure · GCP · OCI · RunPod · Kubernetes · WSO2 · React · Next.js · PostgreSQL · MongoDB · Redis |
| Languages | Python · C++ · C · Java · TypeScript · Swift · SQL · Shell |

<details>
<summary>GitHub activity</summary>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake.svg">
  <img alt="Contribution graph rendered as a snake game" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake.svg">
</picture>

<img alt="GitHub metrics: activity, languages, contribution calendar and top repositories" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/github-metrics.svg" width="480">

</div>

</details>

## Contact

Open to research engineering internships and collaboration on language models, model compression, and on-device inference.

[gruheshkurra2@gmail.com](mailto:gruheshkurra2@gmail.com) · [LinkedIn](https://www.linkedin.com/in/gruheshkurra/) · [Portfolio](https://gruheshkurra.com) · [Hugging Face](https://huggingface.co/karthik-2905) · [Kaggle](https://www.kaggle.com/gruheshkurra) · [X](https://x.com/Karthik__kurra)

<sub>Also known as Gruhesh Kurra and Karthik Kurra.</sub>
