# Gruhesh Sri Sai Karthik Kurra

AI research engineer. LLMs, generative models and neural compression, built from the math up.

MSc Computing (AI & ML), Imperial College London, from Sep 2026 · 1st place, Global Challenge Lab 2026

Hyderabad, India → London, UK from Sep 2026 · Remote-friendly

[Portfolio](https://gruheshkurra.com) · [Blog](https://blogs.gruheshkurra.com) · [LinkedIn](https://www.linkedin.com/in/gruheshkurra/) · [Hugging Face](https://huggingface.co/karthik-2905) · [ORCID](https://orcid.org/0009-0002-0558-2882) · [X](https://x.com/Karthik__kurra) · [Email](mailto:gruheshkurra2@gmail.com)

[About](#about) · [Now](#now) · [Selected work](#selected-work) · [Research](#research) · [Writing](#writing) · [Experience](#experience) · [Stack](#stack) · [Education](#education) · [Certifications](#certifications) · [Awards](#awards) · [Contact](#contact)

---

## About

I build machine learning systems from the maths up: transformers, diffusion and attention re-implemented from the equations, then shipped on-device and into production. Work spans model optimisation, neural weight compression, deepfake forensics and synthetic data.

IEEE ICCCMLA 2025 (MANCE: morphology-aware embeddings), plus open work on Zenodo and IJNRD, and 2 papers under review at IEEE NEPCON 2026. B.Tech CSE, KL University, CGPA 9.72/10 (2021–2025). AWS, TensorFlow, Oracle, Red Hat and WSO2 certified.

---

## Now

| | |
|:---|:---|
| Sep 2026 | MSc Computing (AI & ML), Imperial College London |
| Writing | *AI from Scratch*: 15-post series, linear algebra through a working GPT-2 |
| Under review | Two papers at IEEE NEPCON 2026: Prism Tuning · Dual-Stream |
| Open to | Remote AI/ML research engineering roles; collaboration on transformers, compression and on-device inference |

---

## Selected work

| Project | What it is | Links |
|:---|:---|:---|
| MANCE | Nested character embeddings for morphology: +58.3 pp on morphological-variant sentiment (91.67% vs 33.33% Word2Vec); 99.0% DBpedia, 93.5% AG News, 81.3% F1 on GermEval NER | [Code](https://github.com/GruheshKurra/MANCE-NLP) · [IEEE](https://ieeexplore.ieee.org/document/11580466) |
| Dual-Stream Deepfake Detection | Two orthogonal evidence branches (Sobel boundary + FFT artifact) with iterative cross-attention refinement: 96.3% AUC video-disjoint, 87.1% on Celeb-DF alone; 227k frames, code, corpus and leakage audit all public | [Code + data](https://github.com/GruheshKurra/radar_deepfake) · [Kaggle](https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames) |
| DeepGuard | On-device iOS deepfake detector: EfficientNet-B1 compressed to a 26 MB Core ML model, &lt;500 ms inference, 98.62% on 25k images | [Code](https://github.com/GruheshKurra/Deepguard) · [Demo](https://www.youtube.com/watch?v=4MmHJNjLRy4) |
| BIE | Bit-index encoding for neural weight compression: 40× at 95% sparsity, MSE &lt; 10⁻⁶ | [Code](https://github.com/GruheshKurra/bit-index-encoding-research-) · [Zenodo](https://zenodo.org/records/17217218) |
| OrbitOps | Reliability layer for satellite onboard AI: pre-launch fault injection, in-orbit known-answer monitoring, rollback. Won Global Challenge Lab 2026 | [Demo code](https://github.com/GruheshKurra/OrbitOps-Demo) |
| Hybrid RAG Deepfake | Privacy-first detector: retrieval plus visual anomaly scoring; 94.8% accuracy / 94.6% F1 | [Zenodo](https://zenodo.org/records/16732053) |
| GPT-2 / Transformers | GPT-2 (124M) and *Attention Is All You Need* rebuilt from scratch, with BLEU benchmarks | [GPT-2](https://github.com/GruheshKurra/FirstGPTFromScratch) · [Transformers](https://github.com/GruheshKurra/TransformersFromScratch) |
| NL2SQL | Natural language → SQL with attention; perplexity 1.42 | [Code](https://github.com/GruheshKurra/nl2sql-pretrained) |
| Clinical Trial Similarity | PubMedBERT + FAISS retrieval over trial records: 99.5% similarity accuracy | [Code](https://github.com/GruheshKurra/Clinical-Trial-Similarity-Analysis) |
| Zynthetix SWE Agent | Agent that reads GitHub issues and opens PRs via GitHub Actions only (no always-on server) | [Code](https://github.com/GruheshKurra/Zynthetix-SWE-Agent) |

<details>
<summary>From-scratch collection: 25+ algorithms rebuilt from the math, no library wrappers</summary>

**Deep learning and generative models**

| Repo | Topic |
|:---|:---|
| [AttentionMechanisms](https://github.com/GruheshKurra/AttentionMechanisms) | Scaled dot-product and multi-head attention |
| [DiffusionModelFromScratch](https://github.com/GruheshKurra/DiffusionModelFromScratch) | DDPM, forward and reverse process |
| [GAN_Implementation](https://github.com/GruheshKurra/GAN_Implementation) | GAN / DCGAN on MNIST |
| [VariationalAutoencoders](https://github.com/GruheshKurra/VariationalAutoencoders) | VAE, ELBO and latent-space analysis |
| [SequenceModeling](https://github.com/GruheshKurra/SequenceModeling) | RNN, LSTM, GRU |
| [GraphNeuralNetworks-GNN-](https://github.com/GruheshKurra/GraphNeuralNetworks-GNN-) | GCN, GraphSAGE, GAT |

**Reinforcement learning**

| Repo | Topic |
|:---|:---|
| [TemporalDifferenceLearning](https://github.com/GruheshKurra/TemporalDifferenceLearning) | TD(0), SARSA, Q-learning |
| [MonteCarloMethods](https://github.com/GruheshKurra/MonteCarloMethods) | Monte Carlo prediction and control |
| [MarkovsDecisionPlane](https://github.com/GruheshKurra/MarkovsDecisionPlane) | MDPs, value and policy iteration |

**Classical ML**

| Repo | Topic |
|:---|:---|
| [SVM-Implementation-From-Scratch](https://github.com/GruheshKurra/SVM-Implementation-From-Scratch) | Hard margin, soft margin, kernels, multi-class |
| [random-forest-from-scratch](https://github.com/GruheshKurra/random-forest-from-scratch) | Bagging, out-of-bag error, feature importance |
| [decision-trees-from-scratch](https://github.com/GruheshKurra/decision-trees-from-scratch) | CART, entropy and Gini splitting |
| [logistic-regression-impl](https://github.com/GruheshKurra/logistic-regression-impl) · [linear-regression-impl](https://github.com/GruheshKurra/linear-regression-impl) | Gradient descent from first principles |
| [naive-bayes-implementation](https://github.com/GruheshKurra/naive-bayes-implementation) · [knn-implementation](https://github.com/GruheshKurra/knn-implementation) | Gaussian / multinomial NB, instance-based learning |
| [Bayesian-Networks](https://github.com/GruheshKurra/Bayesian-Networks) · [AnomalyDetection](https://github.com/GruheshKurra/AnomalyDetection) | Probabilistic graphical models, outlier detection |

**Clustering and dimensionality reduction**

| Repo | Topic |
|:---|:---|
| [k-means-clustering](https://github.com/GruheshKurra/k-means-clustering) · [dbscan-clustering](https://github.com/GruheshKurra/dbscan-clustering) · [hierarchical-clustering](https://github.com/GruheshKurra/hierarchical-clustering) | Full clustering suite |
| [tsne-from-scratch](https://github.com/GruheshKurra/tsne-from-scratch) · [umap-dimensionality-reduction](https://github.com/GruheshKurra/umap-dimensionality-reduction) · [dimensionality-reduction](https://github.com/GruheshKurra/dimensionality-reduction) | PCA, t-SNE, UMAP |

</details>

---

## Research

ORCID [0009-0002-0558-2882](https://orcid.org/0009-0002-0558-2882)

**Published**

| Paper | Venue | Year |
|:---|:---|:---|
| [Morphology-Aware Nested Character Embeddings for Word Representation](https://ieeexplore.ieee.org/document/11580466) | IEEE ICCCMLA | 2025 |
| [BIE: Bit-Index Encoding for Neural Network Weight Compression](https://zenodo.org/records/17217218) | Zenodo | 2025 |
| [Hybrid RAG-Enhanced Deepfake Detection](https://zenodo.org/records/16732053) | Zenodo | 2024 |
| Dynamic Auto-Finetuning of Language Models Based on Confidence-Driven Knowledge Integration | IJNRD | 2024 |
| Global Remote RAM Sharing: A Novel Framework for Distributed Computational Systems | IJNRD | 2024 |
| Voice-Activated AI for Seamless Computer Interaction | IJNRD | 2025 |

**Under review and working papers** (none of these are accepted or published)

| Paper | Status | Year |
|:---|:---|:---|
| Prism Tuning: Repulsion-Trained Seed Embeddings in a Frozen Transformer for Non-Redundant Generation | Under review, IEEE NEPCON 2026. Architecture and formulation paper; no empirical results claimed | 2026 |
| [Dual-Stream Artifact Detection with Iterative Evidence Refinement for Frame-Level Deepfake Recognition](https://github.com/GruheshKurra/radar_deepfake) | Under review, IEEE NEPCON 2026. Code, 227k-frame corpus and leakage self-audit released | 2026 |
| RADAR: Reasoning-Augmented Deepfake Artifact Recognition via Multi-Branch Evidence Aggregation | Preprint: design paper, no measured results | 2026 |

---

## Writing

I write at [blogs.gruheshkurra.com](https://blogs.gruheshkurra.com): AI and machine learning from the maths up. Derive the equations by hand, work a numeric example, then implement it. No wrappers, no hand-waving.

### AI from Scratch: the full series

Linear algebra through a working GPT-2, in order. Every post has the derivation, real numbers from the actual config, and code that runs.

| # | Post | Topic |
|:--|:---|:---|
| 1 | [Linear Algebra for Machine Learning: The Visual Intuition](https://blogs.gruheshkurra.com/blog/essence-of-linear-algebra/) | Vectors, matrices, eigenvectors |
| 2 | [Backpropagation from Scratch: Build an Autograd Engine](https://blogs.gruheshkurra.com/blog/build-autograd-from-scratch/) | Reverse-mode autodiff |
| 3 | [Numerical Gradient Checking: Debug Your Autograd Engine](https://blogs.gruheshkurra.com/blog/numerical-gradient-checking-explained/) | Central differences |
| 4 | [Byte Pair Encoding (BPE) Explained: How GPT Tokenizers Work](https://blogs.gruheshkurra.com/blog/byte-pair-encoding-from-scratch/) | Tokenisation, merge rules |
| 5 | [Token Embeddings Explained: How LLMs Turn IDs Into Vectors](https://blogs.gruheshkurra.com/blog/token-embeddings-explained/) | Embedding matrix, gather vs one-hot |
| 6 | [Positional Encoding Explained: How Transformers Learn Order](https://blogs.gruheshkurra.com/blog/positional-encoding-explained/) | Learned vs sinusoidal |
| 7 | [Cross-Entropy Loss Explained: From Logits to LLM Training](https://blogs.gruheshkurra.com/blog/cross-entropy-loss-explained/) | Softmax and NLL |
| 8 | [Adam and AdamW Explained: How LLMs Update Their Weights](https://blogs.gruheshkurra.com/blog/adam-optimizer-explained/) | Momentum, RMSprop, decoupled decay |
| 9 | [How GPT Actually Works: A Visual Guide to Transformers](https://blogs.gruheshkurra.com/blog/what-is-a-gpt-visual-intro/) | Plain-English transformer tour |
| 10 | [How Attention Works in Transformers: Queries, Keys, Values](https://blogs.gruheshkurra.com/blog/attention-in-transformers-explained/) | Attention pattern, multi-head |
| 11 | [Transformer from Scratch: Forward Pass and Backprop by Hand](https://blogs.gruheshkurra.com/blog/transformer-from-scratch-forward-backward-math/) | One training step, fully worked |
| 12 | [GPT Math Explained: The Full Forward Pass Beyond Attention](https://blogs.gruheshkurra.com/blog/gpt-math-beyond-attention/) | Token IDs → loss → AdamW |
| 13 | [Build GPT-2 from Scratch in PyTorch: A Full Walkthrough](https://blogs.gruheshkurra.com/blog/build-gpt2-from-scratch/) | 124M params, trained locally |
| 14 | [DeepSeek V4 Explained: Long-Context Engineering and Math](https://blogs.gruheshkurra.com/blog/deepseek-v4-engineering-explained/) | Sparse attention, KV-cache scaling |
| 15 | [Build a Mini LLM from Scratch in NumPy: RoPE, GQA, SwiGLU](https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/) | 3.87M-param chat model, pure NumPy |

[All posts](https://blogs.gruheshkurra.com/ai-explanations/) · [Series](https://blogs.gruheshkurra.com/series/ai/) · [Topics](https://blogs.gruheshkurra.com/tags/) · [Library](https://blogs.gruheshkurra.com/library/) · [RSS](https://blogs.gruheshkurra.com/feed.xml)

### Latest posts

<!-- BLOG-POST-LIST:START --><a href="https://blogs.gruheshkurra.com/blog/natural-language-inference-explained/">Natural Language Inference Explained: Entailment in NLP</a><br/><a href="https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/">Build a Mini LLM from Scratch in NumPy: RoPE, GQA, SwiGLU</a><br/><a href="https://blogs.gruheshkurra.com/blog/gpt-math-beyond-attention/">GPT Math Explained: The Full Forward Pass Beyond Attention</a><br/><a href="https://blogs.gruheshkurra.com/blog/adam-optimizer-explained/">Adam and AdamW Explained: How LLMs Update Their Weights</a><br/><a href="https://blogs.gruheshkurra.com/blog/cross-entropy-loss-explained/">Cross-Entropy Loss Explained: From Logits to LLM Training</a><br/><!-- BLOG-POST-LIST:END -->

---

## Experience

<table>
<tr>
<td width="44"><img src="assets/org-logos/infoajax.png" width="40" height="40" alt="InfoAjax"></td>
<td><b>AI Integration Engineer</b> · InfoAjax Consulting · Oct – Nov 2025<br/>Azure OpenAI agents for schema mapping; React + FastAPI live monitoring; Azure AD via Microsoft Graph.</td>
</tr>
<tr>
<td><img src="assets/org-logos/wso2.png" width="40" height="40" alt="WSO2"></td>
<td><b>WSO2 API Developer</b> · InfoAjax Consulting · Oct 2024 – Jun 2025<br/>Production REST APIs for PLDT (Philippines telecom); GitLab vault secrets; Choreo hosting under SLA.</td>
</tr>
<tr>
<td><img src="assets/org-logos/iiith.png" width="40" height="40" alt="IIIT Hyderabad"></td>
<td><b>Research Intern</b> · IIIT Hyderabad · Dec 2024 – Jun 2025<br/>Layout-preserving document translation: ViT layout detection at 92%, multi-language OCR; research expo demos.</td>
</tr>
<tr>
<td><img src="assets/org-logos/ihub.png" width="40" height="40" alt="iHub-Data"></td>
<td><b>AI &amp; ML Research Trainee</b> · iHub-Data, IIIT Hyderabad · May – Oct 2024<br/>Six-month research apprenticeship: LLM fundamentals, fine-tuning, quantisation, deployment.</td>
</tr>
<tr>
<td><img src="assets/org-logos/zynthetix.png" width="40" height="40" alt="Zynthetix"></td>
<td><b>Founder &amp; CEO</b> · Zynthetix · Mar 2024 – Jan 2025<br/>Synthetic data platform (GANs, VAEs) with hierarchical generators and transformer-based PII detection; led a 3-person team.</td>
</tr>
<tr>
<td><img src="assets/org-logos/xtraleap.jpg" width="40" height="40" alt="Xtraleap"></td>
<td><b>Data Science Intern</b> · Xtraleap India · Jul – Oct 2023<br/>ML pipelines: EDA, feature engineering, model evaluation with pandas, NumPy and scikit-learn.</td>
</tr>
</table>

---

## Stack

| Area | Tools |
|:---|:---|
| Primary | Python · PyTorch · Hugging Face Transformers · NumPy · FastAPI · Docker · Git · Azure OpenAI |
| Research focus | LLMs · transformers and attention · generative models (GANs, VAEs, DDPM) · GNNs · model compression (quantisation, pruning, bit-index encoding) · RAG · on-device inference and Core ML · synthetic data |
| Also use | TensorFlow · LangChain · LlamaIndex · scikit-learn · React · Next.js · TypeScript · React Native · Swift · TailwindCSS |
| Cloud and data | AWS (EC2, S3, Lambda) · Azure (AD, OpenAI, Graph) · GCP · WSO2 · CI/CD · PostgreSQL · MongoDB · Supabase · Redis |
| Languages | Python · TypeScript / JavaScript · Java · C · C++ · SQL · R · Swift |

---

## Education

MSc Computing (Artificial Intelligence and Machine Learning), Imperial College London · Sep 2026 – Sep 2027 *(incoming)*

B.Tech, Computer Science and Engineering, KL University, Hyderabad · Aug 2021 – May 2025 · CGPA 9.72 / 10

IELTS Academic 7.5 · Cambridge C1 Advanced

---

## Certifications

20 certificates. Click a card to open the full image.

<table>
  <tr>
    <td align="center" valign="top" width="25%"><a href="certificates/global-challenge-lab-2026.jpg"><img src="certificates/global-challenge-lab-2026.jpg" width="160" alt="Global Challenge Lab 2026, 1st place"></a><br/>Global Challenge Lab 2026<br/><sub>Imperial · 1st place · 2026</sub></td>
    <td align="center" valign="top" width="25%"><a href="certificates/tensorflow-developer.jpg"><img src="certificates/tensorflow-developer.jpg" width="160" alt="TensorFlow Developer Certificate"></a><br/>TensorFlow Developer<br/><sub>Google · 2024</sub></td>
    <td align="center" valign="top" width="25%"><a href="certificates/aws-solutions-architect.jpg"><img src="certificates/aws-solutions-architect.jpg" width="160" alt="AWS Solutions Architect Associate"></a><br/>Solutions Architect – Associate<br/><sub>AWS · 2023</sub></td>
    <td align="center" valign="top" width="25%"><a href="certificates/aws-cloud-practitioner.jpg"><img src="certificates/aws-cloud-practitioner.jpg" width="160" alt="AWS Cloud Practitioner"></a><br/>Cloud Practitioner<br/><sub>AWS · 2023</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top"><a href="certificates/oracle-architect.jpg"><img src="certificates/oracle-architect.jpg" width="160" alt="Oracle Cloud Infrastructure Architect Associate"></a><br/>OCI Architect Associate<br/><sub>Oracle · 2023</sub></td>
    <td align="center" valign="top"><a href="certificates/oracle-genai.jpg"><img src="certificates/oracle-genai.jpg" width="160" alt="Oracle Generative AI Professional"></a><br/>OCI Generative AI Professional<br/><sub>Oracle · 2024</sub></td>
    <td align="center" valign="top"><a href="certificates/redhat-developer.jpg"><img src="certificates/redhat-developer.jpg" width="160" alt="Red Hat Enterprise Application Developer"></a><br/>Enterprise Application Developer<br/><sub>Red Hat · 2024</sub></td>
    <td align="center" valign="top"><a href="certificates/wso2-mi-developer.jpg"><img src="certificates/wso2-mi-developer.jpg" width="160" alt="WSO2 Micro Integrator Developer V4"></a><br/>Micro Integrator Developer V4<br/><sub>WSO2 · 2025</sub></td>
  </tr>
</table>

<details>
<summary>More certificates (12)</summary>

<table>
  <tr>
    <td align="center" valign="top" width="25%"><a href="certificates/wso2-mi-practitioner.jpg"><img src="certificates/wso2-mi-practitioner.jpg" width="160" alt="WSO2 Micro Integrator Practitioner V4"></a><br/>Micro Integrator Practitioner V4<br/><sub>WSO2 · 2025</sub></td>
    <td align="center" valign="top" width="25%"><a href="certificates/oracle-database.jpg"><img src="certificates/oracle-database.jpg" width="160" alt="Oracle Database Certificate"></a><br/>Oracle Database<br/><sub>Oracle · 2023</sub></td>
    <td align="center" valign="top" width="25%"><a href="certificates/automation-anywhere.jpg"><img src="certificates/automation-anywhere.jpg" width="160" alt="Certified Advanced Automation Professional"></a><br/>Advanced Automation Professional<br/><sub>Automation Anywhere · 2024</sub></td>
    <td align="center" valign="top" width="25%"><a href="certificates/iiit-training.jpg"><img src="certificates/iiit-training.jpg" width="160" alt="IIIT Hyderabad 6-month AI/ML training"></a><br/>6-Month AI/ML Training<br/><sub>IIIT Hyderabad · 2024</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top"><a href="certificates/visuara-llms.jpg"><img src="certificates/visuara-llms.jpg" width="160" alt="Build Large Language Models From Scratch"></a><br/>Build LLMs From Scratch<br/><sub>Visuara · 2024</sub></td>
    <td align="center" valign="top"><a href="certificates/hf-mcp-unit1.jpg"><img src="certificates/hf-mcp-unit1.jpg" width="160" alt="Hugging Face MCP Unit 1"></a><br/>MCP Unit 1<br/><sub>Hugging Face · 2024</sub></td>
    <td align="center" valign="top"><a href="certificates/hf-mcp-unit3.jpg"><img src="certificates/hf-mcp-unit3.jpg" width="160" alt="Hugging Face MCP Unit 3"></a><br/>MCP Unit 3<br/><sub>Hugging Face · 2024</sub></td>
    <td align="center" valign="top"><a href="certificates/google-gdg.jpg"><img src="certificates/google-gdg.jpg" width="160" alt="Google Developer Groups certificate"></a><br/>Google GDG<br/><sub>GDG · 2025</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top"><a href="certificates/udemy-linear-algebra.jpg"><img src="certificates/udemy-linear-algebra.jpg" width="160" alt="Become a Linear Algebra Master"></a><br/>Linear Algebra Master<br/><sub>Udemy · 2026</sub></td>
    <td align="center" valign="top"><a href="certificates/udemy-deep-learning.jpg"><img src="certificates/udemy-deep-learning.jpg" width="160" alt="A Deep Understanding of Deep Learning"></a><br/>Deep Learning with Python<br/><sub>Udemy · 2026</sub></td>
    <td align="center" valign="top"><a href="certificates/sac-appreciation.jpg"><img src="certificates/sac-appreciation.jpg" width="160" alt="Student Activity Center certificate"></a><br/>SAC Appreciation<br/><sub>KL University · 2022–23</sub></td>
    <td align="center" valign="top"><a href="certificates/streetcause.jpg"><img src="certificates/streetcause.jpg" width="160" alt="Street Cause volunteer certificate"></a><br/>Volunteer<br/><sub>Street Cause · 2023–24</sub></td>
  </tr>
</table>

</details>

---

## Awards

**1st Place, Global Challenge Lab 2026, Imperial College London** · July 2026

Imperial Enterprise Lab and Futurize's global innovation sprint: 1,000+ students from partner universities worldwide, 14 days, four tracks, one Demo Day. Won with OrbitOps, a vendor-neutral reliability layer for satellite onboard AI:

- Inject radiation-style faults before launch and measure what actually breaks
- Watch the model in orbit with known-answer probes (catches silent degradation that never crashes and never lowers confidence)
- Roll back to a clean copy the moment it breaks

Five people, five disciplines: AI/ML ×2, aerospace, physics and mechanical, finance. I led the AI/ML work and technical strategy.

Earlier: 1st Place, University Webathon (2022) · 2nd Place, Design Expo (2022–23) · Technical Lead, Student Activity Council, KL University (2023–25), SAC Momentum Award.

---

## GitHub activity

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake.svg">
  <img alt="Contribution graph rendered as a snake game" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake.svg">
</picture>

</div>

<details>
<summary>Full stats: activity, languages, contribution calendar, top repositories</summary>

<div align="center">
  <img alt="GitHub metrics: activity, languages, contribution calendar and top repositories" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/github-metrics.svg" width="480">
</div>

</details>

---

## Contact

[gruheshkurra2@gmail.com](mailto:gruheshkurra2@gmail.com) · [LinkedIn](https://www.linkedin.com/in/gruheshkurra/) · [gruheshkurra.com](https://gruheshkurra.com)

Open to remote AI/ML research roles and to collaboration on transformer research, neural compression, on-device AI and agentic systems.

| Where | Link |
|:---|:---|
| Blog: AI/ML deep dives | [blogs.gruheshkurra.com](https://blogs.gruheshkurra.com) |
| Portfolio | [gruheshkurra.com](https://gruheshkurra.com) |
| Hugging Face | [karthik-2905](https://huggingface.co/karthik-2905) |
| Kaggle | [gruheshkurra](https://www.kaggle.com/gruheshkurra) |
| ORCID | [0009-0002-0558-2882](https://orcid.org/0009-0002-0558-2882) |
| LinkedIn | [gruheshkurra](https://www.linkedin.com/in/gruheshkurra/) |
| X | [@Karthik__kurra](https://x.com/Karthik__kurra) |
| DEV | [dev.to/gruhesh_kurra](https://dev.to/gruhesh_kurra_6eb933146da) |

<sub>Also found as Gruhesh Kurra · Karthik Kurra · Gruhesh Sri Sai Karthik · Gruhesh Karthik.</sub>
