<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/header-dark.svg?v=2">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/header-light.svg?v=2">
  <img alt="Gruhesh Sri Sai Karthik Kurra. I write the maths, then I train the model. MSc Computing (AI and ML), Imperial College London, 2026–27." src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/header-light.svg?v=2" width="100%">
</picture>

<p align="center">
  <a href="https://gruheshkurra.com"><b>Portfolio</b></a> &nbsp;·&nbsp;
  <a href="https://blogs.gruheshkurra.com"><b>Blog</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/karthik-2905"><b>Hugging Face</b></a> &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/gruheshkurra/"><b>LinkedIn</b></a> &nbsp;·&nbsp;
  <a href="https://orcid.org/0009-0002-0558-2882"><b>ORCID</b></a> &nbsp;·&nbsp;
  <a href="mailto:gruheshkurra2@gmail.com"><b>Email</b></a>
</p>

Hi, I'm Karthik. I like to understand a model well enough to build it with nothing but NumPy. So I work through the derivation on paper, write the code, train it, and measure what it does. That habit gave me a small language model with its own autograd engine, a Qwen3 fine-tune that calls tools, and papers on word embeddings, weight compression, and deepfake detection.

In September 2026 I start the MSc in Computing (Artificial Intelligence and Machine Learning) at Imperial College London. I'm looking for research engineering internships in model training, evaluation, or efficient inference.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/numbers-dark.svg?v=2">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/numbers-light.svg?v=2">
  <img alt="1st place, Global Challenge Lab 2026. First-author IEEE paper, ICCCMLA 2025. 19 AI from Scratch posts. B.Tech CGPA 9.72 out of 10." src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/numbers-light.svg?v=2" width="100%">
</picture>

## Things I've built

<p align="center">
<a href="https://github.com/GruheshKurra/core-language-model"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/numpy-lm-dark.svg?v=2"><img alt="A language model in pure NumPy: 3.87M parameters on my own autograd engine, 78 unit tests" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/numpy-lm-light.svg?v=2" width="49%"></picture></a>
<a href="https://github.com/GruheshKurra/AL1-model-B"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/qwen-tools-dark.svg?v=2"><img alt="Teaching Qwen3-0.6B to call tools: exact tool calls rose from 6 to 11 of 12 after LoRA fine-tuning" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/qwen-tools-light.svg?v=2" width="49%"></picture></a>
<a href="https://github.com/GruheshKurra/radar_deepfake"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/dual-stream-dark.svg?v=2"><img alt="Dual-Stream deepfake detection: 96.3% AUC on a video-disjoint split of 227,504 frames" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/dual-stream-light.svg?v=2" width="49%"></picture></a>
<a href="https://github.com/GruheshKurra/MANCE-NLP"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/mance-dark.svg?v=2"><img alt="MANCE, morphology-aware embeddings, IEEE ICCCMLA 2025: 99.0% on DBpedia, 93.5% on AG News" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/mance-light.svg?v=2" width="49%"></picture></a>
<a href="https://github.com/GruheshKurra/Deepguard"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/deepguard-dark.svg?v=2"><img alt="DeepGuard for iOS: a 26 MB Core ML deepfake detector, under 500 ms per image" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/deepguard-light.svg?v=2" width="49%"></picture></a>
<a href="https://github.com/GruheshKurra/bit-index-encoding-research-"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/bie-dark.svg?v=2"><img alt="Bit-index encoding: 40 times compression of sparse weights at 95% sparsity" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/assets/cards/bie-light.svg?v=2" width="49%"></picture></a>
</p>

Weights and data: [NumPy LM](https://huggingface.co/karthik-2905/model-a-scratch) · [Qwen3 adapter](https://huggingface.co/karthik-2905/AL1-model-B) · [deepfake frame corpus](https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames) · [DeepGuard demo](https://www.youtube.com/watch?v=4MmHJNjLRy4)

DeepGuard was trained and converted, not shipped as a product. The Qwen3 result comes from 12 test cases.

<details>
<summary><b>More models I've rebuilt</b></summary>
<br>

| Model | What it is |
|:---|:---|
| [GPT-2 from scratch](https://github.com/GruheshKurra/FirstGPTFromScratch) | 124M parameters in PyTorch, with my own BPE tokeniser and training loop |
| [Attention Is All You Need](https://github.com/GruheshKurra/TransformersFromScratch) | The original encoder-decoder transformer, checked with BLEU |
| [LLaMA-style decoder](https://github.com/GruheshKurra/LLamaModel) | RMSNorm, RoPE, and SwiGLU, with a training recipe |
| [NL2SQL](https://github.com/GruheshKurra/nl2sql-pretrained) | Turns plain-English questions into SQL; perplexity 1.42 |
| [Clinical trial search](https://github.com/GruheshKurra/Clinical-Trial-Similarity-Analysis) | PubMedBERT embeddings with FAISS retrieval |
| [Graph neural networks](https://github.com/GruheshKurra/GraphNeuralNetworks-GNN-) | GCN, GraphSAGE, and GAT for node classification |
| [Diffusion](https://github.com/GruheshKurra/DiffusionModelFromScratch) | DDPM with a ~130K-parameter UNet |
| [GANs](https://github.com/GruheshKurra/GAN_Implementation) · [VAEs](https://github.com/GruheshKurra/VariationalAutoencoders) | Generative models on MNIST |
| [Attention](https://github.com/GruheshKurra/AttentionMechanisms) · [RNN, LSTM, GRU](https://github.com/GruheshKurra/SequenceModeling) | The pieces inside sequence models |

**Reinforcement learning:** [TD learning](https://github.com/GruheshKurra/TemporalDifferenceLearning) · [Monte Carlo](https://github.com/GruheshKurra/MonteCarloMethods) · [MDPs](https://github.com/GruheshKurra/MarkovsDecisionPlane)

**Classical ML:** [SVM](https://github.com/GruheshKurra/SVM-Implementation-From-Scratch) · [random forest](https://github.com/GruheshKurra/random-forest-from-scratch) · [decision trees](https://github.com/GruheshKurra/decision-trees-from-scratch) · [logistic](https://github.com/GruheshKurra/logistic-regression-impl) and [linear](https://github.com/GruheshKurra/linear-regression-impl) regression · [naive Bayes](https://github.com/GruheshKurra/naive-bayes-implementation) · [kNN](https://github.com/GruheshKurra/knn-implementation) · [Bayesian networks](https://github.com/GruheshKurra/Bayesian-Networks) · [anomaly detection](https://github.com/GruheshKurra/AnomalyDetection)

**Unsupervised:** [k-means](https://github.com/GruheshKurra/k-means-clustering) · [DBSCAN](https://github.com/GruheshKurra/dbscan-clustering) · [hierarchical clustering](https://github.com/GruheshKurra/hierarchical-clustering) · [t-SNE](https://github.com/GruheshKurra/tsne-from-scratch) · [UMAP](https://github.com/GruheshKurra/umap-dimensionality-reduction) · [PCA](https://github.com/GruheshKurra/dimensionality-reduction)

**Apps and tools:** [AI voice assistant for Windows](https://github.com/GruheshKurra/AI-Voice-Assistant-for-Windows) · [WSO2 on Kubernetes](https://github.com/GruheshKurra/WSO2-Kubernetes-Support) · [FarmCare](https://github.com/GruheshKurra/Farmcare) · [healthcareAI](https://github.com/GruheshKurra/healthcareAI) · [AI learning roadmaps](https://github.com/GruheshKurra/awesome-ai-roadmaps)

</details>

## Papers

**Accepted**

- [**Morphology-Aware Nested Character Embeddings for Word Representation**](https://ieeexplore.ieee.org/document/11580466)<br>Kurra, Moparthi, Chowdary, Pagadala · IEEE ICCCMLA 2025 · [code](https://github.com/GruheshKurra/MANCE-NLP)

**Under review at IEEE NEPCON 2026**

- [**Dual-Stream Artifact Detection with Iterative Evidence Refinement for Frame-Level Deepfake Recognition**](https://github.com/GruheshKurra/radar_deepfake)<br>Kurra, Pagadala, Batumalay, Boddapati · code, data, and leakage audit are public
- **Prism Tuning: Repulsion-Trained Seed Embeddings in a Frozen Transformer for Non-Redundant Generation**<br>Kurra, Pagadala, Batumalay, Koduru · method paper, no experimental results yet

**Preprints** (not peer-reviewed)

- [**BIE: Bit-Index Encoding for Efficient Neural Network Weight Compression**](https://zenodo.org/records/17217218) · Zenodo, 2025
- [**Hybrid RAG-Enhanced Deepfake Detection**](https://zenodo.org/records/16732053) · Zenodo, 2025 · design and evaluation plan
- **RADAR: Reasoning-Augmented Deepfake Artifact Recognition via Multi-Branch Evidence Aggregation** · 2026 · design paper; its measured results are in Dual-Stream

## Where I've been

| When | Role | Where |
|:---|:---|:---|
| 2026 – 27 | MSc Computing (AI and ML) | Imperial College London |
| Jul 2026 | 1st place, Global Challenge Lab | Imperial College London |
| Oct – Nov 2025 | AI Integration Engineer | InfoAjax Consulting |
| Dec 2024 – Jun 2025 | Research Intern | IIIT Hyderabad |
| Oct 2024 – Jun 2025 | WSO2 API Developer | InfoAjax Consulting |
| May – Oct 2024 | AI and ML Research Trainee | iHub-Data, IIIT Hyderabad |
| Mar 2024 – Jan 2025 | Founder | Zynthetix |
| 2021 – 25 | B.Tech Computer Science, CGPA 9.72 / 10 | KL University, Hyderabad |

<details>
<summary><b>What I did in each role</b></summary>
<br>

**Global Challenge Lab, Imperial College London.** More than 1,000 students spent 14 days building ventures across four tracks. Five of us, as Team Corio, won with OrbitOps, a reliability layer for AI running on satellites. It injects radiation-style faults before launch, sends known-answer probes in orbit to catch silent degradation, and rolls back to a clean copy when something breaks. I led the AI/ML work and technical strategy. The team won £1,500.

**Research Intern, IIIT Hyderabad.** I built a system that translates documents while keeping the layout intact. A Vision Transformer finds the layout, character metrics identify the fonts, and K-means pulls out the colour palette. FastAPI and Tesseract OCR handle English, Hindi, Telugu, and German. I demonstrated it live at the institute's research expo in May 2025. [Demo video](https://youtu.be/rcvSuBcBjyg)

**AI Integration Engineer, InfoAjax.** I connected enterprise apps to cybersecurity services and used Azure OpenAI agents for schema mapping. I linked Android and Apple enterprise apps to Azure, built Azure-to-Salesforce proofs of concept, and made a React, FastAPI, and WebSocket console for watching agents live.

**WSO2 API Developer, InfoAjax.** I shipped REST ticketing APIs for PLDT, a Philippine telecom company, in WSO2 Integration Studio. They ran on WSO2 Choreo, with secrets kept in a GitLab vault, under client service-level agreements.

**Founder, Zynthetix.** I designed a synthetic-data architecture in which a parent model directs hundreds of specialised child models. Together they generate tabular, image, and text data without duplicates. That idea later became the Prism Tuning paper.

**Research Trainee, iHub-Data.** A six-month programme with faculty mentors, covering model architecture, language models, fine-tuning, quantisation, and deployment.

**KL University.** I was Technical Lead of the Student Activity Council from 2023 to 2025. I mentored more than 20 juniors, ran workshops for more than 100 students, and coordinated more than 15 events, and received the SAC Momentum Award. I also won the University Webathon in 2022 and took 2nd place at the Design Expo in 2022–23.

</details>

## Writing

I write [**AI from Scratch**](https://blogs.gruheshkurra.com/series/ai/), a series of 19 posts so far. Each post derives the maths, works through a small example by hand, and then turns it into code. Good places to start:

- [Backpropagation from scratch: build an autograd engine](https://blogs.gruheshkurra.com/blog/build-autograd-from-scratch/)
- [How attention works: queries, keys, values](https://blogs.gruheshkurra.com/blog/attention-in-transformers-explained/)
- [Build GPT-2 from scratch in PyTorch](https://blogs.gruheshkurra.com/blog/build-gpt2-from-scratch/)
- [Build a mini LLM in NumPy: RoPE, GQA, SwiGLU](https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/)

<details>
<summary><b>Latest posts</b></summary>
<br>

[Every post](https://blogs.gruheshkurra.com/ai-explanations/) · [RSS](https://blogs.gruheshkurra.com/feed.xml) · [DEV](https://dev.to/gruhesh_kurra_6eb933146da)

<!-- BLOG-POST-LIST:START --><a href="https://blogs.gruheshkurra.com/blog/convolutional-neural-networks-explained/">Convolutional Neural Networks: CNN Math Explained</a><br/><a href="https://blogs.gruheshkurra.com/blog/deepseek-v4-inside-one-token/">DeepSeek V4 Inside: One Token Through Every Block</a><br/><a href="https://blogs.gruheshkurra.com/blog/looped-transformers-explained/">Looped Transformers Explained: Recurrent Depth and Astra</a><br/><a href="https://blogs.gruheshkurra.com/blog/natural-language-inference-explained/">Natural Language Inference Explained: Entailment in NLP</a><br/><a href="https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/">Build a Mini LLM from Scratch in NumPy: RoPE, GQA, SwiGLU</a><br/><!-- BLOG-POST-LIST:END -->

</details>

## Tools I reach for

- **Every day:** Python, PyTorch, NumPy, Hugging Face Transformers, TRL, PEFT/LoRA, FastAPI, Docker, Git
- **Models:** transformers (GQA, RoPE, SwiGLU, KV-cache), BPE tokenisers, diffusion, GANs, VAEs, GNNs, RAG with FAISS
- **Vision and devices:** EfficientNet, Vision Transformers, CLIP, OpenCV, Tesseract, Core ML, Swift
- **Efficiency:** LoRA, quantisation, pruning, distillation, Numba kernels
- **Infrastructure:** AWS, Azure, GCP, Oracle Cloud, RunPod, Kubernetes, WSO2, PostgreSQL, MongoDB, Redis
- **Languages:** Python, C++, C, Java, TypeScript, Swift, SQL, Shell

<details>
<summary><b>Certificates</b></summary>
<br>

**Machine learning:** [TensorFlow Developer](certificates/tensorflow-developer.jpg) (Google, 2024) · [OCI Generative AI Professional](certificates/oracle-genai.jpg) (Oracle, 2024) · [Build LLMs From Scratch](certificates/visuara-llms.jpg) (Visuara, 2024) · [MCP Course Unit 1](certificates/hf-mcp-unit1.jpg) and [Unit 3](certificates/hf-mcp-unit3.jpg) (Hugging Face, 2024) · [Six-month AI/ML training](certificates/iiit-training.jpg) (iHub-Data, 2024) · [Linear Algebra Master](certificates/udemy-linear-algebra.jpg) and [A Deep Understanding of Deep Learning](certificates/udemy-deep-learning.jpg) (Udemy, 2026)

**Cloud:** [AWS Solutions Architect – Associate](certificates/aws-solutions-architect.jpg) and [Cloud Practitioner](certificates/aws-cloud-practitioner.jpg) (2023) · [OCI Architect Associate](certificates/oracle-architect.jpg) and [Oracle Database](certificates/oracle-database.jpg) (2023)

**Integration:** [WSO2 Micro Integrator Developer V4](certificates/wso2-mi-developer.jpg) and [Practitioner V4](certificates/wso2-mi-practitioner.jpg) (2025) · [Red Hat Enterprise Application Developer](certificates/redhat-developer.jpg) (2024) · [Automation Anywhere Advanced Professional](certificates/automation-anywhere.jpg) (2024)

**Community:** [Global Challenge Lab 2026, 1st place](certificates/global-challenge-lab-2026.jpg) · [Google Developer Groups](certificates/google-gdg.jpg) (2025) · [KL University appreciation](certificates/sac-appreciation.jpg) (2022–23) · [Street Cause volunteer](certificates/streetcause.jpg) (2023–24)

English: IELTS Academic 7.5 (CEFR C1).

</details>

<details>
<summary><b>GitHub activity</b></summary>
<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake.svg">
  <img alt="Contribution graph rendered as a snake game" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake.svg" width="100%">
</picture>

<p align="center"><img alt="GitHub metrics: activity, languages, contribution calendar and top repositories" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/main/github-metrics.svg" width="480"></p>

</details>

## Say hello

If you work on language models, compression, or on-device inference and want to talk about an internship or a collaboration, email me at [gruheshkurra2@gmail.com](mailto:gruheshkurra2@gmail.com). I'm also on [LinkedIn](https://www.linkedin.com/in/gruheshkurra/), [Hugging Face](https://huggingface.co/karthik-2905), [Kaggle](https://www.kaggle.com/gruheshkurra), and [X](https://x.com/Karthik__kurra).

<sub>You may also know me as Gruhesh Kurra or Karthik Kurra.</sub>
