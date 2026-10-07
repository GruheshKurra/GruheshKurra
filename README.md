<a href="https://gruheshkurra.com">
  <img width="100%" alt="Karthik Kurra — machine learning from the maths up" src="https://capsule-render.vercel.app/api?type=waving&height=230&section=header&color=0:0b3d2e,55:0a7f5a,100:1f6feb&text=Karthik%20Kurra&fontColor=ffffff&fontSize=60&fontAlignY=38&desc=Machine%20learning%20from%20the%20maths%20up&descSize=20&descAlignY=60&animation=fadeIn" />
</a>

<div align="center">

<a href="https://gruheshkurra.com"><img alt="Typing summary" src="https://readme-typing-svg.demolab.com/?font=JetBrains+Mono&weight=600&size=19&duration=3200&pause=900&color=2EA88A&center=true&vCenter=true&width=640&height=40&lines=MSc+Computing+(AI+%26+ML)+%40+Imperial+College+London;Language+models+%C2%B7+deepfake+forensics+%C2%B7+efficient+inference;Derive+%E2%86%92+implement+%E2%86%92+train+%E2%86%92+evaluate;Open+to+research+engineering+%26+ML+internships" /></a>

<br/>

<a href="https://gruheshkurra.com"><img alt="Portfolio" src="https://img.shields.io/badge/Portfolio-0a7f5a?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>&nbsp;<a href="https://blogs.gruheshkurra.com"><img alt="Blog" src="https://img.shields.io/badge/Blog-1f6feb?style=for-the-badge&logo=rss&logoColor=white" /></a>&nbsp;<a href="https://huggingface.co/karthik-2905"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" /></a>&nbsp;<a href="https://www.linkedin.com/in/gruheshkurra/"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" /></a>&nbsp;<a href="https://orcid.org/0009-0002-0558-2882"><img alt="ORCID" src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" /></a>&nbsp;<a href="mailto:gruheshkurra2@gmail.com"><img alt="Email" src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white" /></a>

</div>

<br/>

<table>
<tr>
<td width="62%" valign="top">

### About me

I build machine learning models from the maths up: derive the equations, write the code, train the model, and check what it learned.

I'm **Gruhesh Sri Sai Karthik Kurra**, based in London and enrolled in the **MSc Computing (Artificial Intelligence and Machine Learning) at Imperial College London** for 2026–2027. My work spans language models, representation learning, deepfake detection, and efficient inference.

I'm looking for **research engineering and ML internships** in model training, evaluation, and inference.

</td>
<td width="38%" valign="top">

### At a glance

- **Now:** MSc AI & ML, Imperial
- **Based in:** London, UK
- **Focus:** LMs · forensics · inference
- **Seeking:** research engineering and ML internships
- **B.Tech:** CGPA 9.72 / 10

</td>
</tr>
</table>

<div align="center">

<img alt="1st place, Global Challenge Lab 2026" src="https://img.shields.io/badge/1st%20place-Global%20Challenge%20Lab%202026-0a7f5a?style=flat-square&logo=starship&logoColor=white" />
<img alt="First-author IEEE paper" src="https://img.shields.io/badge/First--author-IEEE%20ICCCMLA%202025-00629B?style=flat-square&logo=ieee&logoColor=white" />
<img alt="Two papers under review" src="https://img.shields.io/badge/Under%20review-2%20papers%20at%20IEEE%20NEPCON%202026-d29922?style=flat-square" />
<img alt="Imperial College London" src="https://img.shields.io/badge/Imperial-MSc%202026%E2%80%932027-0000CD?style=flat-square" />

</div>

<br/>

## Selected projects

<table>
<tr>
<td width="50%" valign="top">

### [A language model in pure NumPy](https://github.com/GruheshKurra/core-language-model)

<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" /> <img src="https://img.shields.io/badge/autograd-from%20scratch-2ea88a?style=flat-square" alt="Autograd from scratch" /> <img src="https://img.shields.io/badge/tests-78-2ea44f?style=flat-square" alt="78 tests" />

A **3.87M-parameter** decoder-only transformer with my own automatic differentiation engine and byte-pair encoding tokeniser. I implemented grouped-query attention, rotary position embeddings, SwiGLU, QK-Norm, and a key-value cache, then pretrained on DailyDialog and fine-tuned on 19,375 EmpatheticDialogues conversations. The repository includes 78 unit tests for the gradients, model, tokeniser, and training code.

[Model weights](https://huggingface.co/karthik-2905/model-a-scratch) · [Build walkthrough](https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/)

</td>
<td width="50%" valign="top">

### [Teaching Qwen3-0.6B to call tools](https://github.com/GruheshKurra/AL1-model-B)

<img src="https://img.shields.io/badge/TRL-LoRA-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="TRL LoRA" /> <img src="https://img.shields.io/badge/Qwen3-0.6B-615ced?style=flat-square" alt="Qwen3-0.6B" /> <img src="https://img.shields.io/badge/tool%20calls-6%E2%86%9211%20%2F%2012-2ea44f?style=flat-square" alt="6 to 11 of 12" />

A LoRA fine-tune using TRL, assistant-only loss, and 1,423 tool and chat examples. The model learns five file and shell tools in Hermes format. Full tool-call exact match rose from **6/12 to 11/12** on a small, 12-case greedy evaluation against the base model.

[LoRA adapter](https://huggingface.co/karthik-2905/AL1-model-B)

</td>
</tr>
<tr>
<td width="50%" valign="top">

### [Dual-Stream deepfake detection](https://github.com/GruheshKurra/radar_deepfake)

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" /> <img src="https://img.shields.io/badge/frames-227%2C504-1f6feb?style=flat-square" alt="227,504 frames" /> <img src="https://img.shields.io/badge/AUC-96.3%25%20video--disjoint-2ea44f?style=flat-square" alt="96.3% AUC" />

A Sobel-edge boundary stream and a multi-band Fourier frequency stream, combined through iterative cross-attention. I trained on 227,504 frames from FaceForensics++, Celeb-DF v2, and WildDeepfake. After auditing source-video overlap, the model reached **96.3% AUC** on a 5,680-frame video-disjoint subset and **87.1% AUC** on Celeb-DF. These results come from one training run.

[Frame dataset](https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames)

</td>
<td width="50%" valign="top">

### [MANCE: morphology-aware word embeddings](https://github.com/GruheshKurra/MANCE-NLP)

<img src="https://img.shields.io/badge/NLP-embeddings-8957e5?style=flat-square" alt="NLP" /> <img src="https://img.shields.io/badge/IEEE-ICCCMLA%202025-00629B?style=flat-square&logo=ieee&logoColor=white" alt="IEEE ICCCMLA 2025" />

Word representations built from nested character sequences, so related word forms can share information. The work explores recurrent, convolutional, and Transformer encoders for classification and named entity recognition. **First-author paper accepted at IEEE ICCCMLA 2025.**

[Paper on IEEE Xplore](https://ieeexplore.ieee.org/document/11580466)

</td>
</tr>
<tr>
<td width="50%" valign="top">

### [DeepGuard: a deepfake model for on-device use](https://github.com/GruheshKurra/Deepguard)

<img src="https://img.shields.io/badge/Core%20ML-26%20MB-000000?style=flat-square&logo=apple&logoColor=white" alt="Core ML 26 MB" /> <img src="https://img.shields.io/badge/EfficientNet-B1-1f6feb?style=flat-square" alt="EfficientNet-B1" />

An EfficientNet-B1 classifier trained on 25,000 images using Apple Silicon and converted to a **26 MB Core ML model** for iOS inference. This project covers model training and conversion; an end-user app has not shipped.

[Demo](https://www.youtube.com/watch?v=4MmHJNjLRy4)

</td>
<td width="50%" valign="top">

### [Bit-Index Encoding](https://github.com/GruheshKurra/bit-index-encoding-research-)

<img src="https://img.shields.io/badge/Numba-kernels-00A3E0?style=flat-square&logo=numba&logoColor=white" alt="Numba" /> <img src="https://img.shields.io/badge/model-compression-2ea88a?style=flat-square" alt="Model compression" />

Experiments in compressing sparse neural-network weights by storing nonzero bit positions. Includes Numba sparse matrix multiplication kernels and benchmarks for computing with the compressed representation.

[Preprint on Zenodo](https://zenodo.org/records/17217218)

</td>
</tr>
</table>

## Research

I work on word representation, model compression, and the evidence used by detection systems.

| Paper | Status |
| :-- | :-- |
| [Morphology-Aware Nested Character Embeddings for Word Representation](https://ieeexplore.ieee.org/document/11580466) | <img src="https://img.shields.io/badge/Accepted-IEEE%20ICCCMLA%202025-2ea44f?style=flat-square" alt="Accepted, IEEE ICCCMLA 2025" /> |
| [Dual-Stream Artifact Detection with Iterative Evidence Refinement for Frame-Level Deepfake Recognition](https://github.com/GruheshKurra/radar_deepfake) | <img src="https://img.shields.io/badge/Under%20review-IEEE%20NEPCON%202026-d29922?style=flat-square" alt="Under review, IEEE NEPCON 2026" /> |
| Prism Tuning: Repulsion-Trained Seed Embeddings in a Frozen Transformer for Non-Redundant Generation | <img src="https://img.shields.io/badge/Under%20review-IEEE%20NEPCON%202026-d29922?style=flat-square" alt="Under review, IEEE NEPCON 2026" /><br/><sub>Method paper; empirical evaluation pending</sub> |
| [BIE: Bit-Index Encoding for Efficient Neural Network Weight Compression](https://zenodo.org/records/17217218) | <img src="https://img.shields.io/badge/Preprint-Zenodo%202025-6e7781?style=flat-square" alt="Preprint, Zenodo, 2025" /> |
| [Hybrid RAG-Enhanced Deepfake Detection: A Novel Approach Combining Retrieval-Augmented Generation with Visual Inconsistency Analysis](https://zenodo.org/records/16732053) | <img src="https://img.shields.io/badge/Preprint-Zenodo%202025-6e7781?style=flat-square" alt="Preprint, Zenodo, 2025" /><br/><sub>Architecture and evaluation plan</sub> |
| RADAR: Reasoning-Augmented Deepfake Artifact Recognition via Multi-Branch Evidence Aggregation | <img src="https://img.shields.io/badge/Preprint-2026-6e7781?style=flat-square" alt="Preprint, 2026" /><br/><sub>Architecture and evaluation protocol</sub> |

<sub>Preprints have not been peer reviewed. Prism Tuning, Hybrid RAG, and RADAR describe proposed methods; their design targets are not measured model results.</sub>

## Experience

<table>
<tr>
<td width="72" align="center" valign="top"><img src="assets/org-logos/infoajax.png" width="48" alt="InfoAjax Consulting" /></td>
<td valign="top">

**AI Integration Engineer** · InfoAjax Consulting · <sub>Oct–Nov 2025</sub><br/>
Built enterprise and cybersecurity integrations using Azure OpenAI agents for schema mapping. Delivered Azure-to-Salesforce proofs of concept and a React, FastAPI, and WebSocket console for live agent monitoring.

</td>
</tr>
<tr>
<td width="72" align="center" valign="top"><img src="assets/org-logos/iiith.png" width="56" alt="IIIT Hyderabad" /></td>
<td valign="top">

**Research Intern** · IIIT Hyderabad · <sub>Dec 2024–Jun 2025</sub><br/>
Built a document translation pipeline that preserves fonts, colours, and page structure using Vision Transformer layout detection, Tesseract OCR, and FastAPI. Supported English, Hindi, Telugu, and German, and presented a live demo at the institute's May 2025 research expo. [Demo](https://youtu.be/rcvSuBcBjyg)

</td>
</tr>
<tr>
<td width="72" align="center" valign="top"><img src="assets/org-logos/wso2.png" width="48" alt="WSO2" /></td>
<td valign="top">

**WSO2 API Developer** · InfoAjax Consulting · <sub>Oct 2024–Jun 2025</sub><br/>
Shipped REST ticketing APIs for PLDT through WSO2 Integration Studio and Choreo, and supported production integrations under client service-level agreements.

</td>
</tr>
<tr>
<td width="72" align="center" valign="top"><img src="assets/org-logos/zynthetix.png" width="48" alt="Zynthetix" /></td>
<td valign="top">

**Founder** · Zynthetix · <sub>Mar 2024–Jan 2025</sub><br/>
Designed a parent-and-child model architecture for generating synthetic tabular, image, and text data with less duplication. The diversity objective later informed Prism Tuning.

</td>
</tr>
<tr>
<td width="72" align="center" valign="top"><img src="assets/org-logos/ihub.png" width="48" alt="iHub-Data" /></td>
<td valign="top">

**AI and ML Research Trainee** · iHub-Data, IIIT Hyderabad · <sub>May–Oct 2024</sub><br/>
Completed a six-month faculty-mentored programme in model architecture, language models, fine-tuning, quantisation, and deployment.

</td>
</tr>
</table>

## Education and awards

<table>
<tr>
<td width="50%" valign="top">

#### Education

**Imperial College London**<br/>
MSc Computing (Artificial Intelligence and Machine Learning)<br/>
<sub>Sep 2026–Sep 2027, expected · Enrolled for September 2026</sub>

**KL University, Hyderabad**<br/>
B.Tech Computer Science and Engineering<br/>
<sub>Aug 2021–May 2025 · CGPA 9.72/10</sub>

</td>
<td width="50%" valign="top">

#### Awards

**1st place · Global Challenge Lab 2026** · Imperial, July 2026<br/>
<sub>Led AI/ML work and technical strategy for five-person Team Corio. Our project, OrbitOps, proposed a reliability layer for satellite AI using pre-launch fault injection, in-orbit known-answer probes, and rollback. The team won £1,500. [Certificate](certificates/global-challenge-lab-2026.jpg)</sub>

**1st place · University Webathon** · KL University, 2022<br/>
<sub>Built a diet-management platform in four hours, competing against more than 50 teams.</sub>

**2nd place · Design Expo** · KL University, 2022–2023<br/>
<sub>Built an Arduino smart switchboard with energy monitoring and app control.</sub>

**SAC Momentum Award** · KL University, 2023–2025<br/>
<sub>As Student Activity Council Technical Lead, I mentored 20+ juniors, ran workshops for 100+ students, and coordinated 15+ events.</sub>

</td>
</tr>
</table>

## Learning in public

I write [**AI from Scratch**](https://blogs.gruheshkurra.com/series/ai/), a series that moves from derivations and hand-worked examples to runnable implementations.

#### Start with

| Post | What it covers |
| :-- | :-- |
| [Build an autograd engine](https://blogs.gruheshkurra.com/blog/build-autograd-from-scratch/) | Automatic differentiation and gradient checks |
| [Attention in Transformers](https://blogs.gruheshkurra.com/blog/attention-in-transformers-explained/) | The attention calculation, step by step |
| [GPT-2 from scratch](https://blogs.gruheshkurra.com/blog/build-gpt2-from-scratch/) | A 124M-parameter architecture in PyTorch |
| [A mini language model in NumPy](https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/) | Tokenisation, model code, and training |

#### Latest writing

<!-- BLOG-POST-LIST:START --><a href="https://blogs.gruheshkurra.com/blog/convolutional-neural-networks-explained/">Convolutional Neural Networks: CNN Math Explained</a><br/><a href="https://blogs.gruheshkurra.com/blog/deepseek-v4-inside-one-token/">DeepSeek V4 Inside: One Token Through Every Block</a><br/><a href="https://blogs.gruheshkurra.com/blog/looped-transformers-explained/">Looped Transformers Explained: Recurrent Depth and Astra</a><br/><a href="https://blogs.gruheshkurra.com/blog/natural-language-inference-explained/">Natural Language Inference Explained: Entailment in NLP</a><br/><a href="https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/">Build a Mini LLM from Scratch in NumPy: RoPE, GQA, SwiGLU</a><br/><!-- BLOG-POST-LIST:END -->

[All posts](https://blogs.gruheshkurra.com/ai-explanations/) · [RSS](https://blogs.gruheshkurra.com/feed.xml) · [DEV](https://dev.to/gruhesh_kurra_6eb933146da)

<details>
<summary><b>From-scratch implementations</b> — notebooks and repositories</summary>
<br/>

| Area | Implementations |
| :-- | :-- |
| Language models | [GPT-2](https://github.com/GruheshKurra/FirstGPTFromScratch) · [Encoder-decoder Transformers](https://github.com/GruheshKurra/TransformersFromScratch) · [LLaMA-style decoding](https://github.com/GruheshKurra/LLamaModel) · [Attention](https://github.com/GruheshKurra/AttentionMechanisms) |
| Generative and graph models | [Diffusion](https://github.com/GruheshKurra/DiffusionModelFromScratch) · [GANs](https://github.com/GruheshKurra/GAN_Implementation) · [VAEs](https://github.com/GruheshKurra/VariationalAutoencoders) · [Graph neural networks](https://github.com/GruheshKurra/GraphNeuralNetworks-GNN-) |
| ML foundations | [SVMs](https://github.com/GruheshKurra/SVM-Implementation-From-Scratch) · [Random forests](https://github.com/GruheshKurra/random-forest-from-scratch) · [PCA](https://github.com/GruheshKurra/dimensionality-reduction) · [k-means](https://github.com/GruheshKurra/k-means-clustering) · [Temporal-difference learning](https://github.com/GruheshKurra/TemporalDifferenceLearning) |
| Applied work | [Natural language to SQL](https://github.com/GruheshKurra/nl2sql-pretrained) · [Clinical trial retrieval with PubMedBERT and FAISS](https://github.com/GruheshKurra/Clinical-Trial-Similarity-Analysis) · [WSO2 on Kubernetes](https://github.com/GruheshKurra/WSO2-Kubernetes-Support) · [AI learning roadmaps](https://github.com/GruheshKurra/awesome-ai-roadmaps) |

</details>

## Tools I work with

<div align="center">

<img alt="Tech stack" src="https://skillicons.dev/icons?i=py,pytorch,tensorflow,sklearn,opencv,swift,apple,fastapi,react,ts,docker,kubernetes,aws,azure,postgres,mongodb,redis,git&perline=9" />

</div>

| Area | Tools and methods |
| :-- | :-- |
| Model development | Python, PyTorch, NumPy, TensorFlow, scikit-learn, Hugging Face Transformers |
| Training and efficiency | TRL, PEFT, LoRA, quantisation, pruning, distillation, Numba |
| Vision and retrieval | OpenCV, Vision Transformers, Tesseract, CLIP, FAISS |
| On-device models | Core ML, Swift, Apple Silicon / Metal MPS |
| APIs and applications | FastAPI, React, TypeScript, REST, WebSocket, WSO2 |
| Infrastructure and data | Docker, Kubernetes, Git, AWS, Azure, PostgreSQL, MongoDB, Redis |

**Certificates:** [TensorFlow Developer](certificates/tensorflow-developer.jpg) · [AWS Solutions Architect – Associate](certificates/aws-solutions-architect.jpg) · [OCI Generative AI Professional](certificates/oracle-genai.jpg) · WSO2 Micro Integrator V4 [Developer](certificates/wso2-mi-developer.jpg) and [Practitioner](certificates/wso2-mi-practitioner.jpg) · [All certificates](certificates/)

## GitHub activity

<div align="center">

<img height="165" alt="GitHub stats" src="https://github-readme-stats.vercel.app/api?username=GruheshKurra&show_icons=true&disable_animations=true&include_all_commits=true&count_private=true&hide_border=true&bg_color=00000000&title_color=2ea88a&icon_color=2ea88a&text_color=8b949e&ring_color=2ea88a" />
<img height="165" alt="GitHub streak" src="https://streak-stats.demolab.com/?user=GruheshKurra&hide_border=true&background=00000000&ring=2ea88a&fire=2ea88a&currStreakLabel=2ea88a&currStreakNum=8b949e&sideNums=8b949e&sideLabels=8b949e&dates=8b949e&stroke=8b949e40" />

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake.svg" />
  <img alt="Contribution snake" src="https://raw.githubusercontent.com/GruheshKurra/GruheshKurra/output/github-snake.svg" />
</picture>

</div>

<details>
<summary><b>Full metrics</b> — languages, topics, repositories and calendar</summary>
<br/>
<div align="center"><img src="github-metrics.svg" alt="GitHub metrics" /></div>
</details>

## Let's talk

For research engineering opportunities or collaboration on language models, model compression, and on-device inference, reach me at [gruheshkurra2@gmail.com](mailto:gruheshkurra2@gmail.com) or on [LinkedIn](https://www.linkedin.com/in/gruheshkurra/).

<img width="100%" alt="" src="https://capsule-render.vercel.app/api?type=waving&height=120&section=footer&color=0:0b3d2e,55:0a7f5a,100:1f6feb" />
