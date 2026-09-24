### Hi, I'm Karthik

I build machine learning models from the maths up. I derive the equations, write the code, train the model, and measure what it does.

I'm starting the MSc in Computing (Artificial Intelligence and Machine Learning) at Imperial College London in September 2026. I'm looking for research engineering internships in model training, evaluation, or efficient inference.

[Portfolio](https://gruheshkurra.com) · [Blog](https://blogs.gruheshkurra.com) · [Hugging Face](https://huggingface.co/karthik-2905) · [LinkedIn](https://www.linkedin.com/in/gruheshkurra/) · [ORCID](https://orcid.org/0009-0002-0558-2882) · [Kaggle](https://www.kaggle.com/gruheshkurra) · [Email](mailto:gruheshkurra2@gmail.com)

## Projects

- **[A language model in pure NumPy](https://github.com/GruheshKurra/core-language-model)**. A 3.87M-parameter decoder-only transformer that runs on my own autograd engine, with RoPE, grouped-query attention, SwiGLU, QK-Norm, and a KV-cache. I pretrained it on DailyDialog and fine-tuned it on EmpatheticDialogues. It has 78 unit tests. [Weights](https://huggingface.co/karthik-2905/model-a-scratch)
- **[Tool calling for Qwen3-0.6B](https://github.com/GruheshKurra/AL1-model-B)**. A LoRA fine-tune with TRL on 1,423 tool and chat examples, using assistant-only loss and five file and shell tools in Hermes format. Exact tool calls went from 6 to 11 out of 12 test cases. [Adapter](https://huggingface.co/karthik-2905/AL1-model-B)
- **[Dual-Stream deepfake detection](https://github.com/GruheshKurra/radar_deepfake)**. A Sobel boundary stream and a multi-band FFT stream, fused by iterative cross-attention and trained on 227,504 frames from FaceForensics++, Celeb-DF v2, and WildDeepfake. It reaches 96.3% AUC on a video-disjoint split and 87.1% on Celeb-DF. [Dataset](https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames)
- **[MANCE](https://github.com/GruheshKurra/MANCE-NLP)**. Word embeddings built from nested character sequences, so related word forms share what they learn. It scores 99.0% on DBpedia, 93.5% on AG News, and 81.3% F1 on GermEval NER. Accepted at IEEE ICCCMLA 2025.
- **[DeepGuard](https://github.com/GruheshKurra/Deepguard)**. An EfficientNet-B1 deepfake detector trained on 25k images on an Apple M4, reaching 98.62% accuracy. I converted it to a 26 MB Core ML model that runs on iOS in under 500 ms per image. It is a trained and converted model, not a shipped app. [Demo](https://www.youtube.com/watch?v=4MmHJNjLRy4)
- **[Bit-index encoding](https://github.com/GruheshKurra/bit-index-encoding-research-)**. Compresses sparse weight matrices by storing the positions of nonzero bits, with Numba sparse matrix multiplication that works on the compressed form. It reaches 40× compression at 95% sparsity; the bitplane variant keeps reconstruction error below 10⁻⁶.

Other things I've built:

- [GPT-2 from scratch](https://github.com/GruheshKurra/FirstGPTFromScratch): 124M parameters in PyTorch, with my own BPE tokeniser and training loop.
- [Transformers from scratch](https://github.com/GruheshKurra/TransformersFromScratch): the encoder-decoder from *Attention Is All You Need*, checked with BLEU.
- [LLaMA-style decoder](https://github.com/GruheshKurra/LLamaModel): RMSNorm, RoPE, and SwiGLU, with a training recipe.
- [NL2SQL](https://github.com/GruheshKurra/nl2sql-pretrained): plain-English questions to SQL, perplexity 1.42.
- [Clinical trial search](https://github.com/GruheshKurra/Clinical-Trial-Similarity-Analysis): PubMedBERT embeddings with FAISS retrieval.
- [AI voice assistant for Windows](https://github.com/GruheshKurra/AI-Voice-Assistant-for-Windows), [WSO2 on Kubernetes](https://github.com/GruheshKurra/WSO2-Kubernetes-Support), [FarmCare](https://github.com/GruheshKurra/Farmcare), [healthcareAI](https://github.com/GruheshKurra/healthcareAI), and [AI learning roadmaps](https://github.com/GruheshKurra/awesome-ai-roadmaps).

I also keep small from-scratch implementations, written to learn how each method works:

- Deep learning: [attention](https://github.com/GruheshKurra/AttentionMechanisms), [RNN, LSTM, and GRU](https://github.com/GruheshKurra/SequenceModeling), [GANs](https://github.com/GruheshKurra/GAN_Implementation), [VAEs](https://github.com/GruheshKurra/VariationalAutoencoders), [diffusion](https://github.com/GruheshKurra/DiffusionModelFromScratch), [graph neural networks](https://github.com/GruheshKurra/GraphNeuralNetworks-GNN-)
- Reinforcement learning: [TD learning](https://github.com/GruheshKurra/TemporalDifferenceLearning), [Monte Carlo methods](https://github.com/GruheshKurra/MonteCarloMethods), [MDPs](https://github.com/GruheshKurra/MarkovsDecisionPlane)
- Classical ML: [SVM](https://github.com/GruheshKurra/SVM-Implementation-From-Scratch), [random forest](https://github.com/GruheshKurra/random-forest-from-scratch), [decision trees](https://github.com/GruheshKurra/decision-trees-from-scratch), [logistic regression](https://github.com/GruheshKurra/logistic-regression-impl), [linear regression](https://github.com/GruheshKurra/linear-regression-impl), [naive Bayes](https://github.com/GruheshKurra/naive-bayes-implementation), [kNN](https://github.com/GruheshKurra/knn-implementation), [Bayesian networks](https://github.com/GruheshKurra/Bayesian-Networks), [anomaly detection](https://github.com/GruheshKurra/AnomalyDetection)
- Unsupervised learning: [k-means](https://github.com/GruheshKurra/k-means-clustering), [DBSCAN](https://github.com/GruheshKurra/dbscan-clustering), [hierarchical clustering](https://github.com/GruheshKurra/hierarchical-clustering), [t-SNE](https://github.com/GruheshKurra/tsne-from-scratch), [UMAP](https://github.com/GruheshKurra/umap-dimensionality-reduction), [PCA](https://github.com/GruheshKurra/dimensionality-reduction)

## Papers

- [Morphology-Aware Nested Character Embeddings for Word Representation](https://ieeexplore.ieee.org/document/11580466). Kurra, Moparthi, Chowdary, Pagadala. Accepted at IEEE ICCCMLA 2025.
- [Dual-Stream Artifact Detection with Iterative Evidence Refinement for Frame-Level Deepfake Recognition](https://github.com/GruheshKurra/radar_deepfake). Kurra, Pagadala, Batumalay, Boddapati. Under review at IEEE NEPCON 2026.
- Prism Tuning: Repulsion-Trained Seed Embeddings in a Frozen Transformer for Non-Redundant Generation. Kurra, Pagadala, Batumalay, Koduru. Under review at IEEE NEPCON 2026. A method paper with no experimental results yet.
- [BIE: Bit-Index Encoding for Efficient Neural Network Weight Compression](https://zenodo.org/records/17217218). Preprint on Zenodo, 2025.
- [Hybrid RAG-Enhanced Deepfake Detection](https://zenodo.org/records/16732053). Preprint on Zenodo, 2025. Sets out the design and evaluation plan.
- RADAR: Reasoning-Augmented Deepfake Artifact Recognition via Multi-Branch Evidence Aggregation. Preprint, 2026. A design paper; its measured results are in the Dual-Stream paper.

Preprints are not peer-reviewed.

## Experience

- **Research Intern, IIIT Hyderabad** (Dec 2024 – Jun 2025). I built a system that translates documents while keeping the layout: a Vision Transformer finds the layout, character metrics identify fonts, and K-means extracts the colour palette. FastAPI and Tesseract OCR handle English, Hindi, Telugu, and German. I demonstrated it live at the institute's research expo in May 2025. [Demo](https://youtu.be/rcvSuBcBjyg)
- **AI Integration Engineer, InfoAjax Consulting** (Oct – Nov 2025). I connected enterprise apps to cybersecurity services, used Azure OpenAI agents for schema mapping, linked Android and Apple enterprise apps to Azure, built Azure-to-Salesforce proofs of concept, and made a React, FastAPI, and WebSocket console for monitoring agents live.
- **WSO2 API Developer, InfoAjax Consulting** (Oct 2024 – Jun 2025). I shipped REST ticketing APIs for PLDT, a Philippine telecom company, in WSO2 Integration Studio, deployed on WSO2 Choreo with secrets in a GitLab vault, under client service-level agreements.
- **Founder, Zynthetix** (Mar 2024 – Jan 2025). I designed a synthetic-data architecture in which a parent model directs hundreds of specialised child models to generate tabular, image, and text data without duplicates. The idea later became the Prism Tuning paper.
- **AI and ML Research Trainee, iHub-Data, IIIT Hyderabad** (May – Oct 2024). A six-month programme with faculty mentors, covering model architecture, language models, fine-tuning, quantisation, and deployment.

## Education

- **Imperial College London**, MSc Computing (Artificial Intelligence and Machine Learning), Sep 2026 – Sep 2027.
- **KL University, Hyderabad**, B.Tech Computer Science and Engineering, 2021 – 2025. CGPA 9.72 / 10.

IELTS Academic 7.5 (CEFR C1).

## Awards

- **1st place, Global Challenge Lab 2026**, Imperial College London (July 2026). Over 14 days, more than 1,000 students built ventures across four tracks. My five-person team, Team Corio, won with OrbitOps, a reliability layer for AI running on satellites. It injects radiation-style faults before launch, sends known-answer probes in orbit to catch silent degradation, and rolls back to a clean copy when something breaks. I led the AI/ML work and technical strategy. The prize was £1,500 for the team.
- **1st place, University Webathon**, KL University (2022). A diet-management platform built in four hours, against more than 50 teams.
- **2nd place, Design Expo**, KL University (2022 – 2023). An Arduino smart switchboard with energy monitoring and app control.
- **SAC Momentum Award**, KL University (2023 – 2025). As Technical Lead of the Student Activity Council, I mentored more than 20 juniors, ran workshops for more than 100 students, and coordinated more than 15 events.

## Writing

I write [AI from Scratch](https://blogs.gruheshkurra.com/series/ai/), a 19-part series. Each post derives the maths, works a small example by hand, and then turns it into code. Good places to start: [building an autograd engine](https://blogs.gruheshkurra.com/blog/build-autograd-from-scratch/), [how attention works](https://blogs.gruheshkurra.com/blog/attention-in-transformers-explained/), [GPT-2 in PyTorch](https://blogs.gruheshkurra.com/blog/build-gpt2-from-scratch/), and [a mini LLM in NumPy](https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/).

Latest posts:

<!-- BLOG-POST-LIST:START --><a href="https://blogs.gruheshkurra.com/blog/convolutional-neural-networks-explained/">Convolutional Neural Networks: CNN Math Explained</a><br/><a href="https://blogs.gruheshkurra.com/blog/deepseek-v4-inside-one-token/">DeepSeek V4 Inside: One Token Through Every Block</a><br/><a href="https://blogs.gruheshkurra.com/blog/looped-transformers-explained/">Looped Transformers Explained: Recurrent Depth and Astra</a><br/><a href="https://blogs.gruheshkurra.com/blog/natural-language-inference-explained/">Natural Language Inference Explained: Entailment in NLP</a><br/><a href="https://blogs.gruheshkurra.com/blog/build-mini-llm-numpy-from-scratch/">Build a Mini LLM from Scratch in NumPy: RoPE, GQA, SwiGLU</a><br/><!-- BLOG-POST-LIST:END -->

[All posts](https://blogs.gruheshkurra.com/ai-explanations/) · [RSS](https://blogs.gruheshkurra.com/feed.xml) · [DEV](https://dev.to/gruhesh_kurra_6eb933146da)

## Skills

- Daily: Python, PyTorch, NumPy, Hugging Face Transformers, TRL, PEFT and LoRA, FastAPI, Docker, Git
- Models: transformers (GQA, RoPE, SwiGLU, KV-cache), BPE tokenisers, diffusion, GANs, VAEs, graph neural networks, RAG with FAISS
- Vision and on-device: EfficientNet, Vision Transformers, CLIP, OpenCV, Tesseract, Core ML, Swift
- Efficiency: LoRA, quantisation, pruning, distillation, Numba kernels
- Infrastructure: AWS, Azure, GCP, Oracle Cloud, RunPod, Kubernetes, WSO2, PostgreSQL, MongoDB, Redis
- Languages: Python, C++, C, Java, TypeScript, Swift, SQL, Shell

## Certificates

- Machine learning: [TensorFlow Developer](certificates/tensorflow-developer.jpg) (Google, 2024), [OCI Generative AI Professional](certificates/oracle-genai.jpg) (Oracle, 2024), [Build LLMs From Scratch](certificates/visuara-llms.jpg) (Visuara, 2024), [MCP Course Unit 1](certificates/hf-mcp-unit1.jpg) and [Unit 3](certificates/hf-mcp-unit3.jpg) (Hugging Face, 2024), [six-month AI/ML training](certificates/iiit-training.jpg) (iHub-Data, 2024), [Linear Algebra Master](certificates/udemy-linear-algebra.jpg) and [A Deep Understanding of Deep Learning](certificates/udemy-deep-learning.jpg) (Udemy, 2026)
- Cloud: [AWS Solutions Architect – Associate](certificates/aws-solutions-architect.jpg) and [AWS Cloud Practitioner](certificates/aws-cloud-practitioner.jpg) (2023), [OCI Architect Associate](certificates/oracle-architect.jpg) and [Oracle Database](certificates/oracle-database.jpg) (2023)
- Integration: [WSO2 Micro Integrator Developer V4](certificates/wso2-mi-developer.jpg) and [Practitioner V4](certificates/wso2-mi-practitioner.jpg) (2025), [Red Hat Enterprise Application Developer](certificates/redhat-developer.jpg) (2024), [Automation Anywhere Advanced Professional](certificates/automation-anywhere.jpg) (2024)
- Other: [Global Challenge Lab 2026, 1st place](certificates/global-challenge-lab-2026.jpg) (Imperial, 2026), [Google Developer Groups](certificates/google-gdg.jpg) (2025), [KL University appreciation](certificates/sac-appreciation.jpg) (2022 – 2023), [Street Cause volunteer](certificates/streetcause.jpg) (2023 – 2024)

## Contact

If you work on language models, model compression, or on-device inference, I'd like to hear from you: [gruheshkurra2@gmail.com](mailto:gruheshkurra2@gmail.com). I'm also on [LinkedIn](https://www.linkedin.com/in/gruheshkurra/) and [X](https://x.com/Karthik__kurra).

Also known as Gruhesh Kurra or Karthik Kurra.
