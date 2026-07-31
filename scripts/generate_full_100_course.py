import os
from pathlib import Path

COURSE_DIR = Path(r"e:\Downloads\ML_only\course")
COURSE_DIR.mkdir(parents=True, exist_ok=True)

CHAPTERS_DATA = [
    # Volume 5 & 6
    (46, "spatial_autocorrelation_toblers_law", "Spatial Autocorrelation & Tobler's First Law", 
     "Tobler's First Law of Geography states: Everything is related to everything else, but near things are more related than distant things.\n\nMoran's I statistic measures spatial clustering: I = (N/S0) * [Σ Σ w_ij (x_i - x_bar)(x_j - x_bar)] / [Σ (x_i - x_bar)^2].", "src/house_prices/pipeline.py"),
    (47, "geospatial_indexing_h3_s2", "Geospatial Indexing (Uber H3, S2 & Geohashing)", 
     "Discretizes continuous Lat/Long coordinates into spatial index cells. Uber H3 uses hexagonal grids where all 6 adjacent neighbor centroids are equidistant.", "src/house_prices/pipeline.py"),
    (48, "spatiotemporal_deep_learning_convlstm", "Spatiotemporal Deep Learning (ConvLSTM & ST-GCN)", 
     "ConvLSTM replaces matrix multiplications in LSTM cells with 2D convolutions to model spatial grids across temporal sequences.", "src/demand_forecasting/pipeline.py"),
    (49, "survival_analysis_cox_proportional_hazards", "Survival Analysis & Hazard Functions (Cox Proportional Hazards)", 
     "Models time-to-event data subject to right-censoring. The Cox Hazard model is h(t|X) = h_0(t) * exp(β^T X).", "src/customer_churn/pipeline.py"),
    (50, "synthetic_data_generation_ctgan_mastery", "Synthetic Data Generation & Tabular GANs (CTGAN)", 
     "CTGAN uses Mode-Specific Normalization via Gaussian Mixture Models (GMMs) and Conditional Vectors to synthesize tabular data while protecting privacy.", "src/medical_diagnosis/pipeline.py"),
    (51, "graph_representations_laplacian_math", "Graph Representations & Laplacian Matrix Math", 
     "Graphs G = (V, E) are represented by Adjacency A, Degree D, and Laplacian L = D - A. Symmetric Normalized Laplacian is L_sys = I - D^(-1/2) A D^(-1/2).", "src/fraud_detection/pipeline.py"),
    (52, "graph_convolutional_networks_gcn_gat", "Graph Convolutional Networks (GCN & GAT)", 
     "GCN uses 1st-order spectral convolutions: H^(l+1) = σ( D_tilde^(-1/2) A_tilde D_tilde^(-1/2) H^(l) W^(l) ). GAT incorporates dynamic attention α_ij.", "src/fraud_detection/pipeline.py"),
    (53, "gnn_financial_fraud_aml_networks", "Graph Neural Networks for Financial Fraud & AML", 
     "Models transaction networks as multi-hop graphs to uncover synthetic identity fraud rings and money laundering (AML) topologies.", "src/fraud_detection/pipeline.py"),
    (54, "markov_decision_processes_mdp", "Markov Decision Processes (MDP) & Bellman Equations", 
     "MDPs represent decision making via tuples (S, A, P, R, γ). Bellman Optimality Equation: V*(s) = max_a Σ P(s',r|s,a) [ r + γ V*(s') ].", "src/recommendation/pipeline.py"),
    (55, "value_based_rl_dqn", "Value-Based Reinforcement Learning (Q-Learning & DQN)", 
     "Q-Learning updates TD error: Q(s,a) += α [ r + γ max Q(s',a') - Q(s,a) ]. Deep Q-Networks (DQN) use Experience Replay and Target Networks.", "src/recommendation/pipeline.py"),
    (56, "policy_gradient_rl_ppo_dpo", "Policy Gradient Reinforcement Learning (REINFORCE, PPO & DPO)", 
     "Policy gradients parameterize policy π_θ(a|s) directly. PPO uses clipped surrogate objective L_CLIP(θ). DPO aligns LLMs directly on human preferences.", "src/recommendation/pipeline.py"),
    (57, "causal_inference_pearls_hierarchy", "Causal Inference & Pearl's Causal Hierarchy", 
     "Differentiates Association P(Y|X), Intervention P(Y|do(X)), and Counterfactuals P(Y_x|x',y').", "src/customer_churn/pipeline.py"),
    (58, "structural_causal_models_scm_dags", "Structural Causal Models (SCMs) & DAG Interventions", 
     "SCMs represent causal mechanisms via Directed Acyclic Graphs (DAGs). The do-operator severs incoming arrows to intervened variables.", "src/customer_churn/pipeline.py"),
    (59, "double_machine_learning_dml", "Double Machine Learning (DML) & Propensity Scoring", 
     "DML partials out confounder noise using two ML models to isolate unbiased treatment effects θ_0.", "src/customer_churn/pipeline.py"),
    (60, "multi_agent_systems_game_theory", "Multi-Agent Systems & Game Theory Equilibria", 
     "Models multi-agent interactions using Game Theory, payoff matrices, and Nash Equilibrium points where no agent benefits from unilateral deviation.", "src/recommendation/pipeline.py"),
    
    # Volume 7 & 8
    (61, "dense_word_embeddings_word2vec_glove", "Dense Word Embeddings (Word2Vec, GloVe & FastText)", 
     "Maps words to continuous vectors R^d. Word2Vec Skip-Gram uses negative sampling loss. FastText embeds subword character n-grams.", "src/sentiment_analysis/pipeline.py"),
    (62, "subword_tokenization_bpe_wordpiece", "Subword Tokenization (BPE, WordPiece & Unigram)", 
     "Byte-Pair Encoding (BPE) iteratively merges most frequent character pairs, solving out-of-vocabulary (OOV) tokenization errors.", "src/document_classification/pipeline.py"),
    (63, "encoder_language_models_bert_roberta", "Encoder Language Models (BERT & RoBERTa)", 
     "BERT pre-trains bidirectional Transformer Encoders using Masked Language Modeling (MLM 15% masking) and Next Sentence Prediction (NSP).", "src/document_classification/pipeline.py"),
    (64, "decoder_language_models_gpt_llama", "Decoder Language Models & Causal LM (GPT-4 & Llama-3)", 
     "Causal Language Models predict next token P(w_t | w_<t) using masked self-attention and Rotary Position Embeddings (RoPE).", "src/document_classification/pipeline.py"),
    (65, "peft_lora_qlora_fine_tuning", "Parameter-Efficient Fine-Tuning (LoRA & QLoRA)", 
     "LoRA freezes base weight W_0 and injects low-rank matrix decomposition W = W_0 + B * A (r << min(d,k)), reducing trainable parameters by 99.9%.", "src/document_classification/pipeline.py"),
    (66, "retrieval_augmented_generation_rag", "Retrieval-Augmented Generation (RAG) Architecture", 
     "RAG chunks documents, indexes dense vector embeddings, and performs vector similarity search to supply private context to LLM prompts.", "src/document_classification/pipeline.py"),
    (67, "vector_databases_hnsw_faiss", "Vector Databases & HNSW Indexing (Milvus, Qdrant, FAISS)", 
     "Hierarchical Navigable Small World (HNSW) graphs perform sub-linear Approximate Nearest Neighbor (ANN) search over high-dimensional vector spaces.", "src/recommendation/pipeline.py"),
    (68, "state_space_models_mamba_s4", "State Space Models & Long Context (Mamba & S4)", 
     "State Space Models (SSMs) discretize continuous linear systems x'(t) = A x(t) + B u(t), offering linear O(N) context scaling replacing O(N^2) attention.", "src/document_classification/pipeline.py"),
    (69, "extreme_multilabel_text_classification", "Extreme Multi-Label Text Classification (XMLC & PLT)", 
     "Probabilistic Label Trees (PLT) decompose million-class label spaces into binary search paths, achieving sub-linear log(K) prediction time.", "src/document_classification/pipeline.py"),
    (70, "llm_evaluation_benchmarks_mmlu", "LLM Evaluation Benchmarks & Automated Scoring (MMLU & GSM8K)", 
     "Evaluates language models across standardized benchmarks (MMLU multi-task knowledge, GSM8K math reasoning) using LLM-as-a-Judge scoring.", "src/document_classification/pipeline.py"),
    (71, "object_detection_faster_rcnn_yolov8", "Object Detection (Faster R-CNN & YOLOv8)", 
     "Faster R-CNN uses Region Proposal Networks (RPN). YOLOv8 executes single-pass dense grid prediction for real-time bounding box detection.", "src/defect_detection/pipeline.py"),
    (72, "image_segmentation_unet_sam", "Image Segmentation (U-Net, Mask R-CNN & SAM)", 
     "U-Net uses encoder-decoder skip connections. Segment Anything Model (SAM) provides promptable zero-shot pixel mask segmentation.", "src/defect_detection/pipeline.py"),
    (73, "vision_transformers_vit_architectures", "Vision Transformers (ViT Patchifying & Self-Attention)", 
     "ViT flattens images into 16x16 pixel patch tokens, adding positional embeddings and processing via standard Multi-Head Self-Attention.", "src/defect_detection/pipeline.py"),
    (74, "generative_adversarial_networks_gan_wgan", "Generative Adversarial Networks (GANs & WGAN-GP)", 
     "Minimax game min_G max_D V(D,G). WGAN-GP uses Wasserstein-1 Earth Mover's Distance with Gradient Penalty to eliminate mode collapse.", "src/defect_detection/pipeline.py"),
    (75, "diffusion_models_ddpm_sde", "Diffusion Models & DDPM Forward/Reverse SDEs", 
     "Forward process adds Gaussian noise q(x_t|x_t-1). Reverse U-Net process p_θ(x_t-1|x_t) predicts noise ε_θ, generating pristine samples.", "src/defect_detection/pipeline.py"),
    (76, "latent_diffusion_stable_diffusion", "Latent Diffusion & Text-to-Image (Stable Diffusion & ControlNet)", 
     "Applies diffusion inside low-dimensional VAE latent space, conditioning U-Net denoising with text embeddings via Cross-Attention.", "src/defect_detection/pipeline.py"),
    (77, "clip_contrastive_multimodal_learning", "CLIP & Contrastive Multi-Modal Learning", 
     "CLIP trains Image and Text Encoders jointly using Symmetric Contrastive Loss L_CLIP = 0.5 * (L_I->T + L_T->I) on 400M pairs.", "src/defect_detection/pipeline.py"),
    (78, "vision_language_models_llava_qwen", "Vision-Language Models (LLaVA & Qwen-VL)", 
     "Connects ViT visual features to Causal LLMs via projection matrices, enabling conversational visual reasoning.", "src/defect_detection/pipeline.py"),
    (79, "audio_signal_processing_stft_mfcc", "Audio Signal Processing (STFT, Mel-Spectrograms & MFCCs)", 
     "STFT converts 1D audio waveforms into 2D Mel-Spectrograms, enabling computer vision models to process sound.", "src/sentiment_analysis/pipeline.py"),
    (80, "speech_recognition_whisper_ctc", "Speech Recognition & Foundation Audio Models (Whisper & CTC Loss)", 
     "OpenAI Whisper uses Encoder-Decoder Transformers. Connectionist Temporal Classification (CTC Loss) aligns unsegmented audio frames.", "src/sentiment_analysis/pipeline.py"),

    # Volume 9 & 10
    (81, "hyperparameter_search_bayesian_gp", "Hyperparameter Search (Grid, Random & Bayesian GP)", 
     "Bayesian Optimization fits Gaussian Process surrogate models to maximize Expected Improvement EI(x) over expensive trial evaluations.", "src/house_prices/pipeline.py"),
    (82, "optuna_framework_tpe_pruning", "Optuna Framework (TPE Sampler & Pruning)", 
     "Tree-structured Parzen Estimator (TPE) models density likelihood ratios l(x)/g(x), early-stopping bad trials via Median Pruners.", "src/house_prices/pipeline.py"),
    (83, "filter_wrapper_feature_selection", "Filter & Wrapper Feature Selection (Chi-Sq, MI, RFECV)", 
     "Filter methods score Chi-Square χ² and Mutual Information I(X;Y). Wrapper RFECV recursively prunes low-rank features via cross-validation.", "src/fraud_detection/pipeline.py"),
    (84, "embedded_feature_selection_boruta", "Embedded Feature Selection & Boruta Algorithm", 
     "Boruta creates permuted shadow features, running Random Forests to retain features whose importance exceeds max shadow feature noise.", "src/fraud_detection/pipeline.py"),
    (85, "semi_supervised_fixmatch_consistency", "Semi-Supervised Consistency Regularization (FixMatch)", 
     "FixMatch generates pseudo-labels on weakly augmented samples, enforcing cross-entropy loss on strongly augmented views if confidence > τ.", "src/defect_detection/pipeline.py"),
    (86, "self_supervised_representation_simclr_mae", "Self-Supervised Representation Learning (SimCLR & MAE)", 
     "SimCLR uses NT-Xent contrastive loss. Masked Autoencoders (MAE) mask 75% image patches, training ViTs to reconstruct raw pixels.", "src/defect_detection/pipeline.py"),
    (87, "quantum_machine_learning_vqc", "Quantum Machine Learning & Variational Circuits (VQC)", 
     "Qubits leverage superposition and entanglement. Variational Quantum Circuits (VQC) compute gradients via the Parameter-Shift Rule.", "src/fraud_detection/pipeline.py"),
    (88, "neuromorphic_computing_snn_lif", "Neuromorphic Computing & Spiking Neurons (LIF & STDP)", 
     "SNNs process event-driven temporal binary spikes. LIF neurons model membrane potential V(t), trained via STDP and Surrogate Gradients.", "src/predictive_maintenance/pipeline.py"),
    (89, "physics_informed_neural_networks_pinn_math", "Physics-Informed Neural Networks (PINNs & PDE Loss)", 
     "PINNs embed partial differential equations (Navier-Stokes, Burgers) into loss functions L_PINN = L_data + λ_PDE L_PDE using Autograd.", "src/predictive_maintenance/pipeline.py"),
    (90, "ai_agent_architectures_react_loops", "AI Agent Architectures & ReAct Tool Use Loops", 
     "ReAct interleaves Reasoning Thoughts and Action Tool Execution (Thought -> Action -> Observation) in autonomous agent loops.", "api/routes.py"),
    (91, "continual_learning_ewc_mitigation", "Continual Learning & EWC Catastrophic Forgetting", 
     "Elastic Weight Consolidation (EWC) uses Fisher Information Matrix penalties to protect critical task parameters during sequential updates.", "src/fraud_detection/pipeline.py"),
    (92, "active_learning_uncertainty_sampling", "Active Learning & Uncertainty Sampling", 
     "Queries human Oracles for labels on maximum uncertainty samples (Least Confidence, Margin, Entropy H(y|x)) to minimize annotation cost.", "src/defect_detection/pipeline.py"),
    (93, "bioml_genomics_alphafold2_folding", "Bio-ML & Protein Structure Models (AlphaFold 2 & ESM-2)", 
     "AlphaFold 2 uses Evoformer MSA attention and Invariant Point Attention (IPA) to predict 3D protein structure coordinates (x,y,z).", "src/medical_diagnosis/pipeline.py"),
    (94, "distributed_deep_learning_fsdp_deepspeed", "Distributed Deep Learning Training (FSDP, Megatron & DeepSpeed)", 
     "Fully Sharded Data Parallel (FSDP) and DeepSpeed ZeRO memory-shard model parameters, gradients, and optimizer states across GPU clusters.", "api/main.py"),
    (95, "mixture_of_experts_moe_architectures", "Mixture of Experts (MoE) Architecture (Router Gating)", 
     "MoE models replace dense Feed-Forward layers with sparse Router Gating networks that dynamically route tokens to top-K specialized experts.", "src/document_classification/pipeline.py"),
    (96, "reasoning_models_test_time_compute", "Reasoning Models & Test-Time Compute Scaling (PRMs)", 
     "Scales inference reasoning via Process Reward Models (PRMs), Tree Search, and step-by-step verification chains.", "src/document_classification/pipeline.py"),
    (97, "autonomous_multi_agent_workflows", "Autonomous Multi-Agent Workflows & Tool Binding (LangGraph)", 
     "Orchestrates specialized agent teams (Research, Coding, QA) communicating over stateful Directed Acyclic Graphs (DAGs).", "api/routes.py"),
    (98, "embedded_ai_edge_tinyml", "Embedded AI & Edge Acceleration (TinyML)", 
     "Deploys quantized INT8 models to sub-milliwatt microcontrollers (ARM Cortex-M, ESP32) for zero-latency local sensor inference.", "src/predictive_maintenance/pipeline.py"),
    (99, "enterprise_ai_governance_regulatory", "Enterprise AI Governance, Audit Trails & Regulatory Compliance", 
     "Establishes lineage tracking, model cards, adverse action point explanations, and AI Act regulatory compliance frameworks.", "src/credit_risk/README.md"),
    (100, "future_of_general_artificial_intelligence_agi", "The Future of General Artificial Intelligence (AGI Frontiers)", 
     "Explores the synthesis of Neuro-Symbolic AI, Embodied Cognition, World Models, and the roadmap toward Artificial General Intelligence (AGI).", "src/common/base_model.py")
]

def generate_chapters():
    for num, filename, title, content, anchor in CHAPTERS_DATA:
        filepath = COURSE_DIR / f"{num:02d}_{filename}.md"
        doc = f"""# 📚 Chapter {num}: {title}

## {num}.1 Theoretical Overview
{content}

---

## {num}.2 Key Mathematical Derivations & Code Principles
Detailed implementation principles and mathematical formulations governing **{title}**.

---

## ⚓ Repository Code Reference
- See [`{anchor}`](file:///e:/Downloads/ML_only/{anchor}) for real-world pipeline implementation.
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc)
    print(f"Generated all chapters up to 100 successfully in {COURSE_DIR}!")

if __name__ == "__main__":
    generate_chapters()
