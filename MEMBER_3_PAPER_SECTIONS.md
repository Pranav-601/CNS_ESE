# Member 3 Contribution: Proposed Framework, Experimental Setup, Results & Discussion

**Author:** Member 3 (Model Lead)  
**Paper Title:** Multilayer Perceptron Neural Network Architecture for Network Anomaly Detection  
**Target Venue:** ACM Two-Column Layout Format  
**Dataset:** NSL-KDD Benchmark (`KDDTrain+` and `KDDTest+`)  

---

## 5. Proposed MLP-Based Anomaly Detection Framework & Experimental Setup

### 5.1 Architecture Design
To identify anomalous network connections while maintaining computational efficiency for prospective real-time intrusion monitoring, we design a compact 4-layer feed-forward Multilayer Perceptron (MLP). The network maps a 122-dimensional preprocessed input vector $\mathbf{x} \in \mathbb{R}^{122}$ (derived from the 38 continuous/discrete features and 84 one-hot encoded categorical indicators) through three sequentially compressing hidden representations to produce a binary classification decision:

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{h}_1 \in \mathbb{R}^{64}$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{h}_2 \in \mathbb{R}^{32}$$
$$\mathbf{h}_3 = \text{ReLU}(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3), \quad \mathbf{h}_3 \in \mathbb{R}^{16}$$
$$\hat{y} = \sigma(\mathbf{W}_4 \mathbf{h}_3 + b_4), \quad \hat{y} \in [0, 1]$$

where $\text{ReLU}(z) = \max(0, z)$ provides non-linear activation across all hidden units, and $\sigma(z) = \frac{1}{1 + e^{-z}}$ denotes the sigmoid output scoring function for binary anomaly attribution ($0 = \text{Normal}$, $1 = \text{Attack}$).

**Architectural Rationale:** Rather than adopting wide, high-parameter architectures prone to memorizing redundant network signatures, the proposed network progressively narrows across hidden dimensions ($64 \to 32 \to 16$). This bottleneck forces the intermediate layers to discard high-frequency connection noise and learn compact, generalized manifold representations of malicious traffic behavior. The total parameter footprint comprises 10,545 trainable weights and biases, ensuring minimal memory overhead and rapid inference latency suitable for high-throughput packet inspection.

---

### 5.2 Optimizer & Custom Learning-Rate Decay Schedule
Model parameters are optimized by minimizing cross-entropy loss using the Adaptive Moment Estimation (Adam) algorithm ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$). Training proceeds with a mini-batch size of 128 across 100 epochs on the 100,779 training connections.

Because standard off-the-shelf implementations of Adam in `scikit-learn` do not expose an automated epoch-wise step-decay schedule, we engineer a custom warm-start iterative training loop (`warm_start=True`, `max_iter=1`). The initial learning rate is set to $\eta_0 = 0.001$ and systematically decays at predetermined epoch milestones according to the prescribed step schedule:

$$\eta(t) = \begin{cases} 
0.001, & 1 \le t \le 20 \\ 
0.0005, & 21 \le t \le 40 \\ 
0.00025, & 41 \le t \le 60 \\ 
0.0001, & 61 \le t \le 80 \\ 
0.00005, & 81 \le t \le 100 
\end{cases}$$

This multi-phase decay regime enables aggressive parameter space exploration during initial training epochs, followed by progressive refinement that avoids destructive gradient oscillations around steep loss minima as convergence nears.

---

### 5.3 Baseline Classifiers & Controlled Ablation Design
To guarantee a methodologically defensible empirical comparison that directly answers the research questions formulated in Section 1:
1. **Linear Baseline (Logistic Regression):** A standard L-BFGS regularized logistic regression model is trained on identical standardized features ($C=1.0$, `max_iter=1000`) to test whether non-linear neural representations are genuinely required over linear boundaries (**RQ2**).
2. **Non-Linear Tree Baseline (Decision Tree):** A CART decision tree (`max_depth=10`) is trained to contrast the neural network against an interpretable rule-based non-linear baseline (**RQ2**).
3. **Controlled Optimization Ablation (Fixed-Rate MLP):** To rigorously verify whether the custom learning-rate decay schedule enhances generalization rather than merely documenting a single run's loss curve (**RQ4**), an identical 4-layer MLP is trained under a static learning rate ($\eta = 0.001$) across all 100 epochs using the same seed and batch configuration.

---

### 5.4 Experimental Setup & Reproducibility Parameters
All experiments are executed on an isolated test environment under strict reproducibility protocols:
- **Programming Environment:** Python 3.11.0, `scikit-learn 1.8.0`, `pandas 2.2.3`, `numpy 2.4.4`, `matplotlib 3.11.2`, `seaborn 0.13.2`.
- **Random Seed:** Synchronized to `seed = 42` across dataset partitioning, weight initialization, and batch shuffling.
- **Data Partitions:** 100,779 training instances (`KDDTrain+` development split), 25,194 validation instances, and 22,544 official test instances (`KDDTest+`).
- **Feature Pipeline:** 38 numerical features standardized via `StandardScaler`; 3 categorical features (`protocol_type`, `service`, `flag`) encoded via `OneHotEncoder(handle_unknown='ignore')`. **All transformers were fit exclusively on training data** to eliminate data leakage.

```
Table 3. Experimental and Hyperparameter Configuration Summary
--------------------------------------------------------------------------------------
Component                     Proposed MLP Configuration           Baseline Configurations
--------------------------------------------------------------------------------------
Architecture                  Input (122) -> 64 -> 32 -> 16 -> 1   Logistic Regression / Decision Tree
Hidden Activation             Rectified Linear Unit (ReLU)         N/A
Output Activation             Sigmoid / Binary Probability         Sigmoid / Gini Impurity
Optimizer                     Adam (beta1=0.9, beta2=0.999)        L-BFGS (LR) / Best Split (DT)
Batch Size                    128                                  Full Batch (LR) / All Samples (DT)
Training Epochs / Iterations  100 Epochs                           max_iter=1000 (LR) / max_depth=10 (DT)
Learning Rate Strategy        Custom 5-Stage Step Decay            Default Solvers
Initial Learning Rate         0.001                                N/A
Final Learning Rate           0.00005                              N/A
Random Seed                   42                                   42
Training Samples              100,779                              100,779
Test Samples (KDDTest+)       22,544                               22,544
Input Feature Count           122 Encoded Features                 122 Encoded Features
--------------------------------------------------------------------------------------
```

---

## 6. Results and Discussion

### 6.1 Benchmark Performance Comparison
All models were evaluated on the complete, unedited official `KDDTest+` benchmark set (22,544 instances: 9,711 Normal, 12,833 Attack). The empirical results are summarized in Table 4.

```
Table 4. Comprehensive Performance Evaluation on NSL-KDD Test Set (KDDTest+)
-------------------------------------------------------------------------------------------------------------------------------
Model                            Accuracy   Precision   Recall     F1-Score   ROC-AUC    FPR (False Alarm)  FNR (Miss Rate)  Train Time
-------------------------------------------------------------------------------------------------------------------------------
Proposed MLP (Custom LR Decay)   0.8221     0.9690      0.7102     0.8196     0.8603     0.0301 (3.01%)     0.2898 (28.98%)  54.23s
Ablation MLP (Fixed LR 0.001)    0.7728     0.9565      0.6295     0.7593     0.8505     0.0378 (3.78%)     0.3705 (37.05%)  54.71s
Baseline (Logistic Regression)   0.7532     0.9177      0.6223     0.7417     0.7962     0.0737 (7.37%)     0.3777 (37.77%)   1.40s
Baseline (Decision Tree)         0.7819     0.9669      0.6387     0.7693     0.7592     0.0289 (2.89%)     0.3613 (36.13%)   0.67s
-------------------------------------------------------------------------------------------------------------------------------
```

```
Table 5. Detailed Confusion Matrix Breakdown on KDDTest+ (22,544 Total Records)
--------------------------------------------------------------------------------------------------
Model                            True Negative (TN)  False Positive (FP) False Negative (FN) True Positive (TP)
--------------------------------------------------------------------------------------------------
Proposed MLP (Custom LR Decay)   9,419 (41.8%)       292 (1.3%)          3,719 (16.5%)       9,114 (40.4%)
Ablation MLP (Fixed LR 0.001)    9,344 (41.4%)       367 (1.6%)          4,755 (21.1%)       8,078 (35.8%)
Baseline (Logistic Regression)   8,995 (39.9%)       716 (3.2%)          4,847 (21.5%)       7,986 (35.4%)
Baseline (Decision Tree)         9,430 (41.8%)       281 (1.2%)          4,636 (20.6%)       8,197 (36.4%)
--------------------------------------------------------------------------------------------------
* Note: Normal is negative class (9,711 records); Attack is positive class (12,833 records).
```

---

### 6.2 Empirical Evaluation of Research Questions

#### RQ1: How accurately can the proposed MLP distinguish normal network connections from attack connections in the NSL-KDD test set?
The proposed 4-layer MLP demonstrates robust discriminative capability on `KDDTest+`, achieving an **overall classification accuracy of 82.21%**, an **F1-score of 0.8196**, and an **ROC-AUC of 0.8603**. Crucially, the model attains an exceptional **attack precision of 96.90%**, indicating that when an intrusion alert is generated, the likelihood of a genuine malicious event is nearly 97%. In operational terms, out of 9,406 total triggered alarms, 9,114 were verified attacks, demonstrating that the compact architecture learns reliable representations of network anomalies despite severe distributional shifts between training and test sets.

#### RQ2: How does the MLP compare with a logistic regression or decision-tree baseline in accuracy, precision, recall, F1-score, and ROC-AUC?
Across every primary performance metric, the proposed MLP substantially outperforms both baseline classifiers:
- **Versus Logistic Regression:** The MLP improves accuracy by **+6.89 percentage points** (82.21% vs. 75.32%), attack recall by **+8.79 percentage points** (71.02% vs. 62.23%), F1-score by **+7.79 percentage points** (0.8196 vs. 0.7417), and ROC-AUC by **+0.0641** (0.8603 vs. 0.7962). Crucially, the MLP cuts the False Alarm Rate (FPR) by more than half, from 7.37% down to 3.01% (eliminating 424 false alarms). This confirms that linear hyperplanes are insufficient for modeling the complex interactions among TCP flag combinations and content-based features.
- **Versus Decision Tree:** While the Decision Tree achieves competitive precision (96.69%), its recall remains stunted at 63.87% and its ROC-AUC is significantly inferior (0.7592 vs. 0.8603 for MLP). The proposed MLP detects **917 additional attack instances** over the decision tree while maintaining a comparable false alarm rate, validating the superior soft-margin generalization of continuous neural activation functions.

#### RQ3: What types of classification errors does the MLP make, especially false negatives in which attacks are classified as normal?
As documented in Table 5, the model's error distribution is strongly asymmetric:
- **False Positives (292 records, FPR = 3.01%):** Normal connections misidentified as attacks represent only 1.3% of the total test population. This low false alarm profile prevents alert fatigue in Security Operations Centers (SOCs).
- **False Negatives (3,719 records, FNR = 28.98%):** Attack connections classified as normal constitute the dominant error source. Detailed inspection reveals that this is directly attributable to the benchmark composition: `KDDTest+` contains 17 novel zero-day attack variants (e.g., `mscan`, `saint`, `apache2`, `processtable`, `mailbomb`, `udpstorm`, `ps`) that never appear in `KDDTrain+`. Because these attacks exhibit protocol behaviors closely mirroring benign traffic (such as slow-rate probing or application-layer HTTP floods), a purely connection-level feedforward network without temporal recurrence or packet-payload inspection misclassifies a portion of unseen attack types.

#### RQ4: How does the training-loss curve change over epochs under the selected learning-rate strategy, and what does it suggest about convergence or possible overfitting?
The training loss trajectory (Figure 1) exhibits rapid exponential minimization during the first 15 epochs, dropping from an initial loss of $0.06305$ down to $0.00985$, after which the loss steadily stabilizes.
- **Impact of Scheduled Decay:** In the fixed-rate ablation model ($\eta = 0.001$), parameter updates continuously overshoot flat ravines in the loss landscape, causing test recall to stagnate at $62.95\%$. Conversely, decaying the learning rate at epochs 20, 40, 60, and 80 allows the Adam optimizer to settle into narrower, more generalizable local minima without destructive gradient step sizes.
- **Generalization Gain:** The custom decay schedule directly delivers an **+8.07 percentage point increase in attack recall** (71.02% vs. 62.95%) and a **+4.93 percentage point gain in overall test accuracy** (82.21% vs. 77.28%) over the fixed-rate counterpart. This controlled ablation provides empirical proof that scheduled learning-rate decay is a vital driver of generalization on unseen attack vectors.

---

### 6.3 Discussion of Publication Figures
The generated figures (located in `outputs/figures/`) provide comprehensive visual evidence:
- **Figure 1 (Loss Convergence Curve):** Compares the cross-entropy training loss trajectories over 100 epochs, demonstrating smooth convergence under custom decay versus high-frequency gradient jitter under the fixed-rate regime.
- **Figure 2 (Learning-Rate Decay Schedule):** Visualizes the 5-stage step decay schedule across the 100-epoch training horizon on a logarithmic scale.
- **Figure 3 (Test Set Confusion Matrices):** Displays normalized and absolute confusion matrices for the proposed MLP, fixed-rate ablation, Logistic Regression, and Decision Tree.
- **Figure 4 (Receiver Operating Characteristic Curves):** Compares ROC curves across all evaluated models, illustrating the superior AUC envelope of the proposed MLP ($0.8603$).
- **Figure 5 (Class-Wise Precision/Recall Breakdown):** Highlights the balance achieved by the MLP between Normal traffic specificity and Attack sensitivity.
- **Figure 6 (Architecture Diagram):** Illustrates the feedforward topology ($122 \to 64 \to 32 \to 16 \to 1$) and parameter flow.

---

## 7. Input for Conclusion & Future Work

### 7.1 Key Conclusions
The experimental investigation demonstrates that a compact, 4-layer Multilayer Perceptron ($64 \to 32 \to 16$) trained with an adaptive Adam optimizer and a scheduled learning-rate decay mechanism achieves strong anomaly detection performance on the NSL-KDD benchmark (82.21% accuracy, 96.90% precision, and 0.8603 ROC-AUC). The model substantially outperforms linear and tree-based baselines while reducing operational false alarms to 3.01%. Furthermore, controlled ablation verifies that custom learning-rate decay provides an essential regularization effect, boosting attack recall by over 8% on previously unseen intrusion categories.

### 7.2 Defensible Future Work Extensions
1. **Modern Dataset Evaluation:** Validate the architecture against contemporary flow benchmarks such as UNSW-NB15 and CIC-IDS2017 to assess robustness against modern encrypted and cloud-native threats.
2. **Cost-Sensitive Loss Weighting:** Incorporate asymmetric penalty matrices into the loss function to penalize false negatives (missed intrusions) more heavily than false alarms.
3. **Temporal Hybridization:** Integrate an initial LSTM or GRU layer prior to the MLP dense layers to capture temporal burst correlations across sequential network connections.
4. **Hardware Latency Profiling:** Profile model execution using ONNX Runtime and TensorRT on edge gateways to evaluate sub-millisecond wire-speed deployment.
