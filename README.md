# Causal-Probabilistic World Model for Clinical Treatment Reasoning

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Clinical%20AI%20Platform-00e5ff?style=for-the-badge&logo=googlechrome&logoColor=white)](https://auramasters.github.io/Causal-Probabilistic-World-Model-for-Clinical-Treatment-Reasoning/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deployed-success?style=for-the-badge&logo=github)](https://auramasters.github.io/Causal-Probabilistic-World-Model-for-Clinical-Treatment-Reasoning/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19.0-61dafb?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178c6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-8.0+-646cff?style=for-the-badge&logo=vite&logoColor=white)](https://vite.dev/)

> **A scientifically validated Causal Artificial Intelligence decision support system designed to simulate interventional counterfactuals ($do$-calculus), optimize individualized antiretroviral therapy (ART), and quantify robustness against unmeasured confounding for HIV-1 clinical management.**

---

### 🌐 Live Interactive Web Platform
👉 **Access the Live Platform:** **[https://auramasters.github.io/Causal-Probabilistic-World-Model-for-Clinical-Treatment-Reasoning/](https://auramasters.github.io/Causal-Probabilistic-World-Model-for-Clinical-Treatment-Reasoning/)**

*The application features full offline & static resilience with an embedded client-side high-precision SCM inference engine, alongside full Flask API support for local full-stack development.*

---

## 🔬 Clinical Overview & Trial Context

The system is trained and evaluated on the landmark **AIDS Clinical Trials Group (ACTG 175)** randomized controlled trial ($N = 2,139$ HIV-1 infected patients) comparing four distinct antiretroviral treatment arms:
* **Arm 0**: Zidovudine (ZDV / AZT monotherapy — historical standard of care)
* **Arm 1**: Zidovudine + Didanosine (ZDV + ddI dual therapy)
* **Arm 2**: Zidovudine + Zalcitabine (ZDV + ddC dual therapy)
* **Arm 3**: Didanosine (ddI monotherapy)

**Clinical Endpoint (`label`)**: Severe disease progression (defined as a confirmed decline in CD4 cell count $\ge 50\%$, development of an AIDS-defining clinical event, or all-cause mortality). Primary trial event rate: **24.3%**.

---

## 🏛️ Dual-Model Causal Architecture

To address faculty advisor feedback regarding information loss from numerical binning, the platform provides two rigorously benchmarked causal models:

```
                                  ┌─────────────────────────────────────────────────────────┐
                                  │                Patient Clinical Profile                 │
                                  │ (CD4, CD8, Age, Weight, Karnofsky, Pre-ART Days, etc.)  │
                                  └───────────────────────────┬─────────────────────────────┘
                                                              │
                                     ┌────────────────────────┴────────────────────────┐
                                     │                                                 │
                                     ▼                                                 ▼
             ┌───────────────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
             │       Model B: Continuous / Hybrid SCM        │ │     Model A: Baseline Discretized BN (23 Edges) │
             │  (G-Computation + L2-Regularized Surfaces)   │ │    (BDeu ESS=10 + Exact Variable Elimination)   │
             ├───────────────────────────────────────────────┤ ├───────────────────────────────────────────────┤
             │ • Preserves exact continuous biomarkers       │ │ • Continuous variables binned into 3 quantiles │
             │ • Zero discretization step-function artifacts │ │ • Discrete Bayesian Network structure learning │
             │ • Exact analytical risk derivatives ∂P/∂X_j   │ │ • Evidence ablation delta sensitivities        │
             │ • VanderWeele & Ding E-Value sensitivity      │ │ • Exact Pearlian do-calculus intervention      │
             │ • Multi-arm What-If CD4 trajectory simulator  │ │ • Preserved as comparative baseline           │
             └───────────────────────┬───────────────────────┘ └───────────────────────┬───────────────────────┘
                                     │                                                 │
                                     └────────────────────────┬────────────────────────┘
                                                              │
                                                              ▼
                                  ┌─────────────────────────────────────────────────────────┐
                                  │             Individualized Treatment Utility            │
                                  │   EU = P(Success | do(trt=t)) × 1.0 + P(Fail) × 0.0     │
                                  │   Optimal Regimen Recommendation + Robustness Bounds    │
                                  └─────────────────────────────────────────────────────────┘
```

### 1. Model B: Continuous / Hybrid SCM (Primary)
* **Information Preservation**: Keeps all 6 numerical biomarkers (`cd40`, `cd80`, `age`, `wtkg`, `karnof`, `preanti`) as true continuous measurements. Preprocessing standard scalers are fitted **strictly on the development partition** ($N = 1,711$) to eliminate test leakage.
* **Causal Engine**: Formulates counterfactual potential outcomes via G-computation response surfaces across treatment arms ($t \in \{0, 1, 2, 3\}$).
* **Local Gradient Sensitivity**: Analytical partial derivatives $\frac{\partial P}{\partial X_j} = P(1 - P) \cdot \frac{\beta_j}{\sigma_j}$, yielding exact per-unit risk slopes (e.g., risk change per 50 CD4 cells or 10 Karnofsky points).
* **E-Value Sensitivity Analysis**: Formulates the VanderWeele & Ding (2017) E-value:
  $$RR = \frac{P(\text{Progression} \mid \text{do}(\text{trt}=0))}{P(\text{Progression} \mid \text{do}(\text{trt}^*))}, \quad E = RR + \sqrt{RR(RR - 1)}$$
  Quantifies the minimum strength of association that an unmeasured confounder must have with both treatment and outcome to explain away the causal effect.
* **Continuous What-If Simulator**: Real-time multi-arm risk trajectory curves across CD4 counts ($50$ to $800$ cells/mm³).

### 2. Model A: Baseline Discretized Bayesian Network
* **DAG Architecture**: 23 causal edges learned via score-based structure search with clinical tier topological constraints.
* **Parameter Learning**: Bayesian Estimator with BDeu equivalent sample size ($ESS = 10$).
* **Inference**: Exact Variable Elimination under Pearl's $do$-operator: $P(Y \mid do(\text{trt} = t), \mathbf{X} = \mathbf{x})$.

---

## 📊 Head-to-Head Benchmark Validation

Evaluated strictly on the **$N = 428$ held-out test cohort** (80/20 train/test split fixed at random seed `42`). Bootstrapped 95% confidence intervals generated across $B = 1,000$ resamples:

| Metric / Dimension | Model A (Discretized BN) | Model B (Continuous SCM) | Advantage / Scientific Gain |
| :--- | :---: | :---: | :--- |
| **Numerical Preservation** | 3 Quantile Bins | **Exact Continuous Values** | Eliminates step-function boundary artifacts |
| **ROC-AUC** | 0.6372 (0.576 – 0.697) | **0.6878 (0.626 – 0.744)** | **+0.0506 AUC Gain** |
| **PR-AUC (Prevalence: 24.3%)** | 0.3687 (0.285 – 0.455) | **0.4130 (0.326 – 0.511)** | **+0.0443 PR-AUC Gain** |
| **Brier Score (Probabilistic Error)** | 0.1753 (0.156 – 0.196) | **0.1699 (0.150 – 0.191)** | **Lower squared prediction error** |
| **Expected Calibration Error (ECE)** | 4.45% | **3.96%** | **Tighter empirical agreement** |
| **Calibrated Specificity** | 50.31% ($\tau^* = 0.20$) | **62.65% ($\tau^* = 0.24$)** | **+12.34% Specificity at ~67% Sensitivity** |
| **E-Value for Confounding** | 3.46 | **7.47** | **High stability against unmeasured bias** |
| **Local Explainability** | Evidence Ablation ($\Delta P$) | **Exact Gradient ($\partial P / \partial X$)** | True instantaneous rate of risk change |
| **Counterfactual Advantage** | 75.0% (321 / 428 patients) | **73.1% (313 / 428 patients)** | Consistently identifies dual-therapy over mono |

---

## 🚀 Key Interface Capabilities

1. **Model Switcher Pill Toggle**: Switch seamlessly in real-time between Model B (Continuous SCM) and Model A (Discretized BN).
2. **Clinical Archetype Presets**: One-click patient profiles (*ART-Naïve Asymptomatic*, *Moderate Progression*, *Advanced Low-CD4 High-Risk*).
3. **Interactive What-If CD4 Trajectory Simulator**: Dynamic interactive SVG visualization displaying predicted progression probability for all 4 treatment arms as CD4 count varies from 50 to 800 cells/mm³.
4. **E-Value Confounding Robustness Card**: Visual metric gauge indicating whether findings are robust against unmeasured confounding ($E \ge 1.5$).
5. **CATE Benefit Tier Badge**: Identifies patient's absolute risk reduction percentile compared against the entire development population.
6. **Side-by-Side Model Benchmark Tab**: Comprehensive comparison table with confidence intervals and Decision Curve Analysis (DCA).

---

## 💻 Local Development & Execution

### Prerequisites
* Python 3.11+
* Node.js 18+ and npm

### 1. Clone & Setup
```bash
git clone https://github.com/AuraMasters/Causal-Probabilistic-World-Model-for-Clinical-Treatment-Reasoning.git
cd Causal-Probabilistic-World-Model-for-Clinical-Treatment-Reasoning

# Python Virtual Environment
python -m venv .venv
.venv\Scripts\activate       # On Windows
# source .venv/bin/activate  # On Linux/macOS
pip install -r requirements.txt
```

### 2. Run the Full-Stack Application
A single command launches both the Flask inference API (port 5000) and the Vite React frontend (port 5173):
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

### 3. Master Pipeline CLI Runner
Execute end-to-end model training, validation, benchmarking, and unit testing via the master CLI:
```bash
# Run everything
python src/pipeline.py --all

# Or run individual stages:
python src/pipeline.py --train      # Fit Continuous SCM strictly on development partition
python src/pipeline.py --validate   # Run full test cohort validation for both models
python src/pipeline.py --compare    # Compile side-by-side benchmark artifact
python src/pipeline.py --test       # Run automated unit tests (15/15 tests)
```

### 4. Automated Test Suite
```bash
python -m unittest tests/test_validation_pipeline.py
```
*Guarantees zero test set leakage, DAG acyclicity, metric correctness, threshold sweep calibration, E-value calculations, and live API endpoints.*

---

## 📁 Repository Structure

```
├── .github/workflows/
│   └── deploy.yml                       # Automated GitHub Actions Pages deployment
├── data/                                # ACTG175 raw and preprocessed datasets
├── frontend/                            # React 19 + TypeScript + Vite UI
│   ├── src/
│   │   ├── components/                  # UI components (Header, What-If, DAG, etc.)
│   │   ├── lib/
│   │   │   ├── api.ts                   # Resilient API client with static fallback
│   │   │   ├── clientInference.ts       # Embedded high-precision continuous SCM
│   │   │   ├── modelWeights.json        # Precomputed continuous model coefficients
│   │   │   └── precomputedOverview.json # Bundled validation benchmark artifacts
│   │   └── types.ts                     # Strict TypeScript interfaces
│   └── vite.config.ts                   # Vite configuration with relative base
├── results/
│   ├── continuous_model/                # Model B serialized artifacts
│   └── validation/
│       ├── comparison/                  # Side-by-side Model A vs B report
│       ├── comprehensive/               # Model A test cohort validation results
│       └── continuous/                  # Model B test cohort validation results
├── scripts/
│   ├── deploy_gh_pages.py               # Native Git GitHub Pages deployment utility
│   ├── run_api.sh                       # Backend launch script
│   └── run_frontend.sh                  # Frontend launch script
├── src/
│   ├── api/                             # Flask REST API endpoints
│   ├── continuous_model/                # Model B continuous SCM implementation
│   ├── parameter_learning/              # Model A BDeu parameter estimation
│   ├── validation/                      # Benchmark validation & threshold sweep
│   └── pipeline.py                      # Master CLI pipeline orchestrator
└── tests/
    └── test_validation_pipeline.py      # 15 automated validation unit tests
```

---

## 📜 Clinical Disclaimer & Methodology Notes

1. **Investigational Tool**: This software is developed for research and educational purposes to demonstrate causal inference and probabilistic decision-making. It does not constitute medical advice or a prescriptive treatment guideline.
2. **Causal Assumptions**: Interventional estimates rest on the standard identification assumptions: Positivity, Consistency, and Conditional Exchangeability (No Unmeasured Confounders). The E-Value analysis is explicitly provided to assess vulnerability to unmeasured confounding.
3. **Threshold Calibration**: Because HIV disease progression occurred in 24.3% of trial patients, the standard decision cutoff ($\tau = 0.50$) produces low clinical sensitivity. Threshold optimization on development data established $\tau^* = 0.24$ for Model B to balance sensitivity (67.3%) and specificity (62.6%).

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
