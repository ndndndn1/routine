# routine

주제별 arXiv, top journal 논문 루틴 보고서 저장소.

## 현재 주제

**sequential variation propagation** — 다대일(many-to-one) 순방향 문제를 일대다(one-to-many) 역방향으로 풀 때 생기는 해의 다중성(ambiguity)을 분산(불확실성) 제어로 억제하는 방법.

## 선정 규칙

- 다음 키워드 중 **2개 이상** 일치하는 논문만 포함
  - process variation / physics-informed / bayesian optimization / Invertible Neural Networks /
    bayesian adaptive design / active learning / Multimodal inverse design /
    sequential design of experiments / sensitivity-aware sampling /
    global sensitivity analysis / surrogate modeling / cs.LG and physics.comp-ph /
    Tandem Structure / Mixture Density Networks / Determinantal Point Processes
- 이전 루틴 결과와 중복되는 논문은 제외
- 코드 구현이 공개되어 있지 않으면 핵심 방법을 재현하는 간단한 참조 구현을 함께 작성

## 폴더 = 검색 키워드 조합

각 폴더명은 검색에 사용한 키워드 조합을 `_`로 연결한 것이다.

| 폴더 | 키워드 |
|---|---|
| `physics-informed/` | physics-informed / inverse-design / surrogate-modeling (seed: 260415) |
| `surrogate-modeling/` | surrogate-modeling / Bayesian UQ (seed: 260415) |
| `inverse-design/` | cross-link from physics-informed (seed: 260415) |
| `active-learning_sequential-design-of-experiments_surrogate-modeling_bayesian-optimization/` | active-learning × SDoE × surrogate × BO (routine: 260416) |
| `active-learning_surrogate-modeling_bayesian-adaptive-design/` | active-learning × surrogate × Bayesian adaptive design (routine: 260416) |
| `active-learning_tandem-structure_surrogate-modeling/` | active-learning × Tandem Structure × surrogate (routine: 260416) |
| `multimodal-inverse-design_tandem-structure_mixture-density-networks_invertible-neural-networks/` | Multimodal inverse design × Tandem × MDN × INN (routine: 260416) |
| `active-learning_global-sensitivity-analysis_surrogate-modeling/` | active-learning × GSA × surrogate (routine: 260416) |
| `sequential-design-of-experiments_bayesian-adaptive-design_surrogate-modeling_physics-informed/` | SDoE × Bayesian adaptive × surrogate × physics-informed (routine: 260416) |
| `sequential-design-of-experiments_bayesian-adaptive-design/` | SDoE × Bayesian adaptive design (routine: 260416) |
| `sensitivity-aware-sampling_surrogate-modeling/` | sensitivity-aware sampling × surrogate (routine: 260416) |
| `determinantal-point-processes_bayesian-optimization/` | DPP × bayesian optimization (routine: 260417) |
| `invertible-neural-networks_surrogate-modeling/` | INN × surrogate modeling (routine: 260417) |
| `physics-informed_invertible-neural-networks/` | physics-informed × INN/normalizing flow (routine: 260417) |
| `process-variation_physics-informed_surrogate-modeling/` | process variation × physics-informed × surrogate (routine: 260417) |
| `active-learning_surrogate-modeling/` | active learning × surrogate modeling (routine: 260417) |
| `determinantal-point-processes_active-learning/` | DPP × active learning (routine: 260417) |
| `sequential-design-of-experiments_surrogate-modeling_bayesian-adaptive-design/` | SDoE × surrogate × Bayesian adaptive design (routine: 260420) |
| `bayesian-adaptive-design_global-sensitivity-analysis/` | Bayesian adaptive design × GSA (routine: 260420) |
| `physics-informed_bayesian-optimization/` | physics-informed × BO (routine: 260420) |
| `bayesian-optimization_surrogate-modeling/` | BO × surrogate modeling (routine: 260420) |

## 논문 인덱스 (중복 방지)

| arXiv | Title | Added |
|---|---|---|
| 2603.21210 | Pretrained Video Models as Differentiable Physics Simulators for Urban Wind Flows | 260415 |
| 2603.10987 | MCMC Informed Neural Emulators for Uncertainty Quantification in Dynamical Systems | 260415 |
| 2603.21180 | ALMAB-DC: Active Learning, Multi-Armed Bandits, and Distributed Computing for Sequential Experimental Design and Black-Box Optimization | 260416 |
| 2603.18259 | ALABI: Active Learning for Accelerated Bayesian Inference | 260416 |
| 2502.15643 | AutoTandemML: Active Learning Enhanced Tandem Neural Networks for Inverse Design Problems | 260416 |
| 2411.09429 | AI-driven inverse design of materials: Past, present and future | 260416 |
| 2601.11790 | Gradient-based Active Learning with Gaussian Processes for Global Sensitivity Analysis | 260416 |
| 2603.16756 | Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models | 260416 |
| 2504.13320 | Gradient-Free Sequential Bayesian Experimental Design via Interacting Particle Systems | 260416 |
| 2503.04181 | Boosting Offline Optimizers with Surrogate Sensitivity | 260416 |
| 2406.08799 | Pareto Front-Diverse Batch Multi-Objective Bayesian Optimization | 260417 |
| 2510.26704 | How Regularization Terms Make Invertible Neural Networks Bayesian Point Estimators | 260417 |
| 2511.03241 | A Unified Physics-Informed Generative Operator Framework for General Inverse Problems (IGNO) | 260417 |
| 2510.26586 | Physics-Informed Mixture Models and Surrogate Models for Precision Additive Manufacturing | 260417 |
| 2603.13646 | Surrogate-Based Bayesian Inference: Uncertainty Quantification and Active Learning | 260417 |
| 2603.22160 | Data Curation for Machine Learning Interatomic Potentials by Determinantal Point Processes | 260417 |
| 2402.16520 | Sequential Design for Surrogate Modeling in Bayesian Inverse Problems | 260420 |
| 2406.13425 | Coupled Input-Output Dimension Reduction: Goal-oriented Bayesian Experimental Design and Global Sensitivity Analysis | 260420 |
| 2407.09739 | Active Learning for Derivative-Based Global Sensitivity Analysis with Gaussian Processes | 260420 |
| 2503.00420 | A Physics-Informed Bayesian Optimization Method for Rapid Development of Electrical Machines | 260420 |
| 2509.04651 | Sensitivity-Driven Adaptive Surrogate Modeling for Simulation and Optimization of Dynamical Systems | 260420 |
| 2602.04537 | An Efficient Bayesian Framework for Inverse Problems via Optimization and Inversion | 260420 |

## 트리거

- 일자 명시 ±1개월 범위의 논문을 수집. 일자 없으면 루틴 실행 일자(현재 `2026-04-20`, 태그 `260420`)를 기준으로 -24개월 윈도우.
- 키워드 ≥2 매칭, 이전 인덱스와 중복 제외.

## 구조

- 각 폴더 하위에 보고서(`*.md`)와 참조 코드 구현(`*.py`)을 함께 저장
- 동일 키워드 하위에 여러 보고서가 누적될 수 있음
