# routine

주제별 arXiv, top journal 논문 루틴 보고서 저장소.

## 현재 주제

**sequential variation propagation** — 다대일(many-to-one) 순방향 문제를 일대다(one-to-many) 역방향으로 풀 때 생기는 해의 다중성(ambiguity)을 분산(불확실성) 제어로 억제하는 방법.

## 선정 규칙

- 다음 키워드 중 **3개 이상** 일치하는 논문만 포함
  - process variation / physics-informed / bayesian optimization / Invertible Neural Networks /
    bayesian adaptive design / active learning / Multimodal inverse design /
    sequential design of experiments / sensitivity-aware sampling /
    global sensitivity analysis / surrogate modeling / cs.LG and physics.comp-ph /
    Tandem Structure / Mixture Density Networks / Determinantal Point Processes /
    Interaction-Integrated Gradients (I-IG) / Transformer-Specific Attribution
- 이전 루틴 결과와 중복되는 논문은 제외
- 코드 구현이 공개되어 있지 않으면 핵심 방법을 재현하는 간단한 참조 구현을 함께 작성

## 폴더 = 검색 키워드 조합

각 폴더명은 검색에 사용한 키워드 조합을 `_`로 연결한 것이다.

| 폴더 | 키워드 |
|---|---|
| [`physics-informed/`](./physics-informed/) | physics-informed / inverse-design / surrogate-modeling (seed: 260415) |
| [`surrogate-modeling/`](./surrogate-modeling/) | surrogate-modeling / Bayesian UQ (seed: 260415) |
| [`inverse-design/`](./inverse-design/) | cross-link from physics-informed (seed: 260415) |
| [`active-learning_sequential-design-of-experiments_surrogate-modeling_bayesian-optimization/`](./active-learning_sequential-design-of-experiments_surrogate-modeling_bayesian-optimization/) | active-learning × SDoE × surrogate × BO (routine: 260416) |
| [`active-learning_surrogate-modeling_bayesian-adaptive-design/`](./active-learning_surrogate-modeling_bayesian-adaptive-design/) | active-learning × surrogate × Bayesian adaptive design (routine: 260416) |
| [`active-learning_tandem-structure_surrogate-modeling/`](./active-learning_tandem-structure_surrogate-modeling/) | active-learning × Tandem Structure × surrogate (routine: 260416) |
| [`multimodal-inverse-design_tandem-structure_mixture-density-networks_invertible-neural-networks/`](./multimodal-inverse-design_tandem-structure_mixture-density-networks_invertible-neural-networks/) | Multimodal inverse design × Tandem × MDN × INN (routine: 260416) |
| [`active-learning_global-sensitivity-analysis_surrogate-modeling/`](./active-learning_global-sensitivity-analysis_surrogate-modeling/) | active-learning × GSA × surrogate (routine: 260416) |
| [`sequential-design-of-experiments_bayesian-adaptive-design_surrogate-modeling_physics-informed/`](./sequential-design-of-experiments_bayesian-adaptive-design_surrogate-modeling_physics-informed/) | SDoE × Bayesian adaptive × surrogate × physics-informed (routine: 260416) |
| [`sequential-design-of-experiments_bayesian-adaptive-design/`](./sequential-design-of-experiments_bayesian-adaptive-design/) | SDoE × Bayesian adaptive design (routine: 260416) |
| [`sensitivity-aware-sampling_surrogate-modeling/`](./sensitivity-aware-sampling_surrogate-modeling/) | sensitivity-aware sampling × surrogate (routine: 260416) |
| [`determinantal-point-processes_bayesian-optimization/`](./determinantal-point-processes_bayesian-optimization/) | DPP × bayesian optimization (routine: 260417) — sequence-aware API, see [How to use](#how-to-use) |
| [`invertible-neural-networks_surrogate-modeling/`](./invertible-neural-networks_surrogate-modeling/) | INN × surrogate modeling (routine: 260417) |
| [`physics-informed_invertible-neural-networks/`](./physics-informed_invertible-neural-networks/) | physics-informed × INN/normalizing flow (routine: 260417) |
| [`process-variation_physics-informed_surrogate-modeling/`](./process-variation_physics-informed_surrogate-modeling/) | process variation × physics-informed × surrogate (routine: 260417) |
| [`active-learning_surrogate-modeling/`](./active-learning_surrogate-modeling/) | active learning × surrogate modeling (routine: 260417) |
| [`determinantal-point-processes_active-learning/`](./determinantal-point-processes_active-learning/) | DPP × active learning (routine: 260417) |
| [`interaction-integrated-gradients_sensitivity-aware-sampling_surrogate-modeling_process-variation/`](./interaction-integrated-gradients_sensitivity-aware-sampling_surrogate-modeling_process-variation/) | I-IG × sensitivity-aware × surrogate × process variation (routine: 260502) |
| [`tandem-structure_multimodal-inverse-design_surrogate-modeling/`](./tandem-structure_multimodal-inverse-design_surrogate-modeling/) | Tandem × Multimodal inverse × surrogate (routine: 260502) |
| [`process-variation_global-sensitivity-analysis_surrogate-modeling_bayesian-optimization/`](./process-variation_global-sensitivity-analysis_surrogate-modeling_bayesian-optimization/) | PV × GSA × surrogate × BO (routine: 260502) |
| [`bayesian-optimization_sequential-design-of-experiments_surrogate-modeling_process-variation/`](./bayesian-optimization_sequential-design-of-experiments_surrogate-modeling_process-variation/) | BO × SDoE × surrogate × PV (routine: 260502) |
| [`bayesian-adaptive-design_sequential-design-of-experiments_surrogate-modeling_active-learning/`](./bayesian-adaptive-design_sequential-design-of-experiments_surrogate-modeling_active-learning/) | BAD × SDoE × surrogate × AL (routine: 260502) |
| [`sequential-design-of-experiments_bayesian-adaptive-design_surrogate-modeling_sensitivity-aware-sampling/`](./sequential-design-of-experiments_bayesian-adaptive-design_surrogate-modeling_sensitivity-aware-sampling/) | SDoE × BAD × surrogate × sensitivity-aware (routine: 260502) |
| [`physics-informed_multimodal-inverse-design_surrogate-modeling_process-variation/`](./physics-informed_multimodal-inverse-design_surrogate-modeling_process-variation/) | PI × Multimodal inverse × surrogate × PV (routine: 260502) |
| [`multimodal-inverse-design_surrogate-modeling_active-learning/`](./multimodal-inverse-design_surrogate-modeling_active-learning/) | Multimodal inverse × surrogate × AL (routine: 260502) |
| [`bayesian-optimization_global-sensitivity-analysis_surrogate-modeling/`](./bayesian-optimization_global-sensitivity-analysis_surrogate-modeling/) | BO × GSA × surrogate (routine: 260502) |

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
| 2510.22176 | Interpretable Geometry Sensitivity for Inverse Design of Integrated Photonics | 260502 |
| 2506.10044 | Improving the performance of optical inverse design of multilayer thin films using CNN-LSTM tandem neural networks | 260502 |
| 2512.13354 | Data-driven inverse uncertainty quantification: application to the Chemical Vapor Deposition Reactor Modeling | 260502 |
| 2504.04244 | From Automation to Autonomy in Smart Manufacturing: A Bayesian Optimization Framework | 260502 |
| 2511.23141 | Automated Discovery of Laser Dicing Processes with Bayesian Optimization for Semiconductor Manufacturing | 260502 |
| 2507.17713 | Sequential Bayesian Design for Efficient Surrogate Construction in the Inversion of Darcy Flows | 260502 |
| 2409.09141 | Sequential Infinite-Dimensional Bayesian Optimal Experimental Design with DILANO | 260502 |
| 2506.00056 | Toward Knowledge-Guided AI for Inverse Design in Manufacturing | 260502 |
| 2409.15307 | An ILUES-based adaptive Gaussian process method for multimodal Bayesian inverse problems | 260502 |
| 2603.17516 | Maximum-Projection-Based Bayesian Optimization Utilizing Sensitivity Analysis for Turbine Design | 260502 |

## How to use

일부 폴더의 참조 구현은 사용자 친화적 API를 함께 제공한다. 현재 제공:

### `determinantal-point-processes_bayesian-optimization/sequence_aware_impl.py`

시퀀스별(레시피/웨이퍼/공정) DOE 제안을 받고 싶을 때. 과거 측정 데이터
`(sequence, x, y)`를 넣으면 다음에 시도할 `(sequence, x)` 배치를 돌려준다.

```python
import numpy as np
from determinantal_point_processes_bayesian_optimization.sequence_aware_impl import (
    suggest_next_batch,
)
# 또는 파일을 바로 sys.path에 추가해 from sequence_aware_impl import ...

S_hist = np.array([0, 0, 1, 1, 2, 2])          # 각 행이 속한 시퀀스 id
X_hist = np.array([[0.1, 0.5], [0.3, 0.7], ...])  # (N, d) 원공간 파라미터
Y_hist = np.array([...])                          # (N,) 또는 (N, n_obj) 목적값 (최대화)

bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
bounds_list = [bounds, bounds, bounds]            # 시퀀스별 파라미터 경계

S_new, X_new = suggest_next_batch(
    S_hist, X_hist, Y_hist,
    bounds_list=bounds_list,
    m_per_seq=[2, 2, 2],    # 시퀀스별 DOE 크기
    rho=0.3,                # 0=독립 GPs, →1=완전 공유
    acq="ucb",              # 또는 "ei"
    seed=None,              # 연속 호출 시 None을 넘겨야 난수 상태가 진행됨
)
```

주의:
- `X_new`는 `bounds_list`와 같은 **원공간(original space)** 값이다 (내부 정규화 없음).
- 루프에서 같은 `seed=<int>`를 반복 전달하면 매번 동일한 배치가 나온다. 서로
  다른 배치를 원하면 `seed=None`(기본)으로 두거나 외부 `rng`를 넘겨라.
- `Y_hist`는 1D(단일 목적) 또는 2D(다중 목적) 둘 다 허용된다.

## 트리거

- 일자 명시 ±1개월 범위의 논문을 수집. 일자 없으면 루틴 실행 일자(최신 `2026-05-02`, 태그 `260502`)를 기준으로 -24개월 윈도우.
- 키워드 ≥3 매칭, 이전 인덱스와 중복 제외.
- 신규 키워드: Interaction-Integrated Gradients (I-IG), Transformer-Specific Attribution.

## 구조

- 각 폴더 하위에 보고서(`*.md`)와 참조 코드 구현(`*.py`)을 함께 저장
- 동일 키워드 하위에 여러 보고서가 누적될 수 있음
