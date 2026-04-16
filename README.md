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

## 논문 인덱스 (중복 방지)

| arXiv | Title | Added |
|---|---|---|
| 2603.21210 | Pretrained Video Models as Differentiable Physics Simulators for Urban Wind Flows | 260415 |
| 2603.10987 | MCMC Informed Neural Emulators for Uncertainty Quantification in Dynamical Systems | 260415 |
| 2603.21180 | ALMAB-DC: Active Learning, Multi-Armed Bandits, and Distributed Computing for Sequential Experimental Design and Black-Box Optimization | 260416 |
| 2603.18259 | ALABI: Active Learning for Accelerated Bayesian Inference | 260416 |

## 트리거

- PR 제목/본문에 `\d{6}\s` 형식(예: `260520 `)이 포함되면 그 일자 ±1개월 범위의 논문을 수집.
- 없으면 루틴 실행 일자(현재 `2026-04-16`, 태그 `260416`)를 기준으로 -24개월 윈도우.
- 키워드 ≥2 매칭, 이전 인덱스와 중복 제외.

## 구조

- 각 폴더 하위에 보고서(`*.md`)와 참조 코드 구현(`*.py`)을 함께 저장
- 동일 키워드 하위에 여러 보고서가 누적될 수 있음
