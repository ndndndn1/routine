# routine

주제별 최근 논문 보고서 저장소 (topic: **variation propagation**).

## 구조

- 핵심 키워드를 폴더명으로 사용 (예: `physics-informed/`, `surrogate-modeling/`, `inverse-design/` ...)
- 각 폴더 하위에 보고서(`*.md`)와 참조 코드 구현(`*.py`)을 함께 저장
- 동일 키워드 하위에 여러 보고서가 누적될 수 있음

## 선정 규칙

- 주제: variation propagation
- 다음 키워드 중 **2개 이상** 일치하는 논문만 포함
  - process variation / physics-informed / bayesian optimization /
    bayesian adaptive design / active learning / inverse design /
    sequential design of experiments / sensitivity-aware sampling /
    global sensitivity analysis / surrogate modeling / cs.LG and physics.comp-ph
- 이전 루틴 결과와 중복되는 논문은 제외
- 코드 구현이 공개되어 있지 않으면 핵심 방법을 재현하는 간단한 참조 구현을 함께 작성

## 트리거

- 풀 리퀘스트 제목/본문에 `\d{6}\s` (예: `260415 `) 형식의 일자가 포함되면,
  해당 일자 **전후 1개월** 내 발표된 논문을 검색/정리하여 커밋
- PR 트리거가 없는 초기/수동 실행 시에는 실행 일자를 기준으로 동일 규칙 적용

## 현재 인덱스

| 일자(search ref) | 폴더 | 파일 | 논문 |
| --- | --- | --- | --- |
| 260415 | `physics-informed/` | `2603.21210_pretrained_video_cfd.md` | Perini et al., "Pretrained Video Models as Differentiable Physics Simulators for Urban Wind Flows" (2026-03-22) |
| 260415 | `inverse-design/` | `2603.21210_pretrained_video_cfd.md` | (동일 논문, 키워드 교차) |
| 260415 | `surrogate-modeling/` | `2603.10987_mcmc_neural_emulators.md` | Haario et al., "MCMC Informed Neural Emulators for UQ in Dynamical Systems" (2026-03-11) |
