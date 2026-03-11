# cifar10c-aug-robustness

- 목적: 데이터 증강(augmentation) 기반 강건성(robustness) 검증
- 범위: 분포 이동 환경(CIFAR-10-C) 중심 비교 실험

## Goal

- 증강 기법이 **clean accuracy**와 **CIFAR-10-C robustness**에 주는 영향을 분리해 측정
- 증강별 성능 비교 및 trade-off 분석
- 재현 가능한 실험 파이프라인 구성

## Step 1) Fixed Experiment Spec (v0)

- 고정 항목:
- 모델: ResNet-18
- Optimizer: SGD(momentum 0.9, weight decay 5e-4)
- Scheduler: Cosine + warmup 5 epochs
- Epoch: 200
- Batch size: train 128 / test 256
- Seed 운용: 기본 seed 42, 통계용 3 seeds
- 평가셋: CIFAR-10(clean) + CIFAR-10-C(robust)
- 평가지표: clean_acc, corruption_mean_acc, severity_acc

- 가변 항목:
- augmentation policy

- 비교 원칙:
- augmentation 외 조건 동일
- 동일 학습 예산 유지
- multi-seed 평균/분산 기준 비교

## Current Status

- 상태: 실험용 스캐폴딩

- `src/data.py`: 데이터셋/증강 파이프라인
- `src/model.py`: 모델 정의
- `src/train.py`: 학습/평가 루프
- `src/utils.py`: 로깅/시드/유틸리티
- `configs/baseline.yaml`: 기본 실험 설정

- 구현 상태: 위 파일 초기화만 완료(내용 비어 있음)
- 구현 기준: 아래 실험 프로토콜

## Project Structure

- `configs/`: 실험 설정 파일(yaml)
- `src/`: 학습 코드
- `results/`: 실험 결과(로그, 메트릭, 체크포인트)

## Environment Setup

1. Python 가상환경 생성 및 활성화
2. 패키지 설치

```bash
pip install -r requirements.txt
```

의존성:

- torch==2.10.0
- torchvision==0.25.0
- tqdm==4.67.3
- pyyaml==6.0.3

## Recommended Experiment Protocol

### 1) 공통 평가셋

- Clean: CIFAR-10 test
- Robust: CIFAR-10-C (15 corruption x severity 1~5)

### 2) 핵심 지표

- Clean Accuracy
- Corruption 평균 Accuracy
- Severity별 Accuracy
- (선택) mCE

### 3) 비교군 구성

- Baseline: no-aug 또는 최소 증강
- Single augmentation: 각 기법 단독
- Combined augmentation: 유망 조합

비교 공정성 기준(고정 항목):

- 모델 아키텍처
- optimizer/lr scheduler
- epoch/학습 예산
- seed 개수(권장 3개 이상)

### 4) 결과 저장 규칙

실험별 디렉토리(`results/<run_name>/`) 저장 항목:

- config snapshot (`config.yaml`)
- clean/robust 요약 메트릭 (`metrics.json`)
- corruption별 상세 결과 (`corruption_metrics.csv`)
- 체크포인트 (`best.pt`, `last.pt`)

## Suggested Next Steps

1. `src/data.py`에 CIFAR-10 및 CIFAR-10-C 로더 구현
2. `src/model.py`에 baseline 모델 구현
3. `src/train.py`에 학습 + clean/robust 평가 루프 구현
4. `results/` 집계 스크립트 추가

## Notes

- 초기 단계: augmentation 후보 소수 유지, 엄격 비교
- 결론 기준: 단일 실행 배제, multi-seed 평균/분산 우선
