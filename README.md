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
- Epoch: 30
- Batch size: 128
- Seed 운용: 3 seeds (0, 1, 42)
- 평가셋: CIFAR-10(clean) + CIFAR-10-C(robust)
- 평가지표: clean_acc, corruption_mean_acc, severity_acc, mCE

- 가변 항목:
- augmentation policy

- 비교 원칙:
- augmentation 외 조건 동일
- 동일 학습 예산 유지
- multi-seed 평균/분산 기준 비교

## Current Status

- 상태: 5가지 aug_type × 3 seeds 실험 완료

- `src/data.py`: 데이터셋/증강 파이프라인 구현 완료
- `src/model.py`: ResNet-18 구현 완료
- `scripts/train.py`: 학습 + clean/robust 평가 루프 구현 완료
- `src/utils.py`: 디바이스 설정, 로깅, 시드, 유틸리티 구현 완료
- `configs/baseline.yaml`: 기본 실험 설정 완료
- `scripts/compare.py`: 실험 결과 비교 스크립트 완료
- `scripts/plot.py`: mCE bar, severity line, corruption heatmap 시각화 완료

## Results (seed 0, 1, 42 평균)

| aug_type | clean_acc | corruption_mean_acc | mCE |
|---|---|---|---|
| none | 0.781 | 0.696 | 0.304 |
| basic | 0.849 | 0.734 | 0.266 |
| augmix | 0.794 | 0.736 | 0.264 |
| autoaugment | 0.816 | 0.747 | 0.253 |
| **randaugment** | **0.841** | **0.757** | **0.243** |

- clean accuracy 1위: basic (0.849)
- robustness 1위: randaugment (mCE 0.243)
- clean-robust trade-off: basic은 clean은 높지만 robustness는 augmix와 유사

## Project Structure

- `configs/`: 실험 설정 파일(yaml)
- `scripts/`: 학습, 비교, 시각화, 데이터 다운로드 스크립트
- `src/`: 모델, 데이터, 유틸리티 코드
- `results/`: 실험 결과(로그, 메트릭, 체크포인트)

## Environment Setup

1. Python 가상환경 생성 및 활성화
2. 패키지 설치

**CPU / CUDA**
```bash
pip install -r requirements.txt
```

**Intel Arc GPU - XPU (Windows 네이티브)**
```bash
pip install -r requirements-xpu.txt
```

3. 데이터 다운로드

CIFAR-10은 학습 시 자동으로 다운로드됩니다.
CIFAR-10-C는 아래 스크립트로 받습니다 (약 2.7GB):

**Linux / WSL2**
```bash
bash scripts/download_data.sh        # ./data/ 에 저장 (기본값)
bash scripts/download_data.sh ./data # 저장 경로 직접 지정
```

**Windows**
```bat
scripts\download_data.bat
scripts\download_data.bat .\data
```

의존성 (CPU/CUDA):

- torch==2.10.0
- torchvision==0.25.0
- numpy<2.4
- tqdm>=4.64.1
- pyyaml==6.0.3

## Usage

**단일 실험**
```bash
python scripts/train.py --config configs/baseline.yaml --aug_type basic --seed 42 --epochs 30
```

**전체 실험 (5 aug × 3 seeds)**
```bash
# Linux / WSL2
bash scripts/run_all.sh

# Windows
scripts\run_all.bat
```

**결과 비교**
```bash
python scripts/compare.py
```

**시각화**
```bash
python scripts/plot.py
```

## Recommended Experiment Protocol

### 1) 공통 평가셋

- Clean: CIFAR-10 test
- Robust: CIFAR-10-C (19 corruption x severity 1~5)

### 2) 핵심 지표

- Clean Accuracy
- Corruption 평균 Accuracy
- Severity별 Accuracy
- mCE

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

## Notes

- 결론 기준: 단일 실행 배제, multi-seed 평균/분산 우선
