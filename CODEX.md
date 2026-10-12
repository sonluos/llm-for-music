# LLM for Music 저장소 검토 가이드

## 저장소

- 경로: `C:\Users\USER\Desktop\LLM for Music`
- 환경: Windows PowerShell
- Python: `.venv\Scripts\python.exe`
- 브랜치: `master`

## Reviewer 역할

이 저장소에서 주차별로 다른 에이전트가 구현한 코드를 실행하고 검토한다.

- 기존 설계를 임의로 재설계하지 않는다.
- 사용자가 "검토"만 요청하면 파일을 수정하지 않는다.
- 코드, 테스트, 실제 데모 실행 결과를 근거로 결함을 보고한다.
- 테스트 통과만으로 승인하지 말고 테스트가 놓친 경계 조건도 확인한다.
- 수정 후 "재검토" 요청에는 이전 지적사항의 수정 여부와 회귀 테스트 추가 여부를 먼저 확인한다.
- 커밋이나 푸시는 사용자가 명시적으로 요청했을 때만 수행한다.
- 사용자의 기존 변경사항을 보존한다.

## 기본 검토 순서

### 1. 변경 범위 확인

```powershell
git status --short --branch
git log -6 --oneline
git diff --stat
git diff --check
git diff
```

신규 및 수정된 소스와 테스트를 모두 읽는다.

### 2. 코드 검토 항목

- import 및 package export 오류
- Windows 경로 문제
- tensor shape, dtype, device 문제
- TorchAudio, Transformers, PEFT API 사용 오류
- FFT 및 DSP 수식 오류
- NaN, Inf, divide-by-zero, 불안정한 IIR 등 수치 안정성
- 입력 검증과 경계 조건
- 테스트가 구현을 그대로 반복하는 형식적 테스트인지 여부
- 모델 학습 시 label masking, optimizer state, train/eval 상태
- 평가 지표가 무효 출력을 잘못 보상하는지 여부
- 문서, 스키마, 데모, 테스트 사이의 불일치

### 3. 기본 테스트

```powershell
.venv\Scripts\python.exe -m pytest -q
```

### 4. 통합 테스트

```powershell
.venv\Scripts\python.exe -m pytest -m integration -q
```

통합 테스트는 Hugging Face Qwen 모델과 PEFT를 실제로 로딩하므로 네트워크나 권한 승인이 필요할 수 있다.

### 5. 전체 데모

```powershell
.venv\Scripts\python.exe scripts\demo_audio.py
```

### 6. 산출물 확인

최소 확인 대상:

- `data\output\test_signal.wav`
- `data\output\waveform.png`
- `data\output\spectrum.png`
- `data\output\model_processed_audio.wav`
- `data\output\reverb_demo.wav`
- `data\output\reverb_impulse_response.png`

WAV는 단순 존재 여부뿐 아니라 다음도 확인한다.

- 파일 크기가 0보다 큰지
- 다시 로드되는지
- sample rate와 shape가 올바른지
- NaN 또는 Inf가 없는지
- 비정상적인 clipping이 없는지

데모 출력에서 FFT peak가 약 `440 Hz`와 `1000 Hz`에 나타나는지도 확인한다.

## 현재까지 발견하고 수정한 주요 문제

### 13주차: LoRA 및 audio-conditioned model

발견됐던 문제:

- `train_one_epoch` 호출마다 AdamW optimizer가 다시 생성되어 epoch마다 momentum과 variance 상태가 초기화됨
- prompt와 full text를 따로 토큰화해 target label 경계가 잘못될 가능성
- greedy decoding인데 sampling 전용 generation option 경고 발생

수정 확인 사항:

- `build_optimizer`를 따로 만들고 같은 optimizer를 여러 epoch에서 재사용
- offset mapping을 사용해 prompt 영역을 masking
- optimizer step 누적 회귀 테스트 추가
- greedy decoding 설정 경고 제거

최종 결과:

- 기본 테스트: `142 passed`
- 통합 테스트: `10 passed`
- 데모 정상 완료

### 14주차: reverb 및 model-to-DSP dispatcher

검토 대상:

- Schroeder/Freeverb 형태 reverb
- 모델의 effect JSON을 실제 EQ/reverb DSP로 연결하는 dispatcher
- reverb 스키마 확장

최종 결과:

- 기본 테스트: `162 passed`
- 통합 테스트: `10 passed`
- reverb plot 및 처리된 WAV 정상 생성
- 차단 버그 없음
- TorchAudio 2.9 관련 deprecation warning만 존재

### 15주차: 평가 지표

검토 대상:

- `parameter_error`
- `evaluate_predictions`
- `repetition_stability`
- `acoustic_feature_change`
- sampling을 위한 `temperature` 인자

발견됐던 문제:

- `repetition_stability([{}, {}, ...])`가 모든 무효 출력을 `effect=None`으로 묶어 `agreement=1.0`으로 평가함
- 반복된 실패를 높은 안정성으로 잘못 보상하는 문제였음

수정 확인 사항:

- 이름 있는 effect가 하나도 없으면 `agreement=0.0`
- 무효 출력은 agreement의 분모에는 포함되지만 동의 출력으로는 포함되지 않음
- 전부 무효인 경우와 일부 무효인 경우의 회귀 테스트 추가

최종 결과:

- 기본 테스트: `178 passed`
- 통합 테스트: `10 passed`
- 전체 데모 정상 완료
- 커밋 준비 완료 판정

## 커밋 준비 판정 기준

다음 조건을 모두 만족할 때만 "커밋 준비 완료"로 판정한다.

- `git diff --check`에 실제 whitespace 오류가 없음
- 기본 테스트 전체 통과
- 통합 테스트 전체 통과
- `scripts/demo_audio.py` 정상 종료
- 필수 산출물 생성 확인
- 새 기능의 핵심 경계 조건 테스트 존재
- 재현 가능한 기능 또는 지표 오류가 없음

LF에서 CRLF로 변경될 수 있다는 Git 안내와 TorchAudio 차기 버전 경고는 현재 차단 사유가 아니다. 실제 API 실패가 발생하면 수정 대상으로 취급한다.

## 최종 보고 형식

사용자는 간결한 결과를 선호한다. 최종 답변에는 다음 네 항목만 작성한다.

- 파일 수정: 검토 과정에서 수정한 파일 및 주차 변경 파일 수
- 버그 발견: 심각도와 재현 조건 또는 "차단 버그 없음"
- 테스트 결과: 기본/통합 테스트 수, 데모, FFT 및 산출물 결과
- 커밋 준비 상태: 준비 완료 또는 수정 필요

예시:

```text
- 파일 수정: 검토 과정에서 수정한 파일 없음. 15주차 변경 파일 6개 확인.
- 버그 발견: 이전의 반복 안정성 무효 출력 처리 문제 수정 및 회귀 테스트 확인. 추가 차단 버그 없음.
- 테스트 결과: 기본 178 passed, 통합 10 passed, 데모 정상 완료, FFT peak 440/1000 Hz 확인.
- 커밋 준비 상태: 준비 완료.
```

핵심은 테스트 개수만 확인하는 것이 아니라, 테스트가 빠뜨린 의미상 오류를 작은 입력으로 직접 재현하는 것이다.
