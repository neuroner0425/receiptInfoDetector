# Receipt OCR Service

Flask + Celery + Redis + PaddleOCR 기반의 **영수증 인식 서비스**입니다.  
영수증 이미지를 업로드하면 비동기적으로 OCR을 수행하고,  
매장명 / 주소 / 전화번호 / 일시 / 결제수단 / 품목 목록 / 총액 을 추출하여 JSON 형태로 반환합니다.

---

## 🚀 Features

- **REST API**
  - `POST /detect` : 영수증 이미지 업로드 → 세션 생성 및 OCR 비동기 처리 시작
  - `GET /process?session=<id>` : 세션 상태 확인 및 결과 조회

- **비동기 처리**
  - Celery + Redis로 작업을 큐잉하고 워커에서 OCR 실행

- **OCR**
  - PaddleOCR (`korean`, `en`) 동시 사용 → 누락 최소화
  - OCR 결과에서 영수증 항목을 라인 단위로 재구성

- **데이터 파싱**
  - 매장 정보: 상호, 주소, 전화번호
  - 거래 정보: 일시, 결제수단, 고객정보
  - 품목 리스트: 상품명, 수량, 단가, 소계
  - 총 결제 금액