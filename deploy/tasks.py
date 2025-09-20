from celery_app import app
from paddleocr import PaddleOCR
import cv2, os, json, math, re, unicodedata
from rapidfuzz import process as fuzz_process, fuzz as fuzz_ratio
from datetime import datetime
from collections import defaultdict

BASE_DIR = "/opt/homebrew/lib/masters"
MIN_CONF = 0.55

# -----------------------------
# Fuzzy Correction Configuration
# -----------------------------
# 자주 등장하는 키워드/헤더/필드명을 vocabulary 로 둬서 OCR 오탈자 보정
FUZZY_KEYWORDS = list({
    # 결제/합계 관련
    "결제금액", "신용판매", "총결제금액", "총 결제금액", "받을금액", "합계", "총액", "청구금액", "결제", "면세", "과세", "부가세",
    # 테이블 헤더
    "품목", "상품명", "내역", "단가", "수량", "금액", "합계", "소계", "할인", "쿠폰",
    # 메타 정보
    "영수증", "매출전표", "현금영수증", "승인", "승인번호", "카드번호", "일시불", "할부",
    # 결제수단/브랜드
    "카드", "신용", "체크", "현금", "포인트", "카카오페이", "네이버페이", "토스페이", "PAYCO",
    # 기타
    "상호", "가맹점명", "매장명", "주소", "소재지", "사업장소재지", "가맹점주소", "고객", "회원", "적립", "포인트"
})

# 보정 대상이 될 최소/최대 길이 및 점수 기준
FUZZY_MIN_LEN = 2
FUZZY_MAX_LEN = 12  # 지나치게 긴 문장은 보정하지 않음 (부분 단어 단위 정규화만)
FUZZY_SCORE_CUTOFF = 86  # 0~100 스코어, 임계 이상일 때만 대치

# Tuning 가이드:
# - FUZZY_SCORE_CUTOFF: 너무 낮추면 과보정(잘못된 치환) 위험. 80~90 권장.
# - FUZZY_MAX_LEN: 긴 문장은 의미 단위가 다양해져 잘못 치환될 수 있으므로 10~15 내에서 조정.
# - Vocabulary: 도메인(편의점, 식당 등)에 따라 추가 키워드 확장 가능.
# - fuzzy_normalize_text_line: 금액/숫자 라인, 한글 없는 라인은 skip 하여 성능 보전.

def fuzzy_correct_token(token: str):
    """단일 토큰에 대해 rapidfuzz fuzzy matching 으로 가장 유사한 키워드로 보정.

    조건:
    - 길이가 FUZZY_MIN_LEN 이상 FUZZY_MAX_LEN 이하
    - 완전히 숫자/기호만이 아닌 경우
    - 스코어가 FUZZY_SCORE_CUTOFF 이상
    """
    raw = token.strip()
    if not raw:
        return token
    if len(raw) < FUZZY_MIN_LEN or len(raw) > FUZZY_MAX_LEN:
        return token
    # 숫자/특수문자만 구성 시 제외
    if re.fullmatch(r'[\d\W_]+', raw):
        return token
    # 이미 exact hit
    if raw in FUZZY_KEYWORDS:
        return raw
    # best match 탐색
    best = fuzz_process.extractOne(raw, FUZZY_KEYWORDS, scorer=fuzz_ratio.QRatio, score_cutoff=FUZZY_SCORE_CUTOFF)
    if best:
        corrected, score, _ = best
        return corrected
    return token

def fuzzy_normalize_text_line(text: str) -> str:
    """한 OCR 라인의 텍스트를 공백 단위 토큰별 fuzzy 보정.

    - 이미 숫자 금액 라인 (숫자/쉼표/원 위주) 은 그대로 둔다.
    - 토큰 개별 보정 후 다시 합침.
    """
    if not text:
        return text
    # 금액 라인 여부 (숫자/원/콤마/공백/부호 비율이 높고 한글이 거의 없는 경우) -> skip
    han = len(re.findall(r'[가-힣]', text))
    digits = len(re.findall(r'[0-9]', text))
    if digits > 0 and han == 0:
        return text
    tokens = re.split(r'(\s+)', text)  # 공백 토큰 보존
    out = []
    for tk in tokens:
        if tk.isspace() or tk == '':
            out.append(tk)
            continue
        out.append(fuzzy_correct_token(tk))
    return ''.join(out)

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_image(path):
    return cv2.imread(path)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def to_fullwidth_fixed(s: str) -> str:
    # 전각/특수문자 → 반각 치환
    return unicodedata.normalize("NFKC", s)

def fix_ocr_number_typos(s: str) -> str:
    # 숫자 OCR 흔한 오탈자 보정 (좌우가 숫자일 때 보정)
    rep = {
        'O': '0', 'o': '0', 'D': '0', 'Q': '0',
        'I': '1', 'l': '1', '|' : '1', '¹': '1',
        'S': '5', '$': '5',
        'B': '8',
        '—': '-', '–': '-', '−': '-', '―': '-',
        ',,': ',', '  ': ' '
    }
    out = []
    for ch in s:
        out.append(rep.get(ch, ch))
    s = ''.join(out)
    # 1,OOO → 1,000 류 보정
    s = re.sub(r'(\d)[O](\d{2,3})', r'\g<1>0\g<2>', s)
    s = re.sub(r'[^0-9\-\.,원\s@x개X*•·~()영합총결제금액가-za-zA-Z가-힣]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_text(s: str) -> str:
    s = to_fullwidth_fixed(s)
    s = s.replace('￦', '₩').replace('원', ' 원')
    s = fix_ocr_number_typos(s)
    return s

def amount_from_text(s: str):
    # "10,000원", "총액: 12,340", "-2,000", "2 000" 등
    s = clean_text(s)
    # 괄호 음수 "(2,000)" → -2000
    negative = False
    if re.search(r'\(|-|\b환급\b|\b할인\b', s):
        negative = True
    m = re.findall(r'(?<![a-zA-Z])[-]?\d{1,3}(?:[,\s]\d{3})+|\d+', s)
    if not m:
        return None
    num = m[-1]  # 라인 끝쪽 숫자 우선
    num = num.replace(',', '').replace(' ', '')
    try:
        val = int(num)
        if negative and val > 0:
            val = -val
        return val
    except:
        return None

def parse_korean_phone(s: str):
    s2 = clean_text(s).replace(' ', '')
    # 02-XXXX-XXXX / 0XX-XXX-XXXX / 010-XXXX-XXXX / 010XXXX....
    s2 = s2.replace('.', '-').replace(')', '-').replace('(', '')
    m = re.search(r'(0\d{1,2}-?\d{3,4}-?\d{4})', s2)
    return m.group(1) if m else None

def parse_datetime_candidates(s: str):
    s = clean_text(s)
    # 자주 나오는 패턴들 (승인/거래/영수/발행 일시 포함)
    pats = [
        r'(\d{4})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})[일\s]*[T\s]*?(\d{1,2})[:시](\d{1,2})(?:[:분](\d{1,2}))?',
        r'(\d{4})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})[일]?',
        r'(\d{2})[.\-/]\s*(\d{1,2})[.\-/]\s*(\d{1,2})[^\d]',
        r'(\d{1,2})[.\-/월]\s*(\d{1,2})[.\-/일]\s*(\d{4})[^\d]'
    ]
    out = []
    for p in pats:
        for m in re.finditer(p, s + " "):  # 안전한 lookahead
            g = list(m.groups())
            try:
                if len(g) >= 6 and g[5] is not None:
                    yyyy, mm, dd, HH, MM, SS = int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5])
                    dt = datetime(yyyy if yyyy > 99 else 2000 + yyyy, mm, dd, HH, MM, SS)
                elif len(g) >= 5 and g[4] is not None:
                    yyyy, mm, dd, HH, MM = int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4])
                    dt = datetime(yyyy if yyyy > 99 else 2000 + yyyy, mm, dd, HH, MM, 0)
                elif len(g) >= 3:
                    # date only
                    yyyy, mm, dd = int(g[0]), int(g[1]), int(g[2])
                    if yyyy <= 31:  # yy mm dd 패턴
                        yy, mm, dd = yyyy, mm, dd
                        yyyy = 2000 + yy
                    dt = datetime(yyyy, mm, dd, 0, 0, 0)
                else:
                    continue
                out.append(dt)
            except Exception:
                continue
    return out

def is_probably_address(s: str):
    s2 = clean_text(s)
    # "주소", "소재지", 도/시/구/동/읍/면/리/로/길/번지 포함
    if re.search(r'(주소|소재지|사업장소재지|가맹점주소)', s2):
        return True
    if re.search(r'(도|시|군|구|읍|면|동|리|로|길)\s*\d', s2):
        return True
    if re.search(r'\d+-\d+번지', s2):
        return True
    return False

def calc_box_rect(qbox):
    # 4점 → (x1,y1,x2,y2), h, cx, cy
    xs = [p[0] for p in qbox]
    ys = [p[1] for p in qbox]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    h = y2 - y1
    w = x2 - x1
    return x1, y1, x2, y2, w, h, (x1+x2)/2.0, (y1+y2)/2.0

def group_lines(ocr_items, y_merge_ratio=0.6):
    """
    ocr_items: [{'text','conf','box'}]
    각 단어/조각을 y 중심 기준으로 라인 그룹핑 후, x 정렬해 문장화
    """
    # 각 아이템에 y, h 계산
    enriched = []
    for it in ocr_items:
        x1,y1,x2,y2,w,h,cx,cy = calc_box_rect(it['box'])
        enriched.append({**it, 'x1':x1,'y1':y1,'x2':x2,'y2':y2,'w':w,'h':h,'cx':cx,'cy':cy})
    # y 기준 정렬
    enriched.sort(key=lambda x: (x['cy'], x['x1']))

    lines = []
    for w in enriched:
        if not lines:
            lines.append([w])
            continue
        last_line = lines[-1]
        avg_h = sum(t['h'] for t in last_line)/len(last_line)
        avg_y = sum(t['cy'] for t in last_line)/len(last_line)
        # 같은 라인인지 판단 (y 거리 < ratio * 평균 높이)
        if abs(w['cy'] - avg_y) <= max(8, y_merge_ratio * max(avg_h, w['h'])):
            last_line.append(w)
        else:
            lines.append([w])

    # 각 라인 x 정렬 & 텍스트 결합
    merged = []
    for ln in lines:
        ln.sort(key=lambda t: t['x1'])
        # 기존 clean_text 후 fuzzy 보정 적용
        raw_join = ' '.join(clean_text(t['text']) for t in ln if t['text'])
        text = fuzzy_normalize_text_line(raw_join)
        x1 = min(t['x1'] for t in ln)
        x2 = max(t['x2'] for t in ln)
        y1 = min(t['y1'] for t in ln)
        y2 = max(t['y2'] for t in ln)
        merged.append({
            'text': text.strip(),
            'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
            'h': y2-y1, 'w': x2-x1,
            'parts': ln
        })
    return merged

def read_with_paddle(image_path):
    results = []
    for lang in ["korean", "en"]:
        ocr = PaddleOCR(lang=lang, use_gpu=False, show_log=False)
        out = ocr.ocr(image_path, cls=True)
        if not out or not out[0]:
            continue
        for line in out[0]:
            box = line[0]
            text = line[1][0]
            conf = line[1][1]
            if conf >= MIN_CONF and text.strip():
                # 단어 단위에서 너무 짧은 pure digit/금액 패턴은 fuzzy 대상이 아님.
                results.append({'text': text, 'conf': conf, 'box': box})
    # 텍스트/박스 중복 제거(대강)
    uniq = []
    seen = set()
    for r in results:
        key = (r['text'], tuple(int(p) for xy in r['box'] for p in xy))
        if key not in seen:
            uniq.append(r)
            seen.add(key)
    return uniq

# -----------------------------
# Header / meta extraction
# -----------------------------
CARD_KEYWORDS = [
    "카드", "신용", "체크", "승인", "승인번호", "카드번호", "할부", "일시불",
    "현대", "국민", "농협", "신한", "우리", "하나", "BC", "롯데", "삼성",
    "카카오페이", "네이버페이", "토스페이", "PAYCO", "PAY"
]
PAYMENT_PRIORITIES = [
    "결제금액", "신용판매", "총 결제금액", "받을금액", "합계", "총액", "청구금액", "결제"
]

ITEM_HEADER_HINTS = ["품목", "상품명", "내역", "단가", "수량", "금액", "합계", "소계"]
TABLE_STOP_HINTS = ["합계", "총액", "결제", "부가세", "면세", "과세", "현금", "카드", "포인트", "승인", "영수증", "현금영수증", "쿠폰", "할인"]

def pick_store_name(lines):
    # 1) 상단 25% 영역 중 가장 글자 크기/폭이 큰 라인
    if not lines:
        return ""
    ys = [ln['y2'] for ln in lines]
    top_cut = sorted(ys)[0] + (max(ys) - min(ys)) * 0.25
    top_lines = [ln for ln in lines if ln['y2'] <= top_cut]
    cand = top_lines if top_lines else lines[:5]
    cand = [ln for ln in cand if len(ln['text']) >= 2]
    # "영수증", "매출전표" 같은 일반 헤더는 제외
    cand = [ln for ln in cand if not re.search(r'(영수증|매출전표|신용판매전표|현금영수증)', ln['text'])]
    if not cand:
        cand = lines[:5]
    cand.sort(key=lambda l: (l['h'] * l['w']), reverse=True)
    txt = cand[0]['text'].strip()
    # "상호: XXX" 등 키워드 기반
    m = re.search(r'(상호|가맹점명|매장명)\s*[:：]?\s*(.+)', txt)
    if m:
        return m.group(2).strip()
    return txt

def pick_address(lines):
    for ln in lines:
        if is_probably_address(ln['text']):
            t = re.sub(r'^(주소|소재지|사업장소재지|가맹점주소)\s*[:：]?\s*', '', ln['text'])
            return t.strip()
    # 못찾으면 빈값
    return ""

def pick_phone(lines):
    # 주소 라인 인접(±2라인)도 탐색
    addr_idx = None
    for i, ln in enumerate(lines):
        if is_probably_address(ln['text']):
            addr_idx = i
            break
    search_order = list(range(len(lines)))
    if addr_idx is not None:
        nearby = list(range(max(0, addr_idx-2), min(len(lines), addr_idx+3)))
        search_order = nearby + [i for i in search_order if i not in nearby]
    for i in search_order:
        ph = parse_korean_phone(lines[i]['text'])
        if ph:
            # 010XXXXXXXX → 하이픈 포맷 정리
            ph = re.sub(r'(\d{2,3})(\d{3,4})(\d{4})', r'\1-\2-\3', ph.replace('-', ''))
            return ph
    return ""

def pick_datetime(lines):
    # 승인/거래/영수/발행 등의 키워드가 있는 라인 우선
    dt_cand = []
    for ln in lines:
        if re.search(r'(승인|거래|영수|발행|결제)\s*(일시|시간|일자)?', ln['text']):
            dt_cand.extend(parse_datetime_candidates(ln['text']))
    if not dt_cand:
        for ln in lines:
            dt_cand.extend(parse_datetime_candidates(ln['text']))
    if not dt_cand:
        return ""
    # 가장 "늦은" 시간(보통 승인/발행 시간) 선택
    dt = sorted(dt_cand)[-1]
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def pick_payment_method(lines):
    # 카드/현금/간편결제 키워드 & 카드사명 조합
    text_all = " | ".join(ln['text'] for ln in lines[:])
    # 카드사 추출
    brand = None
    for k in ["신한","국민","농협","우리","하나","현대","롯데","삼성","BC"]:
        if re.search(k, text_all):
            brand = k
            break
    # 간편결제
    for pay in ["카카오페이","네이버페이","토스페이","PAYCO","PAY"]:
        if re.search(pay, text_all, re.I):
            return f"{pay}"
    if re.search(r'(현금|현금영수증)', text_all):
        return "현금"
    if re.search(r'(카드|신용|체크|승인|일시불|할부)', text_all):
        return f"카드({brand})" if brand else "카드"
    return ""

def pick_user_info(lines):
    # 회원/고객/적립 키워드 라인 수집
    info = []
    for ln in lines:
        if re.search(r'(회원|고객|적립|포인트|멤버십|바코드)', ln['text']):
            # "회원번호: xxx", "고객명: yyy" 등만 추출
            snippet = re.sub(r'\s{2,}', ' ', ln['text'])
            info.append(snippet)
    return ' / '.join(sorted(set(info)))[:200]  # 너무 길면 자름

# -----------------------------
# Items table parsing
# -----------------------------
def detect_table_region(lines):
    """
    품목/상품명/단가/수량/금액 등의 헤더 라인을 찾고,
    그 아래 영역을 테이블로 간주
    """
    header_idx = None
    for i, ln in enumerate(lines):
        if sum(1 for h in ITEM_HEADER_HINTS if h in ln['text']) >= 2:
            header_idx = i
            break
    if header_idx is None:
        # 힌트 1개만 있어도 헤더로 가정 (보수적)
        for i, ln in enumerate(lines):
            if any(h in ln['text'] for h in ITEM_HEADER_HINTS):
                header_idx = i
                break
    if header_idx is None:
        return None, None
    # stop 지점 탐색
    stop_idx = None
    for j in range(header_idx+1, len(lines)):
        if any(h in lines[j]['text'] for h in TABLE_STOP_HINTS):
            stop_idx = j
            break
    if stop_idx is None:
        stop_idx = len(lines)
    return header_idx, stop_idx

def split_columns_by_alignment(line_parts):
    """
    한 라인의 단어 파트를 x좌표로 보고
    문자열/숫자 영역을 rough하게 분리
    반환: (name_part_text, qty_text, unit_text, sub_total_text)
    """
    # 오른쪽으로 갈수록 금액류가 모이는 경향
    # parts: list of word tokens with x1
    parts = sorted(line_parts, key=lambda p: p['x1'])
    texts = [clean_text(p['text']) for p in parts if p['text']]
    joined = ' '.join(texts)
    # 1) 명확한 패턴 우선
    #   상품명 .... 2 1,000 2,000
    m = re.search(r'(.+?)\s+(\d+)\s+([@\s]?\d[\d,\s]*)\s+(\-?\d[\d,\s]*)$', joined)
    if m:
        return m.group(1).strip(), m.group(2), m.group(3), m.group(4)
    #   상품명 .... 2개 1,000 2,000
    m = re.search(r'(.+?)\s+(\d+)\s*개\s+([@\s]?\d[\d,\s]*)\s+(\-?\d[\d,\s]*)$', joined)
    if m:
        return m.group(1).strip(), m.group(2), m.group(3), m.group(4)
    # 2) 숫자 토큰 2~3개가 라인 끝에 몰린 경우
    nums = [t for t in texts if re.search(r'\d', t)]
    if len(nums) >= 2:
        # 뒤에서 2~3개가 qty/unit/subtotal일 가능성
        sub = nums[-1]
        unit = nums[-2]
        qty = None
        if len(nums) >= 3:
            qty = nums[-3]
        else:
            # "x2" 또는 "@1000" 형태로 섞여있을 수 있음
            m = re.search(r'(?:x|X|\*)\s*(\d+)', joined)
            if m: qty = m.group(1)
        name = joined
        # 라인에서 숫자 부분 제거하여 이름 추출 (너무 공격적이면 품목 손실)
        tail = (qty or '') + ' ' + (unit or '') + ' ' + (sub or '')
        name = name.replace(tail, '').strip()
        return name, qty, unit, sub
    # 3) 실패 시 전부 이름으로 간주
    return joined, None, None, None

def parse_item_line(line):
    name, qty, unit, sub = split_columns_by_alignment(line['parts'])
    # 숫자 정리
    q = None
    if qty:
        m = re.search(r'(\d+)', qty)
        if m:
            q = int(m.group(1))
    u = amount_from_text(unit) if unit else None
    st = amount_from_text(sub) if sub else amount_from_text(line['text'])
    # 품목명에서 일반 헤더/합계/결제 등 제외
    if any(h in name for h in TABLE_STOP_HINTS + ITEM_HEADER_HINTS):
        return None
    # 너무 짧고 숫자만 있으면 제외
    if len(re.sub(r'[\W\d_]+', '', name)) < 1:
        return None
    # 결과 구성
    item = {
        "productName": name.strip(),
        "quantity": int(q) if q else 1,
        "unitPrice": int(u) if u is not None else None,
        "subTotal": int(st) if st is not None else None
    }
    # unitPrice가 없고 subTotal / quantity로 역산
    if item["unitPrice"] is None and item["subTotal"] is not None and item["quantity"]:
        item["unitPrice"] = int(round(item["subTotal"] / item["quantity"]))
    # subTotal이 없고 unitPrice * quantity로 생성
    if item["subTotal"] is None and item["unitPrice"] is not None:
        item["subTotal"] = int(item["unitPrice"] * item["quantity"])
    # 모두 None이면 품목 무효
    if item["unitPrice"] is None and item["subTotal"] is None:
        return None
    return item

def parse_items(lines):
    header_idx, stop_idx = detect_table_region(lines)
    if header_idx is None:
        # 헤더를 못찾으면 전체에서 시도하되, 총액/결제/합계 전까지만
        stop_idx = None
        for i, ln in enumerate(lines):
            if any(h in ln['text'] for h in TABLE_STOP_HINTS):
                stop_idx = i
                break
        rng = lines[:stop_idx] if stop_idx else lines
    else:
        rng = lines[header_idx+1:stop_idx]
    items = []
    for ln in rng:
        it = parse_item_line(ln)
        if it:
            items.append(it)
    # 중복/할인 제거(이름에 할인/쿠폰/증정 등은 품목 제외)
    items = [i for i in items if not re.search(r'(할인|쿠폰|증정|사은|포인트)', i['productName'])]
    return items

def pick_receipt_total(lines, items_sum):
    # 결제/합계 키워드가 있는 라인에서 가장 신뢰도 높은 금액
    candidates = []
    for ln in lines[::-1]:  # 하단부터 검색
        if any(k in ln['text'] for k in PAYMENT_PRIORITIES):
            val = amount_from_text(ln['text'])
            if isinstance(val, int):
                # 음수는 총액으로 보기 어려움
                if val >= 0:
                    priority = 0
                    # 결제금액 > 합계 > 총액 우선순위
                    if "결제금액" in ln['text'] or re.search(r'(결제\s*금액|받을금액)', ln['text']):
                        priority = 3
                    elif "합계" in ln['text']:
                        priority = 2
                    elif "총액" in ln['text'] or "총 금액" in ln['text']:
                        priority = 1
                    candidates.append((priority, val))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][1]
    # 후보 없으면 품목 합계 사용
    if items_sum and items_sum > 0:
        return items_sum
    # 마지막 시도로, 맨 아래 큰 숫자
    bottom_nums = []
    for ln in lines[-8:]:
        v = amount_from_text(ln['text'])
        if isinstance(v, int) and v >= 0:
            bottom_nums.append(v)
    if bottom_nums:
        return max(bottom_nums)
    return 0

# -----------------------------
# Main Task
# -----------------------------
@app.task
def process_receipt_task(session):
    session_dir = os.path.join(BASE_DIR, session)
    ensure_dir(session_dir)
    image_path = os.path.join(session_dir, "0.jpeg")
    result_path = os.path.join(session_dir, "result.json")

    # (선택) 간단한 전처리: 대비 향상 + 선명화
    img = read_image(image_path)
    if img is not None:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            sharp = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0,0), 1.0), -0.5, 0)
            tmp_path = os.path.join(session_dir, "0_pre_1.jpeg")
            cv2.imwrite(tmp_path, sharp)
            image_for_ocr = tmp_path
        except Exception:
            image_for_ocr = image_path
    else:
        image_for_ocr = image_path

    # OCR (ko + en 통합)
    ocr_items = read_with_paddle(image_for_ocr)
    # 라인 그룹핑
    lines = group_lines(ocr_items)

    # 메타 추출
    store_name = pick_store_name(lines) or ""
    address = pick_address(lines) or ""
    phone = pick_phone(lines) or ""
    timestamp = pick_datetime(lines) or ""
    payment = pick_payment_method(lines) or ""
    user_info = pick_user_info(lines) or ""

    # 품목/합계
    items = parse_items(lines)
    items_sum = sum(i['subTotal'] for i in items if isinstance(i.get('subTotal'), int))
    total = pick_receipt_total(lines, items_sum)

    # 결과 스키마 정리 (스펙에 맞춤)
    normalized_items = []
    for it in items:
        normalized_items.append({
            "productName": it.get("productName", ""),
            "quantity": int(it.get("quantity", 1)),
            "unitPrice": int(it.get("unitPrice") or 0),
            "subTotal": int(it.get("subTotal") or (it.get("unitPrice") or 0) * int(it.get("quantity", 1)))
        })

    data = {
        "storeName": store_name,
        "address": address,
        "phoneNumber": phone,
        "timestamp": timestamp,  # yyyy-MM-dd HH:mm:ss
        "paymentMethod": payment,
        "userInformation": user_info,
        "itemList": normalized_items,
        "receiptTotal": int(total or 0)
    }

    save_json(result_path, data)
    return "done"
