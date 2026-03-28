# HANDOFF PROTOCOL — Hand-Wash Detection CV Project
> **Autor:** David Ryšánek | **Datum předání:** 2026-03-24

---

## 1. Cíle Bakalářské Práce (Oficiální Zadání)

1. Přehled a analýza současných metod AI a CV ve výrobních procesech. ✅ (rešerše)
2. Navrhnout architekturu systému pro zpracování obrazových dat z kamery. ✅
3. Vyvinout prototyp systému pro analýzu dynamických scén. ✅ (3 detektory hotovy)
4. **Implementovat zpracování obrazových dat s využitím nástrojů AI.** ⚠️ **PRÁVĚ PROBÍHÁ** — přechod na YOLO26 Deep Learning klasifikátor.
5. Otestovat funkčnost, přesnost a praktickou využitelnost. ✅ (evaluační pipeline hotový)

---

## 2. Přehled Codebase

```
c:\Git\CV_Bc_project\
├── src/
│   ├── config.py                  # Centrální konfigurace (DetectionParams dataclass)
│   ├── soap_trigger_detector.py   # HLAVNÍ detektor (MediaPipe + pravidla + per-station tracking)
│   ├── baseline_motion.py         # Baseline detektor (MOG2 only)
│   ├── mediapipe_detector.py      # Starší MediaPipe-only detektor
│   ├── evaluate.py                # Clip-based evaluace (stará metoda, 20s klipy)
│   ├── evaluate_full.py           # FULL-VIDEO evaluace (nová, přesná metoda)
│   ├── annotate.py                # Anotační nástroj pro 20s klipy
│   ├── annotate_full.py           # Anotační nástroj pro celé video (POUŽÍVAT TENTO)
│   ├── debug_viewer.py            # Debug vizualizace detektoru na klipech
│   ├── generate_yolo_dataset.py   # Generátor YOLO datasetu z anotovaného videa
│   ├── pseudo_label_dataset.py    # Pseudo-labeling z neanotovaného videa (MÁ BUG, viz níže)
│   ├── roi_select.py              # Výběr ROI zón
│   ├── cut_clips.py               # Řezání surového videa na 20s klipy
│   ├── tune_params.py / tune_diagnose.py  # Grid search / diagnostika parametrů
│   └── batch_run.py               # Hromadné spouštění detektoru
├── datasets/
│   └── yolo_cls/                  # YOLO classification dataset
│       ├── train/washing/         # Trénovací data "mytí" (UŽIVATEL PRÁVĚ ČISTÍ)
│       ├── train/not_washing/     # Trénovací data "nemytí"
│       ├── val/washing/           # Validační data "mytí"
│       └── val/not_washing/       # Validační data "nemytí"
├── outputs/
│   ├── roi.json                   # ROI definice (soap_zones, sink_zones)
│   ├── full_video_gt.json         # Ground Truth pro tp00002 (29 eventů)
│   ├── annotations.json           # Starší clip-based anotace
│   ├── eval_full_video_summary.json  # Poslední evaluační výsledky
│   └── debug_clips/               # Extrahované debug klipy (FP/FN)
├── data_clips/2026-02-06/         # 60 surových 48min videí (tp00001-tp00060), každé 256 MB
├── thesis_goals.md                # Cíle BP
└── .venv/                         # Python 3.13 virtual environment
```

---

## 3. ROI Definice (`outputs/roi.json`)

Kamera je **overhead** (shora dolů), snímá 2 umyvadla s dávkovači mýdla.

| Zóna | Station | x | y | w | h |
|------|---------|---|---|---|---|
| soap_zone[0] | 0 | 1815 | 508 | 89 | 103 |
| soap_zone[1] | 1 | 1891 | 750 | 70 | 115 |
| sink_zone[0] | 0 | 1669 | 594 | 148 | 159 |
| sink_zone[1] | 1 | 1730 | 879 | 165 | 203 |
| Global ROI | — | 1445 | 509 | 504 | 778 |

---

## 4. Detekční Pipeline (soap_trigger_detector.py)

### Architektura (State Machine per-station):
```
IDLE → [hand in soap_zone] → PENDING → [instant confirm, soap_post_trigger_confirm_sec=0] → WASHING → [no motion OR no hand for 6s] → event END
```

### Klíčové parametry (config.py, aktuální hodnoty):
| Parametr | Hodnota | Popis |
|---|---|---|
| `wash_sec_on` | 5.0s | Min pohyb pro start |
| `wash_sec_off` | 2.0s | Klid pro ukončení |
| `soap_post_trigger_confirm_sec` | **0.0** | Post-trigger validace VYPNUTA (experimentálně ověřeno, že škodí recallu) |
| `post_trigger_min_sink_sec` | **0.0** | Kumulativní sink validace VYPNUTA |
| `soap_min_event_duration_sec` | 3.0s | Min délka eventu |
| `station_hand_timeout_sec` | 6.0s | Per-station timeout |
| `hand_detection_grace_sec` | 1.0s | Grace period pro MediaPipe |
| `merge_gap_sec` | 3.0s | Merge blízkých eventů |

### Důležitá modifikace (poslední session):
Řádek 330 v `soap_trigger_detector.py`: per-station `should_end` nyní kombinuje:
```python
should_end = (st.hand_missing_cnt >= hand_timeout_frames) or (st.still_cnt >= off_frames)
```
Dříve se per-station ukončoval JEN při ztrátě ruky. Teď se ukončí i při ztrátě pohybu (MOG2 motion), což zkracuje přetažené eventy (mirror-checking FP).

---

## 5. Evaluační Výsledky (Historie)

### Clip-based evaluace (stará, evaluate.py):
| Metoda | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (MOG2) | — | — | ~0.20 |
| MediaPipe základní | 0.60 | 0.63 | 0.61 |
| Soap Trigger (per-station) | 0.87 | 0.63 | 0.73 |

### Full-video evaluace (nová, evaluate_full.py na tp00002, 29 GT eventů):

| Konfigurace | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| Bez motion-end (základ) | 27 | 10 | 6 | 0.73 | 0.82 | 0.77 |
| Post-trigger 0.5s sink | 23 | 5 | 10 | 0.82 | 0.70 | 0.75 |
| Post-trigger 1.0s sink | 21 | 4 | 12 | 0.84 | 0.64 | 0.72 |
| Post-trigger 2.0s sink | 14 | 4 | 16 | 0.78 | 0.47 | 0.58 |
| Post-trigger 3.0s sink | 9 | 2 | 21 | 0.82 | 0.30 | 0.44 |
| **S motion-end (AKTUÁLNÍ)** | **29** | **9** | **5** | **0.76** | **0.85** | **0.81** |

> **Závěr:** Post-trigger validace drasticky snižuje recall (MediaPipe nevidí ruce v umyvadle kvůli pěně/vodě). Motion-end je lepší přístup. Heuristiky dosáhly stropu kolem F1 = 0.80.

---

## 6. Ground Truth Data

### tp00002 — PLNĚ ANOTOVÁNO (29 eventů)
- Soubor: `outputs/full_video_gt.json`
- Video: `data_clips/2026-02-06/20260127_193759_tp00002.mp4` (48 min)

### tp00003 — NENÍ ANOTOVÁNO ❌
- Video: `data_clips/2026-02-06/20260127_234623_tp00003.mp4`
- **Uživatel plánuje anotovat tento soubor jako další krok.**
- Příkaz: `.venv\Scripts\python.exe src/annotate_full.py data_clips/2026-02-06/20260127_234623_tp00003.mp4`

---

## 7. YOLO Dataset (Aktuální Stav)

- **Vygenerováno** z tp00002 pomocí `generate_yolo_dataset.py`
- Uloženo v `datasets/yolo_cls/{train,val}/{washing,not_washing}/`
- **Uživatel je právě manuálně čistí** (mazání chybně klasifikovaných obrázků)
- Dataset je VELMI MALÝ po čištění (uživatel musel smazat ~40% washing fotek, protože skript přiřadil washing i prázdnému umyvadlu na druhé stanici)
- **Je třeba přidat data z dalších videí (tp00003+) pro dostatečný objem.**

### Známý bug v `pseudo_label_dataset.py`:
```python
# Řádek 79: KeyError: 'station_id'
active_stations.add(ev["station_id"])  # ŠPATNĚ
# Mělo být:
active_stations.add(ev["station"])     # SPRÁVNĚ
```
Uživatel se nakonec rozhodl NEpoužívat pseudo-labeling, ale ručně anotovat tp00003 a pak generovat dataset standardně.

---

## 8. Nainstalované Závislosti

```
ultralytics    # YOLO (verze 8.4.26, podporuje YOLO26)
torch          # PyTorch 2.11.0
torchvision    # 0.26.0
mediapipe      # MediaPipe Hands
opencv-python  # OpenCV
scipy, numpy, matplotlib, polars
```
Python 3.13, venv v `.venv/`.

---

## 9. CO DĚLAT JAKO DALŠÍ KROK (Priority)

### Krok 1: Uživatel anotuje tp00003
```bash
.venv\Scripts\python.exe src/annotate_full.py data_clips/2026-02-06/20260127_234623_tp00003.mp4
```
Ovládání: Z/M = start/end, šipky = ±5s, PageUp/Down = ±1min, mezerník = pauza, S = uložit.

### Krok 2: Vygenerovat YOLO dataset z tp00003
Opravit `generate_yolo_dataset.py` tak, aby:
- Bral GT soubor jako parametr (ne hardcoded path)
- Přiřazoval `washing` label POUZE stanici, kde je skutečně pohyb (ne oběma)
- Přidával výstup do existující složky `datasets/yolo_cls/` (append, ne overwrite)

### Krok 3: Natrénovat YOLO26-cls
```python
from ultralytics import YOLO
model = YOLO("yolo11n-cls.pt")  # nebo yolo26n-cls.pt pokud dostupný
model.train(data="datasets/yolo_cls", epochs=50, imgsz=224, batch=16)
```

### Krok 4: Integrace + Finální evaluace
- Napojit YOLO klasifikátor do detekčního pipeline
- Temporal smoothing (0.5s rolling average na výstupní probability)
- Spustit `evaluate_full.py` a porovnat s heuristikou (F1=0.81)

---

## 10. Typické Problémy & Edge Cases

| Problém | Popis | Řešení |
|---|---|---|
| Mirror-checking FP | Člověk si kontroluje obličej u zrcadla, ruce projdou soap zonou | Motion-end timeout (aktuálně implementováno) |
| Foam occlusion FN | MediaPipe ztratí ruku při mydlení → event se rozpadne | YOLO by měl řešit (pěna = feature) |
| Passerby FP | Někdo projde kolem a letmo sahne na dávkovač | `soap_min_event_duration_sec=3.0` filtruje krátké eventy |
| Split detections | Jedno mytí = 2-3 krátké detekce | `merge_gap_sec=3.0` je sloučí |
| Dataset station mismatch | generate_yolo_dataset.py označí washing i prázdné umyvadlo | Opravit: používat per-station motion check |

---

## 11. Výzkumný Kontext (Pro text BP)

Progresivní metodologie:
1. **Baseline (MOG2)** — naive pixel tracking → F1 ~ 0.20
2. **Heuristika (MediaPipe + pravidla)** — pravidlový přístup → F1 = 0.81 (strop)
3. **Deep Learning (YOLO26-cls)** — neuronová síť → F1 = ? (cíl 0.90+)

Klíčový argument pro BP: *"Klasické CV metody (pravidla) dosahují stropu kolem F1=0.80 kvůli nespolehlivosti detekce rukou za přítomnosti pěny a vody. Konvoluční neuronová síť překonává tento strop tím, že se naučí rozpoznávat pěnu a vodu jako pozitivní prediktivní znaky mytí, nikoli jako překážku."*
