# Počítačové vidění ve výrobním prostředí – Detekce mytí rukou

Tento repozitář obsahuje zdrojové kódy a dokumentaci k bakalářské práci na téma **Využití metod umělé inteligence a počítačového vidění ve výrobním prostředí** (The use of artificial intelligence and computer vision methods in manufacturing environments).

**Autor:** David Ryšánek

---

## 🎯 Propojení projektu s cíli bakalářské práce

Tento projekt představuje praktickou implementaci zadání bakalářské práce a zaměřuje se na nasazení pokročilých algoritmů v průmyslové praxi. Klíčovým scénářem (use-case) je **automatizace kontroly hygienických procesů (mytí rukou)**, která slouží jako demonstrace možností moderního počítačového vidění při zvyšování standardů ve výrobním prostředí.

Následující přehled mapuje technickou realizaci projektu na jednotlivé oficiální cíle práce:

### 1. Přehled a analýza současných metod (Teoretická část)

- **Zadání:** Provést přehled a analýzu současných metod AI a CV využitelných ve výrobě.
- **Stav projektu:** Tato část je řešena rešerší v textu práce. Zahrnuje porovnání tradičních metod počítačového vidění (Background Subtraction, MOG2, zpracování regionů zájmu - ROI) s moderními AI přístupy (detekce klíčových bodů s MediaPipe, modely rodiny YOLO).

### 2. Návrh architektury systému (Metodika)

- **Zadání:** Navrhnout architekturu systému pro zpracování obrazových dat z kamerového zařízení v reálném výrobním prostředí.
- **Stav projektu:** Architektura je navržena jako pipeline:
  1.  Kamerový záznam (pohled na umyvadla)
  2.  Předzpracování (ořez a definice ROI - Region of Interest pres `roi_select.py`)
  3.  Modul detekce (zpracování framů a identifikace pohybu/rukou)
  4.  Počítání průchodů osob přes exit zónu (YOLO person detector)
  5.  Korelační modul (propojení wash událostí s průchody → compliance)
  6.  Vyhodnocovací a reportovací modul (statistiky, časové záznamy událostí)

### 3. Prototyp systému pro analýzu dynamických scén (Implementace)

- **Zadání:** Vyvinout prototyp systému využívající metody CV pro analýzu dynamických scén.
- **Stav projektu:** Projekt aktuálně obsahuje:
  - **Nástroj pro přípravu a extrakci dat (FFmpeg)**
  - **Anotační nástroj (`annotate.py`)** pro tvorbu Ground Truth dat (ruční fixace časových oken, kdy probíhá mytí).
  - **Baseline model (`baseline_motion.py`)**, který využívá Background Subtraction (MOG2) a analýzu obrysů (contours) v nastaveném ROI. Tento model tvoří spolehlivý základ díky stabilnímu osvětlení ve sledované zóně.
  - **AI model (`mediapipe_detector.py`)**, který kombinuje MOG2 detekci pohybu s detekcí rukou pomocí Google MediaPipe Hands. Událost je zaznamenána pouze pokud je v ROI detekován pohyb **A ZÁROVEŇ** jsou přítomny ruce — tím se eliminují falešné poplachy.
  - **Soap Trigger model (`soap_trigger_detector.py`)** — nejúspěšnější přístup. Využívá definované trigger zóny kolem dávkovačů mýdla. Událost začíná, když MediaPipe detekuje ruku v kontaktu s dávkovačem, a končí, když pohyb v ROI klesne. Kombinace domain-specific znalosti s AI detekcí.

### 4. Zpracování v reálném čase pomocí AI knihoven (Vylepšená AI)

- **Zadání:** Implementovat zpracování obrazových dat v reálném čase s využitím knihoven a nástrojů umělé inteligence.
- **Stav projektu:** V rámci extenze baseline modelu byla začleněna pokročilá AI vrstva Google MediaPipe Hands ve dvou variantách:
  1. **MediaPipe Hands** — detekce rukou v celém ROI jako filtr falešných poplachů.
  2. **Soap Trigger** — detekce rukou specificky u dávkovačů mýdla jako přesný trigger začátku události. Aktuálně nejlepší přístup (F1=0.73, Precision=0.87, Recall=0.63).

### 5. Testování a vyhodnocení (Hodnocení kvality)

- **Zadání:** Otestovat funkčnost, přesnost a praktickou využitelnost navrženého systému.
- **Stav projektu:** Součástí projektu je metodika evaluace:
  1.  Tvorba Ground Truth datasetu uživatelem pomocí `annotate.py` (cíl: ~100-150 anotovaných klipů s různými směnami).
  2.  Spuštění skriptu `batch_run.py`, který na těchto klipech spustí navržené modely (Baseline, a v budoucnu AI model).
  3.  Automatizované zpracování metrik (Precision, Recall, F1-Score) porovnáním detekcí s ručními anotacemi. Tyto výsledky budou tvořit klíčovou argumentační část v kapitole "Testování a vyhodnocení" samotné textové práce.

---

## 🛠 Struktura projektu

- `data_raw/` - Složka pro původní, dlouhé hrubé videozáznamy z provozu.
- `data_clips/` - Složka pro vystřihnuté krátké klipy (~20 vteřin) využívané pro anotaci a trénování/testování. Klipy se získávají pomocí FFmpeg.
- `models/` - Stažené AI modely (MediaPipe HandLandmarker apod.).
- `notebooks/` - Jupyter notebooky pro analýzu a porovnání výsledků.
  - `Data_Exploration.ipynb` - Explorace datasetu (distribuce, pokrytí, statistiky GT).
  - `Detector_Comparison.ipynb` - Porovnání detektorů (grafy, tabulky, F1/IoU analýza).
- `outputs/` - Složka obsahující výstupy práce:
  - `roi.json` - Definice regionu zájmu + soap zones.
  - `annotations.json` - Ground Truth anotace vytvořené uživatelem.
  - `eval_*.csv` / `eval_*_summary.json` - Výsledky evaluace detektorů.
- `src/` - Zdrojové kódy:
  - `config.py` - Globální definice cest a proměnných.
  - `roi_select.py` - Nástroj pro manuální výběr ROI a soap zones (`--soap-zone`).
  - `annotate.py` - Interaktivní anotační nástroj pro tvorbu Ground Truth dat.
  - `baseline_motion.py` - Detektor #1: MOG2 Background Subtraction + hysteresis.
  - `mediapipe_detector.py` - Detektor #2: MOG2 + MediaPipe Hands (Tasks API).
  - `soap_trigger_detector.py` - Detektor #3 (nejlepší): soap zone trigger + motion tracking.
  - `evaluate.py` - Evaluační skript s metrikami. Podporuje `--detector baseline|mediapipe|soap_trigger`.
  - `batch_run.py` - Dávkové spouštění detekce na všech klipech.
  - `debug_viewer.py` - Vizuální diagnostický nástroj pro ladění parametrů detektorů.
  - `tune_diagnose.py` - Diagnostický nástroj pro detailní analýzu state-machine přechodů.

---

## 🚀 Jak systém používat

### 1. Definice oblasti (ROI)

Nejprve je nutné definovat na obrazu místo, kde se nachází umyvadlo.

```bash
python src/roi_select.py
```

_(Nakreslete obdélník, potvrďte klávesou Enter. Výsledek se uloží do `outputs/roi.json`)_

### 2. Anotace videí (Ground Truth)

Pro získání měřitelných výsledků je nutné vytvořit testovací sadu dat (Ground truth). Nástroj postupně otevře nahrané klipy a umožní vyznačit časové úseky.

```bash
python src/annotate.py --skip-annotated
```

- **S** - Start události (začátek mytí)
- **E** - Konec události
- **Q** - Uložit klip a přejít na další
- **SPACE** - Pauza / Play

### 3. Vyhodnocení (Batch Run)

Po ověření a označení klipů je nutné spustit modely napříč celou sadou pro vyhodnocení.

```bash
python src/batch_run.py
```

Tento příkaz aplikuje nastavené detekční mechanismy na všechny klipy.

### 4. Evaluace detektorů

Porovnat výkon Baseline vs. AI modelu:

```bash
python src/evaluate.py --detector baseline
python src/evaluate.py --detector mediapipe
python src/evaluate.py --detector soap_trigger
```

Výsledky se uloží do `outputs/eval_<detektor>.csv` a `eval_<detektor>_summary.json`.

**Aktuální výsledky evaluace (128 hodnocených klipů, 43 GT událostí):**

| Detektor | Precision | Recall | F1-Score | Mean IoU | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Baseline (MOG2) | 0.5333 | 0.7442 | 0.6214 | 0.3940 | 32 | 28 | 11 |
| MediaPipe Hands | 0.5091 | 0.6512 | 0.5714 | 0.4009 | 28 | 27 | 15 |
| **Soap Trigger (optimized)** | **0.8710** | **0.6279** | **0.7297** | **0.6614** | **27** | **4** | **16** |

**Evoluce Soap Trigger detektoru:**

| Iterace | Precision | Recall | F1 | Změna |
|---|---|---|---|---|
| Soap Trigger (base) | 0.846 | 0.512 | 0.638 | — |
| + Temporal smoothing & Event merging | **0.871** | **0.628** | **0.730** | F1 +14% |

> ℹ️ **Poznámka:** Recall 63 % znamená, že systém zachytí přibližně 2 ze 3 skutečných mytí rukou. Pro produkční nasazení compliance monitoringu by bylo nutné dosáhnout Recall ≥ 85 %, což vyžaduje buď rozšíření datasetu, nebo přechod na end-to-end deep learning přístup (např. action recognition). Detailní analýza chyb a ablační studie jsou v [`Detector_Comparison.ipynb`](notebooks/Detector_Comparison.ipynb).

### 5. Debug & Ladění parametrů

Pro vizuální diagnostiku a ladění parametrů detektorů slouží `debug_viewer.py`:

```bash
python src/debug_viewer.py data_clips/2026-02-06/tp00000.pic outputs/roi.json --detector soap_trigger
```

**Overlays:** ROI box (zelený/červený dle stavu), soap zones (žlutá/cyan), hand landmarks (magenta), motion heatmap.

**Klávesové ovládání:**
- **SPACE** - Pauza / Play
- **D / A** - Krok vpřed / vzad
- **M** - Zapnout/vypnout motion heatmap
- **H** - Zapnout/vypnout hand landmarks
- **G** - Zapnout/vypnout ground truth timeline
- **+/-** - Zvýšit/snížit motion threshold
- **S** - Uložit aktuální frame
- **Q** - Ukončit

### 6. Požadavky

```bash
pip install -r requirements.txt
```

---

## 📊 Compliance statistiky pro výrobní provoz

Systém kromě samotné detekce mytí rukou poskytuje firmě prakticky využitelné statistiky pro kontrolu hygieny zaměstnanců.

### Varianta A: Anonymní compliance monitoring (implementováno)

Systém nepotřebuje identifikovat konkrétní osoby — pracuje zcela anonymně:

1. **Exit zóna** — v obraze je definována průchozí oblast, kudy zaměstnanci odchází z hygienické zóny do výroby.
2. **Person detector (YOLO)** — počítá průchody osob přes exit zónu. Směr pohybu (z místnosti do výroby) je rozpoznán na základě trajektorie.
3. **Korelace** — pokud v posledních N sekundách před průchodem proběhla wash událost → **compliant** (umyl si ruce). Jinak → **non-compliant**.

**Dostupné statistiky:**

| Statistika               | Popis                               | Přínos pro firmu                       |
| ------------------------ | ----------------------------------- | -------------------------------------- |
| **Compliance rate**      | % průchodů s předchozím mytím rukou | Klíčový KPI hygieny provozu            |
| **Compliance per směna** | Rozpad po ranní / odpolední / noční | Identifikace problémových směn         |
| **Compliance v čase**    | Hodinový breakdown přes den         | Kdy se nejvíc/nejméně dodržuje hygiena |
| **Průměrná délka mytí**  | Sekundy na jedno mytí               | WHO doporučuje ≥20s — splňují to?      |
| **Trend v čase**         | Denní/týdenní compliance graf       | Měření efektu školení a opatření       |
| **Peak traffic**         | Počet průchodů za hodinu            | Plánování kapacity umyvadel            |
| **Wash-to-exit time**    | Čas mezi mytím a odchodem do výroby | Kontrola přímočarosti procesu          |

**Výhody:** Plná kompatibilita s GDPR — žádné rozpoznávání obličejů, žádná identifikace osob, pouze anonymní agregované počty.

### Varianta B: Pseudonymní tracking (rozšíření)

> ℹ️ **Poznámka:** Tato varianta není součástí bakalářské práce, ale architektura systému je navržena tak, aby ji bylo možné snadno doplnit jako rozšíření.

> ℹ️ **Právní požadavky:** Pseudonymní tracking pracuje s osobními údaji a vyžaduje stanovení zákonného titulu podle [čl. 6 GDPR](https://www.privacy-regulation.eu/cs/6.htm) (např. oprávněný zájem či souhlas). Před nasazením je nezbytná konzultace s DPO a doporučeno provedení posouzení vlivu na ochranu osobních údajů (DPIA).


Pro firmy, které požadují sledování na úrovni jednotlivců (např. v regulovaných odvětvích jako farmacie nebo potravinářství), je možné systém rozšířit o:

1. **Multi-Object Tracking (MOT)** — přiřazení pseudonymního ID (číslo, ne jméno) každé detekované osobě pomocí algoritmů jako ByteTrack nebo DeepSORT na bázi YOLO detekcí.
2. **Trajektorie a směr pohybu** — sledování celé cesty osoby přes záběr kamery. Rozlišení směru: příchod k umyvadlu vs. odchod do výroby.
3. **Per-ID korelace** — propojení konkrétního track ID s wash událostí: „Osoba #7 si umyla ruce v 14:32:15, délka mytí 22s, odchod do výroby v 14:32:45."
4. **Denní ID report** — anonymizovaný, ale konzistentní: „Dnes prošlo 45 unikátních osob, 38 si umylo ruce (84.4% compliance)."

**Technické požadavky pro Variantu B:**

- `ultralytics` (YOLOv8/v11) — detekce osob
- `supervision` nebo vlastní implementace ByteTrack — multi-object tracking
- Re-identifikační model (volitelně) — pro konzistentní ID napříč kamerami


