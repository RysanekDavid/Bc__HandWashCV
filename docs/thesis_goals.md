# Cíle bakalářské práce

**Název tématu:**
- **CS:** Využití metod umělé inteligence a počítačového vidění ve výrobním prostředí
- **EN:** The use of artificial intelligence and computer vision methods in manufacturing environments
- **Autor:** David Ryšánek

## Oficiální zadání (Cíle práce)

1. **Provést přehled a analýzu současných metod umělé inteligence a počítačového vidění využitelných ve výrobních procesech.**
   - *Status:* Teoretická část (rešerše v textu BP).

2. **Navrhnout architekturu systému pro zpracování obrazových dat z kamerového zařízení v reálném výrobním prostředí.**
   - *Status:* Hotovo (využíváme ROI zóny pre-processingu atd.).

3. **Vyvinout prototyp systému využívající metody počítačového vidění pro analýzu dynamických scén ve výrobním prostoru.**
   - *Status:* Hotovo (MOG2 baseline + MediaPipe heuristika v `soap_trigger_detector.py`). Poslouží jako referenční porovnání.

4. **Implementovat zpracování obrazových dat v reálném čase s využitím knihoven a nástrojů umělé inteligence.**
   - *Status:* **PROBÍHÁ.** Generujeme YOLO Deep Learning model (vylaďujeme image classification na výřezech umyvadel).

5. **Otestovat funkčnost, přesnost a praktickou využitelnost navrženého systému v reálném výrobním prostředí.**
   - *Status:* Máme nachystaný skript `evaluate_full.py`, kterým nakonec porovnáme Pravidla (F1=0.80) vs UI / YOLO (F1=?).
