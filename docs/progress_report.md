# Progress report

License plate detection and recognition on CCPD2019.
Status as of 17 October 2026, for the 20 October "second approach" milestone.

## Milestone status

The second approach is working: a plate-specific CRNN/CTC recognizer, trained
deterministically on the 3,500 original-scale training crops, now outperforms
the pretrained EasyOCR baseline under both recognition protocols. Detection
remains the strong stage; recognition was the weak spot and is now substantially
improved.

| Stage | Model | Status |
|---|---|---|
| Dataset | CCPD 5,000-image subset, 15,000 scale-variant images, leak-free splits | complete |
| Detection (E1 baseline) | fine-tuned YOLO11s, frozen | complete |
| Recognition baseline (E2) | pretrained EasyOCR, frozen | complete |
| Second approach (E2/E2b) | CRNN + BiLSTM + CTC, trained on CCPD chars | complete |
| Preprocessing ablation (E3) | none / gray+contrast / Canny | complete |
| Size and tilt sensitivity (E4/E5) | per-bucket and per-bin metrics | complete |
| Confidence threshold (E6) | accuracy-coverage sweep, validation-selected tau | complete |
| Detector fine-tuning comparison (E1) | pretrained vs fine-tuned YOLO11 | remaining |

## Test results

### Detection (frozen YOLO11s)

| Split | mAP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Validation | 0.9245 | 0.9402 | 0.8986 | 0.9189 |
| Test | 0.9073 | 0.9408 | 0.8938 | 0.9167 |

### Recognition (EasyOCR vs CRNN)

Both engines are evaluated through the same harness: mean per-character
confidence, tau selected on validation by accepted exact accuracy + coverage
- 1 (both select mode `none`, tau 0.50), test reported once per protocol.

| Protocol | Engine | Coverage | Character accuracy | Exact plates | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ocr_only | EasyOCR | 0.7300 | 0.3474 | 0 / 800 | 0.0377 | 0.0924 | 0.0535 |
| ocr_only | CRNN | 1.0000 | 0.7538 | 269 / 800 | 0.5500 | 1.0000 | 0.7097 |
| end_to_end | EasyOCR | 0.7025 | 0.3240 | 0 / 800 | 0.0267 | 0.0593 | 0.0368 |
| end_to_end | CRNN | 0.9762 | 0.6561 | 133 / 800 | 0.3624 | 0.9371 | 0.5226 |

Per-bucket and per-tilt-bin CRNN numbers, and the validation ablation, are in
`models/crnn/REPORT.md`.

## Observations

- CRNN lifts test character accuracy to 0.7538 (ocr_only) and 0.6561
  (end_to_end), with 269 and 133 exact plates where EasyOCR read zero.
- The end-to-end gap to OCR-only is the detector floor plus box geometry.
- Large plates degrade sharply (ocr_only 0.5722 vs 0.8635 for small) because
  the fixed 48-px input downsamples them.
- The 10-19 degree tilt bin is the weakest recognition bin.
- Canny preprocessing collapses for the trained model; the selection
  correctly picks `none` for both protocols.

## Remaining work

- E1: fine-tune YOLO11 on the train split and compare against the frozen
  detector; re-run end-to-end recognition with the better detector.
- Large-plate input geometry (taller CRNN input) as a recognition improvement.
- Final submission: failure-case analysis, latency/memory logging, clean-clone
  reproducibility check, two-page summary and demo (10 November).
