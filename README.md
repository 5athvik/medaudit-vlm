# MedAudit-VLM: Clinical Vision-Language Pipeline & NIST AI RMF Assurance Suite

An enterprise-ready multimodal pipeline designed to extract structured clinical entities from prescription and document scans, featuring a custom PyTorch autograd loss with clinical risk penalties and a NIST AI RMF statistical assurance testing harness.

---

## Technical Highlights

- **PyTorch Autograd & Custom Loss:** Implemented `RiskWeightedClinicalLoss` to penalize dosage and metric unit mispredictions up to 3.5x higher than general text errors.
- **Parameter-Efficient Fine-Tuning (PEFT):** LoRA target projections on transformer attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with gradient accumulation and mixed-precision execution.
- **Statistical Risk Assurance:** Automated evaluation framework based on the NIST AI RMF and ICFR principles, computing Wilson score binomial confidence intervals to verify strict error thresholds.
- **Production API:** Asynchronous FastAPI service with Pydantic schema validation, real-time latency logging, and device health monitoring.

---

## Project Structure

```text
medaudit-vlm/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI service with /audit/extract & /health
├── audit/
│   ├── __init__.py
│   └── eval_nist.py         # NIST AI RMF & ICFR statistical error bound tests
├── data/
│   ├── generate_mock_data.py # Synthetic clinical document generator
│   └── raw/                 # Generated test samples
├── model/
│   ├── __init__.py
│   ├── custom_loss.py       # Custom PyTorch autograd loss
│   └── test_loss.py         # Loss gradient verification unit test
├── train_vlm.py             # PEFT/LoRA training configuration
├── requirements.txt
└── README.md