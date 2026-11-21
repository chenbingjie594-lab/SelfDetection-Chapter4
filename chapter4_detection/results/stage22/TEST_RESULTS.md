# Stage22 test results

- Command: `PYTHONPATH=. /mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -m pytest -q chapter4_detection/tests/test_stage22_cir.py`
- Result: **3 passed in 4.79 s**
- Syntax check: `python -m py_compile chapter4_detection/models/cir.py chapter4_detection/scripts/stage22_*.py`
- Result: **passed**

The tests cover aligned crop geometry, all four CIR forward/backward paths, zero-initialized identity behavior, residual box decoding, and checkpoint save/reload on CPU.
