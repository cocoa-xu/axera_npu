# Changelog

## v0.1.0 (2026-11-21)

* The `ax620e-npu` driver: `/dev/npu` with the vendor's interface (models,
  tasks, waiting for them), memory from the device tree's pool, uncached or
  cached with the caches kept in step around each task, and a reset of the
  NPU after a task fails
* An open runtime for `.axmodel` files whose graph is one NPU program: its
  results are byte for byte the vendor's on MobileNetV2, YOLOv5s and PP-OCR
* `AxeraNPU`: `load/1`, `load_binary/1`, `describe/1` (without the NPU) and
  `run/2`, with binaries or Nx tensors
* `AxeraNPU.OCR`: PP-OCR text detection and recognition; lines wider than the
  recognizer's input are read in pieces, cut between letters
* A Buildroot package for the driver; the `axrun` and `ocr` tools, and
  `neu.py` to decode NPU programs
