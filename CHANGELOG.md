# Changelog

## v0.2.0 (2026-11-28)

* `models/ppocrv5`: compiling PP-OCRv5 for `AxeraNPU.OCR` with Pulsar2 4.2,
  text detection at 640 x 640 and recognition, both taking u8 RGB: a screen
  page of six lines in about half a second on the NanoKVM-Go+
* The driver loads programs and submits tasks with 64-bit user pointers
  (`NPU_PROGRAM_LOAD`, `NPU_TASK_SUBMIT`), which the vendor's interface lacks,
  and serves 32-bit processes on a 64-bit kernel too. The runtime uses them,
  so it runs in 64-bit processes and needs this driver
* The driver reads a task's id before another thread may have run and freed
  the task

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
