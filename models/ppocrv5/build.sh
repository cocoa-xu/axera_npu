#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Inside pulsar2:4.2: the detection model at 640x640 (U8 activations) and the recognizer, with u8 RGB NHWC inputs.
set -e
cd /data
python3 - <<PY
import onnx
m = onnx.load("onnx/det_mobile_sim_static.onnx")
for t in (m.graph.input[0], m.graph.output[0]):
    d = t.type.tensor_type.shape.dim
    d[2].dim_value = 640
    d[3].dim_value = 640
del m.graph.value_info[:]
m = onnx.shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save(m, "onnx/det_640.onnx")
print("det_640.onnx:", [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])
PY
for c in det640_u8 rec_u16; do
	pulsar2 build --config $c.json > out/$c.log 2>&1 && echo "$c done" || echo "$c FAILED"
done
