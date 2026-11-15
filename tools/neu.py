"""neu.py FILE: decodes a .neu program (the FlatBuffers blob in an .axmodel
that the runtime hands to /dev/npu) as the ax620e-npu driver reads it: its
command queues, OCM use, runtime values, tensors and parameters.

Field numbers are vtable slots; names are ours. Extract the blob from an
.axmodel with the onnx package: the initializer that npu_graph_info's
"neu_key" names.
"""
import struct
import sys


class Table:
    def __init__(self, buf, pos):
        self.buf, self.pos = buf, pos
        self.vt = pos - struct.unpack_from("<i", buf, pos)[0]
        self.vsize = struct.unpack_from("<H", buf, self.vt)[0]

    def _off(self, field):
        slot = 4 + 2 * field
        if slot + 2 > self.vsize:
            return 0
        return struct.unpack_from("<H", self.buf, self.vt + slot)[0]

    def has(self, field):
        return self._off(field) != 0

    def u8(self, field, default=0):
        off = self._off(field)
        return self.buf[self.pos + off] if off else default

    def u32(self, field, default=0):
        off = self._off(field)
        return struct.unpack_from("<I", self.buf, self.pos + off)[0] if off else default

    def u16(self, field, default=0):
        off = self._off(field)
        return struct.unpack_from("<H", self.buf, self.pos + off)[0] if off else default

    def _target(self, field):
        off = self._off(field)
        if not off:
            return None
        at = self.pos + off
        return at + struct.unpack_from("<I", self.buf, at)[0]

    def table(self, field):
        t = self._target(field)
        return Table(self.buf, t) if t is not None else None

    def string(self, field):
        t = self._target(field)
        if t is None:
            return None
        n = struct.unpack_from("<I", self.buf, t)[0]
        return self.buf[t + 4:t + 4 + n].decode(errors="replace")

    def vector(self, field):
        """(start of elements, count)"""
        t = self._target(field)
        if t is None:
            return None, 0
        return t + 4, struct.unpack_from("<I", self.buf, t)[0]

    def tables(self, field):
        start, n = self.vector(field)
        out = []
        for i in range(n):
            e = start + 4 * i
            out.append(Table(self.buf, e + struct.unpack_from("<I", self.buf, e)[0]))
        return out

    def bytes(self, field, width=1):
        start, n = self.vector(field)
        return self.buf[start:start + n * width] if start is not None else b""


def tensor(t, kind):
    return {"kind": kind, "name": t.string(0), "f1": t.u32(1), "f2": t.u32(2),
            "where": "OCM" if t.u8(3) else "DDR"}


def decode(buf):
    root = Table(buf, struct.unpack_from("<I", buf, 0)[0])
    model = {"vtable_size": root.vsize}
    soc = root.table(1)
    model["soc_type"] = soc.u8(0) if soc else None
    model["f3_u8"] = root.u8(3)
    model["name"] = root.string(8)
    model["control_mode"] = root.u8(9)

    # The command queue and its queues.
    mcode = root.table(5)
    words = mcode.bytes(2, 8) if mcode else b""
    model["mcode_bytes"] = len(words)
    model["queues"] = [{"id": q.u8(0), "f1": q.u32(1), "f2": q.u32(2)} for q in mcode.tables(1)] if mcode else []
    model["mcode_f0"] = mcode.u32(0) if mcode and mcode.has(0) else None

    # Runtime values: OCM use and the named slots.
    rtv = root.table(6)
    names = {}
    if rtv:
        model["ocm_usage"] = rtv.u32(0)
        model["ocm_reserved"] = (rtv.u32(1) + 0x1f) & ~0x1f
        for e in rtv.tables(2):
            names[e.string(0)] = e.u32(1)
    model["rtv"] = names
    model["ocm_base_rtv"] = names.get("_ocm_base", 0xffff)

    # IO, ring buffer and params.
    io = root.table(2)
    model["io"], model["params"], model["ringbuffer"] = [], [], None
    if io:
        for t in io.tables(0):
            model["io"].append(tensor(t, "input"))
        for t in io.tables(1):
            model["io"].append(tensor(t, "output"))
        rings = io.tables(2)
        if rings:
            model["ringbuffer"] = {"name": rings[0].string(0), "size": rings[0].u32(2),
                                   "f1": rings[0].u32(1)}
        for t in io.tables(3):
            p = tensor(t, "param")
            model["params"].append(p)
    for entry in model["io"] + model["params"]:
        entry["rtv"] = names.get(entry["name"]) if entry["where"] == "DDR" else None
    if model["ringbuffer"]:
        model["ringbuffer"]["rtv"] = names.get(model["ringbuffer"]["name"])

    # Param data, matched to params by name.
    data = {}
    for t in root.tables(4):
        data[t.string(1)] = len(t.bytes(0))
    model["param_data"] = data
    return model


if __name__ == "__main__":
    import json
    blob = open(sys.argv[1], "rb").read()
    print(json.dumps(decode(blob), indent=1))
