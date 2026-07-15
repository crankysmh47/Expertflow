# Exhaustive Q4 expert layout

The pinned `gguf-py` reader enumerated the real model without loading weight arrays. The resulting artifact contains every expert object: 30 layers × 128 experts = 3,840 objects, with three exact source spans per object.

- Inventory: `C:\models\expertflow\runs\expert-layout-q4\inventory.json`
- Bytes: 6,803,857
- SHA-256: `daf9a54c1d03a933a667644de412038fb1530ee90ef1761f2f74dbdacb5f1b7a`
- Model SHA-256: `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- Result: all 3,840 encoded objects are exactly 3,345,412 bytes; all 3,840 projected CUDA slots are exactly 3,346,048 bytes

## Per-object reconciliation

| Component | Encoded bytes | CUDA row-end padding | Projected allocation |
| --- | ---: | ---: | ---: |
| Down Q4_0 weight | 1,115,136 | 180 | 1,115,316 |
| Fused gate/up Q4_0 weight | 2,230,272 | 144 | 2,230,416 |
| Down F32 scale | 4 | 0 | 4 |
| **Encoded total** | **3,345,412** | **324** | — |

The pinned CUDA backend uses `MATRIX_ROW_PADDING=512` elements for quantized matrix tails and 128-byte tensor starts. Packing the three projected tensors in down, gate/up, scale order adds 76 bytes before gate/up, 112 bytes before the scale, and 124 bytes at the slot end. Row padding plus alignment is 636 bytes, giving the 3,346,048-byte slot.

For each object, the inventory records layer ID, expert ID, tensor name/type/shape, exact GGUF source start/end, encoded bytes, CUDA row padding, projected component allocation, slot start, and alignment padding. The verifier checked that every component span remains within its source tensor and that every layer has the same three components and 128 experts.

## Static-96 physical consistency

The independent projection matches the capacity curve exactly:

- Target layers: 21 (`0–20`)
- Slots per layer: 96
- Total slots: 2,016
- Slot bytes: 3,346,048
- Projected cache: 6,745,632,768 bytes / 6,433.136719 MiB

This is a source-derived allocation projection. No live llama.cpp slot arena exists yet, so it is not described as measured VRAM consumption.
