from __future__ import annotations

from typing import List

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from typing_extensions import NotRequired


class SampleDict(TypedDict):
    src: List[int]
    tgt: List[int]
    src_sent_labels: List[int]
    segs: List[int]
    clss: List[int]
    src_txt: List[str]
    tgt_txt: List[str]
    mask_src: NotRequired[List[int]]
    mask_cls: NotRequired[List[int]]
    mask_tgt: NotRequired[List[int]]
