## Demo of an InstanceNorm class using the NVFuser Python frontend

The pull request [#1309 on APEX: NVFuser JIT eager mode
InstanceNorm3d](https://github.com/NVIDIA/apex/pull/1309#diff-157f61b2609318940a485fff536ff926ba51949087a24c26fbb242b151ebbbc5)
implements InstanceNorm using NVFuser's C++ interface. In this repository, I'll
demonstrate a pure Python library that does not require a
[CUDAExtension](https://pytorch.org/tutorials/advanced/cpp_extension.html) and
achieves the same functionality.
