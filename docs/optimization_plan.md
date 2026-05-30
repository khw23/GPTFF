# GPTFF 兼容性与性能优化计划

本文档记录本分支的优化边界、优先级和验证方式。当前分支名为
`perf/gptff-inference-training-optimization`。

## 目标

1. 先保证新版 Python、PyTorch 和 ASE 下的基础推理可用。
2. 保持 `pretrained/gptff_v1.pth` 与 `pretrained/gptff_v2.pth` 直接可加载。
   - V1：不启用 transformer。
   - V2：启用 transformer。
3. 修复单原子、少原子体系的 ASE calculator 失败问题。
4. 逐步减少旧 API warning，例如 ASE 的 `set_calculator` 用法和新版 PyTorch 的 `torch.load` 行为变化。
5. 后续再做推理速度、训练速度、内存占用和 batched workflow 的优化。

## 环境策略

本项目的 DGX Spark 宿主机 Python 缺少 `Python.h` 开发头文件，直接用 host venv
编译 GPTFF 的 Cython 扩展会失败。因此当前开发验证优先使用 Docker：

- 旧版对照环境：项目已有镜像 `mlip-mlff-suite:pt25.09-cuda13.0.1-arm64`，
  其中 ASE 固定为 `3.26.0`。
- 新版开发环境：基于同一 CUDA/PyTorch 镜像，将 ASE 升级到当前稳定版
  `3.28.0` 后安装本分支。

宿主机如果需要直接使用 venv/conda，应确保 Python 开发头文件可用，或使用
conda/mamba 创建环境，因为 conda Python 通常自带匹配的 headers。

## 优先级

### P0：正确性与兼容性

- `ASECalculator` 能在 ASE `3.26.0` 和 `3.28.0` 下运行。
- Python packaging 当前声明 `ase>=3.26,<3.29`，覆盖已验证的旧版对照和当前最新版，
  不默认放开到未测试的未来 ASE 主版本。
- V1/V2 checkpoint 能在 CPU 上加载并完成 smoke test。
- 单原子体系和两原子体系返回有限能量与正确形状的力。
- README 示例改为 `atoms.calc = calc`，避免旧 ASE calculator API 用法。
- `torch.load` 显式使用 `weights_only=False`，避免新版 PyTorch 默认行为变化。

### P1：轻量性能优化

- 推理模式冻结参数 `requires_grad=False`，减少 autograd 管理开销。
- 将容易造成单样本维度坍缩的裸 `squeeze()` 改为保形状写法。
- 后续可进一步区分 force-only 和 stress 计算路径：MD 通常只需要力，不一定需要
  每步计算应力。
- 针对 `predict_energies_batched` 做更系统的 batch 大小和内存测试。

### P2：训练与框架扩展

- 更新训练脚本中的 AMP API，兼容新版 PyTorch。
- 缓存或预处理图构建结果，降低 DataLoader worker 压力。
- 对 batch size、workers、GPU 利用率、CPU 数据准备时间做可复现实验。
- TorchSim 接入放到后续阶段；当前先优化 GPTFF 自身。

## 版权与借鉴边界

可以参考 MACE、CHGNet、MatterSim 等模型暴露出的通用工程问题，例如：

- force-only 推理应避免不必要的 stress autograd；
- batch 推理要避免 Python 循环成为瓶颈；
- 数据加载和邻居表构建需要独立计时。

但本分支不复制这些项目的实现代码。优化应来自 GPTFF 自身 profiling、接口整理和
PyTorch/ASE 通用 API 使用。

## 验证命令

在旧版 ASE 环境中：

```bash
docker run --rm \
  -v /home/khw/projects/ml/external/gptff-khw23:/workspace/GPTFF \
  -w /workspace/GPTFF \
  mlip-mlff-suite:pt25.09-cuda13.0.1-arm64 \
  bash -lc 'python -m pip install --no-build-isolation --no-deps -e . && pytest -q'
```

在新版 ASE 环境中：先构建包含 `ase==3.28.0` 的开发镜像，再运行同样的测试。
