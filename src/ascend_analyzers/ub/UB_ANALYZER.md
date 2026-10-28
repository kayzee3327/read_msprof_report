# Triton / BiSheng UB 分析工具

Python 3.9+，仅依赖标准库，无需 NPU、CANN、matplotlib 或联网。HTML 报告在现代浏览器中离线使用。

```bash
python3 ub_analyzer.py memory_info_aiv.json -o ub_report
# 已知目标可用 UB 容量时才提供，单位为 byte：
python3 ub_analyzer.py memory_info_aiv.json -o ub_report --capacity-bytes 196608
python3 -m unittest -v test_ub_analyzer.py
```

输出 `ub_report/report.html`（交互曲线、任意 IR 时刻地址分布、buffer 生命周期与源位置）、`analysis.json`（完整可复查数据）、`timeline.csv`（占用保持不变的闭区间分段）。容量参数示例不代表自动判断了设备型号；不会默认把 910B、910C 或其他芯片容量混用。

Python 接口：`analyze_document(json.load(file), capacity_bytes=None)`，再调用 `write_outputs(report, directory)`。多个 UB Record 独立报告；不把不同 scope 或规划记录叠加。失败规划、无 offset、非法负数、反向生命周期、非字节对齐 extent 均报错并返回退出码 2，防止静默给出不完整结论。

## 本样本结果

| 指标 | 结果 |
|---|---:|
| dump 可见地址高水位 | **77,632 B = 75.8125 KiB** |
| 最大同时存活地址并集 | **39,040 B = 38.125 KiB，IR 95** |
| tmp 自身最大地址并集 | **4,096 B = 4 KiB，IR 95 和 114** |
| 原始条目 / 完全重复 / 去重条目 | 28 / 6 / 22 |
| 按相同地址、大小、生命周期和 tmp 属性合并的组 / 非空物理槽 | 18 / 23 |

IR 95 的存活占用可手算校验：4 个 8192 B 槽 + 4 个 512 B 槽 + 2 个 64 B 槽 + 1 个 4096 B tmp = 39,040 B。这里仍包含生命周期在 95 结束的 `%alloc_19`，同时包含在 95 产生的输出及 scratch。IR 95 的存活地址高水位为 77,024 B，与该时刻存活并集不是同一个指标。

全局最高地址来自 77,568 处的 64 B 存储条目，末端为 77,632。不要将 `max(offset + extent)` 当作字节数，也不要把所有 buffer 的 size 求和后称为 UB 峰值。

## 字段语义与源码证据

已实际阅读 Ascend 官方开源编译器，固定在 commit `cb4fc6a96e157953fe63457d7c5f832d086fc0c3`。输入没有编译器版本，所以无法保证与生成该 JSON 的二进制逐行一致；本样本的尺寸、别名、多缓冲重复和单时刻 tmp 均与源码行为相符。

1. **extent 单位是 bit，offset 单位是 byte。** [MemoryDisplay.cpp / generateMemoryInfo](https://github.com/Ascend/AscendNPU-IR/blob/cb4fc6a96e157953fe63457d7c5f832d086fc0c3/bishengir/lib/Dialect/HIVM/Transforms/MemoryDisplay.cpp#L133) 导出 `AlignUp(constBits, kBitsToByte)`，没有除 8；[PlanMemory.cpp / UpdateBuffer2Offsets](https://github.com/Ascend/AscendNPU-IR/blob/cb4fc6a96e157953fe63457d7c5f832d086fc0c3/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp#L1646) 把 bitsOffset 向上取整为 bytes。样本 `16x128xf32` 为 8192 B，对应 extent 65536；下一地址正好为 8192。
2. **offset 数组是多个实际槽，而非任选一个地址。** [ExpandMultiBufferStorageEntry](https://github.com/Ascend/AscendNPU-IR/blob/cb4fc6a96e157953fe63457d7c5f832d086fc0c3/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp#L1808) 建立额外存储条目，共享 bufferLifeVec；UpdateBuffer2Offsets 收集多个地址，特殊复用也可能增加槽。工具将每个不同地址展开为 `[offset, offset + extent/8)`。槽的精确轮转时刻没有导出，因此统计的是所报告生命周期内的规划占用。
3. **生命周期为闭区间。** [GetOverlapBufferLife](https://github.com/Ascend/AscendNPU-IR/blob/cb4fc6a96e157953fe63457d7c5f832d086fc0c3/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp#L2805) 用 `lo <= hi` 判断交集；边界相交可能经原地复用规则合法共享。不能直接按 `[start,end)` 统计；工具在 start 加入，在 end+1 移除。`[95,95]` tmp 覆盖 IR 95，绝不是空生命周期。
4. **重复与别名不能重复计费。** [collectMemoryInfoForDebug](https://github.com/Ascend/AscendNPU-IR/blob/cb4fc6a96e157953fe63457d7c5f832d086fc0c3/bishengir/lib/Dialect/HIVM/Transforms/MemoryDisplay.cpp#L156) 遍历 storage entry 的 inplaceBuffers；多槽展开也会重复导出相同 buffer。完全相同 JSON 行先去重；同布局同生命周期条目合并展示，但最终始终用地址并集计算，包括部分交叠。相同 `%alloc_3` 名称在不同 IR/布局中出现时保留为不同条目，不以 SSA 打印名全局去重。
5. **extent 优先于 memref 形状反推。** 导出大小来自存储条目，而打印的 buffer 可以是其别名。样本 `%alloc_36` 的 `16xbf16` 逻辑 payload 为 32 B，但 extent 是 512 bit，即 64 B，且与 `16x1xf32` 的 `%alloc_32` 共址同生命周期。工具保留 64 B，诊断形状差异，绝不擅自缩小为 32 B。
6. **不要再次用 alloc 的 alignment=64 逐项补齐。** JSON 已给出地址，部分地址并非 64 B 对齐。真正的规划对齐使用 scope 的 `alignedConstBits`，见 [PlanMemory.cpp](https://github.com/Ascend/AscendNPU-IR/blob/cb4fc6a96e157953fe63457d7c5f832d086fc0c3/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp#L1889)，但该字段不在 JSON 中。因此报告的 77,632 B 是可见地址高水位；无法仅凭 JSON 证明等于最终设备资源元数据中的预留量。中间洞可由地址观察，最后一块额外对齐尾部不可还原。

## 计算口径

对时刻 t，取所有满足 `start <= t <= end` 的槽的字节区间集合 A(t)：

- `live_bytes(t) = length(union(A(t)))`，峰值是该值在所有事件分段中的最大值。
- `live_high_water_bytes(t) = max(offset + size)`，空集合时为 0。
- 全局 `address_high_water_bytes = max(所有槽末端)`，以 UB 基址 0 为起点；包括低地址洞。
- `tmp_union_bytes` 为所有存活 tmp 区间的并集长度。
- `tmp_incremental_bytes = live_bytes - normal_bytes`。tmp 与普通 buffer 共址时，此值与 tmp 自身并集不同；两种数据都保留。
- `holes_below_live_high_water_bytes = live_high_water_bytes - live_bytes`。它只是该时刻高水位下未被存活数据覆盖的地址，不等同于一定能回收的碎片。
- `ever_covered_bytes` 是所有时刻地址区间的总并集，不能解释为同时存活量。

报告不把 IR 序号转换成微秒，不从单次 dump 推断循环次数、跨 core 总量或真实运行时峰值。完全相同布局合组是展示优化，不作为证明 SSA alias 身份的手段。额外交叠会列出诊断；原地复用可能合法，所以不武断报告内存冲突。

## 验证与规模

测试包含当前样本回归、闭区间边界、单时刻 tmp、多槽重复、别名、部分交叠、复用、形状不一致、独立 Record、异常输入与 HTML 转义；随机小样本用逐字节集合逐 IR 暴力计算核对事件扫描结果。

实现只遍历事件分段，不枚举巨大的 IR 编号区间，也不逐字节分配内存。当前实现每个事件重新对活跃区间排序，另有 O(N²) 的交叠诊断，适合常规 kernel 的数十至数千槽；极大 dump 可进一步换成坐标压缩线段树和扫描线交叠诊断。
