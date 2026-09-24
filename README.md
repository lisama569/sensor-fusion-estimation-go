# Sensor Fusion Estimation (Go)

一套**本地运行**的运动平台状态估计流程：用本地确定性生成的测量数据模拟多路
传感器，在同一套流程里完成线性与非线性测量的（扩展）卡尔曼融合，并对多采样率、
延迟、时间戳乱序/重复、病态噪声配置以及单路传感器故障给出**可区分、可解释**的
处理结果。

无外部服务、无网络依赖；`go test` 与 `go run ./cmd/demo` 即可在本机复现。

## 快速开始

```bash
# 端到端演示：多速率 + 延迟 + 乱序 + polar 路发散故障 + 自动恢复
go run ./cmd/demo

# 全部单元/集成测试
go test ./...

# 竞态检测（并发提交/推进，题目要求的验收方式）
go test -race ./...

# 静态检查
go vet ./...
```

演示会打印：最终估计 vs 真值、误差、协方差特征值与不对称度、每路传感器的
接受/拒绝/隔离/恢复统计，以及完整的判定轨迹（每个事件的时间、传感器、原因、NIS）。

## 支持的模型

| 类别 | 模型 | 代码 |
| --- | --- | --- |
| 状态模型 | 4 维 2D 恒速 `[x, vx, y, vy]`，白噪声加速度过程噪声（谱密度 `q`） | `internal/models` `NewConstantVelocity2D` |
| 线性测量 | `z = H x + b + v`，如 GPS 位置 `[x,y]`、速度 `[vx,vy]` | `models.LinearModel` |
| 非线性测量 | `z = h(x) + b + v`，相对信标的距离-方位角 `[r, θ]`，方位角按 `[-π,π)` 归一化 | `models.NonlinearModel`（`AngleRows`） |

线性与非线性测量在**同一条**更新流程中处理（同一 `Fuser`、同一 `Filter`）。

## 时间对齐规则

- 每条测量带绝对**采样时间戳**；每路通道有独立采样周期与恒定**延迟**，
  延迟只影响“到达时刻 = 时间戳 + 延迟”，不改动时间戳。
- 对齐缓冲维护单调**水位线 (watermark)**，只释放 `timestamp <= watermark` 的测量；
  释放顺序确定性地按 `(timestamp, sensorID, arrivalSeq)`，与 goroutine 调度无关。
- 演示/测试的推荐用法：**先并发提交全部测量**（此时水位线为 -∞，不会误判过期），
  再沿“到达时刻”单线程推进时钟——这是乱序流的标准水位线驱动方式。
- 异常时间输入**显式拒绝并计数**，绝不静默丢弃：
  - `duplicate_timestamp`：同一 `(sensor, timestamp)` 已接受过；
  - `stale_timestamp`：时间戳早于 `watermark - MaxDelay`；
  - `bad_timestamp` / `non_finite_measurement`：NaN/Inf。

## 数值与容错策略

- 噪声矩阵 `Q/R/初始P` 校验：形状、有限性、对称性（容差）、特征值半正定；
  非法配置以 `noise_not_spd` / `dimension_mismatch` 拒绝，**零特征值（半正定）合法**。
- 预测 `P = FPF' + Q`（过程噪声按 `dt^4/dt^3/dt^2` 积分），更新用
  **Cholesky 求解** `S = HPH' + R`；`S` 奇异（退化/病态测量）返回
  `degenerate_update`，不产生 NaN。
- 更新采用 **Joseph 形式** `(I-KH)P(I-KH)' + KRK'`，对任意 K 保持对称半正定；
  每步对称化并在必要时用 Higham 最近半正定投影修复（特征值下限）。
- 非线性模型在当前状态求雅可比；方位角残差自动卷绕到 `[-π,π)`。
- **故障隔离**：非有限值或 NIS 连续超门限的通道被隔离；隔离期间该路只在
  **scratch 副本**上试更新，绝不污染全局估计；连续若干次健康试更新后自动恢复。
- 所有拒绝原因见 `internal/reject`，错误携带稳定 code 与人类可读 detail，
  可通过 `reject.Of(err)` 程序化区分。

## 并发模型

`Submit` 可被任意 goroutine 并发调用，内部用互斥锁串行化“入队 + 记账”；
所有改变滤波器状态的工作集中在 `AdvanceTo/Pump/Run/Flush` 中、按确定性顺序执行。
因此同一输入序列无论怎样并发提交，最终状态与统计量逐位一致；`-race` 下无数据竞争。

## 包结构

```
internal/reject      可区分的拒绝原因（稳定 code）
internal/models      状态模型、线性/非线性测量模型
internal/noise       协方差校验、对称化、最近半正定投影
internal/alignment   水位线时间对齐缓冲（乱序/重复/过期）
internal/filter      数值稳定的 EKF（Cholesky 更新、Joseph 协方差、NIS）
internal/fusion      并发接入、故障隔离/恢复、统计与事件日志
internal/synth       确定性本地数据生成（GPS/速度/极坐标、延迟、故障注入）
cmd/demo             端到端演示
```

详细设计与判定依据见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
