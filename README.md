# 本地多传感器状态估计（Go）

一套完全本地运行、零第三方依赖的运动平台状态估计流程。用本地确定性生成的测量数据
模拟多路不同采样率、带延迟、乱序与时间戳重复的传感器（线性 + 非线性测量），在统一的
扩展卡尔曼滤波（EKF）流程中完成融合，并对非法噪声、退化/病态测量、非有限值和持续
发散的传感器给出**可区分、可复现**的拒绝与隔离行为。

- 模块路径：`github.com/lisama569/sensor-fusion-estimation-go`
- 仅依赖 Go 标准库（矩阵运算在 `internal/mat` 内自行实现）。

## 目录结构

| 路径 | 作用 |
| --- | --- |
| `internal/mat` | 稠密矩阵/向量、转置乘加、对称化、LDLᵀ 分解与 SPD 求解/求逆、Jacobi 特征值、PSD 与条件数检查、有限值检查 |
| `pkg/model` | 状态定义；2D 恒速运动模型；线性位置测量模型与非线性距离/方位测量模型 |
| `pkg/engine` | EKF 预测/更新、时间对齐器（乱序/重复/过期）、NIS 门限、逐路故障隔离与恢复状态机、并发安全引擎门面；以及长期运行的**会话层**：存档/续跑、历史补数（回放）、有界去重记忆 |
| `pkg/sim` | 基于固定随机种子的本地确定性多传感器数据集生成（多速率、延迟、抖动、乱序、重复、故障注入） |
| `cmd/demo` | 端到端命令行演示，打印输入序列、逐条判定依据与最终估计/统计 |

## 状态与测量模型

### 状态向量与运动模型

恒速 2D 模型，状态为

```
x = [px, py, vx, vy]^T
```

状态转移 `x_k = F x_{k-1}`，其中

```
F = [ 1 0 dt 0
      0 1 0  dt
      0 0 1  0
      0 0 0  1 ]
```

过程噪声采用连续白噪声加速度模型（功率谱密度 `QPSD`，要求有限且 `>= 0`）：

```
Q = QPSD * [ dt^3/3 I2  dt^2/2 I2
             dt^2/2 I2  dt     I2 ]
```

`dt` 必须有限且非负；`dt == 0` 为空操作；不允许向后预测。

预测协方差：`P = F P F^T + Q`，随后做对称化 `P <- (P + P^T)/2` 以消除浮点不对称。

### 测量模型（线性与非线性在同一流程）

- **线性位置** `model.LinearPosition`：`h(x) = H x`，
  `H = [1 0 0 0; 0 1 0 0]`，噪声协方差 `R`。
- **非线性距离/方位** `model.RangeBearing`（相对传感器原点 `(Sx, Sy)`）：
  ```
  h(x) = [ sqrt(dx^2+dy^2), atan2(dy, dx) ],  dx = px-Sx, dy = py-Sy
  ```
  雅可比在当前状态处解析计算；距离退化为零（方位无定义）的奇异点会被拒绝。

两类模型都实现同一接口 `model.MeasurementModel`，由同一个更新步骤处理。

## 时间对齐规则

`engine.aligner` 负责把多路、不同采样率、带延迟且乱序到达的测量对齐：

1. `Submit` 只做与到达顺序无关的结构性校验（来源/模型/维度、NaN/Inf 时间戳），
   然后把测量**缓冲**。重复/过期/故障等**顺序相关**判定一律推迟到 `Advance`。
2. `Advance(t)` 取出所有时间戳 `<= t` 的缓冲测量，按 **(时间戳, 来源名)**
   稳定排序，因此融合顺序是全局确定的，与 goroutine 调度无关。
3. 延迟窗口：设已提交时钟为 `t`、`MaxDelayHorizon = H`，
   时间戳 `< t - H` 的测量判为 `expired_measurement`；仍在窗口内的“迟到”测量
   是合法的乱序测量，会按自身时间戳参与融合（预测被惰性地推进到该时间点）。
4. 重复：同一 `(来源, 精确时间戳)` 在同一批中出现多于一次时，**该重复组的所有副本
   都被拒绝**（无法判定哪一份权威），原因为 `duplicate_timestamp`。
5. 所有拒绝都返回带机器可读 `RejectReason` 的 `*RejectError` 并计入统计，
   **绝不静默丢弃**。

## 数值稳定与容错策略

- **对称半正定**：每次预测和更新后都强制对称化；更新使用 **Joseph 形式**
  `P = (I-KH) P (I-KH)^T + K R K^T`，只要 `R` 半正定即可保证 `P` 对称半正定。
- **新息协方差** `S = H P H^T + R` 先对称化，再用 Jacobi 特征值检查：
  - 必须有限；必须对称半正定；
  - 最小特征值 `<= 0` → `degenerate_measurement`（零信息/奇异）；
  - 条件数 `λmax/λmin > CondLimit`（默认 `1e12`）→ `ill_conditioned`。
  求解使用数值更稳的 LDLᵀ 分解（而非直接求逆）。
- **门限拒绝（先验门控）**：更新前先做只读 NIS 计算 `y^T S^{-1} y`。
  超过门限（`NIS > GateSigma^2`）的离群测量**不会写入状态**，记为 `out_of_gate`；
  连续 `FaultStreak` 次越界则隔离该路（`stream_faulty`）。
- **非有限值硬故障**：测量含 `NaN/+Inf/-Inf` 时，该路立即隔离，
  拒绝原因为 `non_finite_value`，全局状态不被触碰。
- **故障隔离与恢复**：
  - 隔离期间该路测量不参与融合，而是作为**只读恢复探针**评估 NIS（门内即计数），
    拒绝原因统一为 `stream_faulty`，故故障路不可能污染全局估计；
  - 连续 `RecoverStreak` 个结构合法且门内探针后自动恢复（记录 `Recoveries`），
    恢复后的下一个测量正常融合。
- **非法噪声**：初始协方差、`Q`、`R` 都会检查有限性、对称性与半正定性，
  非法即拒绝（`illegal_noise`），流程继续处理其他测量。
- **可区分的拒绝原因**：
  `non_finite_value`、`duplicate_timestamp`、`expired_measurement`、
  `bad_timestamp`、`dimension_mismatch`、`illegal_noise`、`degenerate_measurement`、
  `ill_conditioned`、`stream_faulty`、`model_evaluation_failed`、`out_of_gate`。

## 长期会话：存档 / 续跑 / 历史补数

在一次性 `Engine` 之上，`engine.Session` 提供一段能长期活着、随时存档、断了还能在
**全新进程**里接着跑的估计会话。它满足三类硬不变量：

### 1. 存档与续跑：与「从未中断」逐位一致

- `NewSession(initial, motion, motionDescriptor, bindings, cfg)` 创建会话；
  每路来源用 `SourceBinding{Source, Descriptor, Model}` 注册模型与一段
  **自描述描述符**（由应用决定如何标识模型/参数版本）。
- `Snapshot() []byte` / `Save(io.Writer)` 导出存档；
  `RestoreSession(io.Reader, motion, motionDescriptor, bindings, cfg)` 在新进程恢复。
- 存档在**单一互斥锁**内生成，即使别的 goroutine 正在持续 `Submit`，导出的也永远是
  某个真实时间点的一致镜像；恢复后续跑同样输入（同一测量集合、同一时钟步进），
  状态均值、协方差、各路统计与拒绝记录与「从头一直没中断过」**逐位一致**。

存档为自描述的二进制格式（仅用标准库）：

```
"SFES" | 版本 uint16 | 载荷长度 uint32 | JSON 载荷 | SHA-256(载荷)
```

- 所有 `float64` 以 IEEE-754 原始位（uint64）序列化，NaN/Inf 也能精确往返，
  不存在十进制重解析改变任何一位的可能；
- 载荷内含引擎配置、运动模型描述符、逐路模型描述符、当前 EKF 状态、对齐器缓冲、
  健康状态机、全部统计与拒绝记录、内部检查点与可重放操作日志。

**兼容范围（明确拒绝，绝不带着半截状态继续算）**，用 `errors.Is` 区分：

| 哨兵错误 | 触发情况 |
| --- | --- |
| `ErrArchiveVersion` | 存档声明的格式版本不是本构建支持的 `ArchiveFormatVersion`（=1），无论更老还是更新 |
| `ErrArchiveCorrupted` | magic 不对、文件截断、声明长度与实际不符、载荷无法解码，或 **SHA-256 校验不通过**（被写坏/被篡改） |
| `ErrArchiveBinding` | 恢复时提供的运动模型描述符、任一来源的模型描述符、来源集合或引擎配置与存档记录不一致 |

模型对象本身不进存档（它是代码）；恢复时由应用按描述符重新提供，描述符对不上即拒绝，
因此不可能用「同名但不同参数」的模型静默续跑。

### 2. 历史测量补数（回放）：与实时到达逐位一致

`Backfill(batch)` 把一段历史测量作为一批离线补入。其结果等价于这些样本在对应时刻
**本就实时到达**：内部从检查点重建一台临时引擎，让补入样本与保留的真实操作日志一起按
`(时间, 来源)` 确定序重放，再原子替换当前世界。最终均值、协方差、统计与实时情形
逐位一致（顺序相关的拒绝在拒绝日志里的**位置**可能不同，但拒绝集合与计数一致）。

> 注意：空的 `Advance` 也是输入——预测按测量时刻惰性合成，`dt` 的分段不同会有 1-ULP
> 差异。因此「实时参照」必须与补数会话走相同的时钟步进表（测试中严格覆盖了这一点）。

补数批次先做**整批结构校验**（未注册来源、NaN/Inf/未来时间戳、维度不符直接返回 error，
一条都不应用）；随后对每条给出可区分的拒绝：

- `expired_measurement`：时间戳早于「当前时钟 − `MaxDelayHorizon`」**或**早于折叠检查点，
  即超出可接受时间范围、无法再用的老数据；
- `duplicate_timestamp`：该样本此前已经处理过（存档前实时处理过、上一个补数批次补过），
  或在**本批次内重复出现**（整组副本全部拒绝，与对齐器语义一致）。

同一条测量永远不会被融合两次，且重复与过期两种原因严格分开、绝不静默放行。

### 3. 有界记忆与并发一致

- 「这条测量处理过没有」的记忆由 `SessionConfig.MemoryLimit` 设硬上界：超出后内部
  **检查点折叠（checkpoint folding）**把前缀重放成新的基线并丢弃最老的已处理样本键。
  跑得再久，保留的操作日志与去重记忆都不会超过该上界（折叠是摊销线性开销）。
- 代价是语义明确的：早于折叠检查点的历史已无法回放到正确的 dt 分段，这类补数以
  `expired_measurement` 明确拒绝，而不是悄悄算错或融合两次。
- 全部入口共用一把会话锁；快照可与持续喂数并发进行。顺序摄入与多 goroutine
  屏障式摄入同一测量集合，最终结果逐位一致；`-race` 下无数据竞争。

### 会话最小示例

```go
bindings := []engine.SourceBinding{
    {Source: "gps", Descriptor: "linear-position R=diag(0.04,0.04) v=1",
        Model: &model.LinearPosition{R: mat.From([][]float64{{0.04, 0}, {0, 0.04}})}},
}
sess, _ := engine.NewSession(initial, &model.ConstantVelocity{QPSD: 0.05},
    "cv-qpsd=0.05", bindings, engine.DefaultSessionConfig())

_ = sess.Submit("gps", 0.1, mat.Vec(0.1, 0.02))
_, _, _ = sess.Advance(0.1)

arc, _ := sess.Snapshot()           // 随时存档（也可 sess.Save(file)）
// ... 全新进程 ...
sess2, err := engine.RestoreSession(bytes.NewReader(arc),
    &model.ConstantVelocity{QPSD: 0.05}, "cv-qpsd=0.05", bindings,
    engine.DefaultSessionConfig())  // 版本/损坏/描述符不符 -> 明确错误
if err != nil { /* ErrArchiveVersion / ErrArchiveCorrupted / ErrArchiveBinding */ }

n, rejs, err := sess2.Backfill([]engine.HistoricalMeasurement{
    {Time: 0.2, Source: "gps", Z: mat.Vec(0.2, 0.0)},
})
```

## 并发模型与可复现性

- `engine.Engine` 用单一互斥锁保护所有状态迁移；`Submit / Advance / State /
  Stats / Rejections` 均可并发调用。
- 关键不变量：**所有依赖顺序的决策都在 `Advance` 的 (时间戳, 来源) 确定序中发生**。
  因此对同一批测量（multiset），无论用多少 goroutine、以何种顺序 `Submit`，
  只要 `Advance` 由同一固定时钟驱动，最终状态、协方差与统计量**逐位一致**。
- 测试 `TestConcurrentDeterministicSameMultiset` 对比顺序摄入与 8 路并发摄入，
  要求逐位相等；`TestConcurrentAPISafety` 在竞态检测下并发轰击全部公共 API。
- 会话层同样单锁串行化所有迁移；`TestSessionConcurrentSnapshotDuringIngestion` 在
  持续喂数过程中并发导出/恢复存档，要求每个导出都能恢复成自洽镜像，并在 `-race` 下
  验证无数据竞争；喂数采用每步提交本步全部样本后再推进的屏障式并发，与单线程同一
  测量集合的结果逐位一致。

## 本地验证方法

需要 Go 1.25+（开发环境为 Go 1.26）。

```bash
# 单元/集成测试
go test ./...

# 只看本轮新增会话行为（带行为判定日志）
go test -run TestSession -v ./pkg/engine/

# 竞态检测（需求要求必须通过；建议多跑几次）
go test -race -count=10 ./...

# 静态检查
go vet ./...

# 端到端演示：打印输入序列、逐条判定依据、最终估计、协方差与统计
go run ./cmd/demo
```

### 测试覆盖说明

**会话层新增行为（`session_test.go`，用 `-v` 可看到每个行为的判定日志）：**

- 存档/续跑 vs 不中断：在 0.3/1.2/2.5/3.0s 多个断点存档，新进程恢复后续跑相同输入，
  断言均值/协方差/各路统计与拒绝集合与不中断参照**逐位一致**
  （`TestSessionSaveResumeMatchesUninterrupted`）。
- 历史补数 vs 实时到达：把一整段历史离线补入，断言与「样本当初实时到达」逐位一致；
  另含一批同时带越门限离群值与 NaN 硬故障的历史，断言融合数、拒绝原因、隔离状态与
  统计完全相同（`TestSessionBackfillMatchesRealtime`、
  `TestSessionBackfillWithRejectedSamples`）。
- 重复测量的可区分拒绝：存档前后的重复 Submit、跨两个补数批次的重复、同一批次内重复
  组（全部副本拒绝），均判 `duplicate_timestamp`；超出延迟窗口/折叠地板的老数据判
  `expired_measurement`，两种原因在同一用例中被明确区分，且不重复计数
  （`TestSessionDuplicateReasonsDistinguishable`）。
- 有界记忆：小 `MemoryLimit` 下反复折叠，断言保留操作记录与去重记忆不超上界、
  估计仍逐位一致；窗口内迟到补数仍与实时一致，而早于折叠地板的样本被
  `expired_measurement` 明确拒绝（`TestSessionMemoryBoundedAndCorrect`、
  `TestSessionRejectionLogConsistentAfterFolding`）。
- 存档安全：截断、magic 错误、载荷翻转、校验和翻转、未来版本号、垃圾内容分别以
  `ErrArchiveCorrupted` / `ErrArchiveVersion` 明确拒绝；模型描述符/运动描述符/来源
  集合/引擎配置不匹配以 `ErrArchiveBinding` 拒绝；初始空会话存档也可往返
  （`TestSessionArchiveRejectsBadInput`、`TestSessionArchiveRoundTripFile`、
  `TestSessionEmptyArchiveRoundTrip`）。
- 并发导出一致性与无竞争：`TestSessionConcurrentSnapshotDuringIngestion`。

**原有用例（无回归）：**

- 乱序到达后按 `(时间,来源)` 确定序融合；重复时间戳显式拒绝；
  超窗口过期显式拒绝；合法窗口内迟到测量被接受（`aligner_test.go`）。
- 病态/非法噪声：非对称、不定、非有限 `R`、奇异 `S`、超高条件数，
  以及距离/方位奇异点，均被以不同原因拒绝（`ekf_test.go`）。
- 协方差性质：预测/更新后断言对称、半正定；线性更新使位置方差下降。
- 单路故障与恢复：非有限值硬隔离、持续发散软隔离，隔离期不污染全局，
  探针计数达标后恢复（`health_test.go`）。
- 可复现性：顺序与并发摄入同一序列逐位一致；仿真数据集同种子逐位复现，
  不同种子产生不同乱序（`engine_test.go`、`integration_test.go`、`sim_test.go`）。
- 测试日志用 `-v` 可查看完整输入序列与每条测量的判定原因
  （见 `integration_test.go` 的 `TestEndToEndScenarioAndReproducibility`）。

## 最小使用示例

```go
initial := &model.State{
    Time: 0,
    X:    mat.Vec(0, 0, 1, 0),
    P:    mat.Identity(4),
}
motion := &model.ConstantVelocity{QPSD: 0.05}
eng, _ := engine.New(initial, motion, engine.DefaultConfig())

gps := &model.LinearPosition{R: mat.From([][]float64{{0.04, 0}, {0, 0.04}})}
if err := eng.Submit("gps", 0.1, mat.Vec(0.1, 0.02), gps); err != nil {
    // err 是 *engine.RejectError，可按 err.Reason 区分拒绝原因
    log.Printf("rejected: %v", err)
}
state, rejections, err := eng.Advance(0.1) // 按 (时间,来源) 确定序融合
```
