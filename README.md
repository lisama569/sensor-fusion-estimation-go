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
| `pkg/engine` | EKF 预测/更新、时间对齐器（乱序/重复/过期）、NIS 门限、逐路故障隔离与恢复状态机、并发安全引擎门面 |
| `pkg/sim` | 基于固定随机种子的本地确定性多传感器数据集生成（多速率、延迟、抖动、乱序、重复、故障注入） |
| `pkg/session` | **长期会话层**：版本化 + 校验和的存档/续跑（内存字节或本地文件）、历史测量批量回放（补数）、跨存档/跨批次去重、有显式上界的已处理记忆、并发一致快照 |
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
  `ill_conditioned`、`stream_faulty`、`model_evaluation_failed`、`out_of_gate`，
  以及会话层的 `already_processed`（跨存档/跨补数批次的重复测量；区别于同批
  `duplicate_timestamp` 与过老的 `expired_measurement`）。

## 并发模型与可复现性

- `engine.Engine` 用单一互斥锁保护所有状态迁移；`Submit / Advance / State /
  Stats / Rejections` 均可并发调用。
- 关键不变量：**所有依赖顺序的决策都在 `Advance` 的 (时间戳, 来源) 确定序中发生**。
  因此对同一批测量（multiset），无论用多少 goroutine、以何种顺序 `Submit`，
  只要 `Advance` 由同一固定时钟驱动，最终状态、协方差与统计量**逐位一致**。
- 测试 `TestConcurrentDeterministicSameMultiset` 对比顺序摄入与 8 路并发摄入，
  要求逐位相等；`TestConcurrentAPISafety` 在竞态检测下并发轰击全部公共 API。

## 长期会话：存档 / 续跑 / 历史补数（`pkg/session`）

`session.Session` 在引擎门面之上再加一层「能长期活着、能随时存档、断了还能
接着跑」的能力。它不改变引擎的任何数值路径，只负责**记忆、回放与持久化**，
因此续跑 / 补数的结果与「从头一直没中断过」的跑法**逐位一致**（均值、协方差、
各路统计、拒绝日志全部相等）。

### 存档与续跑

- `Export() ([]byte, error)`：在会话锁内一次性深拷贝**全部**进度——滤波状态
  `(time, x, P)`、运动模型、配置、对齐器（已提交时钟、每源水位、所有挂起测量）、
  逐路故障状态机、统计量与完整拒绝日志、以及会话层的已处理记忆——序列化成一个
  **自描述信封**。
- `SaveFile(path)`：写临时文件 + `fsync` + 原子 `rename`（权限 `0600`），崩溃
  中途也不会在目标路径留下半截存档。
- `Import(data)` / `LoadFile(path)`：在**任何状态被构造出来之前**完成全部校验，
  校验不过就明确拒绝，绝不用半截状态继续算（见下「兼容范围」）。
- 模型自描述：参与存档的模型实现 `model.Serializable`（`Kind()` + `Params()`），
  内置的 `ConstantVelocity`、`LinearPosition`、`RangeBearing` 已自动注册
  （kind 带 `/vN` 后缀）；自定义模型用
  `model.RegisterMotionDecoder / RegisterMeasurementDecoder` 注册解码器。
  未注册的 kind 会被拒绝而不是静默还原成别的模型。

信封格式（纯标准库 `encoding/json` + `crypto/sha256`，无第三方依赖）：

```
{ "magic": "SFE-SESSION", "version": 1, "created": <RFC3339>,
  "checksum_algo": "sha256", "sha256": <body 的十六进制摘要>,
  "body": { "engine": <完整引擎快照>, "memory": <已处理记忆>,
            "inflight_*": <已提交未推进的键>, "horizon": ... } }
```

校验和覆盖 body 的**精确字节**（body 采用紧凑 JSON，规避 `json.RawMessage` 在
信封缩进重编码时被改写导致的摘要漂移）。

### 历史补数（回放）与去重

- `Replay(items)`：把一段历史测量作为一批离线补进来；先逐条提交，再按引擎既有的
  **(时间戳, 来源) 确定序**一次性融合到本批最新时间。只要补数发生在时钟越过这些
  时间戳之前，结果与这些测量当初实时到达**逐位一致**（批内打乱提交顺序也不影响）。
- 「处理过没有」的记忆分两层，拒绝原因彼此可区分：
  - **永久记忆**：只在一次 `Advance` 真正对某条测量做出确定性裁决后才提交
    （融合、门限拒绝、隔离探针、以及除同批重复外的任何拒绝都算「已裁决」）。
  - **在途记忆（inflight）**：已 `Submit` 但还没 `Advance` 的键只存在会话层缓冲，
    不进永久记忆。因此同一条测量的两个副本出现在**同一批**时仍能到达引擎，由
    引擎按既有规则拒绝为 `duplicate_timestamp`；而它若在**更早的补数批次**或
    **某个存档之前**已被裁决过，则在会话层被拒绝为
    `already_processed`（新增原因），永远不会融合第二次。
  - 时间戳老到超出延迟窗口、根本无法再使用的样本，仍优先判为
    `expired_measurement`（与「重复」明确区分），由引擎计数。
- 永久记忆随存档一起保存恢复，所以跨进程/跨存档边界的重复样本照样被挡住。

### 记忆上界（不会无限膨胀）

已处理记忆有**显式上界**，与会话寿命无关（`Options.MaxProcessedPerSource`，
默认每源 4096 个键）：

1. **窗口剪枝**：每次 `Advance` 后丢弃 `t <= now - MaxDelayHorizon` 的键——这类
   时间戳不可能再合法到达（再来只会被判 expired）；它们由每源单调 **watermark**
   继续覆盖，防重复能力不丢，只是不再占 map。
2. **硬上界**：某源仍超过上限时，把最小的一批时间戳折进 watermark，map 键数恒不
   超过上限。该折进只在输入速率长期远超预算时触发；折进后一个「真正全新但极老」
   的样本会被当作 `already_processed` 拒绝（宁拒勿重融）。

### 一致快照与并发

- 会话用单一互斥锁串行化所有迁移；`Export` 在锁内取快照，因此即使旁边有 goroutine
  持续 `Submit/Advance`，导出的也永远是**某个真实时间点**的完整一致图像，导入后
  立即可用。
- 所有依赖顺序的决策依旧发生在引擎的 (时间戳, 来源) 确定序中，故并发下最终结果
  逐位一致；`TestConcurrentExportConsistencyNoRace` 在 `-race` 下边写边导出边
  导入，验证无数据竞争且每份快照内部自洽。

### 兼容范围：版本不匹配与存档损坏

`Import/LoadFile` 在构造任何引擎状态之前按下述顺序拒绝，绝不带半截状态续算：

| 情况 | 返回 |
| --- | --- |
| 不是 JSON / 被截断 / 写到一半 | `ErrArchiveCorrupt`（`archive corrupt or truncated`，附解析原因） |
| `magic` 不对 | `ErrArchiveCorrupt`（bad magic） |
| body 校验和与信封 `sha256` 不符（任何字节被改/损坏） | `ErrArchiveCorrupt`（checksum mismatch） |
| 信封 `version` 不是当前支持版本 | `ErrArchiveVersion`（archive version mismatch） |
| 引擎快照 `FormatVersion` 不兼容 | `ErrArchiveVersion` |
| 模型 `kind` 在本进程未注册解码器 | `ErrArchiveCorrupt`（无法还原模型） |
| 维度/有限性/PSD/内部计数自洽性校验失败 | `ErrArchiveCorrupt` |

用 `errors.Is(err, session.ErrArchiveVersion)` /
`errors.Is(err, session.ErrArchiveCorrupt)` 区分两类失败。

## 本地验证方法

需要 Go 1.25+（开发环境为 Go 1.26）。

```bash
# 单元/集成测试
go test ./...

# 竞态检测（需求要求必须通过；建议多跑几次）
go test -race -count=10 ./...

# 只看新增会话能力（存档/续跑、回放、去重、损坏/版本拒绝）的详细覆盖日志
go test -v ./pkg/session/ ./pkg/engine/ -run 'Session|Replay|Duplicate|Expired|Archive|Snapshot|Concurrent'

# 静态检查
go vet ./...

# 端到端演示：打印输入序列、逐条判定依据、最终估计、协方差与统计
go run ./cmd/demo
```

### 测试覆盖说明

- 乱序到达后按 `(时间,来源)` 确定序融合；重复时间戳显式拒绝；
  超窗口过期显式拒绝；合法窗口内迟到测量被接受（`aligner_test.go`）。
- 病态/非法噪声：非对称、不定、非有限 `R`、奇异 `S`、超高条件数，
  以及距离/方位奇异点，均被以不同原因拒绝（`ekf_test.go`）。
- 协方差性质：预测/更新后断言对称、半正定；线性更新使位置方差下降。
- 单路故障与恢复：非有限值硬隔离、持续发散软隔离，隔离期不污染全局，
  探针计数达标后恢复（`health_test.go`）。
- 可复现性：顺序与并发摄入同一序列逐位一致；仿真数据集同种子逐位复现，
  不同种子产生不同乱序（`engine_test.go`、`integration_test.go`、`sim_test.go`）。
- **会话存档/续跑/补数（`pkg/session/session_test.go`、`pkg/engine/snapshot_test.go`）**：
  - `TestSaveResumeBitIdenticalToUninterrupted`：中途存档、在「新进程」里读回
    续跑，均值/协方差/统计与不中断运行**逐位相等**；
  - `TestMultipleSaveResumeWithPending`：多轮存档循环且携带挂起测量，仍逐位一致；
  - `TestReplayEqualsLiveArrival`：历史一批回放（故意打乱提交顺序）与实时到达
    逐位一致；
  - `TestCrossArchiveAndCrossBatchDuplicates`：跨补数批次、跨存档重复样本被拒为
    `already_processed`，同批双副本被拒为 `duplicate_timestamp`，二者可区分，
    且被拒样本绝不计数、不影响后续真样本融合；
  - `TestExpiredIsDistinctFromDuplicate`：超窗口老样本拒为 `expired_measurement`，
    与重复原因区分；
  - `TestCorruptArchiveRejected` / `TestVersionMismatchRejected`：截断、随机字节、
    翻转字节、篡改 body、错误 magic 一律 `ErrArchiveCorrupt`；信封/引擎版本号不
    匹配一律 `ErrArchiveVersion`；
  - `TestConcurrentExportConsistencyNoRace`：8 路并发边写边导出边导入，快照始终
    代表一个自洽时间点（`-race` 下无数据竞争）；
  - `TestSaveLoadFile`：文件存档往返逐位一致且权限为 `0600`；
  - 记忆层 `TestMemory*`：窗口剪枝与每源硬上界 + watermark 保证显式上界且永不
    重复融合，DTO 往返不丢键；
  - 引擎层 `TestSnapshotRoundTripBitIdentical`、`TestSnapshotCarriesFaultedStream`：
    快照携带挂起测量与隔离中流的故障状态机/计数，恢复后续跑逐位一致。
- 测试日志用 `-v` 可查看完整输入序列与每条测量的判定原因，以及每个新增行为
  打印的覆盖说明
  （如 `pkg/session` 的 `go test -v ./pkg/session/`）。

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

### 长期会话：存档续跑与历史补数示例

```go
initial := &model.State{Time: 0, X: mat.Vec(0, 0, 1, 0), P: mat.Identity(4)}
motion := &model.ConstantVelocity{QPSD: 0.05}

sess, _ := session.New(initial, motion, session.DefaultOptions())

// 在线喂入……
_ = sess.Submit("gps", 0.1, mat.Vec(0.1, 0.02),
    &model.LinearPosition{R: mat.From([][]float64{{0.04, 0}, {0, 0.04}})})
_, _, _ = sess.Advance(0.1)

// 随时存档（文件为原子写入；也可用 Export() 只拿字节留在内存）
if err := sess.SaveFile("session.json"); err != nil { log.Fatal(err) }

// ---- 全新进程 ----
sess2, err := session.LoadFile("session.json")
switch {
case errors.Is(err, session.ErrArchiveVersion):
    log.Fatal("存档版本不受支持，拒绝续跑")
case errors.Is(err, session.ErrArchiveCorrupt):
    log.Fatal("存档损坏/被截断，拒绝带着半截状态续算")
}

// 离线补一段历史：与这些测量当初实时到达的结果逐位一致
rejs, _ := sess2.Replay([]session.ReplayItem{{
    Source: "gps", Time: 0.2,
    Z:     mat.Vec(0.21, 0),
    Model: &model.LinearPosition{R: mat.From([][]float64{{0.04, 0}, {0, 0.04}})},
}})
for _, r := range rejs {
    // r.Reason 可能是 already_processed / duplicate_timestamp /
    // expired_measurement / out_of_gate …，彼此可区分
    log.Printf("replayed sample rejected: %s", r.Reason)
}
```
