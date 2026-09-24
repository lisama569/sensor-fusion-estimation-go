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

## 并发模型与可复现性

- `engine.Engine` 用单一互斥锁保护所有状态迁移；`Submit / Advance / State /
  Stats / Rejections` 均可并发调用。
- 关键不变量：**所有依赖顺序的决策都在 `Advance` 的 (时间戳, 来源) 确定序中发生**。
  因此对同一批测量（multiset），无论用多少 goroutine、以何种顺序 `Submit`，
  只要 `Advance` 由同一固定时钟驱动，最终状态、协方差与统计量**逐位一致**。
- 测试 `TestConcurrentDeterministicSameMultiset` 对比顺序摄入与 8 路并发摄入，
  要求逐位相等；`TestConcurrentAPISafety` 在竞态检测下并发轰击全部公共 API。

## 本地验证方法

需要 Go 1.25+（开发环境为 Go 1.26）。

```bash
# 单元/集成测试
go test ./...

# 竞态检测（需求要求必须通过；建议多跑几次）
go test -race -count=10 ./...

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
