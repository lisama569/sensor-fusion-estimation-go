# 架构与判定依据

本文说明状态/测量模型、时间对齐规则、数值稳定性、容错与故障隔离策略，
以及本地验证方法。所有结论都可由仓库中的测试与演示复现。

## 1. 总体数据流

```
        goroutine A (gps)  ─┐
        goroutine B (vel)  ─┼─► Fuser.Submit ─► alignment.Buffer (watermark)
        goroutine C (polar)─            │ (互斥锁，只入队/记账)
                                          │
              Fuser.AdvanceTo / Pump / Run / Flush
                                          │  (单线程、确定性顺序)
                            ┌─────────────▼──────────────┐
                            │ scratch EKF 试更新 + NIS 门限 │
                            │ 健康 ► 提交 live filter      │
                            │ 异常 ► 拒绝/隔离（不污染状态） │
                            └─────────────┬──────────────┘
                                          ▼
                        状态 (x,P) + Stats + 事件日志 (Event)
```

关键不变量：

1. `Submit` 从不修改滤波器状态，只做静态校验、重复/过期判定与入队。
2. 滤波器状态只在时钟推进路径上修改，输入顺序由对齐缓冲确定性排序确定。
3. 每个更新都先在 `Filter.Clone()` 的副本上试算，只有“有限、可逆、NIS 未越限”
   的结果才替换 live filter，因此任何坏测量都无法污染全局估计。

## 2. 状态与测量模型

### 2.1 状态模型

4 维 2D 恒速模型 `x = [x, vx, y, vy]`，离散转移

```
F0 = [1 1 0 0; 0 1 0 0; 0 0 1 1; 0 0 0 1]   (单位时间)
```

白噪声加速度（连续谱密度 `q`）的单位时间过程噪声（每轴）

```
Q0 = q*[1/4 1/2; 1/2 1]  以 2x2 块的形式置于位置/速度分量。
```

任意 `dt` 下：`F(dt) = I + (F0-I)dt`；`Q(dt)` 按条目的 `dt` 幂积分——
位置-位置项 `dt^4`、位置-速度交叉项 `dt^3`、速度-速度项 `dt^2`
（见 `filter.processNoiseDT` / `dtPowerHint`，按
`[位置,速度,位置,速度]` 的索引奇偶推断幂次）。

### 2.2 测量模型

- 线性：`z = H x + b + v`（`models.LinearModel`）
  - `gps`：`H` 选位置分量，`z=[x,y]`
  - `vel`：`z=[vx,vy]`
- 非线性：`z = h(x)+b+v`，雅可比由调用方在当前状态给出（`models.NonlinearModel`）
  - `polar`（相对信标 `(bx,by)`）：
    ```
    h(x) = [ hypot(x-bx, y-by), atan2(y-by, x-bx) ]
    H    = [ dx/r, 0, dy/r, 0 ;
            -dy/r², 0, dx/r², 0 ]
    ```
  - 通过 `AngleRows()` 声明方位角分量；更新残差卷绕到 `[-π,π)`，
    使测量落在 `+π/-π` 切口两侧时残差仍正确。

线性与非线性测量使用同一个 `Filter.Update` 流程。

## 3. 时间对齐规则（alignment.Buffer）

- `Measurement.Timestamp` 是真实采样时刻；通道 `Delay` 只用于产生
  `Arrival = Timestamp + Delay`（见 `synth.Generator.Arrival`）。
- 已接受的 `(sensorID, timestamp)` 永久记入 `seen`，跨水位线也判重。
- `Accept` 判定顺序（任一失败即返回、不入队、计数并写事件日志）：
  1. 时间戳是否有限（`bad_timestamp`）；
  2. 测量向量是否有限（`non_finite_measurement`）；
  3. 是否已存在同一 `(sensor,timestamp)`（`duplicate_timestamp`）；
  4. 是否 `timestamp < watermark - MaxDelay`（`stale_timestamp`）。
- `AdvanceTo(t)` 单调推进水位线，释放所有 `timestamp<=t` 的测量，
  稳定排序键 `(timestamp, sensorID, arrivalSeq)`，`arrivalSeq` 是入队先后，
  从而即使同一纳秒到达也有总序。
- `Flush` 释放剩余全部（空缓冲返回 `no_data`）。

> 设计要点：先并发提交、时钟不动（水位线 -∞），任何时间戳都不会过期；
> 提交全部完成后再沿“到达时刻”推进水位线。这样既保留了真实并发（锁竞争、
> 随机入队顺序），又保证结果只取决于输入而不取决于线程调度。

## 4. 数值稳定性（filter.EKF）

- **有限性**：初值、预测状态、雅可比、预测测量、残差、更新后状态全部检查。
- **预测**：`P ← FPF' + Q(dt)`；结果若因舍入轻微越界，用最近半正定投影修复。
- **更新**：
  ```
  ν  = z - h(x) (角度行 wrapPi)
  S  = H P H' + R
  K  = P H' S^{-1}        // Cholesky 求解；失败 => degenerate_update
  NIS = ν' S^{-1} ν
  x  ← x + K ν
  P  ← (I-KH) P (I-KH)' + K R K'    // Joseph 形式，对称半正定
  ```
- **退化检测**：`S` 不能 Cholesky 分解即判为退化测量（如零 `R` 且该方向已被
  观测到方差塌缩），返回 `degenerate_update`，先验状态原样保留。
- **对称半正定**：Joseph 形式对任意增益都成立；每步再做对称化/投影，
  特征值不低于 `EigenFloor`。测试逐次断言特征值非负、`P_ij=P_ji`。
- **非法噪声**：`noise.ValidateSPD` 用 `EigenSym` 检查最小特征值；
  非方阵给 `dimension_mismatch`，非对称/含 NaN/负特征值给 `noise_not_spd`。
  零特征值（完全已知方向）合法。

## 5. 容错、故障隔离与恢复（fusion.Fuser）

通道级参数：`DivergeNIS`（NIS 门限）、`DivergeCount`（连续坏几次隔离）、
`RecoverCount`（隔离后连续好几次恢复）。

- **非有限测量**：`Submit` 立即以 `non_finite_measurement` 拒绝；故障时间戳记入
  `pendingFaults`，等水位线到达该时刻才计入坏 streak 并可能隔离——避免一个
  “未来时间戳”的坏样本把更早的健康测量也错误隔离。
- **NIS 越限**：试更新的 `NIS > 门限` 判为 `channel_diverging`，结果丢弃、
  坏 streak +1；连续 `DivergeCount` 次触发 `isolated`。
- **隔离期**：该路测量不参与 live filter；每个测量在 scratch 副本上试更新：
  - 试更新有限、可逆且 NIS 正常 => 好 streak +1；
  - 否则好 streak 清零，事件记 `channel_isolated`；
  - 连续 `RecoverCount` 次健康 => `recovered`，把最后一次健康试算的状态
    提交为 live 状态，通道重新纳入。
- 其他通道在隔离期间正常工作，全局估计不受污染（集成测试断言故障运行相对
  无故障参考运行的状态偏差有界）。

### 5.1 可区分的拒绝原因（internal/reject）

| Reason | 触发条件 |
| --- | --- |
| `non_finite_measurement` | 测量向量含 NaN/Inf |
| `bad_timestamp` | 时间戳 NaN/Inf |
| `stale_timestamp` | 超出 `watermark - MaxDelay` |
| `duplicate_timestamp` | 同通道重复时间戳 |
| `noise_not_spd` | Q/R/P 非对称、非有限或负特征值 |
| `dimension_mismatch` | 矩阵/向量形状不一致、未知通道等 |
| `degenerate_update` | 新息协方差 S 奇异 |
| `channel_diverging` | NIS 连续超门限 |
| `channel_isolated` | 隔离期到达的测量 |
| `non_finite_innovation` | h(x)/雅可比/残差非有限 |
| `not_initialized` | Init 之前预测/更新 |
| `no_data` | 空缓冲 Flush |

`reject.Of(err)` 返回稳定 code，便于测试与上层路由；`Error.Detail` 给出
具体维度/位置/特征值等判定依据。

## 6. 可复现性与并发一致性

- 数据生成器使用固定种子 `math/rand`；同种子生成逐位相同序列
  （`synth` 测试断言）。
- 事件释放顺序是纯函数 `(timestamp, sensorID, arrivalSeq)`；
  状态转移/更新中没有读入 goroutine 调度或墙钟（`Event.Wall` 仅日志用）。
- 因此“顺序提交”和“多 goroutine 并发提交”对同一输入给出**逐位相同**的最终
  状态与统计（`TestReproducibleConcurrentEqualsSequential`、
  `TestIntegrationReproducibleAcrossSubmitStyles`）。
- 所有可变状态访问都在 `Fuser.mu` 下；`-race` 下全绿。

## 7. 本地验证方法

```bash
# 单元 + 集成测试（打印每个用例的输入序列与判定依据）
go test ./... -v

# 竞态检测（题目要求）
go test -race ./...

# 端到端演示
go run ./cmd/demo
```

测试覆盖矩阵：

| 关注点 | 测试 |
| --- | --- |
| 乱序/延迟/重复/过期/确定性顺序 | `internal/alignment/*_test.go` |
| 非法/病态噪声区分、最近半正定修复 | `internal/noise/*_test.go` |
| 模型形状/噪声校验 | `internal/models/*_test.go` |
| 预测/更新 PSD、退化更新、非有限拒绝、方位角卷绕、可复现 | `internal/filter/*_test.go` |
| 并发=顺序、单路故障隔离与恢复、拒绝原因计数、竞态压力 | `internal/fusion/*_test.go` |
| 生成器同种子可复现、延迟语义、故障窗口 | `internal/synth/*_test.go` |
| 线性+非线性同流程、混合异常输入、端到端隔离/恢复 | `sensor_fusion_integration_test.go` |

## 8. 局限与扩展点

- 状态模型当前内置 2D 恒速；`processNoiseDT` 的幂次按
  `[位置,速度,...]` 奇偶布局推断，更换状态布局时应提供通用的 `Q(dt)`。
- 非线性更新用一阶 EKF（无迭代）；对强非线性可在 scratch 中加 IEKF 迭代。
- 时间模型为离散步进 + 水位线；未实现外推补偿的实时线程，实时场景由调用方
  按到达时刻调用 `AdvanceTo`。
