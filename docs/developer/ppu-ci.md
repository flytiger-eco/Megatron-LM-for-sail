# SAIL PPU CI 维护

本分支聚焦正式 PPU CI：复用上游训练脚本与比较器，在 SAIL runner 上执行。
维护提交统一使用 `DavidSophus <95245426@qq.com>`。历史专项诊断和本地调试工具
已移出交付内容。

## 独立测试分支

`davidsophus/ppu-ci-integration` 用于集成验证，普通 push 只运行 CPU 控制测试。
测试分支与 main/dev 使用同一套入口、授权及执行组件；不向 main/dev 推送测试提交。
硬件执行使用仓库级 `sail-ppu-hardware` 串行队列；排队的 PR 在进入设备前重新检查 head/base 和状态归属。

手动触发此分支的 `PPU Build`，选择 `test_kind=smoke`，其余输入保持默认：

- UT 只运行上游 `tests/unit_tests/test_model_configs.py` bucket。
- FT 只运行 `gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files`，一次完整数值比较。
- 两者使用同一份构建产物；UT 失败则 FT 不启动。
- 这是调试证据，不是完整 UT/FT 覆盖；此模式不能用于发布 PR 成功状态。

## 同步与 PR 的统一入口

main/dev 同步发布成功后，worker 输出 `sync_request`（被测 SHA、两个父提交、发布分支），
父 workflow 直接调用 `ppu-build.yml` 的 `workflow_call`，再复用同一套 Build→UT→FT。
不依赖 `GITHUB_TOKEN` 的 push 触发后续 Actions。main 和 dev 的 artifacts、NAS 目录分别使用
`sync-main-` / `sync-dev-` 前缀，共享同一个硬件队列。

普通分支更新、同步及无标签 PR 共用 UT 的 `quick` 范围；FT 默认使用相同的上游 L0 测试集、一次重复、30 分钟预算。
其中上游标注为 lightweight 的 case 只记录观察结果，不计为数值精度通过；其余 case 使用正式 golden 比较。
`Run tests` 保留显式轻量观察模式，`Run functional tests` 可扩大到 L1；常规路径不需要标签。
文档和纯 CI 编排变更仅走 CPU 检查；执行器、镜像/依赖、recipe、golden、被测代码变更仍运行硬件测试。
改名同时检查原路径，变更列表不完整时不豁免。

GitHub 自 2025-12-08 起从默认分支加载 `pull_request_target`，即使 PR 合入 dev，控制代码仍来自 main。
因此入口用 PR 的 `base.ref` 判断目标分支，用默认分支 SHA 固定控制代码。
独立分支通过真实 PR 的 full-SHA 手动授权验证执行和状态；自动 PR 事件入口必须在 main 上线后验收。
参考：[GitHub 变更说明](https://github.blog/changelog/2025-11-07-actions-pull_request_target-and-environment-branch-protections-changes/)。

测试同步接线时，在独立分支调度同步 workflow 并设置 `sandbox_handoff=true`。
它只读取固定的 `davidsophus/ppu-sync-main` / `davidsophus/ppu-sync-dev` 测试引用，要求两个父提交分别是
当前 main/dev 基线与当前控制 SHA，再调用同一流水线的最小诊断模式；不发布到 main/dev，不创建同步 PR。
此模式验证发布后的接线和 SHA 隔离，不代表实际上游合并或完整测试集通过。
本轮只维护 dev 的固定测试引用；main 使用集成分支自身的 smoke 入口。未准备并绑定 main 引用时，不使用 `target=all`。

```bash
# 使用集成分支自身的 main 产品源码验证共享流水线。
gh workflow run ppu-build.yml --repo flytiger-eco/Megatron-LM-for-sail \
  --ref davidsophus/ppu-ci-integration -f test_kind=smoke

# 使用已绑定当前 control SHA 的 dev 测试引用验证同步后的交接。
gh workflow run copilot-upstream-sync.yml --repo flytiger-eco/Megatron-LM-for-sail \
  --ref davidsophus/ppu-ci-integration -f target=dev -f sandbox_handoff=true -f publish=false
```

测试引用使用合成 merge commit，只保留当前产品分支的源码并补齐已有 PPU baseline；第二父提交仅标识
控制代码。它们不能合入产品分支。修改 control 或同步基线后，需重新绑定测试引用，旧引用会在 CPU 授权阶段被拒绝。
运行期间不要移动被测引用；先等待该轮结束，再更新并记录新旧完整 SHA。

dev 的 UT recipe 目前还包含 `examples/multimodal_dev/tests` 和 2/4 卡分组。
选择器直接读取 recipe 的 `gpus`，GTP bucket 按上游脚本显式设置的 `GPUS_PER_NODE=4` 执行；
这条例外集中在 `ppu_policy.py`。新增测试根目录、超出单机 1/2/4/8 卡的拓扑会在 CPU 选例阶段报错。
NVRx 预检查同时支持旧 `get_async_strategy()` 与 dev 的直接 NVRx 接口，环境报告记录 `source_api`。
两条路径均检查上游能力和原始 API 签名，不创建 checkpoint worker，也不因接口变化跳过检查。
构建阶段的推理通信预检查也集中在 `ppu_functional_environment.py`：根据实际 engine 类的来源，
检查旧 `dynamic_engine` 或 dev 拆分后的 `dynamic.coordinator` 中的 ZMQ/MessagePack 能力；
未知布局和缺少能力均失败，日志记录本轮检查的模块。

## 执行路径

```mermaid
flowchart TD
    S[同步成功：发布 SHA 与父提交] --> A[ppu-build.yml 共用入口]
    P[main/dev PR 或分支更新] --> A
    A --> B[pr_authorize 固定 control / target SHA]
    B --> Q{文档或纯 CI 改动？}
    Q -->|是| X[CPU 检查与硬件豁免状态]
    Q -->|否| C[ppu-execute.yml 选例与 manifest]
    C --> D[CPU 资产预检查]
    C --> E[构建镜像与封装源码]
    E --> F[SAIL Pod: UT]
    E --> G[SAIL Pod: functional_run.sh]
    D --> G
    F -->|UT 成功后调度| G
    G --> H[functional_runtime: 隔离目录、预算、进程清理]
    H --> I[上游 run_ci_test.sh 与 pytest]
    I --> J[ci_export 脱敏导出]
    J --> K[CPU functional_results 复核]
    K --> L[汇总正式结果]
    F --> L
```

combined 流程仅在 UT 成功后启动 functional；最终组合结果要求 UT 和 functional 都成功。
`control_sha` 标识控制代码，`target_sha` 标识被测源码；训练 PR 在旧 control 下通过，
不能证明该 PR 修改的 CI 脚本已被执行。新控制代码需要单独的 CPU 验证和开发环境验证，
不能为此放开正式授权规则。
FT 的 scope、cadence 和预算来自 control 中受信任的 recipe；被测分支的 recipe 可以是另一版本。
manifest 的 `control_files`、`source_files` 分别记录两者摘要，case 的 `recipe_sha256` 对应被测源码。
执行前仍逐项验证 source；不能拿 main/control 的摘要要求 dev/source 文件相同。

## 保留的脚本与责任

以下文件均位于 `.github/scripts/`。CPU 测试在其 `tests/test_ppu_*.py` 下。

| 文件 | 维护职责 |
|---|---|
| `ppu_policy.py` | 受信任分支、最小诊断 case、CPU 豁免路径和同步产物前缀 |
| `ppu_pr_authorize.py` | 授权、SHA、过期状态与 PR 状态归属 |
| `ppu_ci.py`、`ppu_pytest_policy.py` | UT 选例、manifest、排除策略、结果门槛和证据 |
| `ppu_ci_build.sh`、`ppu_ci_run.sh` | 构建准备和 UT 执行 |
| `ppu_ci_container.py`、`.sh` | 镜像指纹、构建、导出/恢复和产物校验 |
| `ppu_ci_paths.py`、`ppu_ci_export.py` | CPU/Pod 存储映射与公共产物脱敏 |
| `ppu_functional.py` | recipe/golden 选择、计划与资产清单 |
| `ppu_functional_snapshot.py` | baseline 配置和 golden 的 SHA256 清单 |
| `ppu_functional_hf.py` | 固定版本 tokenizer、receipt 与离线缓存 |
| `ppu_functional_environment.py` | 包/API/工具/设备/编译环境检查 |
| `ppu_functional_run.sh` | 正式 Pod 执行入口：镜像、环境、runtime |
| `ppu_functional_runtime.py` | 上游 shell 调用、重复/预算/重试、进程族清理 |
| `ppu_functional_pytest.py`、`ppu_functional_results.py` | pytest 证据记录及完整结果复核 |

进程族发现与信号清理函数现位于 runtime 内，不依赖诊断脚本；保留跨 session、孤儿进程、
PID 重用、超时、取消和实际子进程清理测试。正常日志、错误分类、环境检查不属于可删除的
临时 debug 内容。模型配置、golden、比较器及正式门槛未随此次清理修改。

维护时先按失败阶段定位，避免重跑整套硬件测试：

| 失败位置 / 证据 | 首先检查什么 | 负责文件 |
|---|---|---|
| authorize / `authorization.json` | PR 状态、head/base SHA、调用 workflow、是否属于 CPU 豁免路径 | `ppu_pr_authorize.py`、`ppu_policy.py` |
| configure / `manifest.json`、`functional-manifest.json` | 上游 recipe 变化、被选 case、拓扑、baseline/tokenizer 锁定文件 | `ppu_ci.py`、`ppu_functional.py` |
| runtime-environment / Docker 构建日志 | 基础镜像 digest、包约束、离线安装与编译 | `Dockerfile.ppu-ci`、`ppu_ci_container.py` |
| functional-assets / `assets-report.json` | NAS 映射、数据文件、tokenizer receipt 与缓存 | `ppu_ci_paths.py`、`ppu_functional_hf.py` |
| build / Pod 状态与构建日志 | 先区分排队、镜像拉取、实际执行，再检查包导入、设备接口与构建产物 | `ppu_ci_build.sh`、`ppu_functional_environment.py` |
| UT Gate / `summary.json` | 实际执行数、failed/skipped、失败方法与各 rank 日志 | `ppu_ci.py`、`ppu_pytest_policy.py` |
| Functional gate / `failure_categories`、`execution.json`、`pytest-regular.json` | 训练退出、比较器结果、证据是否完整、case 预算 | `ppu_functional_runtime.py`、`ppu_functional_results.py` |

只有 `infrastructure_retryable` 可在剩余预算内自动重试；数值不一致、OOM、导入失败、证据错误
不会通过反复重跑变成通过。`observation` 也不会加入 numeric 通过率。修改单个控制模块后，
先运行对应的 CPU 测试文件；完整控制回归的依赖与命令以 `ppu-control-tests.yml` 为准。

## 最小验证顺序

先在具备 Python 3.12 和 `.github/docker/control-requirements.txt` 依赖的 CPU 环境中检查。
不要为控制脚本在宿主机安装 Megatron GPU 依赖或同步整个项目环境。

```bash
# 只验证选例与锁定文件，不申请 Pod，也不检查 NAS 是否可读。
PPU_INPUTS='{"functional_cases":"gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files"}' \
  python .github/scripts/ppu_functional.py catalogue > /tmp/ppu-catalogue.json

# 控制脚本回归；训练与数值检查使用合成证据，不占用 PPU。
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  .github/scripts/tests/test_ppu_functional.py \
  .github/scripts/tests/test_ppu_functional_runtime.py \
  .github/scripts/tests/test_ppu_functional_results.py
```

需要 NAS 或设备验证时，在 `ppu-build.yml` 的受信任分支上手动选择下列输入。
功能测试控制 Build/UT，其他 UT 开关保持默认值；不要提交 `all` 个 case。

| 输入 | 首先检查资产 | 确需训练时 |
|---|---|---|
| `test_kind` | `functional` | `functional` |
| `functional_mode` | `assets` | `regression` |
| `functional_scope` | `all`（由下列精确 case 再过滤） | 同左 |
| `functional_cases` | `gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files` | 同左 |
| `functional_repeat` | `1` | `1` |
| `functional_timeout` | `5` | `5` 分钟总训练预算 |

这个普通 GPT case 为 TP1/PP1、12 层、hidden size 512、50 步；选择它是为了减少特性组合，
不是声称它已经实测最快。需要数据、vocab/merges 和固定的 GPT2 tokenizer 缓存。
现有执行器仍要求单机 **8 卡**。`repeat=1` 不关闭正式基础设施重试：最多 3 次 attempt
共用预算；排队、构建、资产预检查和退出清理另有开销。冷编译可能耗尽预算，不能据此判精度失败。

最小集成调试使用本页的 `smoke` 入口；历史四步专项诊断不属于当前交付。
需要时独立 checkout 使用，并记录实际 control/target SHA；其 smoke 结果不计作精度通过。

已有正式产物直接在 CPU 复核，不重跑训练：

```bash
python .github/scripts/ppu_functional_results.py \
  --manifest /path/to/manifest.json --root /path/to/target --results /path/to/results
```

结果目录保留 `functional-000/` 等子目录中的原始环境、action、rank、pytest、TensorBoard
和导出证据。先定位失败阶段，再选择一个相关 case。`ckpt-resume` 检查保存/加载，不能证明
故障注入后的自动恢复；一次 numeric 或 smoke 也不能证明性能稳定。

## 独立分支验证记录（2026-11-20）

以下均为 GitHub Actions 上的实际执行。CPU 控制测试与 PPU 训练证据分开记录，正式 main/dev 未写入测试提交。

| 验证 | 代码 / 运行记录 | 结果与边界 |
|---|---|---|
| CPU 控制回归 | `b9fdbe4b8`；[44601645148](https://github.com/flytiger-eco/Megatron-LM-for-sail/actions/runs/44601645148) | 1,842 passed；覆盖授权、路径、选例、产物、比较与清理逻辑，不代表训练通过 |
| 草稿 PR | [44571365750](https://github.com/flytiger-eco/Megatron-LM-for-sail/actions/runs/44571365750) | 保持 pending，不构建、不申请 PPU |
| 文档 PR | [44572021070](https://github.com/flytiger-eco/Megatron-LM-for-sail/actions/runs/44572021070) | 按策略豁免硬件测试 |
| 纯 CI PR | `506307b1d` 控制；[44577131713](https://github.com/flytiger-eco/Megatron-LM-for-sail/actions/runs/44577131713) | 明确发布 Docs/CI-only 豁免状态；构建和执行 job 跳过 |
| main 源码最小流程 | control / target 均为 `069979312`；[44571681281](https://github.com/flytiger-eco/Megatron-LM-for-sail/actions/runs/44571681281) | UT：816 passed、0 failed、213 skipped；FT：1 个 numeric case 通过，1 attempt / 1 repeat；组合结果 success |
| dev 跨分支校验修复前 | control `cdc427722` / target `b88d6d178`；[44593647743](https://github.com/flytiger-eco/Megatron-LM-for-sail/actions/runs/44593647743) | 构建成功，UT：244 passed、0 failed、67 skipped；FT 在训练前被 recipe 摘要混用问题拦截。已在 `b9fdbe4b8` 修复并用真实 dev 文件在 CPU 复现验证；此轮不算精度通过 |
| dev 源码最小流程 | control `b9fdbe4b8` / target `891c690a2`；[44602771487](https://github.com/flytiger-eco/Megatron-LM-for-sail/actions/runs/44602771487) | UT：244 passed、0 failed、67 skipped；FT：1 个 numeric case 通过，1 attempt / 1 repeat；100% numeric gate 与组合结果 success |

临时 PR [#21](https://github.com/flytiger-eco/Megatron-LM-for-sail/pull/21) 已关闭，未合并。
PR 探针使用真实 PR 的 head/base SHA 手动授权；不能将它们写成自动 `pull_request_target` 事件已验收。

正式上线还需处理现有同步权限：生产同步 [44563433346](https://github.com/flytiger-eco/Megatron-LM-for-sail/actions/runs/44563433346)
在创建 main 同步 PR 时被仓库设置拒绝（`GitHub Actions is not permitted to create or approve pull requests`）。
这次测试没有修改仓库设置或凭据。沙箱接线成功不能证明该生产发布步骤已恢复。

main 最小运行的首次镜像拉取耗时 20 分 2 秒（压缩镜像约 7.11 GiB），实际构建检查 78 秒。
UT 设备内执行 55 秒，FT 训练与比较约 184 秒。各项计时口径不同，不应相加作为稳定性能指标；
冷启动开销与训练性能分别记录。此运行不覆盖完整 L0、checkpoint resume、FT 故障注入或性能回归。

dev 成功运行的设备内构建检查为 75 秒，FT 训练与比较约 197 秒，Pod 内完整执行 209 秒。
同样只代表单次最小诊断，不作为 main/dev 性能对比。原始环境报告确认使用 dev 的 `direct_nvrx`
接口与新的推理通信模块；配置、golden、数值门槛和依赖版本均未为这轮通过而改动。
最终文档提交只补充说明，实际硬件证据以表中 control/target SHA 为准。
已验证 dev 源码由 `davidsophus/ppu-sync-dev-validated` 保留；可重跑的 `davidsophus/ppu-sync-dev`
会绑定到最新文档提交，其产品源码与已验证版本保持相同。

## 维护约定

我们分别维护平台环境、测试控制、模型基线、发布门禁；变更附对应的环境报告、CPU 测试、
采样证据或 control/target 身份。长期 CODEOWNERS 沿用项目真实授权关系，不虚构维护账号。
四项交付以此说明、现有最小调用方式、CPU 自验证和责任边界为基础逐步完善。

项目依赖由 `pyproject.toml` / `uv.lock` 管理；PPU CI 使用独立的平台环境：Dockerfile 读取被测源码的
project dependencies、test dependency group 和 training extra，再叠加 SAIL 原生包约束与测试工具。
它不会直接执行整个 NV dev extra 的 `uv sync`。CPU 控制解释器使用 `control-requirements.txt`，
tokenizer 则由固定 revision、文件摘要和 receipt 管理。产品镜像 `docker/Dockerfile.ppu` 另有用途。

同一次流水线通过镜像 digest、安装报告、编译产物与环境摘要保证 Build / UT / FT 使用同一环境。
这不等于所有传递依赖都已完全锁定：清空构建缓存后的解析结果仍需检查安装报告，并单独验证升级。
禁止通过现场 pip 修复或改 golden 掩盖失败。历史 `_2nd.json` 保留采样来源；resume 比较本次
第一阶段与恢复阶段。以后清理重复配置或依赖漂移，应另立小改动验证。

本地 CPU 验证不代表远程 Linux runner、SAIL 训练、精度或性能验收；硬件验证单独记录。
