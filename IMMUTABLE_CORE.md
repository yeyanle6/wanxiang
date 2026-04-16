# Immutable Core — 万象的不可协商边界

> SkillForge 可以造任何工具、Director 可以规划任何角色、mining 可以产出任何洞察
> —— 但这些都**不能改变下面列出的文件和接口**。
>
> 改动 immutable core 必须经过人工审核。Pre-commit hook 会在你尝试提交时警告。
> `tests/test_immutable_core.py` 在 CI 里会锁住公开接口签名。

## 为什么需要这个

万象的自进化路径（SkillForge → trace mining → prompt self-tuning）意味着
系统越来越多的行为由运行时数据和 LLM 决定。但"能力空间让 LLM 自由探索"的
前提是"安全边界用代码硬守"。这份清单就是那条硬边界的最高级版本：**守的是系统自己**。

## Protected files

### 1. `wanxiang/core/message.py` — AI-native 消息协议

**受保护的公开接口：**
- `MessageStatus` 枚举值（SUCCESS / NEEDS_REVISION / ERROR）
- `Message` dataclass 字段（id, intent, content, sender, status, metadata, parent_id, context, turn, timestamp）
- `Message.create_reply()` 签名
- `Message.to_dict()` / `Message.to_prompt()` 签名

**不受保护**：内部格式化细节、`to_prompt` 的具体模板文本。

**理由**：Message 是所有 Agent 之间的唯一通讯协议。改动它等于改动万象的语言。

### 2. `wanxiang/core/agent.py` — BaseAgent 执行主循环 + allowlist

**受保护的公开接口：**
- `BaseAgent.__init__()` 签名
- `BaseAgent.execute()` 签名（接收 Message → 返回 Message）
- `BaseAgent._execute_tool_with_allowlist()` —— allowlist 校验逻辑（工具名必须在 allowed set 里才能执行）
- `BaseAgent._render_team_capability_block()` —— team_context 注入机制的存在性

**不受保护**：prompt 模板文本（writer/reviewer instructions）、CLI tool protocol JSON 格式、LLM 调用细节。

**理由**：execute 定义了 Agent "怎么跑"，allowlist 定义了 Agent "能用什么"。这两个是安全边界。

### 3. `wanxiang/core/pipeline.py` — 三种 workflow 模式

**受保护的公开接口：**
- `WorkflowEngine.__init__()` 签名（required: `agents`, `plan`；optional 参数可加可调，但不能改变默认行为）
- `WorkflowEngine.run()` 签名
- 三种模式的存在性：`_run_pipeline`, `_run_review_loop`, `_run_parallel`
- `_validate_plan()` —— 确保 execution_order 里的 agent 都存在

**不受保护**：事件发射的细节、日志格式、可选 kwarg（例如 Phase 7.2 引入的 `parallel_stagger_s`，默认 8.0，用于平滑 TPM 峰值）。

**理由**：三种 workflow 是万象的编排骨架。可以加第四种，但不能改现有三种的语义。Phase 7.2 的 `parallel_stagger_s` 只在 `_run_parallel` 内部加 `asyncio.sleep(i * stagger)`，不改变任一 mode 的语义——只是把并发启动从"同一 tick 全发"变成"按间隔依次发"。

### 4. `wanxiang/core/tools.py` — ToolRegistry 执行 + 安全守卫

**受保护的公开接口：**
- `ToolSpec` dataclass 字段
- `ToolResult` dataclass 字段
- `ToolRegistry.execute()` 签名 + 内部的校验→执行→超时→截断→审计 pipeline
- `ToolRegistry.filter_for_agent()` —— allowlist 过滤
- `ToolRegistry.register()` —— 注册时的 name/timeout/max_output_bytes 校验
- `_safe_truncate_utf8()` —— UTF-8 安全截断（被 sandbox 也依赖）

**不受保护**：审计日志查询接口（`get_audit_log`）、`get_tool_groups`（Phase 5 加的辅助方法）。

**理由**：ToolRegistry.execute 是"工具调用必须经过的唯一大门"。allowlist、schema 校验、超时、输出截断、审计全部在这条路径上。绕过它 = 绕过安全。

### 5. `wanxiang/core/sandbox.py` — 进程级隔离

**受保护的公开接口：**
- `SandboxExecutor.__init__()` 签名
- `SandboxExecutor.execute()` 签名 + 内部的 tempdir→写入→subprocess→timeout→清理 pipeline
- `SandboxExecutor._scrubbed_env()` —— 环境变量白名单（只保留 PATH / PYTHONPATH / LANG / LC_*）

**不受保护**：默认超时值、默认输出上限、日志格式。

**理由**：sandbox 是 SkillForge 生成代码和系统其他部分之间的隔离层。env 白名单保证 API key 不泄露，tempdir 保证文件系统不污染。

## 变更流程

1. Pre-commit hook (`.githooks/pre-commit`) 检测到上述文件被修改时，会打印警告并要求 `ALLOW_CORE_CHANGE=1` 环境变量才能继续
2. `tests/test_immutable_core.py` 锁定所有受保护接口的函数签名；签名变更会导致 CI 失败
3. 签名锁定测试失败时，**先更新 IMMUTABLE_CORE.md 说明为什么需要变更**，再更新测试中的期望签名

这不是"禁止改动"，是"改动必须经过思考和记录"。
