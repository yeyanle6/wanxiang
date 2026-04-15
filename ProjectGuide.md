# ProjectGuide · 万象架构演进记录

> 这份文档是万象（Wanxiang）项目的**架构演进日志**，按关键决策点而非代码模块组织。目标是让未来的自己（或新协作者）在两周/两年后读完能立刻理解"为什么这样设计，代价是什么"。
>
> 不是 API 文档（看 README）、不是实现细节（看代码本身）、不是路线图（看 README Roadmap）。

---

## 0. 起源

项目起点是一篇关于 AI 编程的博文。博文的核心观察：LLM 单体调用已经不够用，真正的生产系统需要多个 Agent 协同。但多数开源实现停留在"N 个硬编码 Agent + 主调度"层面，既不 AI-native 也不灵活。

从读博文到写代码之间，定下一个主要目标：**做一个真正把"多 Agent 协同"当作一等公民的编排引擎**——团队结构、工具分配、评审标准都由 LLM 在运行时决定，而不是程序员在编译时预先写死。

---

## 1. 三个设计支柱

所有后续决策都能追溯到这三个想法。它们不是同时出现的，而是在 Phase 1–3 逐步浮现、互相验证。

### 支柱 1：AI-native Message 协议
**消息即任务**。Agent 之间不传递函数参数或 JSON schema，而是传递一个结构化消息（`intent` / `content` / `status` / `context` / `turn` / `metadata`）。LLM 既能**读**（作为输入理解任务）也能**写**（作为输出）。

对比传统做法：
- ❌ 传统：每个 Agent 定义自己的 input schema 和 output schema，前一个 Agent 的输出要适配下一个 Agent 的输入
- ✅ 本项目：所有 Agent 共享同一个 Message 类型，语义靠 `intent` 字段描述，状态靠 `status` 枚举传递

**代价**：LLM 偶尔会生成不符合期望的 `status`，需要在 BaseAgent 层做状态推断兜底（`_infer_status`）。但这种松耦合换来了 Agent 之间几乎零接口冲突。

### 支柱 2：动态 Agent 生成
**没有预定义的 `WriterAgent` / `ReviewerAgent` 子类**。只有一个 `BaseAgent`。它的"身份"由两部分组成：
- `base_identity`（静态）：角色描述（"You are a reviewer that..."）
- `persona_prompt`（动态）：每次任务执行前，BaseAgent 让 LLM 基于当前任务**自己生成一个任务适配的 persona**

Director Agent (AgentFactory) 在接到用户任务时，产出一个 `TeamPlan`：
```json
{
  "workflow": "review_loop",
  "agents": [
    {"name": "writer", "duty": "...", "base_identity": "..."},
    {"name": "reviewer", "duty": "...", "base_identity": "..."}
  ],
  "execution_order": ["writer", "reviewer"],
  "max_iterations": 3
}
```

**代价**：Director 会犯错（生成 synthesizer 却不叫 writer；给 reviewer 分配 web_search；用 pipeline 做本该 review_loop 的内容任务）。这催生了**策略兜底层**（第 3 节会展开）。但换来的灵活性极高：加一个新场景不需要写新类，只要改 Director prompt。

### 支柱 3：工具感知协作（Tool-aware Collaboration）
**每个 Agent 执行时携带一份 "team_context" 快照**——队友是谁、各自有什么工具、当前 LLM 模式是什么。

这个支柱最后才浮现（Phase 3B++），但本质上是前两个支柱的必然推论：如果 Agent 身份是动态的，那么不同 run 里 Reviewer 面对的 Writer 能力可能完全不同——有时候 writer 有 web_search 能搜实时数据，有时候没有。Reviewer 的评审标准必须感知这个差异，否则会提出物理上不可满足的要求（"请引用 Gartner 2026 报告"，但 writer 根本无法联网）。

---

## 2. 演进时间线

### Phase 1：Message 协议 + BaseAgent
**目标**：跑通"单 Agent 能接收任务、调用 LLM、返回带状态的结果"。

关键文件：`core/message.py`、`core/agent.py`。

这一阶段做对的事：
- `Message.create_reply()` 自动串联 `parent_id` 和 `context`，形成 handoff 链
- 引入 `MessageStatus` 枚举（SUCCESS / NEEDS_REVISION / ERROR）作为 review loop 的收敛信号
- `BaseAgent.refine_persona()` 在每次 `execute()` 内部用 LLM 生成任务 persona

### Phase 2：WorkflowEngine + Factory
**目标**：两个 Agent 能协同（先做 pipeline，后加 review_loop）。

关键文件：`core/pipeline.py`、`core/factory.py`。

这一阶段的关键转折：**放弃预定义子类**。原本想写 `WriterAgent(BaseAgent)` / `ReviewerAgent(BaseAgent)`，写到一半发现——如果 Director 能根据任务生成任意角色，那 Writer/Reviewer 只是 `base_identity` 不同的 BaseAgent 而已。把子类删掉，代码立刻瘦身一半，灵活度反而更高。

`WorkflowEngine` 的三种模式（pipeline / review_loop / parallel）按复杂度递进：
- `pipeline` —— 最简，顺序调用，遇错终止
- `review_loop` —— 两个 Agent 反复迭代，靠 reviewer 的 `status` 收敛
- `parallel` —— `asyncio.gather` 并发，失败分支不拖累其他分支，最后 synthesizer 合并

### Phase 3A：实时可观测
**目标**：用户不再是"提交任务→等 2 分钟→看结果"，而是**实时看到每个 Agent 的思考过程**。

关键文件：`server/app.py`、`server/runner.py`、`server/events.py`、`wanxiang-ui.jsx`。

这一阶段的关键设计：**事件驱动而非轮询**。`WorkflowEngine` 接受 `on_event` 回调，在每个步骤前后发射事件；`RunManager` 把引擎事件转成 `RunEvent` 推进一个 `asyncio.Queue`；FastAPI WebSocket endpoint 作为 queue 的消费者把事件流送到浏览器。

8 种事件类型覆盖了整个生命周期：`run_started` / `agent_started` / `agent_completed` / `tool_started` / `tool_completed` / `iteration_completed` / `parallel_completed` / `run_completed`。

前端（1700+ 行单文件 React + Babel in-browser）实现了：
- 时间线卡片
- DAG 三种布局（pipeline / review_loop / parallel）+ 实时状态高亮
- 工具调用子步骤缩进展示
- 初稿终稿 Diff
- Reviewer Feedback 聚合
- 耗时分析条形图
- 历史记录持久化（JSONL）+ 回放

### Phase 3B：工具系统 + 双 LLM 通道
**目标**：Agent 不只是 LLM 独白，还能调用工具；同时支持 API 和 Claude CLI 两种后端。

关键文件：`core/tools.py`、`core/builtin_tools.py`、`core/llm_client.py`、`server/mcp_status.py`。

这一阶段分三层递进：

1. **ToolRegistry**：本地工具注册、allowlist 过滤、timeout 控制、schema 校验、异常封装。
2. **Claude native tools**（server-side）：`web_search_20250305` 等 Anthropic API 原生工具，不走本地 handler，靠 `stop_reason` 分流（`end_turn` → 直接返回；`tool_use` → 走本地执行）。
3. **LLMClient 抽取**：`auto` / `api` / `cli` 三模式。`api` 走 Messages API；`cli` 走 `claude -p` 子进程 + JSON tool protocol；`auto` 自动检测 `ANTHROPIC_API_KEY` 和 `claude auth status`。

**CLI tool mode 是一个小发明**：Claude CLI 不支持原生 tool_use blocks，我们设计了一套极简 JSON 协议（`{"action":"tool"/"final",...}`）让 CLI 模式也能走工具循环。代价是 CLI 模式不支持 server-side tools（如 web_search），但 registry tools 完全可用。

### Phase 3B++：team_context（工具感知协作）
**目标**：解决一个真实的用户 bug——同一个任务（"从多个角度研究 2026 年 AI Agent 趋势"）在 CLI 模式下跑，reviewer 死咬"必须引用 Gartner/IDC/Nature"，writer 无法满足，3 轮全 NEEDS_REVISION，最终 final_status=NEEDS_REVISION。

关键文件：`core/agent.py`（新增 `team_context` 字段 + `_render_team_capability_block`）、`core/factory.py`（`instantiate_team` 注入 team snapshot）。

修复思路（也是支柱 3 的具象化）：**reviewer 的评审标准必须感知 writer 的工具可达性**。在 reviewer prompt 里注入：
```
Team tool availability:
- writer: native_web_search=no, registry_tools=[]
- llm_mode: cli (server-side native tools disabled)

Evaluation rules for THIS run (tool-aware):
- Writer has NO live web_search. Do NOT require specific citations...
- Evaluate instead: structure, internal logical consistency, ...
```

复测结果：同一任务，2 轮迭代直接 SUCCESS，总耗时 194 秒。

---

## 3. 关键架构决策（Decision Log）

### D1 · AI-native Message vs JSON Schema 契约
**问题**：Agent 之间如何传递任务？

**候选**：
- (A) 每个 Agent 有独立的 input/output schema，靠 adapter 适配
- (B) 所有 Agent 共享同一个 Message 类型，语义靠字符串字段表达

**选择**：B。

**理由**：动态 Agent 生成的前提是 Agent 之间不需要静态类型契约。如果选 A，每加一个新角色都要写 adapter；选 B 之后，新角色只是一个 `base_identity` 字符串。

**代价**：LLM 生成的 `status` 偶尔不规范，需要 `_infer_status` 兜底。值得。

### D2 · Dynamic Agent vs Predefined Subclasses
**问题**：不同角色（writer、reviewer、researcher、synthesizer）怎么区分？

**候选**：
- (A) 每个角色一个 `BaseAgent` 子类，各自 override `build_prompt`
- (B) 只有一个 `BaseAgent`，角色差异靠 Director 在 `TeamPlan` 里指定的 `base_identity` 体现
- (C) 混合——一个基础类加少量专用子类

**选择**：B（完全动态）。

**理由**：子类数量会无限膨胀（研究员、合成器、校对员、翻译员……），而它们的核心逻辑其实一样（接消息 → 调 LLM → 返回 Message）。把差异性完全推到 prompt 层，代码库保持极简。

**代价**：
- reviewer / writer 的角色识别靠关键字匹配（`_is_reviewer_role`），不是强类型
- Director 偶尔生成"字面上叫 synthesizer 但职责是 writer"的角色，靠策略层（D4）纠正

### D3 · LLMClient 单独抽取
**问题**：LLM 调用逻辑该放在 BaseAgent 里还是单独抽一层？

**候选**：
- (A) `BaseAgent.call_llm()` 内联实现 Anthropic API 调用
- (B) 独立 `LLMClient` 类，BaseAgent 持有它作为依赖

**选择**：B，早在 Phase 2 末尾就抽了。

**理由**：Phase 3B 要加 CLI 后端时立刻收益——只改一个文件（`llm_client.py`），BaseAgent 零改动。tool_use 循环、mode 解析、resolve_mode 都集中在一处。**单点抽象比四处复制更经得起需求变化**。

### D4 · Policy 兜底层 vs 纯 Director 规划
**问题**：Director 生成的 `TeamPlan` 偶尔不合理（content 任务用 pipeline、parallel 没有 synthesizer、reviewer 被分配 web_search）。怎么办？

**候选**：
- (A) 改进 Director prompt，让它永远规划正确
- (B) 在 Factory 里加一层 `_apply_planning_policies`，对 Director 产出做规则修正
- (C) 放任不管，相信 LLM

**选择**：B。

**理由**：Director prompt 再详尽也有 corner case；靠 LLM 决定"每种任务该用哪种 workflow"既不可靠也难调试。规则修正层作为确定性的兜底，把 LLM 不擅长的结构化约束（"content → review_loop" / "parallel 末位必须是 synthesizer" / "reviewer 不得有 web_search"）用代码实现。

**代价**：policy 层现在有 200 多行代码（`factory.py`），规则越堆越多。未来可能需要把它抽成独立的 `policies.py`，甚至支持 pluggable rule 系统。

### D5 · Claude native tools vs 外部 MCP 接入
**问题**：想让 Agent 能搜索网页，先上哪条路线？

**候选**：
- (A) 走 Claude API 原生的 `web_search_20250305`——服务端执行，无需本地部署
- (B) 接入真正的 MCP server（通过 stdio 或 SSE 协议）——需要进程管理、协议解析

**选择**：A（先做），B 作为 Phase 3C。

**理由**：MCP server 接入是独立的基础设施层，涉及子进程管理、JSON-RPC 协议、连接池保活。这些和"Agent 能搜索"这个核心能力无关。先用 Claude native tool 验证完整链路，把 UI / 事件 / 策略层都跑通，再考虑 MCP。

**代价**：Claude native tools 只能在 API 模式用，CLI 模式不支持。这直接催生了 D6。

### D6 · CLI graceful degrade vs fail-fast
**问题**：CLI 模式遇到 native tools 时怎么处理？

**候选**：
- (A) 立即抛 `RuntimeError`（"Native tools require API mode"）
- (B) 打 warning 日志，静默剥掉 native tools 继续执行
- (C) 策略层提前感知模式，不分配 native tools

**选择**：B + C 双保险。

**理由**：最初是 A（fail-fast）——这在单元测试语境下是"正确"的，因为错误不会被掩盖。但用户实际用 CLI 模式跑时，writer 带 web_search 就立刻崩溃（final_status=ERROR），整个 run 毫无产出。B（fail-soft）更符合用户价值：**拿到一篇没有搜索支撑的报告，好过看到 ERROR**。同时 C 让策略层在规划阶段就感知模式，理想情况下 B 的兜底永远不会被触发，只是以防万一。

### D7 · team_context 的发明
**问题**：Reviewer 在 CLI 模式下反复要求"请引用 Gartner 报告"，writer 物理上无法满足，死循环。

**候选**：
- (A) 在 reviewer prompt 里硬编码 "if CLI, 不要求 citations"
- (B) 建立通用的 team_context 机制，让所有 Agent 都能感知队友能力
- (C) 让 writer 在 CLI 模式下自己写"数据来源声明"，reviewer prompt 里加一条"看到声明就不追究"

**选择**：B。

**理由**：A 和 C 都是局部补丁——只解决了 CLI 模式下的 reviewer 问题。B 是架构升级：**任何 Agent 都能看到队友的工具能力**，未来加新角色（比如 "fact-checker"）也能自动享受这个机制。实际实现只多了 50 行代码，但给整个架构增加了一层"工具感知协作"的能力。

### D8 · WebSocket 事件粒度
**问题**：实时可观测化里，事件该多细？

**候选**：
- (A) 每次 LLM token 都推一个事件（字符级流式）
- (B) 每个 Agent 完成才推一个事件（粗粒度）
- (C) 关键生命周期节点：started / completed / tool_called / iteration_completed

**选择**：C。

**理由**：A 在后端耗资源、前端难渲染、对调试没多大帮助（LLM token 流是线性的，没有结构信息）。B 太粗，看不到工具调用和 review loop 内部节奏。C 是"用户关心的就是这些"——每个状态变化都值得渲染一张卡片。

---

## 4. 偏离原路线图之处

原路线图（Phase 1–3 的早期设想）和实际实现的差异：

| 原计划 | 实际做了 | 原因 |
|---|---|---|
| `WriterAgent` / `ReviewerAgent` 子类 | 只有一个 `BaseAgent` | Director 能动态指定身份后，子类变成冗余 |
| Message 用 Pydantic BaseModel 做强校验 | 用 `@dataclass`，LLM 输出松校验 | Pydantic 对 LLM 生成的 JSON 太严格，不好兜底 |
| 依赖 `anthropic` 官方 SDK | 直接用 `httpx` 调 REST | SDK 对 streaming / CLI fallback 不友好，自己写更自由 |
| 只支持 API 模式 | 加了 CLI 模式 + auto 检测 | 用户有 Claude Code 登录但未必有 API key |
| 预先规划工具系统 | 到 Phase 3B 才引入 | 早期没工具场景也能跑通，过早抽象会浪费 |
| Director prompt 足够好就行 | 需要 policy 兜底层 | LLM 规划能力有上限，结构约束交给代码更可靠 |
| 前端只做简单时间线 | 加了 DAG / Diff / 回放 / 历史 / MCP 状态 | 每次发现某个信息看不清，就加一块面板 |

---

## 5. 待改进 / 未解问题

### 5.1 未完成
- **Phase 3C**：接入真实外部 MCP server（filesystem → web search via MCP → Notion）。引擎层已完全准备好（`ToolRegistry` 的 handler 签名天然兼容 MCP 的 `tools/call` RPC），缺的是 `mcp_client.py` 的 stdio/SSE 协议实现和进程管理。
- **LICENSE 文件**：README 里声明了 MIT，但仓库里还没实际的 LICENSE 文件。
- **深色模式**：前端只有亮色。

### 5.2 已知但未修
- **Policy 层膨胀**：`factory.py` 的 policy 逻辑已 200+ 行，未来需要拆成独立的 `policies.py`，支持 pluggable 规则。
- **CLI 模式无 native tools**：这不是能修的 bug，是协议限制。team_context 的补丁让它可用但不理想，长期看需要鼓励用户配 API key。
- **无认证**：FastAPI server 没有 auth 层，任何能访问 8000 端口的人都能创建 run。个人/本地用场景下 OK，正式部署前需要加。
- **JSONL history 无上限**：`data/runs.jsonl` 会无限增长。需要加轮转或者清理策略。

### 5.3 架构级思考题
- **多 run 并发**：`RunManager` 是单例，多个 run 能并发跑（每个独立的 `asyncio.Task` 和 queue），但没有压测过。未来如果做 SaaS 版本，需要考虑 Redis 队列 / 多 worker。
- **Agent 持久化**：目前每次 run 都是全新的 Agent 实例。如果要做"有记忆的 Agent"（跨 run 维持上下文），需要加一层 Agent state persistence。
- **工具的细粒度权限**：目前 allowlist 是 agent 级别。未来可能需要场景级别（"reviewer 在 iteration 1 时不能用 web_search，iteration 2 才能"）。

---

## 6. 项目价值 / 能用来做什么

对自己：
- **具体范例理解"多 Agent 协同"**：读到论文或文章提到 Agent 编排时，能对应到自己写过的代码
- **工程实践积累**：从博文到可运行产品，走了一遍完整 0→1
- **Prompt 工程实战**：Director prompt、persona prompt、reviewer prompt 各自的迭代过程都是真实训练

对外：
- **开源学习样本**：完整、测试覆盖、可复现，适合任何想理解多 Agent 架构的人 clone 下来玩
- **Phase 3C 之后的潜力**：如果真接入 Notion/filesystem/web search 的 MCP 服务，可以作为一个"个人知识工作编排器"真正用起来

---

## 7. 核心文件地图

学习代码时的推荐阅读顺序：

1. `wanxiang/core/message.py` —— 最短、最基础，先读完
2. `wanxiang/core/agent.py` —— BaseAgent 的 execute 主循环 + tool loop
3. `wanxiang/core/factory.py` —— Director 怎么规划 + policy 兜底
4. `wanxiang/core/pipeline.py` —— 三种 workflow 怎么编排
5. `wanxiang/core/tools.py` + `builtin_tools.py` —— 工具注册和执行机制
6. `wanxiang/core/llm_client.py` —— 双后端 + mode 解析
7. `wanxiang/server/runner.py` —— 把引擎事件转成前端事件流
8. `wanxiang/server/app.py` —— FastAPI endpoints + WebSocket
9. `wanxiang-ui.jsx` —— 前端单文件，读完前 300 行的 helpers 就能跳着看

测试代码（`tests/`）按核心程度：
- `test_message.py` / `test_pipeline.py` / `test_factory.py` —— 核心行为
- `test_tools.py` / `test_agent_tools.py` —— 工具系统完整边界
- `test_llm_client_modes.py` / `test_mcp_status.py` —— 基础设施
- `test_server_events.py` / `test_runner_tool_events.py` —— 事件流一致性
