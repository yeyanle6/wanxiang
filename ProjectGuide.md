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

### Phase 3C：接入真实外部 MCP server
**目标**：让 Agent 能调用**本地子进程 MCP server 提供的工具**（不止 Claude API 的 server-side tools）。

关键文件：
- `core/mcp_client.py` —— JSON-RPC 2.0 协议层 + stdio transport（reader-task pattern，按 id 路由响应）
- `core/mcp_bridge.py` —— MCP tools → ToolSpec（schema 透传），handler 包装 call_tool + text extraction
- `core/mcp_loader.py` —— YAML 配置解析 + `MCPPool` 管理多 server 生命周期
- `server/app.py` —— FastAPI `lifespan` 绑定 pool 启停

分三步递进：

- **3C.1 协议层**：stdio JSON-RPC 客户端。关键是 reader-task pattern——一个后台 task 消费所有出站字节，按 id 把响应路由到对应的 Future（见 D9）。8 个单测用内存 transport 覆盖握手、tools/list、tools/call、error、timeout、close 等边界
- **3C.2 桥接 + 生命周期**：`register_mcp_tools` 把 `tools/list` 结果转成 ToolSpec，**inputSchema 原样透传**（MCP 的 JSON Schema 格式和 Claude API 的 input_schema 一致，无需转换）；`MCPPool` 绑定 FastAPI lifespan，shutdown 时优雅关闭子进程。配置走 `configs/mcp.yaml`（gitignored），模板在 `mcp.yaml.example`
- **3C.3 Director 感知 + allowed_agents ACL**：ToolSpec 加 `group` 字段（builtin/server 分组），planner prompt 按源分组展示工具**带描述**——Director 现在能看到 "filesystem (MCP server):\n  - read_text_file: Read text file..." 这样的块，自主决定给哪个 agent 分配。新增 `allowed_agents` 字段（名字级 ACL），`_apply_tool_restrictions` policy 在每个规划出口强制裁剪

**收尾两个关键修复**：
- **CLI MCP 隔离**（D10）：Claude CLI `-p` 模式**自动加载用户级 MCP server**（Notion/Gmail/...），导致 tool loop 在两套 MCP 交叉时 hang 5+ 分钟。`--strict-mcp-config --mcp-config <empty>` 隔离
- **reviewer peer 识别泛化**（D11）：Director 命名 producer 为 "analyzer" 不匹配 writer 关键字，team_context 找不到 peer 就静默跳过 capability block。fallback 到 `execution_order[0]` + 新增 source-bounded 评审规则（registry tools 限定下不苛求源外细节）

真实端到端验证：跑 `@modelcontextprotocol/server-filesystem` + 任务"读 demo.txt 并总结"。Director 自动分配 `read_text_file` 给 analyst，工具 9ms 返回真实内容，reviewer 一轮 SUCCESS，总耗时 85s。

### Phase 3D：工具层加固
**目标**：Phase 4 要让 LLM 动态生成工具。如果工具层本身校验松、输出不限、无审计，那从 Synthesizer 出来的 bug 代码会直接污染引擎。

关键文件：`core/tools.py`（新增 / 修改）、`server/app.py`（`/api/tools/audit`）。

三步渐进加固：
- **Step 1 · JSON Schema 完整校验**：扔掉手写的 `_validate_arguments` + `_matches_type`（~40 行），换成 `jsonschema` 库。之前 6 类约束（enum / minimum / pattern / nested required / additionalProperties / anyOf）全部被静默放行；现在都强制生效，且错误消息带字段路径（"at 'user.age'"）
- **Step 2 · 输出守卫**：`ToolSpec.max_output_bytes`（默认 50KB），`_safe_truncate_utf8` 在 UTF-8 字节边界切断（`errors='ignore'` 丢尾部残字节，不抛 `UnicodeDecodeError`），超限后在末尾追加 `[Output truncated: <orig> bytes → <limit> bytes]` 告知 LLM。保护 context window 不被 runaway handler 撑爆
- **Step 3 · 环形审计日志**：`collections.deque(maxlen=1000)` 记录 `ToolCallRecord`（timestamp / success / elapsed_ms / input_bytes / output_bytes / truncated / error），`/api/tools/audit?limit=&tool=` endpoint 按需查询。为 trace mining 提供结构化工具层数据

### Phase 4：运行时工具合成（Level 1 能力组合式自进化）
**目标**：让系统在运行时为自己造工具。不是"改 Agent 的 prompt"或"调参数"，而是真的**注册新的 Python callable** 到 Registry。

关键文件：
- `core/sandbox.py` —— 进程级隔离的 pytest 执行器（12 tests）
- `core/skill_forge.py` —— Synthesizer + Sandbox + Registry 的编排闭环（14 tests）
- `configs/agents/skill_synthesizer.yaml` —— JSON-only 输出契约的 agent 定义
- `core/factory.py` —— `needs_synthesis` 协议 + synthesis 阶段 + max_tool_rounds 下限（7 tests）
- `server/app.py` —— FastAPI lifespan 挂载 SkillForge（feature-flagged）

分 4 个递进 commit：

**4.1 SandboxExecutor**：tempfile.mkdtemp → 写入 handler.py + test_handler.py → asyncio spawn `python -m pytest -q` → `asyncio.wait_for` 强制超时 → `_safe_truncate_utf8` 截断 stdout/stderr → shutil.rmtree in finally。关键安全措施：env 只保留 PATH/PYTHONPATH/LANG/LC_*（ANTHROPIC_API_KEY 等 secrets 绝不泄露给子进程）、`stdin=DEVNULL`（防 `input()` 挂起）、cwd 锁 tempdir。12 个单测覆盖 passing / failing / syntax error / no tests collected / timeout / env scrub / 大输出截断 / tempdir 清理。

**4.2 SkillForge**：`ForgeResult` + `ForgeAttempt` 数据结构记录每轮尝试（为 trace mining 预留）。`parse_synthesizer_response` 三级 fallback（raw / markdown fence / 贪婪 `{...}`）。`forge()` 循环：run synthesizer → parse → name collision check → sandbox → success 则 `exec()` handler 绑定到 `ToolSpec(group="synthesized")` 注册 / failure 则构建结构化反馈给下一轮。14 个单测包括"首次失败 → pytest stderr 反馈 → 第二轮修正 → 通过"的核心闭环。

**4.3 Director 感知 + Factory 触发**：planner prompt 加"OPTIONAL Runtime tool synthesis"段落（告知只用于 pure-Python deterministic capabilities，不要用于网络/文件系统）。`TeamPlan.needs_synthesis: list[SynthesisRequest]`。`_run_synthesis_stage` 串联 forge 调用，成功后把占位的 `suggested_name` 替换为真实注册的 `tool_name` 到 agent 的 `allowed_tools`。

**4.4 Server 接线 + max_tool_rounds 下限**：`WANXIANG_ENABLE_SKILL_FORGE=1` 环境变量开启；`/api/skill-forge/status` 端点。Policy 层新增 `MIN_ROUNDS_FOR_TOOL_USERS = 15`，Director 默认给 5 且对批量任务偏小——此 floor 保证有工具的 agent 至少能跑 15 轮 tool loop（D14）。

真实端到端验证（CLI 模式，批量任务）：
> "请把下列 10 个中文数字批量转换成阿拉伯数字，然后求和：..."

- Director 识别缺口 → `needs_synthesis: [{"requirement": "convert chinese numerals to arabic", "suggested_name": "chinese_numeral_to_int"}]`
- SkillForge 调 synthesizer → LLM 一次给出完整 handler + pytest（含"零"、"万"等 edge case）
- Sandbox 跑 pytest → 通过 → 注册
- converter agent 调用 10 次，全部返回正确结果（包括 `一万零二十 → 10020`、`三十万五千 → 305000`、`七千零八 → 7008` 这些含零/大位数的 tricky case）
- 总和 341,873 正确

**没有人写一行中文数字转换的代码**。这是 Level 1 能力组合式自进化的首次实现。

### Phase 5：离线 trace mining（Level 2 的观察层）
**目标**：系统积累了几十个 run 之后，人眼已经看不出规律了。需要一个离线的数据聚合层，把 `runs.jsonl` + 工具审计日志 + synthesis_log 压缩成结构化报告，回答"系统在哪些地方反复失败 / 哪些工具没人用 / reviewer 收敛得如何"这类问题。这是 Level 2 自进化的**观察部分**——LLM 解读层（prompt self-tuning）要等数据积累够了再做。

关键文件：
- `wanxiang/core/trace_mining.py` —— `TraceMiningReport` dataclass + `mine_traces()` 纯函数（22 tests）
- `wanxiang/server/app.py` —— `GET /api/trace/mining` endpoint（6 tests）
- `wanxiang/server/models.py` —— Pydantic response schema 镜像 `TraceMiningReport` 结构
- `tests/fixtures/trace_mining/` —— 手写 5 个 run + 审计日志 + 合成日志 fixture

分两个 commit 递进：

**5.1 数据层**：`TraceMiningReport` 9 个维度（final_status 分布 / workflow mix / 关键词聚类失败模式 / 按 group 分类的工具使用 / synthesis 成功率 / agent naming 分布 / slowest agents / reviewer 收敛桶 / 时间窗口信息）。关键设计决策（见 D15）：**关键词匹配而非语义聚类**（LLM 贵且不稳）、**tool_groups 可选 map**（让调用者注入分类而非硬编码）、**extra_failure_keywords 可扩展**（新失败模式不改源码）、**after/before 窗口过滤**（避免 `runs.jsonl` 膨胀后全量扫描）。22 个测试覆盖每个维度、窗口过滤、JSON 可序列化、空输入边界。

**5.2 Endpoint**：`GET /api/trace/mining?after=&before=` 走 Pydantic response model。支撑改动极小——`ToolRegistry.get_tool_groups()` 导出 `{name: group}`、`RunManager.read_raw_history()` 公开历史读取。6 个端点测试用 fake RunManager 直调 handler 协程，绕开真实 lifespan 的 MCP 子进程启停（`wanxiang.server.__init__` 把 `app` 重绑定成 FastAPI 实例，需要走 `sys.modules["wanxiang.server.app"]` 拿到模块本身）。

**真实生产数据验证**（23 个真实 run 首次自省）：
- `reviewer_convergence` 暴露 **8 个 review_loop 里 4 个 never_converged**（50%）——比预期差，值得专项分析
- `tool_usage` 里 `chinese_numeral_to_int` 显示为 `group=unknown`——不是 mining bug，是"合成工具跨进程不持久化"的必然后果。Mining 在这里的价值**恰恰是让一个未完成功能的必要性浮出水面**
- `synthesis_stats=0` 同源——`factory.synthesis_log` 只在进程内存里
- `agent_naming` 统计实锤了 D11 fallback 的价值：Director 实际用过 `analyzer` / `analyst` / `converter` / `time_writer` / `writer` / `synthesizer`，光靠 writer 关键字根本覆盖不全
- `common_failure_patterns` 顶部两位都是 "Claude CLI call failed" / "Not logged in" 各 6 次——开发期 CLI 认证断过一段时间。提示未来可以加一层 `infra_errors` vs `logic_errors` 分桶

这是项目第一次用 mining 数据反过来驱动开发决策："合成工具持久化"从 Phase 4 的 nice-to-have 变成下一阶段的首位 backlog，理由不是主观判断而是客观观测。

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

### D9 · MCP client 的 reader-task pattern
**问题**：JSON-RPC 客户端怎么处理并发请求 + server 主动通知？

**候选**：
- (A) 请求后 `await transport.readline()` 拿下一行当响应
- (B) 每次请求起一个独立 `readline` task
- (C) 一个后台 task 消费所有字节，按 id 路由到 Futures

**选择**：C。

**理由**：A 有两个致命缺陷——server 在响应前先发 notification（log/progress）时，`readline` 拿到的是 notification 不是响应；并发请求会互相阻塞。B 的问题是多个 readline 之间谁先拿到响应不确定，可能 A 请求的响应被 B 的 readline 抢走。C 把"读"和"请求"解耦：每个请求持有一个 Future，reader 按 id 给对应 Future `set_result`。这是 HTTP/2 stream multiplexing 的同构模式。

**代价**：close 时必须 fail 所有 pending Futures，避免调用方永久挂起。`test_close_cancels_pending_requests` 和 `test_server_eof_fails_pending_requests` 守住这条不变量。

### D10 · Claude CLI MCP 隔离
**问题**：CLI 模式下 `claude -p` 会**自动加载用户级 MCP server**（Notion/Gmail/oh-my-claudecode/...），当我们同时把自己的 MCP registry 也塞进 tool loop 时，两套 MCP 语义混乱，Claude CLI 进入内部 tool round 等网络 IO 永不返回（观测到 5+ 分钟 hang、CPU 接近 0）。

**候选**：
- (A) 忽略——反正 API 模式能跑就行
- (B) 给 `claude -p` 加 `--strict-mcp-config --mcp-config <empty>`，强制只用空配置
- (C) 用 `--bare` 模式，跳过所有 hooks/plugin/MCP

**选择**：B。

**理由**：A 不可接受——CLI 模式是不需要 API key 就能跑的核心卖点。C 副作用太大，`--bare` 会同时强制 `ANTHROPIC_API_KEY` 作为唯一认证（禁用 OAuth 和 keychain），正好和"CLI 模式 = 不需要 API key"的目标背道而驰。B 精准——只隔离 MCP，保留 OAuth 认证、hooks、settings。实测空配置下 `claude -p` 秒级返回，hang 彻底消除。

**代价**：每次 CLI 调用多两个参数（`--strict-mcp-config --mcp-config <path>`）+ 一个临时 JSON 文件。值得。

### D11 · team_context peer 识别的 fallback + source-bounded 评审
**问题**：reviewer 的 tool-aware 规则依赖识别"队友里的 producer"，但识别靠关键字（writer/write/author/撰写/...）。当 Director 命名 "analyzer" / "analyst" / "summarizer" 时，关键字匹配不到，capability block 被静默跳过，reviewer 回到"通用严格评审"模式，面对"读文件总结"这类任务仍然要求"必须引用 IDC/Gartner"。

**候选**：
- (A) 不断扩充关键字列表
- (B) 关键字失败时回退到 `execution_order[0]`
- (C) 让 Director 在 plan 里显式标注 agent 的 `role` 字段

**选择**：B。

**理由**：A 是 losing battle——Director 每次可能用新名字；穷举不现实。C 需要协议改动（TeamPlan 加字段），且要让 Director 可靠填它。B 最小改动：review_loop 里 `execution_order[0]` 语义上**永远**是 producer。在关键字匹配失败时 fallback，保留原先的匹配优先级作为快路径。

**同时加强**：当识别到的 producer 拿的是 registry tools（如 `read_text_file`）时，reviewer 规则额外明确"你的知识受源材料边界约束，不要苛求源外细节；若 draft 显式声明 Limitations/Scope，视作正确 scoping，返回 SUCCESS"。这样面对"读文件总结"这类 input-bounded 任务，reviewer 不会陷入"要求源外技术细节"的死循环。

**代价**：reviewer prompt 正在变成"各种情境下的评审手册"。长期看需要抽成独立的 reviewer policy module。

### D12 · Sandbox 选择：进程级 vs Docker
**问题**：Phase 4 的 Synthesizer 会让 LLM 生成并执行 Python 代码。用什么级别的隔离？

**候选**：
- (A) 直接 `exec()`，不隔离
- (B) 进程级：tempfile.TemporaryDirectory + subprocess + env scrub + timeout
- (C) Docker 容器
- (D) nsjail / seccomp / gVisor

**选择**：B。

**理由**：A 不可接受——LLM 生成的代码有权访问 `ANTHROPIC_API_KEY`、文件系统、网络。D 太重，对 Phase 4 MVP 是过度工程。C 的启动开销（秒级）在 forge 重试循环里累积很可观（每次 retry 都多等 1-2 秒），且需要宿主机装 Docker 守护进程，部署成本上升。B 刚够——handler_code 被约束为 pure-Python + stdlib + no I/O（在 synthesizer YAML 的 hard constraints 里写死），子进程 env 只保留 PATH/PYTHONPATH/LANG/LC_*（实测验证 `ANTHROPIC_API_KEY` 在子进程里不可见），tempdir + cwd 隔离文件系统。

**代价**：如果未来放宽 handler 约束（允许导入第三方包 / 文件系统写入），B 的防线会不够。那时候该重新评估 C 或 D。现在不做是对的——**YAGNI 原则**。

### D13 · SkillForge 的失败反馈循环
**问题**：Synthesizer 第一次生成错代码是常态。怎么让它修正？

**候选**：
- (A) 固定重试 N 次，每次让它重新生成
- (B) 每次把 pytest stderr/stdout + 结构化 feedback 喂给 synthesizer，让它基于错误信息修正

**选择**：B。

**理由**：A 就是随机撞运气，第二次生成和第一次没有因果关系。B 让每次 retry 都是**条件独立 retry → 条件相关 debug**。`SkillForge._build_feedback` 按 SandboxResult 的形状构造不同措辞：
- `timed_out=True` → "Your test timed out. Avoid infinite loops..."
- 有 `stderr` → 贴上 pytest traceback
- 有 `stdout` → 贴上 pytest 输出
- 名字冲突 → "The tool name '...' is already taken. Choose a different name"
- 解析失败 → "Return a single JSON object; do not include any prose."

**测试验证**：`test_forge_recovers_from_failing_test_on_retry` 明确 assert 第二轮的 prompt 里包含 "Feedback on your previous attempt" 且携带 pytest 的具体 test 名——这是证明闭环有因果关系、不是随机撞运气的决定性证据。

**代价**：反馈作为 user turn 发给 synthesizer 增加 token 开销。实测 3 轮以内收敛，成本可接受。

### D14 · max_tool_rounds 的硬 floor
**问题**：Phase 4 端到端首次真实跑批量任务（10 个数字转换）时，converter 在 3 轮 tool call 后就 `Exceeded max_tool_rounds=3`，整个 pipeline ERROR。每次 tool call 都成功（83ms / 1ms / 2ms），真正的瓶颈是 round 上限。

**候选**：
- (A) 接受这是 Director 的职责，改 planner prompt 让它给批量任务设更高的 max_tool_rounds
- (B) Policy 层兜底：有工具的 agent 自动至少 15 轮
- (C) 动态估算：根据任务文本里"批量 / 10 个 / 每个"之类的关键字自动调整

**选择**：B。

**理由**：A 不可靠——Director 会复制 planner prompt 里的 `"max_tool_rounds": 5` 示例，对批量任务判断不准。这和 D4（policy 兜底 vs 纯 Director 规划）是同一个模式：**LLM 的结构化约束能力有上限，重复出现的失败模式交给代码层兜底更可靠**。C 太花哨，且启发式规则本身容易出错。B 的代价只是 tool-less agent 的少量重复检查（no-op），换来批量 workload 的可用性。

**测试验证**：`test_tool_users_get_max_tool_rounds_lifted_to_floor` 等 3 个测试守住该行为：有工具 + Director 给 3 → 抬到 15；已经 50 的保留 50；无工具不动。

**代价**：15 是个经验魔数。未来可能需要按 tool 类型分级（web_search 重、echo 轻）。不急着做。

### D15 · Trace mining：纯数据层先行，LLM 解读层延后
**问题**：Phase 5 的目标是"让系统观察自己"。最直觉的做法是写一个 meta-agent，吃完整的 `runs.jsonl` 直接用 LLM 产自然语言洞察。但这样做之前要回答：数据从哪来？怎么保证洞察有统计显著性？

**候选**：
- (A) 直接上 meta-agent：读 JSONL → 调 LLM → 输出"系统分析报告"
- (B) 先做**纯数据聚合层**输出结构化 `TraceMiningReport`，meta-agent 留到数据足够再做
- (C) 混合：数据层 + 一个薄 LLM 层马上套上去

**选择**：B。

**理由**：
1. **LLM 层的价值由输入数据质量决定**。当时 `runs.jsonl` 只有 23 个 run，其中 13 个是 CLI 认证 infra error——有效样本太少，LLM 再聪明也只能产生泛泛之论。B 让 LLM 层在有足够信号时再做，不在数据稀疏时浪费调用
2. **纯数据层立刻可用**——`curl /api/trace/mining | jq .` 直接能看清系统状态，前端 dashboard 也能直接渲染。不需要任何 LLM 就有价值
3. **CI 稳定**。纯函数 + dataclass 可以跑 22 个精准测试覆盖每个维度；LLM 层的测试必然松（只能 assert "提到了 reviewer" 这种模糊条件）
4. **给 LLM 层提供稳定输入契约**。未来的 meta-agent 消费的是 `TraceMiningReport` 的 JSON schema，不是 raw JSONL——数据层变更时 LLM 层不用跟改 prompt

**关键设计选择**（都在 D15 这条决策之内，不单独开条目）：
- **关键词聚类 over 语义聚类**：`DEFAULT_FAILURE_KEYWORDS` 是我们真实遇到过的错误字符串（`Exceeded max_tool_rounds` / `Claude CLI call failed` / `Not logged in` / ...），literal 匹配 + Counter 排序。语义聚类要等 LLM 层做
- **`tool_groups` 可选注入 map**：不在 mining 里硬编码"哪些工具是 MCP / 哪些是 synthesized"，让调用者从 `ToolRegistry.get_tool_groups()` 传进来。Mining 只做聚合，不做分类
- **`extra_failure_keywords` 可扩展**：新失败模式可以在调用时追加，不必改源码
- **时间窗口过滤**：`runs.jsonl` 无轮转会膨胀，`after` / `before` 保证 mining 永远可用

**代价**：短期看"少了 AI 含量"——纯统计不如自然语言报告酷。但长期看这是唯一对的做法。等 `runs.jsonl` 积累 50+ 真实 run、infra_errors 和 logic_errors 能分开后，meta-agent 再上场会产出质量高得多的洞察。

**测试验证**：`test_default_failure_keywords_nonempty` + `test_extra_failure_keywords_are_honored` + `test_tool_classification_distinguishes_native_mcp_synthesized` 守住 D15 的三个可扩展性保证。

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
- **Synthesized tool 持久化**（下一步，由 Phase 5 mining 数据直接驱动）：目前合成的工具只在当前 run 内生效。需要把 `handler_code` + `test_code` 写入 `skills/` 目录 + 人工审核流程（审核后持久化，下次启动自动加载）。Mining 报告里反复出现的 `group=unknown` 和 `synthesis_stats=0 after restart` 就是这个缺口的客观信号
- **Prompt self-tuning agent**（Level 2 的解读层）：trace mining 的 LLM 消费者，基于 `TraceMiningReport` 产出 policy 调优建议。按 D15，延后至 `runs.jsonl` 积累 ≥50 个真实（非 infra error）run 后再做
- **UI 面板接入 mining / MCP pool / SkillForge 状态**：endpoint 都在（`/api/trace/mining` / `/api/mcp/wanxiang-pool` / `/api/skill-forge/status`），前端还没展示
- **MCP SSE transport**：目前只支持 stdio。用户 Claude CLI 连接的 Notion / Gmail / Calendar 等云端 MCP server 走 SSE 协议，接入需要在 `mcp_client.py` 加一个 `SSETransport` 实现
- **LICENSE 文件**：README 里声明了 MIT，但仓库里还没实际的 LICENSE 文件
- **深色模式**：前端只有亮色

### 5.2 已知但未修
- **Policy 层膨胀**：`factory.py` 的 policy 逻辑已 200+ 行，未来需要拆成独立的 `policies.py`，支持 pluggable 规则。
- **CLI 模式无 native tools**：这不是能修的 bug，是协议限制。team_context 的补丁让它可用但不理想，长期看需要鼓励用户配 API key。
- **无认证**：FastAPI server 没有 auth 层，任何能访问 8000 端口的人都能创建 run。个人/本地用场景下 OK，正式部署前需要加。
- **JSONL history 无上限**：`data/runs.jsonl` 会无限增长。Phase 5 mining 的 `after` / `before` 窗口过滤让这个问题在读侧可控，但落盘侧仍需要加轮转或清理策略。
- **Failure 分类粒度**：mining 的 `common_failure_patterns` 目前把 infra error（CLI 认证）和 logic error（max_tool_rounds 超限）混在一起。需要加一层 `infra_errors` vs `logic_errors` 分桶，两类问题的修复路径完全不同。

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
2. `wanxiang/core/agent.py` —— BaseAgent 的 execute 主循环 + tool loop + team_context
3. `wanxiang/core/factory.py` —— Director 怎么规划 + policy 兜底 + tool 展示分组
4. `wanxiang/core/pipeline.py` —— 三种 workflow 怎么编排
5. `wanxiang/core/tools.py` + `builtin_tools.py` —— 工具注册和执行机制（group / allowed_agents / max_output_bytes / 审计日志）
6. `wanxiang/core/sandbox.py` —— 进程级隔离的 pytest 执行器（env scrub / timeout / tempdir / UTF-8 截断）
7. `wanxiang/core/skill_forge.py` —— Synthesizer + Sandbox + Registry 的生成→测试→反馈→注册闭环
8. `wanxiang/core/mcp_client.py` —— JSON-RPC 2.0 + stdio transport + reader-task pattern
9. `wanxiang/core/mcp_bridge.py` —— MCP tools → ToolSpec（schema 透传）
10. `wanxiang/core/mcp_loader.py` —— YAML 解析 + MCPPool 生命周期
11. `wanxiang/core/llm_client.py` —— 双后端 + mode 解析 + CLI MCP 隔离
12. `wanxiang/core/trace_mining.py` —— 纯数据聚合层（9 个维度 / 关键词聚类 / 可插拔分类 / 时间窗口）
13. `wanxiang/server/runner.py` —— 把引擎事件转成前端事件流
14. `wanxiang/server/app.py` —— FastAPI endpoints + WebSocket + lifespan 绑定 MCPPool + SkillForge + mining endpoint
15. `wanxiang/server/models.py` —— Pydantic response schema（含 `TraceMiningResponse` 等）
16. `configs/agents/skill_synthesizer.yaml` —— synthesizer 的 JSON-only 输出契约
17. `wanxiang-ui.jsx` —— 前端单文件，读完前 300 行的 helpers 就能跳着看

测试代码（`tests/`）按核心程度：
- `test_message.py` / `test_pipeline.py` / `test_factory.py` —— 核心行为
- `test_tools.py` / `test_agent_tools.py` —— 工具系统完整边界 + reviewer capability block
- `test_sandbox.py` / `test_skill_forge.py` —— Phase 4 自进化层（sandbox 隔离 + forge 闭环）
- `test_trace_mining.py` / `test_trace_mining_endpoint.py` —— Phase 5 观察层（纯数据聚合 + HTTP endpoint）
- `test_mcp_client.py` / `test_mcp_bridge.py` / `test_mcp_loader.py` —— MCP 协议、桥接、生命周期
- `test_llm_client_modes.py` / `test_mcp_status.py` —— 基础设施
- `test_server_events.py` / `test_runner_tool_events.py` —— 事件流一致性
- `tests/fixtures/trace_mining/` —— 5 个手写 run + 审计日志 + 合成日志 fixture
