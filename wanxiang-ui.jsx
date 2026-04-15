const { useCallback, useEffect, useMemo, useReducer, useRef } = React;

function inferDefaultApiBase() {
  if (typeof window !== "undefined" && window.location?.origin) {
    return window.location.origin;
  }
  return "http://localhost:8000";
}

function normalizeEventType(value) {
  return String(value ?? "").trim().toLowerCase();
}

const INITIAL_STATE = {
  status: "idle", // idle | connecting | running | completed | error
  runId: null,
  plan: null,
  events: [],
  replayMode: false,
  replaying: false,
  replayCursor: 0,
  historyRuns: [],
  historyOpen: false,
  historyLoading: false,
  historyError: null,
  historySelectedRunId: null,
  mcpStatus: null,
  mcpLoading: false,
  mcpError: null,
  selectedEventIndex: null,
  taskInput: "",
  apiBase: inferDefaultApiBase(),
  error: null,
};

function extractPlanFromEvents(events) {
  const runStarted = (events ?? []).find(
    (item) => normalizeEventType(item?.type) === "run_started"
  );
  return runStarted?.data?.plan ?? null;
}

function reducer(state, action) {
  switch (action.type) {
    case "SET_TASK":
      return { ...state, taskInput: action.value };
    case "SET_API_BASE":
      return { ...state, apiBase: action.value };
    case "START_RUN":
      return {
        ...state,
        status: "connecting",
        runId: null,
        plan: null,
        events: [],
        replayMode: false,
        replaying: false,
        replayCursor: 0,
        selectedEventIndex: null,
        error: null,
      };
    case "SET_RUN_ID":
      return { ...state, runId: action.runId };
    case "ADD_EVENT": {
      const nextEvents = [...state.events, action.event];
      const eventType = normalizeEventType(action.event?.type);
      let nextStatus = state.status;
      let nextPlan = state.plan;

      if (eventType === "run_started") {
        nextStatus = "running";
        nextPlan = action.event?.data?.plan ?? null;
      } else if (eventType === "run_completed") {
        nextStatus = "completed";
      }

      return {
        ...state,
        status: nextStatus,
        plan: nextPlan,
        events: nextEvents,
      };
    }
    case "SET_ERROR":
      return { ...state, status: "error", error: action.error };
    case "TOGGLE_HISTORY":
      return { ...state, historyOpen: typeof action.open === "boolean" ? action.open : !state.historyOpen };
    case "SET_HISTORY_LOADING":
      return { ...state, historyLoading: Boolean(action.loading) };
    case "SET_HISTORY":
      return {
        ...state,
        historyRuns: Array.isArray(action.runs) ? action.runs : [],
        historyError: null,
      };
    case "SET_HISTORY_ERROR":
      return { ...state, historyError: action.error };
    case "SET_MCP_LOADING":
      return { ...state, mcpLoading: Boolean(action.loading) };
    case "SET_MCP_STATUS":
      return { ...state, mcpStatus: action.status ?? null, mcpError: null };
    case "SET_MCP_ERROR":
      return { ...state, mcpError: action.error };
    case "LOAD_HISTORY": {
      const run = action.run ?? {};
      const events = Array.isArray(run.events) ? run.events : [];
      return {
        ...state,
        status: "completed",
        runId: run.run_id ?? state.runId,
        plan: extractPlanFromEvents(events),
        events,
        replayMode: false,
        replaying: false,
        replayCursor: 0,
        historySelectedRunId: run.run_id ?? null,
        selectedEventIndex: null,
        taskInput: typeof run.task === "string" && run.task ? run.task : state.taskInput,
        error: null,
      };
    }
    case "START_REPLAY":
      if (state.events.length === 0) return state;
      return {
        ...state,
        replayMode: true,
        replaying: true,
        replayCursor: 0,
        selectedEventIndex: null,
        error: null,
      };
    case "STOP_REPLAY":
      return {
        ...state,
        replayMode: false,
        replaying: false,
        replayCursor: 0,
      };
    case "ADVANCE_REPLAY": {
      if (!state.replaying) return state;
      const nextCursor = state.replayCursor + 1;
      if (nextCursor >= state.events.length) {
        return {
          ...state,
          replayMode: false,
          replaying: false,
          replayCursor: state.events.length,
        };
      }
      return {
        ...state,
        replayCursor: nextCursor,
      };
    }
    case "SELECT_EVENT":
      return { ...state, selectedEventIndex: action.index };
    default:
      return state;
  }
}

function toWsUrl(apiBase, runId) {
  const url = new URL(`/api/runs/${runId}/events`, apiBase);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

function statusBadgeClass(status) {
  if (status === "success") {
    return "bg-emerald-100 text-emerald-700 border-emerald-200";
  }
  if (status === "needs_revision") {
    return "bg-amber-100 text-amber-700 border-amber-200";
  }
  if (status === "error") {
    return "bg-red-100 text-red-700 border-red-200";
  }
  return "bg-slate-100 text-slate-600 border-slate-200";
}

function formatDuration(ms) {
  if (typeof ms !== "number") return "-";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function summarizePlanProgress(plan, events) {
  const order = Array.isArray(plan?.execution_order) ? plan.execution_order : [];
  const states = {};
  order.forEach((agent) => {
    states[agent] = { phase: "pending", status: null };
  });

  for (const event of events ?? []) {
    const type = normalizeEventType(event?.type);
    const data = event?.data ?? {};
    const agent = data.agent;
    if (!agent || !states[agent]) continue;

    if (type === "agent_started") {
      states[agent] = { ...states[agent], phase: "running" };
      continue;
    }
    if (type === "agent_completed") {
      states[agent] = {
        ...states[agent],
        phase: "completed",
        status: typeof data.status === "string" ? data.status : null,
      };
    }
  }

  return states;
}

function dagNodeColors(node) {
  if (node?.phase === "running") {
    return { fill: "#dbeafe", stroke: "#2563eb", text: "#1e3a8a" };
  }
  if (node?.status === "success") {
    return { fill: "#dcfce7", stroke: "#16a34a", text: "#14532d" };
  }
  if (node?.status === "needs_revision") {
    return { fill: "#fef3c7", stroke: "#d97706", text: "#78350f" };
  }
  if (node?.status === "error") {
    return { fill: "#fee2e2", stroke: "#dc2626", text: "#7f1d1d" };
  }
  return { fill: "#f8fafc", stroke: "#94a3b8", text: "#334155" };
}

function formatAgentName(name) {
  if (!name) return "agent";
  return String(name).replace(/_/g, " ");
}

function getElapsedSegments(events) {
  return (events ?? [])
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => normalizeEventType(event?.type) === "agent_completed")
    .map(({ event, index }) => {
      const data = event?.data ?? {};
      const elapsedMs = Number(data.elapsed_ms ?? 0);
      return {
        id: `${event.timestamp ?? "ts"}-${index}`,
        index: index + 1,
        agent: String(data.agent ?? "agent"),
        status: String(data.status ?? "unknown"),
        elapsedMs: Number.isFinite(elapsedMs) && elapsedMs >= 0 ? elapsedMs : 0,
        turn: data.turn ?? null,
        iteration: data.iteration ?? null,
      };
    });
}

function segmentBarColor(status) {
  if (status === "success") return "bg-emerald-500";
  if (status === "needs_revision") return "bg-amber-500";
  if (status === "error") return "bg-red-500";
  return "bg-slate-500";
}

function normalizeParagraph(text) {
  return String(text ?? "")
    .replace(/\s+/g, " ")
    .trim();
}

function splitParagraphs(text) {
  return String(text ?? "")
    .replace(/\r\n/g, "\n")
    .split(/\n\s*\n+/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);
}

function buildCountMap(values) {
  const map = new Map();
  for (const value of values) {
    map.set(value, (map.get(value) ?? 0) + 1);
  }
  return map;
}

function computeParagraphDiff(beforeText, afterText) {
  const before = splitParagraphs(beforeText);
  const after = splitParagraphs(afterText);

  const afterCounts = buildCountMap(after.map((p) => normalizeParagraph(p)));
  const beforeView = before.map((text) => {
    const key = normalizeParagraph(text);
    const count = afterCounts.get(key) ?? 0;
    if (count > 0) {
      afterCounts.set(key, count - 1);
      return { text, tag: "same" };
    }
    return { text, tag: "removed" };
  });

  const beforeCounts = buildCountMap(before.map((p) => normalizeParagraph(p)));
  const afterView = after.map((text) => {
    const key = normalizeParagraph(text);
    const count = beforeCounts.get(key) ?? 0;
    if (count > 0) {
      beforeCounts.set(key, count - 1);
      return { text, tag: "same" };
    }
    return { text, tag: "added" };
  });

  return { beforeView, afterView };
}

function getCompletedAgentOutputs(events) {
  return (events ?? [])
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => normalizeEventType(event?.type) === "agent_completed")
    .map(({ event, index }) => {
      const data = event?.data ?? {};
      return {
        id: `${event.timestamp ?? "ts"}-${index}`,
        index: index + 1,
        agent: String(data.agent ?? "agent"),
        status: String(data.status ?? "unknown"),
        content: String(data.content ?? ""),
        turn: data.turn ?? null,
        iteration: data.iteration ?? null,
        timestamp: event.timestamp ?? null,
      };
    });
}

function PlanDag({ plan, events }) {
  const order = Array.isArray(plan?.execution_order) ? plan.execution_order : [];
  if (order.length === 0) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-500">
        TeamPlan 暂无可视化节点。
      </div>
    );
  }

  const workflow = plan?.workflow ?? "pipeline";
  const nodeState = summarizePlanProgress(plan, events);

  const nodeWidth = 136;
  const nodeHeight = 50;
  const gap = 72;
  const padX = 26;
  const topY = 20;
  const centerY = topY + nodeHeight / 2;

  const renderNode = (agent, x, y) => {
    const state = nodeState[agent] ?? { phase: "pending", status: null };
    const colors = dagNodeColors(state);
    const formatted = formatAgentName(agent);
    const label = formatted.length > 18 ? `${formatted.slice(0, 17)}…` : formatted;
    const statusText =
      state.phase === "running"
        ? "RUNNING"
        : state.status
        ? String(state.status).toUpperCase()
        : "PENDING";

    return (
      <g key={agent}>
        <rect
          x={x}
          y={y}
          width={nodeWidth}
          height={nodeHeight}
          rx="10"
          fill={colors.fill}
          stroke={colors.stroke}
          strokeWidth={state.phase === "running" ? 3 : 2}
        />
        <text
          x={x + nodeWidth / 2}
          y={y + 22}
          textAnchor="middle"
          fontSize="13"
          fontWeight="700"
          fill={colors.text}
        >
          {label}
        </text>
        <text
          x={x + nodeWidth / 2}
          y={y + 40}
          textAnchor="middle"
          fontSize="11"
          fontWeight="600"
          fill={colors.text}
        >
          {statusText}
        </text>
        {state.phase === "running" ? (
          <circle cx={x + nodeWidth - 12} cy={y + 12} r="4" fill="#2563eb">
            <animate attributeName="opacity" values="1;0.25;1" dur="1s" repeatCount="indefinite" />
          </circle>
        ) : null}
      </g>
    );
  };

  if (workflow === "parallel" && order.length >= 2) {
    const branchNames = order.slice(0, -1);
    const synthName = order[order.length - 1];
    const branchGap = 14;
    const totalBranchHeight = branchNames.length * nodeHeight + (branchNames.length - 1) * branchGap;
    const branchX = padX;
    const branchTopY = topY;
    const synthX = padX + nodeWidth + 132;
    const synthY = branchTopY + (totalBranchHeight - nodeHeight) / 2;
    const width = padX * 2 + nodeWidth * 2 + 132;
    const height = Math.max(108, topY * 2 + totalBranchHeight);

    return (
      <div className="rounded-lg border border-slate-200 bg-white p-3">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
          <defs>
            <marker
              id="arrow-end-parallel"
              markerWidth="10"
              markerHeight="8"
              refX="10"
              refY="4"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L10,4 L0,8 Z" fill="#64748b" />
            </marker>
          </defs>

          <text x={6} y={height / 2 + 4} fontSize="12" fill="#475569" fontWeight="600">
            Task
          </text>
          <line
            x1={30}
            y1={height / 2}
            x2={branchX}
            y2={height / 2}
            stroke="#64748b"
            strokeWidth="2"
            markerEnd="url(#arrow-end-parallel)"
          />

          {branchNames.map((name, index) => {
            const y = branchTopY + index * (nodeHeight + branchGap);
            const state = nodeState[name] ?? { phase: "pending", status: null };
            const isError = state.status === "error";
            const needsRevision = state.status === "needs_revision";
            const stroke = isError ? "#dc2626" : needsRevision ? "#d97706" : "#64748b";
            const dash = isError || needsRevision ? "6 4" : undefined;
            return (
              <line
                key={`parallel-edge-${name}`}
                x1={branchX + nodeWidth}
                y1={y + nodeHeight / 2}
                x2={synthX}
                y2={synthY + nodeHeight / 2}
                stroke={stroke}
                strokeWidth="2"
                strokeDasharray={dash}
                markerEnd="url(#arrow-end-parallel)"
              />
            );
          })}

          {branchNames.map((name, index) => {
            const y = branchTopY + index * (nodeHeight + branchGap);
            return renderNode(name, branchX, y);
          })}
          {renderNode(synthName, synthX, synthY)}
        </svg>

        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-600">
          <span className="rounded border border-slate-300 bg-slate-50 px-2 py-0.5">
            Parallel Branches: {branchNames.length}
          </span>
          <span className="rounded border border-slate-300 bg-slate-50 px-2 py-0.5">
            Synthesizer: {formatAgentName(synthName)}
          </span>
        </div>
      </div>
    );
  }

  const width = padX * 2 + order.length * nodeWidth + (order.length - 1) * gap;
  const height = workflow === "review_loop" ? 150 : 108;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
        <defs>
          <marker
            id="arrow-end"
            markerWidth="10"
            markerHeight="8"
            refX="10"
            refY="4"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L10,4 L0,8 Z" fill="#64748b" />
          </marker>
        </defs>

        {workflow === "review_loop" && order.length >= 2 ? (
          <>
            <line
              x1={padX + nodeWidth}
              y1={centerY}
              x2={padX + nodeWidth + gap}
              y2={centerY}
              stroke="#64748b"
              strokeWidth="2"
              markerEnd="url(#arrow-end)"
            />
            <path
              d={`M ${padX + nodeWidth + gap} ${centerY + 5}
                  C ${padX + nodeWidth + gap + 30} ${centerY + 44},
                    ${padX - 30} ${centerY + 44},
                    ${padX} ${centerY + 5}`}
              fill="none"
              stroke="#d97706"
              strokeWidth="2"
              strokeDasharray="6 5"
              markerEnd="url(#arrow-end)"
            />
            <text
              x={padX + nodeWidth / 2 + gap / 2}
              y={centerY + 56}
              textAnchor="middle"
              fontSize="12"
              fill="#92400e"
              fontWeight="600"
            >
              ↩ Retry Loop
            </text>
            {order.length > 2
              ? order.slice(1).map((_, idx) => {
                  const i = idx + 1;
                  if (i >= order.length - 1) return null;
                  const x1 = padX + i * (nodeWidth + gap) + nodeWidth;
                  const x2 = padX + (i + 1) * (nodeWidth + gap);
                  return (
                    <line
                      key={`extra-edge-${i}`}
                      x1={x1}
                      y1={centerY}
                      x2={x2}
                      y2={centerY}
                      stroke="#64748b"
                      strokeWidth="2"
                      markerEnd="url(#arrow-end)"
                    />
                  );
                })
              : null}
          </>
        ) : (
          order.map((_, i) => {
            if (i >= order.length - 1) return null;
            const x1 = padX + i * (nodeWidth + gap) + nodeWidth;
            const x2 = padX + (i + 1) * (nodeWidth + gap);
            return (
              <line
                key={`edge-${i}`}
                x1={x1}
                y1={centerY}
                x2={x2}
                y2={centerY}
                stroke="#64748b"
                strokeWidth="2"
                markerEnd="url(#arrow-end)"
              />
            );
          })
        )}

        {order.map((agent, i) => {
          const x = padX + i * (nodeWidth + gap);
          return renderNode(agent, x, topY);
        })}
      </svg>

      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-600">
        <span className="rounded border border-slate-300 bg-slate-50 px-2 py-0.5">PENDING</span>
        <span className="rounded border border-blue-300 bg-blue-50 px-2 py-0.5 text-blue-700">
          RUNNING
        </span>
        <span className="rounded border border-emerald-300 bg-emerald-50 px-2 py-0.5 text-emerald-700">
          SUCCESS
        </span>
        <span className="rounded border border-amber-300 bg-amber-50 px-2 py-0.5 text-amber-700">
          NEEDS_REVISION
        </span>
        <span className="rounded border border-red-300 bg-red-50 px-2 py-0.5 text-red-700">ERROR</span>
      </div>
    </div>
  );
}

function DurationAnalysis({ events }) {
  const segments = useMemo(() => getElapsedSegments(events), [events]);
  const runCompleted = useMemo(
    () => (events ?? []).some((event) => normalizeEventType(event?.type) === "run_completed"),
    [events]
  );

  if (!runCompleted || segments.length === 0) {
    return null;
  }

  const maxElapsed = segments.reduce((acc, item) => Math.max(acc, item.elapsedMs), 0);
  const totalElapsed = segments.reduce((acc, item) => acc + item.elapsedMs, 0);
  const slowest = segments.reduce(
    (acc, item) => (item.elapsedMs > acc.elapsedMs ? item : acc),
    segments[0]
  );

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-4 py-3 text-sm font-semibold text-slate-700">
        区域4 · 耗时分析
      </div>

      <div className="space-y-3 p-4">
        <div className="grid gap-2 text-sm text-slate-700 md:grid-cols-3">
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            Steps: <span className="font-semibold">{segments.length}</span>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            Total: <span className="font-semibold">{formatDuration(totalElapsed)}</span>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            Slowest:{" "}
            <span className="font-semibold">
              {formatAgentName(slowest.agent)} ({formatDuration(slowest.elapsedMs)})
            </span>
          </div>
        </div>

        <div className="space-y-2">
          {segments.map((segment) => {
            const widthPct = maxElapsed > 0 ? Math.max((segment.elapsedMs / maxElapsed) * 100, 4) : 4;
            return (
              <div key={segment.id} className="grid items-center gap-2 md:grid-cols-[230px_1fr_64px]">
                <div className="truncate text-xs font-medium text-slate-700" title={segment.agent}>
                  #{segment.index} {formatAgentName(segment.agent)}
                  {segment.iteration ? ` · iter ${segment.iteration}` : ""}
                  {segment.turn ? ` · turn ${segment.turn}` : ""}
                </div>
                <div className="h-6 rounded-md bg-slate-100">
                  <div
                    className={`flex h-full items-center rounded-md px-2 text-[11px] font-semibold text-white ${segmentBarColor(
                      segment.status
                    )}`}
                    style={{ width: `${widthPct}%` }}
                  >
                    {segment.status.toUpperCase()}
                  </div>
                </div>
                <div className="text-right text-xs font-semibold text-slate-700">
                  {formatDuration(segment.elapsedMs)}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function diffParagraphClass(tag, side) {
  if (tag === "removed" && side === "before") {
    return "border-red-200 bg-red-50";
  }
  if (tag === "added" && side === "after") {
    return "border-emerald-200 bg-emerald-50";
  }
  return "border-slate-200 bg-white";
}

function QualityInsights({ events, plan }) {
  const completed = useMemo(() => getCompletedAgentOutputs(events), [events]);
  const executionOrder = Array.isArray(plan?.execution_order) ? plan.execution_order : [];
  const workflow = plan?.workflow ?? "pipeline";

  if (completed.length === 0) {
    return null;
  }

  if (workflow === "parallel") {
    const lastParallelEvent = [...(events ?? [])]
      .reverse()
      .find((event) => normalizeEventType(event?.type) === "parallel_completed");
    const parallelData = lastParallelEvent?.data ?? {};
    const eventParallelAgents = Array.isArray(parallelData.parallel_agents)
      ? parallelData.parallel_agents.filter(Boolean)
      : [];
    const inferredCompletedOrder = Array.from(new Set(completed.map((item) => item.agent)));

    let synthesizerAgent = null;
    if (executionOrder.length >= 2) {
      synthesizerAgent = executionOrder[executionOrder.length - 1];
    } else if (typeof parallelData.synthesizer === "string" && parallelData.synthesizer) {
      synthesizerAgent = parallelData.synthesizer;
    }

    let branchAgents = [];
    if (executionOrder.length >= 2) {
      branchAgents = executionOrder.slice(0, -1);
    } else if (eventParallelAgents.length > 0) {
      branchAgents = eventParallelAgents;
    } else if (synthesizerAgent) {
      branchAgents = inferredCompletedOrder.filter((agent) => agent !== synthesizerAgent);
    } else {
      branchAgents = inferredCompletedOrder.slice(0, -1);
    }

    branchAgents = Array.from(new Set(branchAgents.filter(Boolean)));
    if (!synthesizerAgent && inferredCompletedOrder.length > 0) {
      const fallback = inferredCompletedOrder.find((agent) => !branchAgents.includes(agent));
      synthesizerAgent = fallback ?? inferredCompletedOrder[inferredCompletedOrder.length - 1];
    }
    if (synthesizerAgent) {
      branchAgents = branchAgents.filter((agent) => agent !== synthesizerAgent);
    }

    const branchSummaries = branchAgents.map((agent) => {
      const outputs = completed.filter((item) => item.agent === agent);
      return {
        agent,
        outputs,
        latest: outputs.length > 0 ? outputs[outputs.length - 1] : null,
      };
    });

    const successfulAgents = Array.isArray(parallelData.successful_agents)
      ? parallelData.successful_agents
      : branchSummaries
          .filter((item) => item.latest?.status === "success")
          .map((item) => item.agent);
    const failedAgents = Array.isArray(parallelData.failed_agents)
      ? parallelData.failed_agents
      : branchSummaries
          .filter((item) => item.latest?.status === "error")
          .map((item) => item.agent);

    const synthesizerOutputs = synthesizerAgent
      ? completed.filter((item) => item.agent === synthesizerAgent)
      : [];
    const synthesizerLatest =
      synthesizerOutputs.length > 0 ? synthesizerOutputs[synthesizerOutputs.length - 1] : null;

    return (
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-100 px-4 py-3 text-sm font-semibold text-slate-700">
          区域5 · 质量对比
        </div>

        <div className="space-y-4 p-4">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="mb-2 text-sm font-semibold text-slate-700">并行分支输出对比</div>
            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-slate-600">
              <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
                Branches: {branchSummaries.length}
              </span>
              <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
                Success: {successfulAgents.length}
              </span>
              <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
                Failed: {failedAgents.length}
              </span>
            </div>

            {branchSummaries.length === 0 ? (
              <div className="rounded-md border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-500">
                当前 run 暂无并行分支输出。
              </div>
            ) : (
              <div className="space-y-2">
                {branchSummaries.map((item) => (
                  <details key={`branch-${item.agent}`} className="rounded-md border border-slate-200 bg-white" open>
                    <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-3 py-2 text-sm">
                      <span className="font-medium text-slate-700">
                        {formatAgentName(item.agent)} · outputs {item.outputs.length}
                        {item.latest?.turn ? ` · turn ${item.latest.turn}` : ""}
                      </span>
                      <span
                        className={`rounded-md border px-2 py-0.5 text-xs font-semibold ${statusBadgeClass(
                          item.latest?.status ?? "unknown"
                        )}`}
                      >
                        {(item.latest?.status ?? "unknown").toUpperCase()}
                      </span>
                    </summary>
                    <div className="border-t border-slate-100 px-3 py-2">
                      {item.latest ? (
                        <pre className="max-h-[220px] overflow-y-auto whitespace-pre-wrap break-words text-sm leading-6 text-slate-700">
                          {item.latest.content}
                        </pre>
                      ) : (
                        <div className="text-sm text-slate-500">该分支暂无完成输出。</div>
                      )}
                    </div>
                  </details>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="mb-2 text-sm font-semibold text-slate-700">Synthesizer 汇总结果</div>
            <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-600">
              <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
                Synthesizer: {formatAgentName(synthesizerAgent)}
              </span>
              <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
                Outputs: {synthesizerOutputs.length}
              </span>
            </div>

            {synthesizerLatest ? (
              <div className="rounded-md border border-slate-200 bg-white">
                <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-3 py-2 text-sm">
                  <span className="font-medium text-slate-700">
                    turn {synthesizerLatest.turn ?? "-"}
                  </span>
                  <span
                    className={`rounded-md border px-2 py-0.5 text-xs font-semibold ${statusBadgeClass(
                      synthesizerLatest.status
                    )}`}
                  >
                    {synthesizerLatest.status.toUpperCase()}
                  </span>
                </div>
                <div className="px-3 py-2">
                  <pre className="max-h-[280px] overflow-y-auto whitespace-pre-wrap break-words text-sm leading-6 text-slate-700">
                    {synthesizerLatest.content}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="rounded-md border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-500">
                当前 run 暂无 synthesizer 输出。
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  const inferredOrder =
    executionOrder.length > 0
      ? executionOrder
      : Array.from(new Set(completed.map((item) => item.agent)));
  const writerAgent = inferredOrder[0] ?? null;
  const reviewerAgent = inferredOrder[1] ?? null;

  const writerOutputs = completed.filter((item) => item.agent === writerAgent);
  const reviewerOutputs = completed.filter((item) => item.agent === reviewerAgent);

  const firstDraft = writerOutputs[0] ?? null;
  const finalDraft = writerOutputs.length > 0 ? writerOutputs[writerOutputs.length - 1] : null;
  const hasDiff = firstDraft && finalDraft && writerOutputs.length >= 2;
  const paragraphDiff = hasDiff
    ? computeParagraphDiff(firstDraft.content, finalDraft.content)
    : { beforeView: [], afterView: [] };

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-4 py-3 text-sm font-semibold text-slate-700">
        区域5 · 质量对比
      </div>

      <div className="space-y-4 p-4">
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-600">
            <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
              Writer: {formatAgentName(writerAgent)}
            </span>
            <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
              Draft Versions: {writerOutputs.length}
            </span>
          </div>

          {hasDiff ? (
            <div className="grid gap-3 lg:grid-cols-2">
              <div>
                <div className="mb-2 text-xs font-semibold text-slate-600">
                  初稿 · turn {firstDraft.turn ?? "-"}
                </div>
                <div className="max-h-[360px] space-y-2 overflow-y-auto pr-1">
                  {paragraphDiff.beforeView.map((item, idx) => (
                    <div
                      key={`before-${idx}`}
                      className={`rounded-md border p-2 text-sm whitespace-pre-wrap text-slate-700 ${diffParagraphClass(
                        item.tag,
                        "before"
                      )}`}
                    >
                      {item.text}
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-2 text-xs font-semibold text-slate-600">
                  终稿 · turn {finalDraft.turn ?? "-"}
                </div>
                <div className="max-h-[360px] space-y-2 overflow-y-auto pr-1">
                  {paragraphDiff.afterView.map((item, idx) => (
                    <div
                      key={`after-${idx}`}
                      className={`rounded-md border p-2 text-sm whitespace-pre-wrap text-slate-700 ${diffParagraphClass(
                        item.tag,
                        "after"
                      )}`}
                    >
                      {item.text}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-md border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-500">
              暂无可对比的初稿和终稿。需要同一 writer 至少两次输出。
            </div>
          )}
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-600">
            <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
              Reviewer: {formatAgentName(reviewerAgent)}
            </span>
            <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
              Feedback Rounds: {reviewerOutputs.length}
            </span>
          </div>

          {reviewerOutputs.length === 0 ? (
            <div className="rounded-md border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-500">
              当前 run 没有 reviewer 反馈事件。
            </div>
          ) : (
            <div className="space-y-2">
              {reviewerOutputs.map((item, idx) => (
                <details key={item.id} className="rounded-md border border-slate-200 bg-white" open={idx === 0}>
                  <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-3 py-2 text-sm">
                    <span className="font-medium text-slate-700">
                      Iteration {item.iteration ?? idx + 1} · turn {item.turn ?? "-"}
                    </span>
                    <span
                      className={`rounded-md border px-2 py-0.5 text-xs font-semibold ${statusBadgeClass(
                        item.status
                      )}`}
                    >
                      {item.status.toUpperCase()}
                    </span>
                  </summary>
                  <div className="border-t border-slate-100 px-3 py-2">
                    <pre className="max-h-[240px] overflow-y-auto whitespace-pre-wrap break-words text-sm leading-6 text-slate-700">
                      {item.content}
                    </pre>
                  </div>
                </details>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function TaskInput({
  task,
  apiBase,
  disabled,
  onTaskChange,
  onApiBaseChange,
  onStart,
  canReplay,
  replaying,
  replayProgress,
  onReplay,
  onStopReplay,
  status,
  runMode,
  mcpStatus,
  mcpLoading,
  mcpError,
  onRefreshMcp,
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 text-sm font-semibold text-slate-700">区域1 · 任务输入</div>
      <div className="grid gap-3 md:grid-cols-[1fr_240px_auto_auto]">
        <input
          className="h-11 rounded-lg border border-slate-300 px-3 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:bg-slate-100"
          placeholder="输入任务，例如：写一篇关于多Agent协作系统的技术博客"
          value={task}
          onChange={(e) => onTaskChange(e.target.value)}
          disabled={disabled}
        />
        <input
          className="h-11 rounded-lg border border-slate-300 px-3 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:bg-slate-100"
          placeholder="API Base (默认 http://localhost:8000)"
          value={apiBase}
          onChange={(e) => onApiBaseChange(e.target.value)}
          disabled={disabled}
        />
        <button
          className="h-11 rounded-lg bg-blue-600 px-4 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          onClick={onStart}
          disabled={disabled || !task.trim()}
        >
          {status === "connecting" || status === "running" ? "执行中..." : "启动运行"}
        </button>
        <button
          className={`h-11 rounded-lg px-4 text-sm font-semibold transition ${
            replaying
              ? "bg-amber-500 text-white hover:bg-amber-600"
              : "bg-slate-700 text-white hover:bg-slate-800"
          } disabled:cursor-not-allowed disabled:bg-slate-400`}
          onClick={replaying ? onStopReplay : onReplay}
          disabled={disabled || !canReplay}
        >
          {replaying ? "停止回放" : "回放"}
        </button>
      </div>
      {canReplay ? (
        <div className="mt-2 text-xs text-slate-500">
          {replaying ? `回放中 ${replayProgress}` : "可回放本次 trace"}
        </div>
      ) : null}
      {runMode ? (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-600">
          <span className="rounded border border-slate-300 bg-slate-50 px-2 py-0.5">
            LLM Mode: {(runMode.effective || runMode.configured || "unknown").toUpperCase()}
          </span>
          {runMode.effective && runMode.configured && runMode.effective !== runMode.configured ? (
            <span className="rounded border border-slate-300 bg-slate-50 px-2 py-0.5">
              configured: {runMode.configured}
            </span>
          ) : null}
        </div>
      ) : null}
      <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs">
        <div className="flex items-center justify-between gap-2">
          <div className="font-semibold text-slate-700">MCP 状态</div>
          <button
            className="rounded border border-slate-300 bg-white px-2 py-0.5 text-[11px] text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
            onClick={onRefreshMcp}
            disabled={Boolean(mcpLoading)}
            type="button"
          >
            {mcpLoading ? "检查中..." : "刷新"}
          </button>
        </div>
        {mcpError ? <div className="mt-1 text-red-700">{mcpError}</div> : null}
        {mcpStatus ? (
          <div className="mt-1 flex flex-wrap items-center gap-2 text-slate-600">
            <span
              className={`rounded border px-2 py-0.5 ${
                mcpStatus.ready
                  ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                  : "border-amber-300 bg-amber-50 text-amber-700"
              }`}
            >
              {mcpStatus.ready ? "READY" : "NOT READY"}
            </span>
            <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
              logged_in: {String(Boolean(mcpStatus.logged_in))}
            </span>
            <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
              connected_servers: {mcpStatus.connected_servers ?? 0}
            </span>
            {Array.isArray(mcpStatus.servers) && mcpStatus.servers.length > 0 ? (
              <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
                servers: {mcpStatus.servers.map((item) => item.name).join(", ")}
              </span>
            ) : null}
          </div>
        ) : (
          <div className="mt-1 text-slate-500">尚未检查 MCP 可用性。</div>
        )}
      </div>
    </div>
  );
}

function formatRunTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function summarizeTask(task, limit = 80) {
  const text = String(task ?? "").trim();
  if (text.length <= limit) return text || "(empty task)";
  return `${text.slice(0, limit - 3)}...`;
}

function summarizeToolArgs(args, limit = 100) {
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    return "{}";
  }
  let text = "";
  try {
    text = JSON.stringify(args);
  } catch (_err) {
    text = String(args);
  }
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit - 3)}...`;
}

function getLatestRunMode(events) {
  const runStarted = [...(events ?? [])]
    .reverse()
    .find((item) => normalizeEventType(item?.type) === "run_started");
  if (!runStarted) return null;
  const data = runStarted.data ?? {};
  const configured = typeof data.llm_mode_configured === "string" ? data.llm_mode_configured : null;
  const effective = typeof data.llm_mode_effective === "string" ? data.llm_mode_effective : null;
  return { configured, effective };
}

function HistoryPanel({
  open,
  runs,
  loading,
  error,
  disabled,
  selectedRunId,
  onToggle,
  onRefresh,
  onSelectRun,
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-4 py-3">
        <button
          className="text-sm font-semibold text-slate-700 hover:text-slate-900"
          onClick={onToggle}
          type="button"
        >
          {open ? "▼" : "▶"} 历史记录
        </button>
        <button
          className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
          onClick={onRefresh}
          disabled={disabled || loading}
          type="button"
        >
          {loading ? "刷新中..." : "刷新"}
        </button>
      </div>

      {open ? (
        <div className="space-y-2 p-3">
          {error ? (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {error}
            </div>
          ) : null}

          {runs.length === 0 && !loading ? (
            <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 px-3 py-2 text-xs text-slate-500">
              暂无历史 run 记录。
            </div>
          ) : null}

          {runs.slice(0, 10).map((run) => (
            <button
              key={run.run_id}
              className={`w-full rounded-md border p-2 text-left transition ${
                selectedRunId === run.run_id
                  ? "border-blue-400 bg-blue-50 ring-1 ring-blue-100"
                  : "border-slate-200 bg-white hover:border-slate-300"
              }`}
              onClick={() => onSelectRun(run.run_id)}
              disabled={disabled}
              type="button"
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <div className="truncate text-xs font-semibold text-slate-700">{run.run_id}</div>
                <span
                  className={`rounded-md border px-2 py-0.5 text-[11px] font-semibold ${statusBadgeClass(
                    run.final_status
                  )}`}
                >
                  {(run.final_status ?? "unknown").toUpperCase()}
                </span>
              </div>
              <div className="mb-1 text-xs text-slate-700">{summarizeTask(run.task)}</div>
              <div className="grid gap-1 text-[11px] text-slate-500 md:grid-cols-3">
                <div>Start: {formatRunTime(run.started_at)}</div>
                <div>End: {formatRunTime(run.completed_at)}</div>
                <div>Steps: {run.step_count ?? "-"}</div>
              </div>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function EventCard({ event, index, selected, onSelect, allEvents }) {
  const ts = event?.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "--:--:--";
  const type = normalizeEventType(event?.type);
  const data = event?.data ?? {};

  if (type === "run_started") {
    const plan = data.plan ?? {};
    return (
      <div className="rounded-lg border border-blue-200 bg-blue-50 p-3">
        <div className="mb-1 text-xs font-semibold text-blue-700">{ts} · run_started</div>
        <div className="mb-2 flex flex-wrap gap-2 text-xs text-slate-700">
          <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
            Workflow: {plan.workflow ?? "-"}
          </span>
          <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
            Max Iterations: {plan.max_iterations ?? "-"}
          </span>
          <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
            LLM: {(data.llm_mode_effective ?? data.llm_mode_configured ?? "-").toUpperCase()}
          </span>
          {data.llm_mode_effective &&
          data.llm_mode_configured &&
          data.llm_mode_effective !== data.llm_mode_configured ? (
            <span className="rounded border border-slate-300 bg-white px-2 py-0.5">
              configured: {data.llm_mode_configured}
            </span>
          ) : null}
        </div>
        <PlanDag plan={plan} events={allEvents} />
      </div>
    );
  }

  if (type === "agent_started") {
    return (
      <button
        className="w-full rounded-lg border border-slate-200 bg-white p-3 text-left transition hover:border-slate-300"
        onClick={() => onSelect(index)}
      >
        <div className="mb-1 flex items-center justify-between">
          <div className="text-xs font-semibold text-slate-500">{ts} · agent_started</div>
          <span className="rounded-md border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs text-blue-700">
            turn {data.turn ?? "-"}
          </span>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-700">
          <span className="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-blue-500" />
          <span className="font-medium">{data.agent ?? "agent"}</span>
          <span className="text-slate-500">正在执行...</span>
        </div>
      </button>
    );
  }

  if (type === "agent_completed") {
    return (
      <button
        className={`w-full rounded-lg border p-3 text-left transition ${
          selected
            ? "border-blue-400 bg-blue-50 ring-2 ring-blue-100"
            : "border-slate-200 bg-white hover:border-slate-300"
        }`}
        onClick={() => onSelect(index)}
      >
        <div className="mb-1 flex items-center justify-between gap-2">
          <div className="text-xs font-semibold text-slate-500">{ts} · agent_completed</div>
          <span
            className={`rounded-md border px-2 py-0.5 text-xs font-semibold ${statusBadgeClass(
              data.status
            )}`}
          >
            {(data.status ?? "unknown").toUpperCase()}
          </span>
        </div>
        <div className="mb-1 text-sm font-medium text-slate-800">
          {data.agent ?? "agent"} · turn {data.turn ?? "-"} · {formatDuration(data.elapsed_ms)}
        </div>
        <div className="text-sm text-slate-600">{data.content_preview ?? "-"}</div>
      </button>
    );
  }

  if (type === "tool_started") {
    const argsPreview = summarizeToolArgs(data.arguments, 100);
    return (
      <button
        className={`ml-8 mr-1 block rounded-lg border p-3 text-left transition ${
          selected
            ? "border-blue-400 bg-blue-50 ring-2 ring-blue-100"
            : "border-blue-200 bg-blue-50 hover:border-blue-300"
        }`}
        onClick={() => onSelect(index)}
      >
        <div className="mb-1 flex items-center justify-between gap-2">
          <div className="text-xs font-semibold text-blue-700">{ts} · tool_started</div>
          <span className="rounded-md border border-blue-200 bg-white px-2 py-0.5 text-xs text-blue-700">
            round {data.tool_round ?? "-"}
          </span>
        </div>
        <div className="mb-1 flex items-center gap-2 text-sm text-slate-700">
          <span className="inline-flex h-5 w-5 animate-pulse items-center justify-center rounded-full bg-blue-100 text-xs">
            🔧
          </span>
          <span className="font-medium">{data.tool ?? "tool"}</span>
          <span className="text-slate-500">调用中...</span>
        </div>
        <div className="text-xs text-slate-600">args: {argsPreview}</div>
      </button>
    );
  }

  if (type === "tool_completed") {
    const success = Boolean(data.success);
    const preview = success ? data.content_preview ?? "-" : data.error ?? data.content_preview ?? "-";
    return (
      <button
        className={`ml-8 mr-1 block rounded-lg border p-3 text-left transition ${
          selected
            ? "border-blue-400 bg-blue-50 ring-2 ring-blue-100"
            : success
            ? "border-emerald-200 bg-emerald-50 hover:border-emerald-300"
            : "border-red-200 bg-red-50 hover:border-red-300"
        }`}
        onClick={() => onSelect(index)}
      >
        <div className="mb-1 flex items-center justify-between gap-2">
          <div className="text-xs font-semibold text-slate-500">{ts} · tool_completed</div>
          <span
            className={`rounded-md border px-2 py-0.5 text-xs font-semibold ${statusBadgeClass(
              success ? "success" : "error"
            )}`}
          >
            {success ? "SUCCESS" : "ERROR"}
          </span>
        </div>
        <div className="mb-1 text-sm font-medium text-slate-800">
          🔧 {data.tool ?? "tool"} · {formatDuration(data.elapsed_ms)}
          {data.tool_round ? ` · round ${data.tool_round}` : ""}
        </div>
        <div className="text-sm text-slate-600">{preview}</div>
      </button>
    );
  }

  if (type === "iteration_completed") {
    const retry = data.reviewer_status === "needs_revision";
    return (
      <div
        className={`rounded-lg border border-dashed p-3 ${
          retry ? "border-amber-300 bg-amber-50" : "border-emerald-300 bg-emerald-50"
        }`}
      >
        <div className="mb-1 text-xs font-semibold text-slate-500">{ts} · iteration_completed</div>
        <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
          <span>Iteration {data.iteration ?? "-"}</span>
          <span
            className={`rounded-md border px-2 py-0.5 text-xs font-semibold ${statusBadgeClass(
              data.reviewer_status
            )}`}
          >
            {(data.reviewer_status ?? "unknown").toUpperCase()}
          </span>
          {retry ? (
            <span className="rounded-md border border-amber-200 bg-amber-100 px-2 py-0.5 text-xs text-amber-700">
              ↩ Retry
            </span>
          ) : null}
        </div>
      </div>
    );
  }

  if (type === "parallel_completed") {
    const successCount = Number(data.success_count ?? 0);
    const failedCount = Number(data.failed_count ?? 0);
    const failedAgents = Array.isArray(data.failed_agents) ? data.failed_agents : [];
    const successAgents = Array.isArray(data.successful_agents) ? data.successful_agents : [];
    const hasFailure = failedCount > 0;
    return (
      <div
        className={`rounded-lg border border-dashed p-3 ${
          hasFailure ? "border-amber-300 bg-amber-50" : "border-emerald-300 bg-emerald-50"
        }`}
      >
        <div className="mb-1 text-xs font-semibold text-slate-500">{ts} · parallel_completed</div>
        <div className="mb-1 flex flex-wrap items-center gap-2 text-sm font-medium text-slate-700">
          <span>Success: {successCount}</span>
          <span>Failed: {failedCount}</span>
          <span className="rounded border border-slate-300 bg-white px-2 py-0.5 text-xs">
            Synthesizer: {data.synthesizer ?? "-"}
          </span>
        </div>
        <div className="text-xs text-slate-600">
          {successAgents.length ? `成功分支: ${successAgents.join(", ")}` : "无成功分支"}
          {failedAgents.length ? ` ｜ 失败分支: ${failedAgents.join(", ")}` : ""}
        </div>
      </div>
    );
  }

  if (type === "run_completed") {
    return (
      <div className="rounded-lg border border-indigo-200 bg-indigo-50 p-3">
        <div className="mb-1 text-xs font-semibold text-indigo-700">{ts} · run_completed</div>
        <div className="grid gap-1 text-sm text-slate-700 md:grid-cols-3">
          <div>Total Steps: {data.total_steps ?? "-"}</div>
          <div>Total Time: {formatDuration(data.total_elapsed_ms)}</div>
          <div>
            Final:{" "}
            <span
              className={`rounded-md border px-2 py-0.5 text-xs font-semibold ${statusBadgeClass(
                data.final_status
              )}`}
            >
              {(data.final_status ?? "unknown").toUpperCase()}
            </span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="text-xs text-slate-500">{ts}</div>
      <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-xs text-slate-600">
        {JSON.stringify(event, null, 2)}
      </pre>
    </div>
  );
}

function EventTimeline({ events, selectedIndex, onSelect, timelineRef }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-4 py-3 text-sm font-semibold text-slate-700">
        区域2 · 实时执行流
      </div>
      <div ref={timelineRef} className="h-[520px] space-y-3 overflow-y-auto p-4">
        {events.length === 0 ? (
          <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-500">
            尚无事件。启动任务后将实时展示 run_started / agent_started / agent_completed /
            tool_started / tool_completed / iteration_completed / parallel_completed / run_completed。
          </div>
        ) : (
          events.map((event, index) => (
            <EventCard
              key={`${event.timestamp}-${index}`}
              event={event}
              index={index}
              selected={selectedIndex === index}
              onSelect={onSelect}
              allEvents={events}
            />
          ))
        )}
      </div>
    </div>
  );
}

function DetailPanel({ event }) {
  const type = normalizeEventType(event?.type);
  const data = event?.data ?? {};

  let title = "区域3 · 详情面板";
  let body = "点击任意 agent_completed 卡片查看完整内容。";

  if (type === "agent_completed") {
    title = `区域3 · ${data.agent ?? "agent"}（完整内容）`;
    body = data.content ?? "";
  } else if (type === "tool_completed") {
    title = `区域3 · 工具结果 · ${data.tool ?? "tool"}`;
    body = JSON.stringify(data, null, 2);
  } else if (type === "tool_started") {
    title = `区域3 · 工具调用 · ${data.tool ?? "tool"}`;
    body = JSON.stringify(data, null, 2);
  } else if (type === "run_completed") {
    title = "区域3 · Run Summary";
    body = JSON.stringify(data, null, 2);
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-4 py-3 text-sm font-semibold text-slate-700">
        {title}
      </div>
      <div className="h-[520px] overflow-y-auto p-4">
        <pre className="whitespace-pre-wrap break-words text-sm leading-6 text-slate-700">{body}</pre>
      </div>
    </div>
  );
}

function App({ defaultApiBase = inferDefaultApiBase() }) {
  const [state, dispatch] = useReducer(reducer, {
    ...INITIAL_STATE,
    apiBase: defaultApiBase,
  });
  const wsRef = useRef(null);
  const timelineRef = useRef(null);

  const isRunning = state.status === "connecting" || state.status === "running";
  const canReplay = state.status === "completed" && state.events.length > 0;
  const replayProgress = `${Math.min(state.replayCursor, state.events.length)}/${state.events.length}`;

  const renderedEvents = useMemo(() => {
    if (!state.replayMode) return state.events;
    return state.events.slice(0, state.replayCursor);
  }, [state.events, state.replayMode, state.replayCursor]);

  const selectedEvent = useMemo(() => {
    if (state.selectedEventIndex === null) return null;
    return renderedEvents[state.selectedEventIndex] ?? null;
  }, [renderedEvents, state.selectedEventIndex]);
  const currentRunMode = useMemo(() => getLatestRunMode(renderedEvents), [renderedEvents]);

  const fetchMcpStatus = useCallback(async () => {
    dispatch({ type: "SET_MCP_LOADING", loading: true });
    dispatch({ type: "SET_MCP_ERROR", error: null });
    try {
      const resp = await fetch(`${state.apiBase}/api/mcp/status`);
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`MCP 检查失败(${resp.status}): ${text}`);
      }
      const payload = await resp.json();
      dispatch({ type: "SET_MCP_STATUS", status: payload });
    } catch (err) {
      dispatch({
        type: "SET_MCP_ERROR",
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      dispatch({ type: "SET_MCP_LOADING", loading: false });
    }
  }, [state.apiBase]);

  const fetchHistory = useCallback(async () => {
    dispatch({ type: "SET_HISTORY_LOADING", loading: true });
    dispatch({ type: "SET_HISTORY_ERROR", error: null });
    try {
      const resp = await fetch(`${state.apiBase}/api/runs?limit=10`);
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`拉取历史失败(${resp.status}): ${text}`);
      }
      const payload = await resp.json();
      dispatch({ type: "SET_HISTORY", runs: Array.isArray(payload?.runs) ? payload.runs : [] });
    } catch (err) {
      dispatch({
        type: "SET_HISTORY_ERROR",
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      dispatch({ type: "SET_HISTORY_LOADING", loading: false });
    }
  }, [state.apiBase]);

  const loadHistoryRun = useCallback(
    async (runId) => {
      dispatch({ type: "SET_HISTORY_ERROR", error: null });
      try {
        const resp = await fetch(`${state.apiBase}/api/runs/${encodeURIComponent(runId)}`);
        if (!resp.ok) {
          const text = await resp.text();
          throw new Error(`加载历史详情失败(${resp.status}): ${text}`);
        }
        const payload = await resp.json();
        dispatch({ type: "LOAD_HISTORY", run: payload });
      } catch (err) {
        dispatch({
          type: "SET_HISTORY_ERROR",
          error: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [state.apiBase]
  );

  useEffect(() => {
    const el = timelineRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [renderedEvents.length]);

  useEffect(() => {
    void fetchHistory();
  }, [fetchHistory]);

  useEffect(() => {
    void fetchMcpStatus();
  }, [fetchMcpStatus]);

  useEffect(() => {
    if (state.status === "completed") {
      void fetchHistory();
    }
  }, [state.status, fetchHistory]);

  useEffect(() => {
    if (!state.runId) return undefined;
    if (!(state.status === "connecting" || state.status === "running")) return undefined;

    const wsUrl = toWsUrl(state.apiBase, state.runId);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        dispatch({ type: "ADD_EVENT", event: payload });
      } catch (err) {
        dispatch({
          type: "SET_ERROR",
          error: `事件解析失败: ${err instanceof Error ? err.message : String(err)}`,
        });
      }
    };

    ws.onerror = () => {
      dispatch({
        type: "SET_ERROR",
        error: "WebSocket 连接异常，请检查后端服务是否可用。",
      });
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [state.runId, state.status, state.apiBase]);

  useEffect(() => {
    if (!state.replaying) return undefined;

    if (state.replayCursor >= state.events.length) {
      dispatch({ type: "STOP_REPLAY" });
      return undefined;
    }

    const nextEvent = state.events[state.replayCursor];
      const nextType = normalizeEventType(nextEvent?.type);
    const delayMs =
      nextType === "agent_completed"
        ? 1200
        : nextType === "agent_started"
        ? 750
        : nextType === "tool_started"
        ? 700
        : nextType === "tool_completed"
        ? 950
        : nextType === "iteration_completed"
        ? 850
        : 950;

    const timer = setTimeout(() => {
      dispatch({ type: "ADVANCE_REPLAY" });
    }, delayMs);

    return () => clearTimeout(timer);
  }, [state.replaying, state.replayCursor, state.events]);

  const onStart = async () => {
    dispatch({ type: "START_RUN" });

    try {
      const resp = await fetch(`${state.apiBase}/api/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: state.taskInput.trim() }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`启动失败(${resp.status}): ${text}`);
      }

      const data = await resp.json();
      if (!data?.run_id) {
        throw new Error("响应缺少 run_id");
      }
      dispatch({ type: "SET_RUN_ID", runId: data.run_id });
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 p-4 md:p-6">
      <div className="mx-auto max-w-7xl space-y-4">
        <TaskInput
          task={state.taskInput}
          apiBase={state.apiBase}
          disabled={isRunning}
          onTaskChange={(value) => dispatch({ type: "SET_TASK", value })}
          onApiBaseChange={(value) => dispatch({ type: "SET_API_BASE", value })}
          onStart={onStart}
          canReplay={canReplay}
          replaying={state.replaying}
          replayProgress={replayProgress}
          onReplay={() => dispatch({ type: "START_REPLAY" })}
          onStopReplay={() => dispatch({ type: "STOP_REPLAY" })}
          status={state.status}
          runMode={currentRunMode}
          mcpStatus={state.mcpStatus}
          mcpLoading={state.mcpLoading}
          mcpError={state.mcpError}
          onRefreshMcp={() => void fetchMcpStatus()}
        />
        <HistoryPanel
          open={state.historyOpen}
          runs={state.historyRuns}
          loading={state.historyLoading}
          error={state.historyError}
          disabled={isRunning}
          selectedRunId={state.historySelectedRunId}
          onToggle={() => dispatch({ type: "TOGGLE_HISTORY" })}
          onRefresh={() => void fetchHistory()}
          onSelectRun={(runId) => void loadHistoryRun(runId)}
        />

        {state.error ? (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {state.error}
          </div>
        ) : null}

        <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
          <EventTimeline
            events={renderedEvents}
            selectedIndex={state.selectedEventIndex}
            onSelect={(index) => dispatch({ type: "SELECT_EVENT", index })}
            timelineRef={timelineRef}
          />
          <DetailPanel event={selectedEvent} />
        </div>

        <DurationAnalysis events={renderedEvents} />
        <QualityInsights events={renderedEvents} plan={state.plan} />
      </div>
    </div>
  );
}

const rootElement = document.getElementById("app");
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(<App defaultApiBase={inferDefaultApiBase()} />);
}
