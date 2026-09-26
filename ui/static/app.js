/* global Chart, echarts */

const state = {
  categoryChart: null,
  noveltyChart: null,
  forecastChart: null,
  trainChart: null,
  validateChart: null,
  incomeChart: null,
  triage: null,
  meta: null,
  trainPoll: null,
};

function $(id) {
  return document.getElementById(id);
}

function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 3600);
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (opts.body && !(opts.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let text = await res.text();
    try {
      const j = JSON.parse(text);
      text = j.detail || text;
    } catch (_) {}
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function destroyChart(chart) {
  if (chart) chart.destroy();
}

function renderLineChart(canvasId, payload, previous) {
  destroyChart(previous);
  const ctx = $(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: payload.labels || [],
      datasets: (payload.datasets || []).map((d) => ({
        pointRadius: 2,
        borderWidth: 2,
        ...d,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 12, font: { family: "IBM Plex Sans", size: 11 } },
        },
      },
      scales: {
        x: {
          ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 10, font: { size: 10 } },
          grid: { color: "rgba(19,35,43,0.06)" },
        },
        y: {
          beginAtZero: true,
          ticks: { precision: 0, font: { size: 10 } },
          grid: { color: "rgba(19,35,43,0.06)" },
        },
      },
      animation: { duration: 550, easing: "easeOutQuart" },
    },
  });
}

function renderBarChart(canvasId, chart, previous, color = "#0f6e56") {
  destroyChart(previous);
  if (!chart) return null;
  const ctx = $(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels: chart.labels || [],
      datasets: [
        {
          label: "Score",
          data: chart.values || [],
          backgroundColor: color,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 1, ticks: { font: { size: 10 } } },
        x: { ticks: { font: { size: 10 }, maxRotation: 30 } },
      },
      animation: { duration: 500 },
    },
  });
}

function renderSurges(noveltyPayload) {
  const ul = $("surgeList");
  const surges = noveltyPayload.surges || [];
  const suggestions = noveltyPayload.suggested_labels || [];
  const items = [];
  surges.slice(0, 6).forEach((s) => {
    items.push(
      `<li><span>${s.kind}: <strong>${s.category}</strong> ×${s.rate_ratio}</span><span>${s.severity}</span></li>`
    );
  });
  suggestions.slice(0, 4).forEach((s) => {
    items.push(
      `<li class="novel"><span>new label: <strong>${s.suggested_label}</strong></span><span>n=${s.cluster_size}</span></li>`
    );
  });
  ul.innerHTML = items.length
    ? items.join("")
    : `<li class="novel"><span>No active surges in this window</span><span>—</span></li>`;
}

function renderDaySummary(payload) {
  const box = $("daySummary");
  const summary = payload.summary || {};
  const metrics = payload.metrics || {};
  const status = summary.status || "unknown";
  box.innerHTML = `
    <span class="status ${escapeAttr(status)}">${escapeHtml(status)}</span>
    <h3>${escapeHtml(summary.headline || "No summary")}</h3>
    <div class="metric-row">
      <span class="metric">actual ${Number(metrics.cumulative_actual || 0).toFixed(0)}</span>
      <span class="metric">expected ${Number(metrics.cumulative_forecast_to_now || 0).toFixed(0)}</span>
      <span class="metric">${Number(metrics.ratio_to_forecast || 0).toFixed(2)}×</span>
      <span class="metric">tomorrow ≈ ${Number(metrics.next_day_forecast_total || 0).toFixed(0)}</span>
    </div>
    <ul>${(summary.bullets || []).map((b) => `<li>${escapeHtml(b)}</li>`).join("")}</ul>
  `;
  $("dayMarker").textContent = `As of ${payload.as_of || "—"} · today ${payload.today || "—"} · tomorrow ${payload.tomorrow || "—"} (UTC)`;
}

function renderIncomeTree(payload) {
  const el = $("incomeSunburst");
  if (!state.incomeChart) {
    state.incomeChart = echarts.init(el);
    window.addEventListener("resize", () => state.incomeChart && state.incomeChart.resize());
  }
  const colors = ["#0F6E56", "#C45C26", "#1F4B7A", "#8B3A4A", "#5B6B2F", "#6B4C9A", "#A67C2A", "#2F6F8F"];
  state.incomeChart.setOption(
    {
      color: colors,
      series: [
        {
          type: "sunburst",
          data: payload.sunburst?.children || [],
          radius: [28, "92%"],
          sort: undefined,
          emphasis: { focus: "ancestor" },
          levels: [
            {},
            {
              r0: "28%",
              r: "58%",
              itemStyle: { borderWidth: 2, borderColor: "#f7fffb" },
              label: { rotate: "tangential", fontSize: 11, fontFamily: "IBM Plex Sans" },
            },
            {
              r0: "58%",
              r: "92%",
              label: { position: "outside", silent: false, fontSize: 10, fontFamily: "IBM Plex Sans" },
              itemStyle: { borderWidth: 1, borderColor: "#f7fffb" },
            },
          ],
          label: { color: "#13232b" },
          itemStyle: { borderRadius: 4 },
        },
      ],
      tooltip: {
        formatter(info) {
          const v = info.value ?? info.data?.value ?? 0;
          const pct = payload.total ? ((100 * v) / payload.total).toFixed(0) : 0;
          return `${info.name}<br/><b>${v}</b> tickets · ${pct}% of period`;
        },
      },
    },
    true
  );

  const box = $("incomeSummary");
  const tops = payload.top_areas || [];
  const max = tops[0]?.count || 1;
  box.innerHTML = `
    <span class="status normal">live income</span>
    <p class="muted" style="margin:0">Selected period · ${escapeHtml(payload.period)}</p>
    <p class="total-hero">${Number(payload.total || 0)}</p>
    <h3>${escapeHtml(payload.headline || "")}</h3>
    <ul class="branch-list">
      ${tops
        .map(
          (t) => `<li>
            <span>${escapeHtml(t.name)}</span><strong>${t.count}</strong>
            <div class="bar"><span style="width:${Math.max(6, (100 * t.count) / max)}%"></span></div>
          </li>`
        )
        .join("")}
    </ul>
    <p class="muted" style="margin:0.4rem 0 0;font-size:0.82rem">Outer ring = labels inside each area. Click a segment to zoom.</p>
  `;
}

async function loadIncomeTree() {
  const period = $("hierarchyPeriod").value;
  const payload = await api(`/api/hierarchy?period=${encodeURIComponent(period)}`);
  renderIncomeTree(payload);
}

async function loadForecast() {
  const dimension = $("forecastDimension").value;
  const payload = await api(`/api/forecast?dimension=${encodeURIComponent(dimension)}`);
  state.forecastChart = renderLineChart("forecastChart", payload, state.forecastChart);
  renderDaySummary(payload);
}

async function loadCharts() {
  const dimension = $("dimension").value;
  const window = $("window").value;
  const [category, novelty] = await Promise.all([
    api(`/api/timeseries?dimension=${encodeURIComponent(dimension)}&window=${encodeURIComponent(window)}`),
    api(`/api/timeseries/novelty?window=${encodeURIComponent(window)}`),
  ]);
  state.categoryChart = renderLineChart("categoryChart", category, state.categoryChart);
  state.noveltyChart = renderLineChart("noveltyChart", novelty, state.noveltyChart);
  renderSurges(novelty);
}

function optionsHtml(options, selected) {
  const selectedList = Array.isArray(selected) ? selected : selected ? [selected] : [];
  const vals = new Set([...(options || []), ...selectedList].filter(Boolean));
  return [...vals]
    .map((v) => `<option value="${escapeAttr(v)}" ${selectedList.includes(v) ? "selected" : ""}>${escapeHtml(v)}</option>`)
    .join("");
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
function escapeAttr(s) {
  return escapeHtml(s).replaceAll("'", "&#39;");
}

function renderTriage(data) {
  state.triage = data;
  const mount = $("triageMount");
  if (!data.items || !data.items.length) {
    mount.innerHTML = `<div class="empty">No tickets in this inbox window.</div>`;
    return;
  }
  const rows = data.items
    .map((item, idx) => {
      const novel = item.is_novel
        ? `<span class="pill">novel / new-type</span>`
        : `<span class="pill ok">known</span>`;
      return `
      <tr data-idx="${idx}">
        <td><div class="key">${escapeHtml(item.issue_key)}</div>${novel}</td>
        <td>
          <p class="summary">${escapeHtml(item.summary || "")}</p>
          <p class="snippet">${escapeHtml(item.description_snippet || "")}</p>
          <button class="expand" type="button" data-action="expand">Expand description</button>
        </td>
        <td>
          <div class="field-stack">
            <select data-field="label">${optionsHtml(data.label_options, item.suggested_label)}</select>
            <span class="muted">ML: ${escapeHtml(item.suggested_label || "—")}</span>
          </div>
        </td>
        <td>
          <div class="field-stack">
            <select data-field="area">${optionsHtml(data.area_options, item.suggested_area)}</select>
            <span class="muted">ML: ${escapeHtml(item.suggested_area || "—")}</span>
          </div>
        </td>
        <td><button class="btn small" type="button" data-action="save">Save</button></td>
      </tr>`;
    })
    .join("");
  mount.innerHTML = `
    <table class="triage-table">
      <thead><tr><th>Ticket</th><th>Summary / description</th><th>Label</th><th>Impacted area</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function loadTriage() {
  const days = $("triageDays").value;
  renderTriage(await api(`/api/triage?days=${encodeURIComponent(days)}&limit=40`));
}

async function saveRow(tr) {
  const idx = Number(tr.dataset.idx);
  const item = state.triage.items[idx];
  const label = tr.querySelector('select[data-field="label"]').value;
  const area = tr.querySelector('select[data-field="area"]').value;
  const btn = tr.querySelector('[data-action="save"]');
  btn.disabled = true;
  try {
    const result = await api("/api/triage/save", {
      method: "POST",
      body: JSON.stringify({
        issue_key: item.issue_key,
        summary: item.summary,
        description: item.description,
        created: item.created,
        selected_labels: label ? [label] : [],
        selected_areas: area ? [area] : [],
        suggested_labels: item.suggested_labels || [],
        suggested_area: item.suggested_area,
        push_to_jira: true,
      }),
    });
    toast(
      `${item.issue_key} saved${result.disagreed ? " · correction queued" : ""}${
        result.jira?.dry_run ? " (local)" : result.jira?.ok ? " → Jira" : ""
      }`
    );
  } catch (err) {
    toast(`Save failed: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
}

function wireTriageClicks() {
  $("triageMount").addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    const tr = btn.closest("tr");
    if (!tr) return;
    if (btn.dataset.action === "expand") {
      const snip = tr.querySelector(".snippet");
      const idx = Number(tr.dataset.idx);
      const item = state.triage.items[idx];
      const expanded = snip.classList.toggle("expanded");
      snip.textContent = expanded ? item.description || "" : item.description_snippet || "";
      btn.textContent = expanded ? "Collapse description" : "Expand description";
      return;
    }
    if (btn.dataset.action === "save") await saveRow(tr);
  });
}

function setModeButtons(mode) {
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  const hints = {
    cache: "Local cached tickets / bundled sample.",
    spreadsheet: "Uploaded Excel/CSV — offline learning & triage.",
    jira: "Live Jira API when reachable; else cache.",
  };
  $("sourceHint").textContent = hints[mode] || "";
}

async function loadMeta() {
  const meta = await api("/api/meta");
  state.meta = meta;
  setModeButtons(meta.source_mode || "cache");
  $("metaBadge").textContent = `${meta.rows} tickets · ${meta.source_mode} · model ${
    meta.model_version || "n/a"
  } · ${meta.jira_writable ? "jira write-on" : "offline/local"}`;
  return meta;
}

async function loadParams() {
  const { params } = await api("/api/train/params");
  $("pAlgorithm").value = params.algorithm;
  $("pC").value = params.C;
  $("pThr").value = params.decision_threshold;
  $("pWord").value = params.max_word_features;
  $("pChar").value = params.max_char_features;
  $("pMinDf").value = params.min_df;
  $("pTest").value = params.test_size;
  $("pMinLab").value = params.min_label_support;
  $("pMinArea").value = params.min_area_support;
}

function collectParams() {
  return {
    algorithm: $("pAlgorithm").value,
    C: Number($("pC").value),
    decision_threshold: Number($("pThr").value),
    max_word_features: Number($("pWord").value),
    max_char_features: Number($("pChar").value),
    min_df: Number($("pMinDf").value),
    test_size: Number($("pTest").value),
    min_label_support: Number($("pMinLab").value),
    min_area_support: Number($("pMinArea").value),
  };
}

function renderTrainStatus(st) {
  $("trainState").textContent = st.state || "idle";
  $("trainLogs").textContent = (st.logs || []).join("\n") || "No logs yet.";
  $("trainLogs").scrollTop = $("trainLogs").scrollHeight;
  if (st.result && st.result.chart) {
    state.trainChart = renderBarChart("trainChart", st.result.chart, state.trainChart);
    const m = st.result.metrics || {};
    $("trainHeadline").textContent = `v${st.result.version} · labels F1µ ${Number(
      m.labels?.f1_micro || 0
    ).toFixed(2)} · area acc ${Number(m.area?.accuracy || 0).toFixed(2)} · window ${st.result.window}`;
  }
}

async function pollTrain() {
  const st = await api("/api/train/status");
  renderTrainStatus(st);
  if (st.state === "running") {
    state.trainPoll = setTimeout(pollTrain, 800);
  } else {
    state.trainPoll = null;
    if (st.state === "completed") {
      await loadMeta();
      toast(`Training complete · ${st.result?.version || ""}`);
    }
    if (st.state === "failed") toast(`Training failed: ${st.error || "unknown"}`);
  }
}

async function loadValidationDays() {
  const tw = $("valTrainWindow").value;
  const data = await api(`/api/validate/days?train_window=${encodeURIComponent(tw)}`);
  const sel = $("valDay");
  sel.innerHTML = (data.days || [])
    .map((d) => `<option value="${escapeAttr(d)}">${escapeHtml(d)}</option>`)
    .join("");
  if (!data.days?.length) {
    sel.innerHTML = `<option value="">No holdout days found</option>`;
  }
}

function renderValidation(result) {
  state.validateChart = renderBarChart("validateChart", result.chart, state.validateChart, "#c45c26");
  $("validateHeadline").textContent = result.headline || "";
  const lm = result.label_metrics || {};
  const am = result.area_metrics || {};
  $("validateSummary").innerHTML = `
    <span class="status ${Number(lm.f1_micro || 0) >= 0.45 ? "normal" : "surge"}">validation</span>
    <h3>${escapeHtml(result.headline || "")}</h3>
    <div class="metric-row">
      <span class="metric">tickets ${result.ticket_count}</span>
      <span class="metric">label F1µ ${Number(lm.f1_micro || 0).toFixed(2)}</span>
      <span class="metric">area match ${Number(am.exact_match_accuracy || 0).toFixed(2)}</span>
    </div>
    <ul>
      <li>Day ${escapeHtml(result.day)} was scored against already-populated labels and impacted area.</li>
      <li>Use Train tab to adjust parameters and re-run if accuracy is below your bar.</li>
    </ul>
  `;
  const rows = (result.rows || [])
    .map(
      (r) => `<tr>
      <td class="key">${escapeHtml(r.issue_key)}</td>
      <td>${escapeHtml((r.true_labels || []).join(", ") || "—")}</td>
      <td>${escapeHtml((r.pred_labels || []).join(", ") || "—")} ${r.label_hit ? "✓" : r.label_hit === false ? "✗" : ""}</td>
      <td>${escapeHtml(r.true_area || "—")}</td>
      <td>${escapeHtml(r.pred_area || "—")} ${r.area_hit ? "✓" : r.area_hit === false ? "✗" : ""}</td>
    </tr>`
    )
    .join("");
  $("validateTable").innerHTML = rows
    ? `<table class="triage-table"><thead><tr><th>Ticket</th><th>True labels</th><th>Pred labels</th><th>True area</th><th>Pred area</th></tr></thead><tbody>${rows}</tbody></table>`
    : "";
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    const on = p.id === `tab-${name}`;
    p.classList.toggle("active", on);
    p.hidden = !on;
  });
  if (name === "train") {
    loadParams().catch(console.error);
    pollTrain().catch(console.error);
  }
  if (name === "validate") loadValidationDays().catch(console.error);
  if (name === "ops") reloadOps().catch(console.error);
}

async function reloadOps() {
  await loadMeta();
  await Promise.all([loadIncomeTree(), loadForecast(), loadCharts(), loadTriage()]);
}

async function boot() {
  wireTriageClicks();
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => switchTab(t.dataset.tab))
  );
  $("dimension").addEventListener("change", () => loadCharts().catch(console.error));
  $("window").addEventListener("change", () => loadCharts().catch(console.error));
  $("hierarchyPeriod").addEventListener("change", () => loadIncomeTree().catch(console.error));
  $("forecastDimension").addEventListener("change", () => loadForecast().catch(console.error));
  $("triageDays").addEventListener("change", () => loadTriage().catch(console.error));
  $("valTrainWindow").addEventListener("change", () => loadValidationDays().catch(console.error));

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        await api("/api/source", { method: "POST", body: JSON.stringify({ mode: btn.dataset.mode }) });
        await reloadOps();
        toast(`Source set to ${btn.dataset.mode}`);
      } catch (err) {
        toast(`Source switch failed: ${err.message}`);
      }
    });
  });

  $("uploadForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const file = $("fileInput").files[0];
    if (!file) return toast("Choose an Excel/CSV export first");
    const body = new FormData();
    body.append("file", file);
    try {
      const result = await api("/api/import/spreadsheet", { method: "POST", body });
      await reloadOps();
      toast(`Imported ${result.rows} rows from ${file.name}`);
    } catch (err) {
      toast(`Import failed: ${err.message}`);
    }
  });

  $("jiraForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    try {
      const result = await api("/api/jira/connect", {
        method: "POST",
        body: JSON.stringify({
          base_url: $("jiraUrl").value,
          jql: $("jiraJql").value,
          email: $("jiraEmail").value || null,
          api_token: $("jiraToken").value || null,
          fetch_limit: 200,
        }),
      });
      await reloadOps();
      toast(`Fetched ${result.rows} issues from Jira`);
    } catch (err) {
      toast(`Jira connect failed — use Excel. ${err.message}`);
    }
  });

  $("btnRefresh").addEventListener("click", async () => {
    $("btnRefresh").disabled = true;
    try {
      await api("/api/refresh", { method: "POST" });
      await reloadOps();
      toast("Data refreshed");
    } catch (err) {
      toast(`Refresh failed: ${err.message}`);
    } finally {
      $("btnRefresh").disabled = false;
    }
  });

  $("btnRetrain").addEventListener("click", async () => {
    $("btnRetrain").disabled = true;
    try {
      const result = await api("/api/retrain-with-feedback", { method: "POST" });
      await reloadOps();
      toast(`Feedback retrain ${result.version}`);
    } catch (err) {
      toast(`Retrain failed: ${err.message}`);
    } finally {
      $("btnRetrain").disabled = false;
    }
  });

  $("btnTrain").addEventListener("click", async () => {
    $("btnTrain").disabled = true;
    try {
      await api("/api/train/start", {
        method: "POST",
        body: JSON.stringify({ window: $("trainWindow").value, params: collectParams() }),
      });
      toast("Training started");
      pollTrain();
    } catch (err) {
      toast(`Train failed: ${err.message}`);
    } finally {
      $("btnTrain").disabled = false;
    }
  });

  $("btnValidate").addEventListener("click", async () => {
    const day = $("valDay").value;
    if (!day) return toast("No validation day available");
    $("btnValidate").disabled = true;
    try {
      const result = await api("/api/validate/day", {
        method: "POST",
        body: JSON.stringify({ day, train_window: $("valTrainWindow").value }),
      });
      renderValidation(result);
      toast(`Validated ${day}`);
    } catch (err) {
      toast(`Validation failed: ${err.message}`);
    } finally {
      $("btnValidate").disabled = false;
    }
  });

  // Ensure spreadsheet data for Kafka demo
  await reloadOps();
}

boot().catch((err) => {
  console.error(err);
  toast(`UI boot failed: ${err.message}`);
});
