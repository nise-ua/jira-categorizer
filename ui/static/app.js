/* global Chart */

const state = {
  categoryChart: null,
  noveltyChart: null,
  forecastChart: null,
  triage: null,
  meta: null,
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
    } catch (_) {
      /* keep text */
    }
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function destroyChart(chart) {
  if (chart) chart.destroy();
}

function renderLineChart(canvasId, payload, previous, extraOptions = {}) {
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
        ...(extraOptions.plugins || {}),
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

async function loadForecast() {
  const dimension = $("forecastDimension").value;
  const payload = await api(`/api/forecast?dimension=${encodeURIComponent(dimension)}`);
  const plugins = {};
  if (typeof payload.tomorrow_start_index === "number") {
    plugins.annotation = undefined; // keep Chart.js free of plugin deps
  }
  // Visual: slightly emphasize tomorrow labels via dataset segment already in API
  state.forecastChart = renderLineChart("forecastChart", payload, state.forecastChart, { plugins });
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
    mount.innerHTML = `<div class="empty">No tickets in this inbox window. Widen the range, import Excel, or refresh.</div>`;
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
        <td>
          <button class="btn small" type="button" data-action="save">Save</button>
        </td>
      </tr>`;
    })
    .join("");

  mount.innerHTML = `
    <table class="triage-table">
      <thead>
        <tr>
          <th>Ticket</th>
          <th>Summary / description</th>
          <th>Label</th>
          <th>Impacted area</th>
          <th></th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function loadTriage() {
  const days = $("triageDays").value;
  const data = await api(`/api/triage?days=${encodeURIComponent(days)}&limit=40`);
  renderTriage(data);
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
    const jiraNote = result.jira?.dry_run
      ? " (local only — no Jira write creds)"
      : result.jira?.ok
        ? " → Jira updated"
        : result.jira?.skipped
          ? ""
          : " → Jira push failed";
    toast(`${item.issue_key} saved${result.disagreed ? " · correction queued" : ""}${jiraNote}`);
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
    if (btn.dataset.action === "save") {
      await saveRow(tr);
    }
  });
}

function setModeButtons(mode) {
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  const hints = {
    cache: "Using local cached tickets / bundled sample. Fine for offline demos.",
    spreadsheet: "Using uploaded Excel/CSV Jira export. Best when Jira API is unavailable.",
    jira: "Using live Jira API when reachable; falls back to cache if the call fails.",
  };
  $("sourceHint").textContent = hints[mode] || "";
}

async function loadMeta() {
  const meta = await api("/api/meta");
  state.meta = meta;
  setModeButtons(meta.source_mode || "cache");
  const src = (meta.source && meta.source.source) || "n/a";
  $("metaBadge").textContent = `${meta.rows} tickets · ${meta.source_mode} · model ${
    meta.model_version || "n/a"
  } · ${meta.jira_writable ? "jira write-on" : "offline/local"}`;
  if ($("jiraUrl") && !($("jiraUrl").value) && src.startsWith("jira:")) {
    /* keep user fields */
  }
  return meta;
}

async function reloadAll() {
  await loadMeta();
  await Promise.all([loadForecast(), loadCharts(), loadTriage()]);
}

async function boot() {
  wireTriageClicks();
  $("dimension").addEventListener("change", () => loadCharts().catch(console.error));
  $("window").addEventListener("change", () => loadCharts().catch(console.error));
  $("forecastDimension").addEventListener("change", () => loadForecast().catch(console.error));
  $("triageDays").addEventListener("change", () => loadTriage().catch(console.error));

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        await api("/api/source", { method: "POST", body: JSON.stringify({ mode: btn.dataset.mode }) });
        await reloadAll();
        toast(`Source set to ${btn.dataset.mode}`);
      } catch (err) {
        toast(`Source switch failed: ${err.message}`);
      }
    });
  });

  $("uploadForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const file = $("fileInput").files[0];
    if (!file) {
      toast("Choose an Excel/CSV export first");
      return;
    }
    const body = new FormData();
    body.append("file", file);
    try {
      const result = await api("/api/import/spreadsheet", { method: "POST", body });
      await reloadAll();
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
      await reloadAll();
      toast(`Fetched ${result.rows} issues from Jira`);
    } catch (err) {
      toast(`Jira connect failed — use Excel import. ${err.message}`);
    }
  });

  $("btnRefresh").addEventListener("click", async () => {
    $("btnRefresh").disabled = true;
    try {
      await api("/api/refresh", { method: "POST" });
      await reloadAll();
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
      await reloadAll();
      toast(
        `Retrained ${result.version} · labels F1 ${Number(result.metrics.labels_f1_macro || 0).toFixed(2)} · area F1 ${Number(result.metrics.area_f1_macro || 0).toFixed(2)}`
      );
    } catch (err) {
      toast(`Retrain failed: ${err.message}`);
    } finally {
      $("btnRetrain").disabled = false;
    }
  });

  await reloadAll();
}

boot().catch((err) => {
  console.error(err);
  toast(`UI boot failed: ${err.message}`);
});
