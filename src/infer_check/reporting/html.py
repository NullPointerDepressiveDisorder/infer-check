"""HTML report generator for infer-check results.

Produces a single self-contained HTML file with inline CSS/JS (no external
dependencies) from cached JSON result files. The report includes an executive
summary, a quantization sweep heatmap, a cross-backend comparison table,
failure cards with a "Copy as GitHub issue" button, and a determinism table.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, Undefined
from pydantic import ValidationError

from infer_check.types import (
    CompareResult,
    ComparisonResult,
    DeterminismResult,
    StressResult,
    SweepResult,
)

__all__ = ["generate_report"]

# ---------------------------------------------------------------------------
# Jinja2 HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>infer-check Report</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0f0f17;
      --surface: #1a1a2e;
      --surface2: #16213e;
      --border: #2a2a4a;
      --accent: #4f8ef7;
      --accent2: #a78bfa;
      --text: #e2e8f0;
      --text-dim: #94a3b8;
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --radius: 8px;
      --font: 'Inter', system-ui, sans-serif;
    }

    body {
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }

    /* ── Header ── */
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 2rem;
      display: flex;
      align-items: center;
      height: 56px;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    header h1 {
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--accent);
    }
    header .badge {
      margin-left: 0.75rem;
      font-size: 0.7rem;
      background: var(--accent2);
      color: #fff;
      border-radius: 999px;
      padding: 2px 10px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    header nav { margin-left: auto; display: flex; gap: 1.5rem; }
    header nav a {
      color: var(--text-dim);
      text-decoration: none;
      font-size: 0.85rem;
      transition: color 0.2s;
    }
    header nav a:hover { color: var(--text); }

    /* ── Layout ── */
    .container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }

    section { margin-bottom: 3rem; }
    section h2 {
      font-size: 1.05rem;
      font-weight: 600;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 1.25rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
    }

    /* ── Executive Summary ── */
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 1.25rem;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1.5rem;
    }
    .stat-card .label {
      font-size: 0.75rem;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.4rem;
    }
    .stat-card .value {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1.1;
    }
    .stat-card .value.ok { color: var(--success); }
    .stat-card .value.warn { color: var(--warning); }
    .stat-card .value.bad { color: var(--danger); }
    .verdict {
      background: var(--surface2);
      border-left: 3px solid var(--accent);
      border-radius: 0 var(--radius) var(--radius) 0;
      padding: 0.75rem 1.25rem;
      font-size: 0.95rem;
      color: var(--text);
    }

    /* ── Tables ── */
    .table-wrap { overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--border); }
    table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
    th {
      background: var(--surface);
      color: var(--text-dim);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 0.75rem;
      padding: 0.6rem 0.9rem;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      text-align: left;
    }
    td {
      padding: 0.55rem 0.9rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255,255,255,0.02); }

    /* Heatmap */
    .heat-cell {
      text-align: center;
      font-size: 0.78rem;
      font-weight: 500;
      border-radius: 4px;
      padding: 0.2rem 0.4rem;
      min-width: 60px;
      display: inline-block;
    }

    /* ── Cliff callout ── */
    .cliff-callout {
      margin-top: 1rem;
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid rgba(239, 68, 68, 0.4);
      border-radius: var(--radius);
      padding: 0.75rem 1.25rem;
      font-size: 0.875rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .cliff-callout .icon { font-size: 1.1rem; }

    /* ── Details / Expandable rows ── */
    .subhead { font-size:0.8rem; text-transform:uppercase; color:var(--text-dim); margin:0.5rem 0 0.75rem; letter-spacing:0.05em; }
    details summary {
      cursor: pointer;
      color: var(--accent);
      font-size: 0.8rem;
      user-select: none;
      padding: 0.15rem 0;
    }
    details[open] summary { margin-bottom: 0.4rem; }
    .output-pre {
      background: #0a0a14;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.75rem;
      font-size: 0.78rem;
      white-space: pre-wrap;
      word-break: break-word;
      color: #c4cad6;
      max-height: 280px;
      overflow-y: auto;
      line-height: 1.5;
    }

    /* ── Failure Cards ── */
    .cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 1.25rem;
    }
    .failure-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }
    .failure-card .card-header {
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .failure-card .prompt-text {
      font-size: 0.82rem;
      color: var(--text-dim);
      flex: 1 1 100%;
      font-style: italic;
      border-left: 2px solid var(--border);
      padding-left: 0.5rem;
      margin-top: 0.2rem;
    }
    .chip {
      font-size: 0.7rem;
      border-radius: 999px;
      padding: 2px 10px;
      font-weight: 600;
      white-space: nowrap;
    }
    .chip-cat { background: rgba(79,142,247,0.18); color: #93c5fd; }
    .chip-fail { background: rgba(239,68,68,0.15); color: #fca5a5; }
    .chip-ok { background: rgba(34,197,94,0.15); color: #86efac; }
    .metrics-row {
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
      font-size: 0.8rem;
    }
    .metric { display: flex; flex-direction: column; }
    .metric .mlabel { color: var(--text-dim); font-size: 0.7rem; text-transform: uppercase; }
    .metric .mval { font-weight: 600; }
    .copy-btn {
      margin-top: auto;
      background: rgba(79,142,247,0.12);
      color: var(--accent);
      border: 1px solid rgba(79,142,247,0.3);
      border-radius: 6px;
      padding: 0.4rem 0.85rem;
      font-size: 0.78rem;
      font-weight: 500;
      font-family: var(--font);
      cursor: pointer;
      transition: background 0.2s, border-color 0.2s;
      align-self: flex-start;
    }
    .copy-btn:hover { background: rgba(79,142,247,0.22); border-color: var(--accent); }
    .copy-btn.copied {
      background: rgba(34,197,94,0.15);
      color: #86efac;
      border-color: rgba(34,197,94,0.3);
    }

    /* ── Footer ── */
    footer {
      margin-top: 4rem;
      border-top: 1px solid var(--border);
      padding: 1.5rem;
      text-align: center;
      color: var(--text-dim);
      font-size: 0.78rem;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  </style>
</head>
<body>

<header>
  <h1>⚡ infer-check</h1>
  <span class="badge">Report</span>
  <nav>
    <a href="#summary">Summary</a>
    {% if sweep_rows %}<a href="#sweep">Sweep</a>{% endif %}
    {% if compare_runs %}<a href="#compare">Compare</a>{% endif %}
    {% if diff_rows %}<a href="#diff">Diff</a>{% endif %}
    {% if failures %}<a href="#cards">Failures</a>{% endif %}
    {% if stress_rows %}<a href="#stress">Stress</a>{% endif %}
    {% if determinism_rows %}<a href="#determinism">Determinism</a>{% endif %}
  </nav>
</header>

<div class="container">

  <!-- ── SECTION 1: Executive Summary ── -->
  <section id="summary">
    <h2>Executive Summary</h2>
    <div class="summary-grid">
      {% if graded_totals %}
      <div class="stat-card">
        <div class="label">Graded Comparisons</div>
        <div class="value">{{ graded_totals.graded }}</div>
      </div>
      <div class="stat-card">
        <div class="label">Regressions</div>
        <div class="value {{ 'bad' if graded_totals.regressions > graded_totals.improvements else 'ok' }}">{{ graded_totals.regressions }}</div>
      </div>
      <div class="stat-card">
        <div class="label">Improvements</div>
        <div class="value">{{ graded_totals.improvements }}</div>
      </div>
      {% endif %}
      <div class="stat-card">
        <div class="label">Comparisons</div>
        <div class="value">{{ total_tests }}</div>
      </div>
      <div class="stat-card">
        <div class="label">Text Similarity &lt; 50%</div>
        <div class="value {{ 'warn' if total_failures > 0 else 'ok' }}">{{ total_failures }}</div>
      </div>
      <div class="stat-card">
        <div class="label">Backends</div>
        <div class="value">{{ backend_count }}</div>
      </div>
      <div class="stat-card">
        <div class="label">Quant Levels</div>
        <div class="value">{{ quant_count }}</div>
      </div>
    </div>
    <div class="verdict">{{ verdict }}</div>
  </section>

  <!-- ── SECTION 2: Quantization Sweep ── -->
  {% if sweep_rows %}
  <section id="sweep">
    <h2>Quantization Sweep</h2>
    {% if sweep_levels %}
    <h4 class="subhead">By level, against the baseline</h4>
    <div class="table-wrap" style="margin-bottom:1.5rem;">
      <table style="font-size:0.85rem;">
        <thead>
          <tr>
            <th>Level</th>
            <th>Accuracy (baseline → level)</th>
            <th>Regressions / improvements</th>
            <th>McNemar p</th>
            <th>Mean KL</th>
            <th>Top-1 agreement</th>
          </tr>
        </thead>
        <tbody>
          {% for row in sweep_levels %}
          <tr>
            <td><code>{{ row.level }}</code>{% if row.is_baseline %} <small style="color:var(--text-dim);">(self-check)</small>{% endif %}</td>
            <td>{% if row.correctness %}{{ row.correctness.a.acc }} → {{ row.correctness.b.acc }} <small style="color:var(--text-dim);">({{ row.correctness.graded }} graded)</small>{% else %}—{% endif %}</td>
            <td>{% if row.correctness %}<span class="chip {{ 'chip-fail' if row.correctness.significant_regression else 'chip-ok' }}">{{ row.correctness.regressions }} / {{ row.correctness.improvements }}</span>{% else %}—{% endif %}</td>
            <td style="font-family:monospace;">{{ row.correctness.p if row.correctness else '—' }}</td>
            <td style="font-family:monospace;">{{ row.kl }}</td>
            <td>{{ row.top1 }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% endif %}
    <h4 class="subhead">Text similarity per prompt</h4>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Prompt ID</th>
            {% for q in quant_cols %}<th>{{ q }}</th>{% endfor %}
          </tr>
        </thead>
        <tbody>
          {% for row in sweep_rows %}
          <tr>
            <td style="font-family:monospace;font-size:0.78rem;">{{ row.prompt_id }}</td>
            {% for cell in row.cells %}
            <td>
              {% if cell.value is not none %}
              <span class="heat-cell" style="background-color:{{ cell.bg }};color:{{ cell.fg }};">
                {{ cell.label }}
              </span>
              {% else %}
              <span style="color:var(--text-dim);font-size:0.75rem;">—</span>
              {% endif %}
            </td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% if degradation_cliff %}
    <div class="cliff-callout">
      <span class="icon">⚠️</span>
      <span><strong>Degradation cliff</strong> at <code>{{ degradation_cliff.level }}</code>: the first
        level (in the order tested) with a statistically significant accuracy loss vs. the baseline —
        {{ degradation_cliff.regressions }} regressions vs {{ degradation_cliff.improvements }} improvements,
        exact McNemar p = {{ '%.2g' % degradation_cliff.mcnemar_p }}.</span>
    </div>
    {% endif %}
  </section>
  {% endif %}

  <!-- ── SECTION: Model Comparison ── -->
  {% if compare_runs %}
  <section id="compare">
    <h2>Model Comparison Runs</h2>
    {% for run in compare_runs %}
    <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1.5rem; margin-bottom:1.5rem;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.25rem; border-bottom:1px solid var(--border); padding-bottom:1rem;">
        <div>
          <h3 style="font-size:1rem; margin-bottom:0.25rem;">{{ run.model_a }} <span style="color:var(--text-dim); font-weight:400;">vs</span> {{ run.model_b }}</h3>
          <p style="font-size:0.8rem; color:var(--text-dim);">{{ run.backend_a }} &middot; {{ run.backend_b }} &middot; {{ run.timestamp }}</p>
        </div>
        <div style="text-align:right;">
          {% if run.correctness %}
          <div class="chip {{ 'chip-fail' if run.correctness.significant_regression else 'chip-ok' }}" style="font-size:0.9rem; padding:4px 12px;">
            {{ run.correctness.regressions }} regressions / {{ run.correctness.improvements }} improvements
          </div>
          {% else %}
          <div class="chip {{ 'chip-fail' if run.flip_rate > 0.1 else 'chip-ok' }}" style="font-size:0.9rem; padding:4px 12px;">
            Flip Rate: {{ run.flip_rate_pct }}%
          </div>
          {% endif %}
        </div>
      </div>

      {% if run.correctness %}
      {% set cr = run.correctness %}
      <h4 class="subhead">Correctness vs answer keys ({{ cr.graded }} graded prompts)</h4>
      <div class="metrics-row" style="margin-bottom:1rem; gap:2rem;">
        <div class="metric">
          <span class="mlabel">Accuracy {{ cr.label_a }}</span>
          <span class="mval" style="font-size:1.25rem;">{{ cr.a.acc }}</span>
          <small style="color:var(--text-dim);">95% CI {{ cr.a.ci }} · {{ cr.a.frac }}</small>
        </div>
        <div class="metric">
          <span class="mlabel">Accuracy {{ cr.label_b }}</span>
          <span class="mval" style="font-size:1.25rem;">{{ cr.b.acc }}</span>
          <small style="color:var(--text-dim);">95% CI {{ cr.b.ci }} · {{ cr.b.frac }}</small>
        </div>
        <div class="metric">
          <span class="mlabel">Exact McNemar p</span>
          <span class="mval" style="font-size:1.25rem;">{{ cr.p }}</span>
          <small style="color:var(--text-dim);">on {{ cr.regressions + cr.improvements }} discordant pairs</small>
        </div>
      </div>
      <div class="table-wrap" style="margin-bottom:0.5rem; max-width:520px;">
        <table style="font-size:0.85rem;">
          <thead><tr><th></th><th>{{ cr.label_b }} right</th><th>{{ cr.label_b }} wrong</th></tr></thead>
          <tbody>
            <tr><th>{{ cr.label_a }} right</th><td>{{ cr.both_right }}</td><td><span class="chip chip-fail">{{ cr.regressions }}</span></td></tr>
            <tr><th>{{ cr.label_a }} wrong</th><td><span class="chip chip-ok">{{ cr.improvements }}</span></td><td>{{ cr.both_wrong }}</td></tr>
          </tbody>
        </table>
      </div>
      {% if cr.notes %}<p style="font-size:0.75rem; color:var(--text-dim); margin-bottom:1.25rem;">{{ cr.notes }}</p>{% endif %}
      {% endif %}

      <div class="metrics-row" style="margin-bottom:1.5rem; gap:2rem;">
        <div class="metric">
          <span class="mlabel">Mean KL</span>
          <span class="mval" style="font-size:1.25rem;">{{ run.mean_kl }}</span>
          <small style="color:var(--text-dim);">{{ run.kl_method }}</small>
        </div>
        <div class="metric">
          <span class="mlabel">Top-1 Agreement</span>
          <span class="mval" style="font-size:1.25rem;">{{ run.top1 }}</span>
        </div>
        <div class="metric">
          <span class="mlabel">Flip Rate</span>
          <span class="mval" style="font-size:1.25rem;">{{ run.flip_rate_pct }}%</span>
        </div>
        <div class="metric">
          <span class="mlabel">Mean Text Similarity</span>
          <span class="mval" style="font-size:1.25rem;">{{ run.mean_similarity_pct }}%</span>
        </div>
        <div class="metric">
          <span class="mlabel">Total Comparisons</span>
          <span class="mval" style="font-size:1.25rem;">{{ run.count }}</span>
        </div>
      </div>

      {% if run.per_category %}
      <h4 style="font-size:0.8rem; text-transform:uppercase; color:var(--text-dim); margin-bottom:0.75rem; letter-spacing:0.05em;">Per-Category Stats</h4>
      <div class="table-wrap">
        <table style="font-size:0.8rem;">
          <thead>
            <tr>
              <th>Category</th>
              <th>Flip Rate</th>
              <th>Mean Similarity</th>
            </tr>
          </thead>
          <tbody>
            {% for cat, stats in run.per_category.items() %}
            <tr>
              <td><code>{{ cat }}</code></td>
              <td>{{ (stats.flip_rate * 100) | round(1) }}%</td>
              <td>{{ (stats.mean_similarity * 100) | round(1) }}%</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endif %}
    </div>
    {% endfor %}
  </section>
  {% endif %}

  <!-- ── SECTION 3: Cross-Backend Comparison ── -->
  {% if diff_rows %}
  <section id="diff">
    <h2>Cross-Backend Comparison</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Prompt ID</th>
            <th>Baseline</th>
            <th>Test Backend</th>
            <th>Similarity</th>
            <th>Answer</th>
            <th>KL</th>
            <th>Outputs</th>
          </tr>
        </thead>
        <tbody>
          {% for row in diff_rows %}
          <tr>
            <td style="font-family:monospace;font-size:0.78rem;">{{ row.prompt_id }}</td>
            <td>
              {{ row.baseline_backend }}<br/>
              <small style="color:var(--text-dim);">{{ row.baseline_quant }}</small>
            </td>
            <td>
              {{ row.test_backend }}<br/>
              <small style="color:var(--text-dim);">{{ row.test_quant }}</small>
            </td>
            <td>
              <span class="chip {{ 'chip-fail' if row.is_failure else 'chip-ok' }}">
                {{ row.similarity }}%
              </span>
            </td>
            <td>{% if row.grades %}<span class="chip {{ 'chip-fail' if row.regressed else 'chip-ok' }}">{{ row.grades }}</span>{% else %}<span style="color:var(--text-dim);">—</span>{% endif %}</td>
            <td style="font-family:monospace;">{{ row.kl }}</td>
            <td>
              <details>
                <summary>Show outputs</summary>
                <div style="margin-top:0.4rem;">
                  <div style="font-size:0.72rem;color:var(--text-dim);margin-bottom:0.2rem;">
                    Baseline ({{ row.baseline_backend }}):
                  </div>
                  <pre class="output-pre">{{ row.baseline_text }}</pre>
                  <div style="font-size:0.72rem;color:var(--text-dim);margin:0.4rem 0 0.2rem;">
                    Test ({{ row.test_backend }}):
                  </div>
                  <pre class="output-pre">{{ row.test_text }}</pre>
                </div>
              </details>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>
  {% endif %}

  <!-- ── SECTION: Stress Test Results ── -->
  {% if stress_rows %}
  <section id="stress">
    <h2>Stress Test Results</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model ID</th>
            <th>Backend</th>
            <th>In flight</th>
            <th>Compared</th>
            <th>Errors</th>
            <th>Identical to solo</th>
            <th>Diverged</th>
          </tr>
        </thead>
        <tbody>
          {% for row in stress_rows %}
          <tr>
            <td style="font-family:monospace;font-size:0.78rem;">{{ row.model_id }}</td>
            <td>{{ row.backend_name }}</td>
            <td>{{ row.concurrency_level }}{% if row.is_reference %} <small style="color:var(--text-dim);">(reference)</small>{% endif %}</td>
            <td>{{ row.compared }}</td>
            <td>
              <span class="chip {{ 'chip-fail' if row.error_count > 0 else 'chip-ok' }}" title="{{ row.error_messages }}">
                {{ row.error_count }}
              </span>
            </td>
            <td>
              {% if row.is_reference %}—{% else %}
              <span class="chip {{ 'chip-ok' if row.consistency_raw == 1.0 else 'chip-fail' }}">
                {{ row.output_consistency }}%
              </span>
              {% endif %}
            </td>
            <td>
              {% if row.divergences %}
              <details>
                <summary>{{ row.divergences | length }} prompt(s)</summary>
                <ul style="font-size:0.75rem; margin:0.4rem 0 0 1rem;">
                  {% for d in row.divergences %}
                  <li><code>{{ d.prompt_id }}</code>: first differing token {{ d.first_divergence }}; sent with {{ d.co_scheduled | join(", ") }}</li>
                  {% endfor %}
                </ul>
              </details>
              {% else %}—{% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>
  {% endif %}

  <!-- ── SECTION 4: Failure Cards ── -->
  {% if failures %}
  <section id="cards">
    <h2>Failure Cards</h2>
    <div class="cards-grid">
      {% for f in failures %}
      <div class="failure-card">
        <div class="card-header">
          <span class="chip chip-cat">{{ f.category }}</span>
          <span class="chip chip-fail">FAIL</span>
        </div>
        <div class="prompt-text">{{ f.prompt_text }}</div>
        <div class="metrics-row">
          <div class="metric">
            <span class="mlabel">Similarity</span>
            <span class="mval">{{ f.similarity }}%</span>
          </div>
          <div class="metric">
            <span class="mlabel">KL Div</span>
            <span class="mval">{{ f.kl }}</span>
          </div>
          <div class="metric">
            <span class="mlabel">Div. Token</span>
            <span class="mval">{{ f.div_index }}</span>
          </div>
        </div>
        <div style="font-size:0.78rem;color:var(--text-dim);">
          {{ f.baseline_backend }} → {{ f.test_backend }}
          {% if f.test_quant %} &nbsp;|&nbsp; <code>{{ f.test_quant }}</code>{% endif %}
        </div>
        <button
          class="copy-btn"
          data-issue="{{ f.github_issue | e }}"
          onclick="copyIssue(this)">
          📋 Copy as GitHub issue
        </button>
      </div>
      {% endfor %}
    </div>
  </section>
  {% endif %}

  <!-- ── SECTION 5: Determinism Results ── -->
  {% if determinism_rows %}
  <section id="determinism">
    <h2>Determinism Results</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Prompt ID</th>
            <th>Backend</th>
            <th>Quant</th>
            <th>Runs</th>
            <th>Determinism Score</th>
            <th>Unique Outputs</th>
            <th>First Divergence (token)</th>
          </tr>
        </thead>
        <tbody>
          {% for row in determinism_rows %}
          <tr>
            <td style="font-family:monospace;font-size:0.78rem;">{{ row.prompt_id }}</td>
            <td>{{ row.backend_name }}</td>
            <td>{{ row.quantization }}</td>
            <td>{{ row.num_runs }}</td>
            <td>
              {% set s_cls = 'chip-ok' if row.score >= 0.9 else
                             ('chip-cat' if row.score >= 0.7 else 'chip-fail') %}
              <span class="chip {{ s_cls }}">
                {{ row.score_pct }}%
              </span>
            </td>
            <td>{{ row.num_unique }}</td>
            <td style="font-family:monospace;">{{ row.first_divergence }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>
  {% endif %}

</div>

<footer>
  Generated by <strong>infer-check</strong> &middot; {{ generated_at }}
</footer>

<script>
  function copyIssue(btn) {
    const text = btn.getAttribute('data-issue');
    if (!navigator.clipboard) {
      console.warn('Clipboard API not available');
      return;
    }
    navigator.clipboard.writeText(text).then(() => {
      btn.textContent = '✅ Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = '📋 Copy as GitHub issue';
        btn.classList.remove('copied');
      }, 2000);
    }).catch(err => console.error('Copy failed:', err));
  }
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helper: colour mapping
# ---------------------------------------------------------------------------


def _similarity_to_hsl(similarity: float) -> tuple[str, str]:
    """Map a [0, 1] similarity to an HSL background and a foreground colour.

    ``1.0`` → green; ``0.0`` → red.  Foreground is always readable against
    the dark background.
    """
    # Clamp.
    sim = max(0.0, min(1.0, similarity))
    # Hue: 0° (red) → 120° (green).
    hue = int(sim * 120)
    bg = f"hsl({hue}, 55%, 22%)"
    fg = f"hsl({hue}, 90%, 75%)"
    return bg, fg


# ---------------------------------------------------------------------------
# File classification (mirrors json_export logic, kept local for independence)
# ---------------------------------------------------------------------------


def _try_load(raw: Any, cls: Any) -> Any | None:
    """Parse *raw* as *cls*, or None if it's a different result type."""
    if not isinstance(raw, dict):
        return None
    try:
        return cls.model_validate(raw)
    except ValidationError:
        return None


def _try_load_list(raw: Any, cls: Any) -> list[Any] | None:
    """Parse *raw* as a list of *cls*, or None if it's a different result type."""
    if not isinstance(raw, list) or not raw:
        return None
    try:
        return [cls.model_validate(item) for item in raw]
    except ValidationError:
        return None


def _load_results(results_dir: Path) -> dict[str, list[Any]]:
    """Load and classify all JSON files in *results_dir*."""
    sections: dict[str, list[Any]] = {
        "sweep": [],
        "diff": [],
        "stress": [],
        "determinism": [],
        "compare": [],
    }

    for path in sorted(results_dir.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        compare = _try_load(raw, CompareResult)
        if compare is not None:
            sections["compare"].append(compare)
            continue

        sweep = _try_load(raw, SweepResult)
        if sweep is not None:
            sections["sweep"].append(sweep)
            continue

        comp_list = _try_load_list(raw, ComparisonResult)
        if comp_list is not None:
            sections["diff"].append(comp_list)
            continue

        stress = _try_load(raw, StressResult)
        if stress is not None:
            sections["stress"].append(stress)
            continue

        stress_list = _try_load_list(raw, StressResult)
        if stress_list is not None:
            sections["stress"].extend(stress_list)
            continue

        det = _try_load(raw, DeterminismResult)
        if det is not None:
            sections["determinism"].append(det)
            continue

        det_list = _try_load_list(raw, DeterminismResult)
        if det_list is not None:
            sections["determinism"].extend(det_list)

    return sections


# ---------------------------------------------------------------------------
# Template context builders
# ---------------------------------------------------------------------------


def _build_sweep_context(
    sweeps: list[SweepResult],
) -> dict[str, Any]:
    """Build heatmap rows and column headers for the sweep section."""
    if not sweeps:
        return {"sweep_rows": [], "quant_cols": [], "degradation_cliff": None}

    # Gather all quant levels (columns).
    quant_set: set[str] = set()
    for sweep in sweeps:
        for comp in sweep.comparisons:
            quant_set.add(comp.test.quantization or "?")
    quant_cols = sorted(quant_set)

    # Index: prompt_id → {quant → ComparisonResult}
    prompt_quant: dict[str, dict[str, ComparisonResult]] = {}
    for sweep in sweeps:
        for comp in sweep.comparisons:
            pid = comp.baseline.prompt_id
            q = comp.test.quantization or "?"
            prompt_quant.setdefault(pid, {})[q] = comp

    # Build row dicts.
    rows = []
    for pid, quant_map in sorted(prompt_quant.items()):
        cells: list[dict[str, Any]] = []
        for q in quant_cols:
            c = quant_map.get(q)
            if c is None:
                cells.append({"value": None, "bg": "", "fg": "", "label": ""})
            else:
                sim = c.text_similarity
                bg, fg = _similarity_to_hsl(sim)
                cells.append(
                    {
                        "value": sim,
                        "bg": bg,
                        "fg": fg,
                        "label": f"{sim * 100:.0f}%",
                    }
                )
        rows.append({"prompt_id": pid[:32], "cells": cells})

    # Degradation cliff from the first sweep's summary if present.
    cliff = None
    for sweep in sweeps:
        c = sweep.summary.get("degradation_cliff") if sweep.summary else None
        if isinstance(c, dict):  # older result files stored a bare string from a different rule
            cliff = c
            break

    return {"sweep_rows": rows, "quant_cols": quant_cols, "degradation_cliff": cliff}


def _fmt_pct(x: float | None, digits: int = 1) -> str:
    return "N/A" if x is None else f"{x * 100:.{digits}f}%"


def _kl_method_label(method: str | None) -> str:
    return {"teacher_forced": "teacher-forced", "prefix": "prefix (approximate)"}.get(method or "", "")


def _kl_text(mean: float | None, method: str | None) -> str:
    if mean is None:
        return "N/A"
    label = _kl_method_label(method)
    return f"{mean:.4f} ({label})" if label else f"{mean:.4f}"


def _correctness_view(table: dict[str, Any] | None, label_a: str, label_b: str) -> dict[str, Any] | None:
    """Display-ready version of a ``paired_correctness`` table (None if no answer keys)."""
    if not table:
        return None

    def side(s: dict[str, Any]) -> dict[str, Any]:
        ci = s.get("accuracy_ci95")
        return {
            "acc": _fmt_pct(s.get("accuracy")),
            "ci": f"{ci[0] * 100:.1f}–{ci[1] * 100:.1f}%" if ci else "N/A",
            "frac": f"{s.get('correct', 0)}/{s.get('scored', 0)}",
        }

    regressions, improvements = table.get("regressions", 0), table.get("improvements", 0)
    p = float(table.get("mcnemar_p", 1.0))
    notes = []
    a, b = table.get("a", {}), table.get("b", {})
    if a.get("unparsed") or b.get("unparsed"):
        notes.append(f"No parseable answer: {a.get('unparsed', 0)} ({label_a}) / {b.get('unparsed', 0)} ({label_b}).")
    if a.get("truncated") or b.get("truncated"):
        notes.append(
            f"Hit the token limit where the other side finished, counted wrong: "
            f"{a.get('truncated', 0)} ({label_a}) / {b.get('truncated', 0)} ({label_b})."
        )
    if table.get("both_truncated"):
        notes.append(f"Both hit the token limit, excluded: {table['both_truncated']}.")
    return {
        "label_a": label_a,
        "label_b": label_b,
        "graded": table.get("n_graded", 0),
        "a": side(a),
        "b": side(b),
        "both_right": table.get("both_right", 0),
        "regressions": regressions,
        "improvements": improvements,
        "both_wrong": table.get("both_wrong", 0),
        "p": f"{p:.3g}",
        "significant_regression": regressions > improvements and p < 0.05,
        "notes": " ".join(notes),
    }


def _build_sweep_levels(sweeps: list[SweepResult]) -> list[dict[str, Any]]:
    """One row per quantization level: correctness and KL against the baseline."""
    rows: list[dict[str, Any]] = []
    for sweep in sweeps:
        summary = sweep.summary or {}
        corr = summary.get("correctness_by_quant") or {}
        kl = summary.get("kl_by_quant") or {}
        if not corr and not kl:
            continue  # results from before these summaries existed
        base = summary.get("baseline_quant") or (sweep.quantization_levels[0] if sweep.quantization_levels else "")
        for level in sweep.quantization_levels:
            k = kl.get(level) or {}
            rows.append(
                {
                    "level": level,
                    "is_baseline": level == base,
                    "correctness": _correctness_view(corr.get(level), base, level),
                    "kl": _kl_text(k.get("mean_kl"), k.get("method")),
                    "top1": _fmt_pct(k.get("mean_top1_agreement"), 2),
                }
            )
    return rows


def _build_compare_context(compare_results: list[CompareResult]) -> list[dict[str, Any]]:
    """Build summary data for the model comparison section."""
    runs = []
    for c in compare_results:
        runs.append(
            {
                "model_a": c.model_a,
                "model_b": c.model_b,
                "backend_a": c.backend_a,
                "backend_b": c.backend_b,
                "flip_rate": c.flip_rate,
                "flip_rate_pct": f"{c.flip_rate * 100:.1f}",
                "mean_similarity_pct": f"{c.mean_text_similarity * 100:.1f}"
                if c.mean_text_similarity is not None
                else "N/A",
                "mean_kl": f"{c.mean_kl_divergence:.4f}" if c.mean_kl_divergence is not None else "N/A",
                "kl_method": _kl_method_label(c.kl_method),
                "top1": _fmt_pct(c.mean_top1_agreement, 2),
                "correctness": _correctness_view(c.correctness, c.model_a, c.model_b),
                "count": len(c.comparisons),
                "timestamp": c.timestamp.strftime("%Y-%m-%d %H:%M"),
                "per_category": c.per_category_stats,
            }
        )
    return runs


def _build_diff_context(diff_batches: list[list[ComparisonResult]]) -> list[dict[str, Any]]:
    """Build rows for the cross-backend comparison table."""
    rows: list[dict[str, Any]] = []
    for batch in diff_batches:
        for comp in batch:
            rows.append(
                {
                    "prompt_id": comp.baseline.prompt_id[:32],
                    "baseline_backend": comp.baseline.backend_name,
                    "baseline_quant": comp.baseline.quantization or "—",
                    "test_backend": comp.test.backend_name,
                    "test_quant": comp.test.quantization or "—",
                    "similarity": f"{comp.text_similarity * 100:.1f}",
                    "similarity_raw": comp.text_similarity,
                    "kl": _kl_text(comp.kl_divergence, comp.kl_method),
                    "grades": (
                        f"{comp.metadata['grade_a']} → {comp.metadata['grade_b']}"
                        if comp.metadata.get("grade_a") and comp.metadata.get("grade_b")
                        else None
                    ),
                    "regressed": comp.metadata.get("grade_a") == "correct"
                    and comp.metadata.get("grade_b") != "correct",
                    "baseline_text": comp.baseline.text,
                    "test_text": comp.test.text,
                    "is_failure": comp.is_failure,
                }
            )
    # Sort worst-first (lowest similarity).
    rows.sort(key=lambda r: float(r["similarity_raw"]))
    return rows


def _build_failure_cards(
    diff_batches: list[list[ComparisonResult]],
) -> list[dict[str, Any]]:
    """Build failure card data including the GitHub issue body."""
    from infer_check.reporting.github_issue import format_issue

    cards = []
    for batch in diff_batches:
        for comp in batch:
            if not comp.is_failure:
                continue
            category = (
                comp.metadata.get("prompt_category")
                or comp.metadata.get("category")
                or comp.test.metadata.get("prompt_category")
                or comp.test.metadata.get("category")
                or comp.baseline.metadata.get("prompt_category")
                or comp.baseline.metadata.get("category")
                or "general"
            )
            prompt_text = comp.metadata.get("prompt_text") or comp.baseline.metadata.get("prompt_text", "")
            prompt_text = comp.baseline.text[:200] if not prompt_text else str(prompt_text)[:200]
            if len(prompt_text) == 200:
                prompt_text += "…"

            kl_str = f"{comp.kl_divergence:.4f}" if comp.kl_divergence is not None else "N/A"
            div_idx = str(comp.token_divergence_index) if comp.token_divergence_index is not None else "N/A"

            github_issue_body = format_issue(comp, include_repro=True)

            cards.append(
                {
                    "category": str(category),
                    "prompt_text": prompt_text,
                    "similarity": f"{comp.text_similarity * 100:.1f}",
                    "kl": kl_str,
                    "div_index": div_idx,
                    "baseline_backend": comp.baseline.backend_name,
                    "test_backend": comp.test.backend_name,
                    "test_quant": comp.test.quantization,
                    "github_issue": github_issue_body,
                }
            )

    # Sort worst-first.
    cards.sort(key=lambda c: float(str(c["similarity"])))
    return cards


def _build_stress_context(stress_results: list[StressResult]) -> list[dict[str, Any]]:
    """Build table rows for the stress section."""
    rows = []
    for s in stress_results:
        rows.append(
            {
                "model_id": s.model_id[:32],
                "backend_name": s.backend_name,
                "concurrency_level": s.concurrency_level,
                "error_count": s.error_count,
                "output_consistency": f"{s.output_consistency * 100:.1f}",
                "consistency_raw": s.output_consistency,
                "num_results": len(s.results),
                "compared": s.compared or len(s.results),
                "is_reference": s.concurrency_level == 1,
                "divergences": s.divergences,
                "error_messages": "; ".join(s.error_messages),
            }
        )
    rows.sort(key=lambda r: (r["backend_name"], r["concurrency_level"]))
    return rows


def _build_determinism_context(
    det_results: list[DeterminismResult],
) -> list[dict[str, Any]]:
    """Build table rows for the determinism section."""
    rows = []
    for d in sorted(det_results, key=lambda r: r.determinism_score):
        num_unique = max(1, d.num_runs - d.identical_count + 1)
        first_div = str(min(d.divergence_positions)) if d.divergence_positions else "—"
        rows.append(
            {
                "prompt_id": d.prompt_id[:32],
                "backend_name": d.backend_name,
                "quantization": d.quantization or "—",
                "num_runs": d.num_runs,
                "score": d.determinism_score,
                "score_pct": f"{d.determinism_score * 100:.1f}",
                "num_unique": num_unique,
                "first_divergence": first_div,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(results_dir: Path, output_path: Path) -> Path:
    """Generate a self-contained HTML report from cached result JSON files.

    Scans *results_dir* for all ``.json`` files, classifies them by result
    type, and renders a single standalone HTML file to *output_path*.
    All CSS and JavaScript are inlined; no external network requests are
    required to view the report.

    Args:
        results_dir: Directory containing cached ``.json`` result files
            produced by ``TestRunner``.
        output_path: Destination for the rendered HTML file.  Parent
            directories are created automatically.

    Returns:
        The resolved *output_path* after writing.

    Raises:
        OSError: If *output_path* cannot be written.

    Examples:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     d = pathlib.Path(tmp)
        ...     out = d / "report.html"
        ...     result = generate_report(d, out)
        ...     result == out and out.exists()
        True
    """

    sections = _load_results(results_dir)
    sweeps: list[SweepResult] = sections["sweep"]
    diff_batches: list[list[ComparisonResult]] = sections["diff"]
    stress_results: list[StressResult] = sections["stress"]
    det_results: list[DeterminismResult] = sections["determinism"]
    compare_results: list[CompareResult] = sections["compare"]

    # ── Executive Summary ──────────────────────────────────────────────────
    all_comparisons: list[ComparisonResult] = []
    for sweep in sweeps:
        all_comparisons.extend(sweep.comparisons)
    for batch in diff_batches:
        all_comparisons.extend(batch)
    for compare in compare_results:
        all_comparisons.extend(compare.comparisons)

    total_tests = len(all_comparisons)
    total_failures = sum(1 for c in all_comparisons if c.is_failure)
    pass_rate = round((total_tests - total_failures) / total_tests * 100, 1) if total_tests else 100.0

    backend_names = {c.test.backend_name for c in all_comparisons} | {c.baseline.backend_name for c in all_comparisons}
    backend_count = len(backend_names)
    quant_count = len({c.test.quantization for c in all_comparisons if c.test.quantization})

    # Answer-key correctness, summed over every non-baseline comparison set.
    from infer_check.runner import correctness_summary

    graded_tables: list[tuple[str, dict[str, Any]]] = []
    for compare in compare_results:
        if compare.correctness:
            graded_tables.append((f"{compare.model_a} → {compare.model_b}", compare.correctness))
    for sweep in sweeps:
        base = (sweep.summary or {}).get("baseline_quant")
        for level, table in ((sweep.summary or {}).get("correctness_by_quant") or {}).items():
            if table and level != base:
                graded_tables.append((f"{base} → {level}", table))
    for batch in diff_batches:
        by_backend: dict[str, list[ComparisonResult]] = {}
        for comp in batch:
            by_backend.setdefault(comp.test.backend_name, []).append(comp)
        for name, comps in by_backend.items():
            table = correctness_summary(comps)
            if table:
                graded_tables.append((f"{comps[0].baseline.backend_name} → {name}", table))

    graded_totals = None
    if graded_tables:
        graded_totals = {
            "graded": sum(t.get("n_graded", 0) for _, t in graded_tables),
            "regressions": sum(t.get("regressions", 0) for _, t in graded_tables),
            "improvements": sum(t.get("improvements", 0) for _, t in graded_tables),
        }
        significant = [
            f"{name} (p={t['mcnemar_p']:.2g})"
            for name, t in graded_tables
            if t.get("regressions", 0) > t.get("improvements", 0) and t.get("mcnemar_p", 1.0) < 0.05
        ]
        verdict = (
            f"Graded against answer keys: {graded_totals['regressions']} regression(s) and "
            f"{graded_totals['improvements']} improvement(s) across {graded_totals['graded']} graded comparison(s). "
            + (
                "Statistically significant accuracy loss: " + ", ".join(significant) + "."
                if significant
                else "No statistically significant accuracy loss (exact McNemar, p < 0.05)."
            )
        )
    else:
        verdict = (
            f"{total_failures} of {total_tests} output(s) differ substantially in text (similarity < 50%) across "
            f"{backend_count} backend(s) and {quant_count} quantization level(s). Text differences alone don't "
            f"mean wrong answers; use a suite with answer keys (e.g. gsm8k) to measure correctness."
        )

    # ── Section data ───────────────────────────────────────────────────────
    sweep_ctx = _build_sweep_context(sweeps)
    sweep_levels = _build_sweep_levels(sweeps)
    compare_runs = _build_compare_context(compare_results)
    diff_rows = _build_diff_context(diff_batches + [c.comparisons for c in compare_results])
    failure_cards = _build_failure_cards(diff_batches + [c.comparisons for c in compare_results])
    stress_rows = _build_stress_context(stress_results)
    determinism_rows = _build_determinism_context(det_results)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # ── Render ─────────────────────────────────────────────────────────────
    env = Environment(  # noqa: S701 — safe: template is a local constant
        autoescape=True,
        undefined=Undefined,
    )
    template = env.from_string(_HTML_TEMPLATE)
    html = template.render(
        total_tests=total_tests,
        total_failures=total_failures,
        pass_rate=pass_rate,
        backend_count=backend_count,
        quant_count=quant_count,
        verdict=verdict,
        graded_totals=graded_totals,
        sweep_levels=sweep_levels,
        sweep_rows=sweep_ctx["sweep_rows"],
        quant_cols=sweep_ctx["quant_cols"],
        degradation_cliff=sweep_ctx["degradation_cliff"],
        compare_runs=compare_runs,
        diff_rows=diff_rows,
        failures=failure_cards,
        stress_rows=stress_rows,
        determinism_rows=determinism_rows,
        generated_at=generated_at,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
