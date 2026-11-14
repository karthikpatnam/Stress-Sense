"""HTML Dashboard generator for StressSense Phase 4 Stage 3.

Compiles all benchmark metrics, spatial patch visualizations, and interactive widgets into a
single, self-contained, high-performance HTML5/Tailwind/Canvas dashboard.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.data_builder import DashboardDataBuilder


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>StressSense | Intelligent Drone Crop Monitoring & Distributed Analytics Dashboard</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {
      --bg-base: #0f172a;
      --bg-surface: #1e293b;
      --bg-surface-elevated: #334155;
      --border-subtle: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --color-primary: #38bdf8;
      --color-success: #22c55e;
      --color-warning: #f59e0b;
      --color-danger: #ef4444;
      --color-accent: #a855f7;
    }
    html.light {
      --bg-base: #f8fafc;
      --bg-surface: #ffffff;
      --bg-surface-elevated: #f1f5f9;
      --border-subtle: #e2e8f0;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --color-primary: #0284c7;
      --color-success: #16a34a;
      --color-warning: #d97706;
      --color-danger: #dc2626;
      --color-accent: #9333ea;
    }
    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .custom-scroll::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    .custom-scroll::-webkit-scrollbar-track {
      background: rgba(0, 0, 0, 0.1);
    }
    .custom-scroll::-webkit-scrollbar-thumb {
      background: var(--border-subtle);
      border-radius: 3px;
    }
    .grid-cell {
      transition: all 0.15s ease-in-out;
    }
    .grid-cell:hover {
      transform: scale(1.05);
      z-index: 20;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
    }
    .tab-btn.active {
      border-bottom: 2px solid var(--color-primary);
      color: var(--color-primary);
      font-weight: 600;
    }
  </style>
</head>
<body class="min-h-screen antialiased flex flex-col custom-scroll">

  <!-- Header Banner -->
  <header class="border-b border-[var(--border-subtle)] bg-[var(--bg-surface)] sticky top-0 z-50 px-6 py-3 shadow-md backdrop-blur-md">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center text-xl shadow-lg shadow-emerald-500/20">
          🌾
        </div>
        <div>
          <h1 class="text-lg font-bold tracking-tight text-[var(--text-main)] flex items-center gap-2">
            StressSense
            <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 font-medium">Phase 4 Stage 3</span>
            <span class="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-medium">Audit: PASS WITH LIMITATIONS</span>
          </h1>
          <p class="text-xs text-[var(--text-muted)]">Cost- & Energy-Aware Drone Crop Health Monitoring via Adaptive Sampling & Apache Spark</p>
        </div>
      </div>

      <!-- Quick Metrics Bar & Theme Switcher -->
      <div class="flex items-center gap-3">
        <div class="hidden lg:flex items-center gap-4 text-xs bg-[var(--bg-surface-elevated)] px-3 py-1.5 rounded-lg border border-[var(--border-subtle)]">
          <div class="flex items-center gap-1.5">
            <span class="text-[var(--text-muted)]">Fleet:</span>
            <span class="font-mono font-bold text-[var(--text-main)]">640 Patches (40 Fields)</span>
          </div>
          <div class="h-3 w-px bg-[var(--border-subtle)]"></div>
          <div class="flex items-center gap-1.5">
            <span class="text-[var(--text-muted)]">Canopy Reduction:</span>
            <span class="font-mono font-bold text-emerald-400">61.76%</span>
          </div>
          <div class="h-3 w-px bg-[var(--border-subtle)]"></div>
          <div class="flex items-center gap-1.5">
            <span class="text-[var(--text-muted)]">Spark Steady:</span>
            <span class="font-mono font-bold text-cyan-400">43.13 p/s</span>
          </div>
          <div class="h-3 w-px bg-[var(--border-subtle)]"></div>
          <div class="flex items-center gap-1.5">
            <span class="text-[var(--text-muted)]">Tests:</span>
            <span class="font-mono font-bold text-emerald-400">111 / 111 Passed</span>
          </div>
        </div>

        <button id="themeToggleBtn" onclick="toggleTheme()" class="p-2 rounded-lg bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] text-[var(--text-muted)] hover:text-[var(--text-main)] transition-colors" title="Toggle Light/Dark Theme">
          <span id="themeIcon">☀️</span>
        </button>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="max-w-7xl mx-auto mt-3 flex items-center gap-6 text-sm border-t border-[var(--border-subtle)]/50 pt-2 overflow-x-auto no-scrollbar">
      <button class="tab-btn active pb-2 flex items-center gap-1.5 whitespace-nowrap transition-colors" onclick="switchTab('tab-overview')">
        <span>📊</span> Executive Overview & Fleet KPIs
      </button>
      <button class="tab-btn pb-2 flex items-center gap-1.5 whitespace-nowrap transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]" onclick="switchTab('tab-inspector')">
        <span>🛰️</span> Interactive Spatial Patch Inspector
      </button>
      <button class="tab-btn pb-2 flex items-center gap-1.5 whitespace-nowrap transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]" onclick="switchTab('tab-sensitivity')">
        <span>🎯</span> Category Sensitivity & Attribution
      </button>
      <button class="tab-btn pb-2 flex items-center gap-1.5 whitespace-nowrap transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]" onclick="switchTab('tab-spark')">
        <span>⚡</span> Apache Spark Distributed Scaling
      </button>
      <button class="tab-btn pb-2 flex items-center gap-1.5 whitespace-nowrap transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]" onclick="switchTab('tab-catalog')">
        <span>📋</span> Fleet Catalog & Patch Table
      </button>
      <button class="tab-btn pb-2 flex items-center gap-1.5 whitespace-nowrap transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]" onclick="switchTab('tab-audit')">
        <span>🔬</span> Scientific Audit Disclosures
      </button>
    </div>
  </header>

  <!-- Main Content Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">

    <!-- ========================================================================= -->
    <!-- TAB 1: EXECUTIVE OVERVIEW & FLEET KPIS -->
    <!-- ========================================================================= -->
    <section id="tab-overview" class="tab-content space-y-6">
      
      <!-- Key KPI Cards Grid -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        
        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)] shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
            <span>Canopy Reduction</span>
            <span class="text-emerald-400 font-bold">Primary</span>
          </div>
          <div class="text-2xl font-bold font-mono text-emerald-400" id="kpi-canopy-red">61.76%</div>
          <p class="text-[11px] text-[var(--text-muted)] mt-1">Valid canopy regions saved from transmission (Median: 62.5%)</p>
        </div>

        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)] shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
            <span>Energy Proxy Savings</span>
            <span class="text-cyan-400 font-bold">Proxy</span>
          </div>
          <div class="text-2xl font-bold font-mono text-cyan-400" id="kpi-energy-red">59.78%</div>
          <p class="text-[11px] text-[var(--text-muted)] mt-1">834.87 J (Full) → 335.79 J (Adaptive) per patch</p>
        </div>

        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)] shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
            <span>High-Stress Capture</span>
            <span class="text-amber-400 font-bold">+47.8%</span>
          </div>
          <div class="text-2xl font-bold font-mono text-amber-400" id="kpi-high-stress">86.58%</div>
          <p class="text-[11px] text-[var(--text-muted)] mt-1">Adaptive vs. 38.83% Equal-Budget under identical budget</p>
        </div>

        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)] shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
            <span>Composite Recall</span>
            <span class="text-purple-400 font-bold">Invariant</span>
          </div>
          <div class="text-2xl font-bold font-mono text-purple-400" id="kpi-recall">48.37%</div>
          <p class="text-[11px] text-[var(--text-muted)] mt-1">Adaptive vs. 41.53% Equal (Any-positive-pixel rule on 130 patches)</p>
        </div>

        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)] shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
            <span>Overall Precision</span>
            <span class="text-sky-400 font-bold">Empirical</span>
          </div>
          <div class="text-2xl font-bold font-mono text-sky-400" id="kpi-precision">45.20%</div>
          <p class="text-[11px] text-[var(--text-muted)] mt-1">Adaptive vs. 38.99% Equal-Budget baseline</p>
        </div>

        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)] shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
            <span>Spark Throughput</span>
            <span class="text-emerald-400 font-bold">Steady</span>
          </div>
          <div class="text-2xl font-bold font-mono text-emerald-400" id="kpi-spark-tput">43.13 p/s</div>
          <p class="text-[11px] text-[var(--text-muted)] mt-1">14.84s steady state compute across 640 patches</p>
        </div>

      </div>

      <!-- Comparative Strategy Comparison Section -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <div class="lg:col-span-2 bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-bold text-[var(--text-main)] flex items-center gap-2">
              <span>🎯</span> Three-Way Sampling Strategy Comparison (Equal-Budget Parity)
            </h2>
            <span class="text-xs text-[var(--text-muted)]">N = 640 Fleet Patches</span>
          </div>
          
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead class="border-b border-[var(--border-subtle)] text-[var(--text-muted)] font-medium">
                <tr>
                  <th class="pb-2">Strategy</th>
                  <th class="pb-2">Canopy Sampling Budget</th>
                  <th class="pb-2">Canopy Reduction</th>
                  <th class="pb-2">High-Stress Coverage</th>
                  <th class="pb-2">Anomaly Recall</th>
                  <th class="pb-2">Proxy Energy / Patch</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-[var(--border-subtle)] font-mono">
                <tr class="hover:bg-[var(--bg-surface-elevated)] transition-colors">
                  <td class="py-2.5 font-bold text-cyan-400">Adaptive Sampling (StressSense)</td>
                  <td class="py-2.5">Dynamic K (35% valid canopy)</td>
                  <td class="py-2.5 text-emerald-400 font-bold">61.76% ± 12.8%</td>
                  <td class="py-2.5 text-amber-400 font-bold">86.58% ± 22.4%</td>
                  <td class="py-2.5 text-purple-400 font-bold">48.37% ± 39.7%</td>
                  <td class="py-2.5 text-emerald-400 font-bold">335.79 J (-59.78%)</td>
                </tr>
                <tr class="hover:bg-[var(--bg-surface-elevated)] transition-colors">
                  <td class="py-2.5 font-bold text-[var(--text-muted)]">Equal-Budget Uniform Baseline</td>
                  <td class="py-2.5">Identical K (Random selection)</td>
                  <td class="py-2.5 text-emerald-400 font-bold">61.76% ± 12.8%</td>
                  <td class="py-2.5 text-[var(--text-muted)]">38.83% ± 18.5%</td>
                  <td class="py-2.5 text-[var(--text-muted)]">41.53% ± 23.6%</td>
                  <td class="py-2.5 text-[var(--text-muted)]">335.79 J (-59.78%)</td>
                </tr>
                <tr class="hover:bg-[var(--bg-surface-elevated)] transition-colors">
                  <td class="py-2.5 font-bold text-rose-400">Full Uniform (Conventional Drone)</td>
                  <td class="py-2.5">100% of Valid Canopy Regions</td>
                  <td class="py-2.5 text-rose-400">0.00% (No savings)</td>
                  <td class="py-2.5 text-rose-400">100.00% (Brute force)</td>
                  <td class="py-2.5 text-rose-400">100.00% (Brute force)</td>
                  <td class="py-2.5 text-rose-400">834.87 J (Baseline)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div class="bg-[var(--bg-surface-elevated)] p-3 rounded-lg border border-[var(--border-subtle)] text-xs text-[var(--text-muted)] space-y-1">
            <p><strong class="text-[var(--text-main)]">Statistical Rigor Disclosure (&alpha; = 0.01):</strong> Both Adaptive and Equal-Budget Uniform strategies evaluate identical per-patch region budgets, guaranteeing mathematical energy parity ($335.79\text{ J}$ identical). Under equal expenditure, Adaptive Sampling captures <strong class="text-amber-400">+47.75% more high-stress regions</strong> ($p < 10^{-15}$, paired Wilcoxon) and <strong class="text-purple-400">+6.84% more ground-truth anomalies</strong>.</p>
          </div>
        </div>

        <!-- Fleet Denominator Breakdown Card -->
        <div class="bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-4">
          <h2 class="text-sm font-bold text-[var(--text-main)] flex items-center gap-2">
            <span>📐</span> Fleet Denominator Taxonomy
          </h2>
          <div class="space-y-2 text-xs">
            <div class="flex justify-between items-center p-2 rounded bg-[var(--bg-surface-elevated)]">
              <span class="text-[var(--text-muted)]">Total Fleet Patches</span>
              <span class="font-mono font-bold text-[var(--text-main)]">640 patches</span>
            </div>
            <div class="flex justify-between items-center p-2 rounded bg-[var(--bg-surface-elevated)]">
              <span class="text-[var(--text-muted)]">Annotated Patches (Train + Val)</span>
              <span class="font-mono font-bold text-[var(--text-main)]">480 patches</span>
            </div>
            <div class="flex justify-between items-center p-2 rounded bg-[var(--bg-surface-elevated)]">
              <span class="text-[var(--text-muted)]">Anomalous Patches (GT > 0)</span>
              <span class="font-mono font-bold text-amber-400">130 patches</span>
            </div>
            <div class="flex justify-between items-center p-2 rounded bg-[var(--bg-surface-elevated)]">
              <span class="text-[var(--text-muted)]">Healthy Patches (0 Anomaly)</span>
              <span class="font-mono font-bold text-emerald-400">350 patches</span>
            </div>
            <div class="flex justify-between items-center p-2 rounded bg-[var(--bg-surface-elevated)]">
              <span class="text-[var(--text-muted)]">Unlabeled Test Split</span>
              <span class="font-mono font-bold text-[var(--text-muted)]">160 patches (Excluded from GT)</span>
            </div>
            <div class="flex justify-between items-center p-2 rounded bg-[var(--bg-surface-elevated)]">
              <span class="text-[var(--text-muted)]">Valid Crop Canopy Regions</span>
              <span class="font-mono font-bold text-cyan-400">8,920 regions (of 10,240)</span>
            </div>
          </div>
          <p class="text-[11px] text-[var(--text-muted)] italic">
            *Ground-truth masks are strictly post-hoc diagnostic references. Unlabeled test patches carry None for recall and are never included in ground-truth denominators.
          </p>
        </div>

      </div>

    </section>

    <!-- ========================================================================= -->
    <!-- TAB 2: INTERACTIVE SPATIAL PATCH INSPECTOR -->
    <!-- ========================================================================= -->
    <section id="tab-inspector" class="tab-content hidden space-y-6">
      
      <!-- Selector & Parameter Controls -->
      <div class="bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 class="text-sm font-bold text-[var(--text-main)] flex items-center gap-2">
              <span>🛰️</span> Representative Crop Patch Selector
            </h2>
            <p class="text-xs text-[var(--text-muted)]">Explore authentic Agriculture-Vision multi-spectral tiles with dynamic client-side sampling re-simulation</p>
          </div>
          
          <!-- Patch Quick Buttons -->
          <div class="flex flex-wrap items-center gap-2" id="patchButtonContainer">
            <!-- Injected dynamically via JS -->
          </div>
        </div>

        <!-- Interactive Sliders Bar -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-[var(--border-subtle)] text-xs">
          <div>
            <div class="flex justify-between mb-1">
              <label class="text-[var(--text-muted)] font-medium">Sampling Budget Ratio (K / Valid):</label>
              <span class="font-mono font-bold text-cyan-400" id="budgetSliderVal">35%</span>
            </div>
            <input type="range" id="budgetRatioSlider" min="10" max="60" step="5" value="35" class="w-full accent-cyan-400" oninput="updateInteractiveBudget(this.value)">
          </div>

          <div>
            <div class="flex justify-between mb-1">
              <label class="text-[var(--text-muted)] font-medium">Healthy Exploration Rate:</label>
              <span class="font-mono font-bold text-purple-400" id="explorSliderVal">5%</span>
            </div>
            <input type="range" id="explorationSlider" min="0" max="20" step="5" value="5" class="w-full accent-purple-400" oninput="updateInteractiveExploration(this.value)">
          </div>

          <div>
            <div class="flex justify-between mb-1">
              <label class="text-[var(--text-muted)] font-medium">Active Layer View:</label>
              <span class="font-mono font-bold text-emerald-400" id="activeLayerLabel">RSDI Spectral Contrast</span>
            </div>
            <select id="layerSelect" class="w-full bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] rounded px-2.5 py-1 text-xs text-[var(--text-main)]" onchange="changeLayer(this.value)">
              <option value="rsdi">RSDI Empirical Contrast Heatmap</option>
              <option value="rgb">RGB Aerial Imagery</option>
              <option value="nir">Near-Infrared (NIR) Band</option>
              <option value="gt">Ground-Truth Anomaly Mask Overlay</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Spatial Visualizer Grid & Detail Panel -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        <!-- Left: Image & 4x4 Grid Overlay (7 cols) -->
        <div class="lg:col-span-7 bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-4">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-[var(--text-main)] flex items-center gap-2">
              <span id="currentPatchTitle">Standing Water (Contiguous Flood)</span>
            </h3>
            <span class="text-xs font-mono text-[var(--text-muted)]" id="currentPatchMeta">128x128 px | Grid: 4x4 (32x32 px/cell)</span>
          </div>

          <!-- Main Interactive Canvas Viewport -->
          <div class="relative w-full aspect-square max-w-[480px] mx-auto rounded-lg overflow-hidden border border-[var(--border-subtle)] shadow-inner bg-black">
            <!-- Background Layer Image -->
            <img id="patchBaseImage" src="" alt="Patch Layer" class="w-full h-full object-cover">
            
            <!-- Ground Truth Mask Overlay Layer -->
            <img id="patchGtOverlay" src="" alt="GT Overlay" class="absolute inset-0 w-full h-full object-cover pointer-events-none opacity-60 hidden">

            <!-- 4x4 Interactive Grid Overlay -->
            <div id="spatialGridOverlay" class="absolute inset-0 grid grid-cols-4 grid-rows-4 pointer-events-auto">
              <!-- 16 region cells injected via JS -->
            </div>
          </div>

          <!-- Legend Bar -->
          <div class="flex flex-wrap items-center justify-center gap-4 text-xs pt-2 border-t border-[var(--border-subtle)]">
            <div class="flex items-center gap-1.5">
              <span class="w-3.5 h-3.5 rounded bg-rose-500/80 border border-rose-300"></span>
              <span class="text-[var(--text-muted)]">High Stress Guarantee (Priority 1)</span>
            </div>
            <div class="flex items-center gap-1.5">
              <span class="w-3.5 h-3.5 rounded bg-amber-500/80 border border-amber-300"></span>
              <span class="text-[var(--text-muted)]">Moderate Stress (Priority 2)</span>
            </div>
            <div class="flex items-center gap-1.5">
              <span class="w-3.5 h-3.5 rounded bg-blue-500/80 border border-blue-300"></span>
              <span class="text-[var(--text-muted)]">Healthy Exploration (Priority 3)</span>
            </div>
            <div class="flex items-center gap-1.5">
              <span class="w-3.5 h-3.5 rounded bg-neutral-600/60 border border-neutral-400"></span>
              <span class="text-[var(--text-muted)]">Unselected Valid</span>
            </div>
            <div class="flex items-center gap-1.5">
              <span class="w-3.5 h-3.5 rounded bg-neutral-900/90 border border-neutral-700"></span>
              <span class="text-[var(--text-muted)]">Invalid / Boundary</span>
            </div>
          </div>
        </div>

        <!-- Right: Cell Inspector & Live Patch Metrics (5 cols) -->
        <div class="lg:col-span-5 space-y-4">
          
          <!-- Hovered Region Details Card -->
          <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)] space-y-3">
            <h4 class="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center justify-between">
              <span>Region Cell Inspector</span>
              <span id="inspectedRegionId" class="font-mono text-xs text-[var(--text-main)]">Hover over a grid cell</span>
            </h4>
            
            <div class="grid grid-cols-2 gap-2 text-xs" id="regionInspectorStats">
              <div class="p-2 rounded bg-[var(--bg-surface-elevated)]">
                <span class="text-[var(--text-muted)] block text-[10px]">Grid Coordinates</span>
                <span class="font-mono font-bold text-[var(--text-main)]" id="regCoord">--</span>
              </div>
              <div class="p-2 rounded bg-[var(--bg-surface-elevated)]">
                <span class="text-[var(--text-muted)] block text-[10px]">RSDI Empirical Score</span>
                <span class="font-mono font-bold text-[var(--text-main)]" id="regRsdi">--</span>
              </div>
              <div class="p-2 rounded bg-[var(--bg-surface-elevated)]">
                <span class="text-[var(--text-muted)] block text-[10px]">Priority Tier</span>
                <span class="font-mono font-bold text-[var(--text-main)]" id="regTier">--</span>
              </div>
              <div class="p-2 rounded bg-[var(--bg-surface-elevated)]">
                <span class="text-[var(--text-muted)] block text-[10px]">Sampling Status</span>
                <span class="font-mono font-bold text-[var(--text-main)]" id="regStatus">--</span>
              </div>
              <div class="p-2 rounded bg-[var(--bg-surface-elevated)] col-span-2">
                <span class="text-[var(--text-muted)] block text-[10px]">Ground-Truth Anomaly</span>
                <span class="font-mono font-bold text-[var(--text-main)]" id="regGt">--</span>
              </div>
            </div>
          </div>

          <!-- Dynamic Patch Re-Simulation Metrics Card -->
          <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)] space-y-3">
            <h4 class="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center justify-between">
              <span>Live Patch Sampling Re-Simulation</span>
              <span class="text-[10px] text-[var(--text-muted)]">Real-Time JS Engine</span>
            </h4>
            
            <div class="space-y-2 text-xs">
              <div class="flex justify-between items-center py-1 border-b border-[var(--border-subtle)]/50">
                <span class="text-[var(--text-muted)]">Valid Canopy Regions:</span>
                <span class="font-mono font-bold text-[var(--text-main)]" id="dynValidCount">--</span>
              </div>
              <div class="flex justify-between items-center py-1 border-b border-[var(--border-subtle)]/50">
                <span class="text-[var(--text-muted)]">Allocated Sampling Budget (K):</span>
                <span class="font-mono font-bold text-cyan-400" id="dynKCount">--</span>
              </div>
              <div class="flex justify-between items-center py-1 border-b border-[var(--border-subtle)]/50">
                <span class="text-[var(--text-muted)]">Canopy Reduction Savings:</span>
                <span class="font-mono font-bold text-emerald-400" id="dynCanopyRed">--</span>
              </div>
              <div class="flex justify-between items-center py-1 border-b border-[var(--border-subtle)]/50">
                <span class="text-[var(--text-muted)]">High-Stress Regions Captured:</span>
                <span class="font-mono font-bold text-amber-400" id="dynHighStressCap">--</span>
              </div>
              <div class="flex justify-between items-center py-1 border-b border-[var(--border-subtle)]/50">
                <span class="text-[var(--text-muted)]">Ground-Truth Anomaly Recall:</span>
                <span class="font-mono font-bold text-purple-400" id="dynGtRecall">--</span>
              </div>
              <div class="flex justify-between items-center py-1">
                <span class="text-[var(--text-muted)]">Estimated Energy (Adaptive vs Full):</span>
                <span class="font-mono font-bold text-emerald-400" id="dynEnergy">--</span>
              </div>
            </div>
          </div>

          <!-- Explanation Box -->
          <div class="p-3 rounded-lg bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] text-[11px] text-[var(--text-muted)] space-y-1">
            <p><strong class="text-[var(--text-main)]">Operational Insight:</strong> The drone guarantees transmission for all high-deficit cells (red rings) before allocating moderate stress and exploration quotas. Notice how adjusting the budget slider re-ranks cells without ever needing a model re-train or network call.</p>
          </div>

        </div>

      </div>

    </section>

    <!-- ========================================================================= -->
    <!-- TAB 3: CATEGORY SENSITIVITY & ATTRIBUTION -->
    <!-- ========================================================================= -->
    <section id="tab-sensitivity" class="tab-content hidden space-y-6">
      
      <!-- Threshold Selector Bar -->
      <div class="bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 class="text-sm font-bold text-[var(--text-main)] flex items-center gap-2">
              <span>🎯</span> Spatial Overlap Sensitivity Sweeps (&tau; = 0%, 5%, 10%, 20%)
            </h2>
            <p class="text-xs text-[var(--text-muted)]">Analyze how varying the spatial overlap threshold impacts category-specific qualifying denominators and recall</p>
          </div>
          
          <div class="flex items-center gap-2 bg-[var(--bg-surface-elevated)] p-1 rounded-lg border border-[var(--border-subtle)]" id="thresholdToggleGroup">
            <button class="px-3 py-1 rounded text-xs font-mono font-medium transition-colors bg-cyan-500 text-white" onclick="selectThreshold('0%')">0% Overlap</button>
            <button class="px-3 py-1 rounded text-xs font-mono font-medium transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]" onclick="selectThreshold('5%')">5% Overlap</button>
            <button class="px-3 py-1 rounded text-xs font-mono font-medium transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]" onclick="selectThreshold('10%')">10% (Default)</button>
            <button class="px-3 py-1 rounded text-xs font-mono font-medium transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]" onclick="selectThreshold('20%')">20% Overlap</button>
          </div>
        </div>

        <!-- Sensitivity Table -->
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs" id="sensitivityTable">
            <thead class="border-b border-[var(--border-subtle)] text-[var(--text-muted)] font-medium">
              <tr>
                <th class="pb-2">Anomaly Category</th>
                <th class="pb-2">Qualifying Patches (N_p)</th>
                <th class="pb-2">Qualifying Regions (N_r)</th>
                <th class="pb-2">Adaptive Macro Mean Recall</th>
                <th class="pb-2">Adaptive Pooled Micro Recall</th>
                <th class="pb-2">Equal-Budget Baseline Recall</th>
                <th class="pb-2">Adaptive Advantage</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-[var(--border-subtle)] font-mono" id="sensitivityTableBody">
              <!-- Dynamically rendered via JS -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Deep Scientific Explanation Cards Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        <div class="bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-3">
          <h3 class="text-xs font-bold uppercase tracking-wider text-amber-400 flex items-center gap-2">
            <span>📉</span> Mechanism of Denominator Contraction & Selection Bias
          </h3>
          <p class="text-xs text-[var(--text-muted)] leading-relaxed">
            As the overlap threshold increases from <span class="font-mono text-[var(--text-main)]">0% → 20%</span>, the total regional denominator contracts from <strong class="text-[var(--text-main)]">881 → 623 regions (-29.3%)</strong>.
          </p>
          <p class="text-xs text-[var(--text-muted)] leading-relaxed">
            For linear erosion channels like <code class="text-cyan-400">waterway</code>, grid cells that only clip the outer edge of a ditch cover $1-40$ pixels ($0.1\%-3.9\%$ of the cell). In these peripheral cells, <strong class="text-[var(--text-main)]">96% to 99% of the pixels are normal, healthy crop canopy</strong>. Because their area-averaged spectral contrast is normal, the uncalibrated heuristic correctly assigns them low deficit priority.
          </p>
          <p class="text-xs text-[var(--text-muted)] leading-relaxed">
            At stricter thresholds (&ge; 20%), these grazing cells are excluded from the test set, leaving only core waterway cells. The measured recall increases from <strong class="text-emerald-400">81.53% to 98.33%</strong> entirely because the evaluation denominator contracted, not because the drone changed its sampling decisions.
          </p>
        </div>

        <div class="bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-3">
          <h3 class="text-xs font-bold uppercase tracking-wider text-purple-400 flex items-center gap-2">
            <span>🔒</span> Mathematical Invariance of Composite Recall (48.37%)
          </h3>
          <p class="text-xs text-[var(--text-muted)] leading-relaxed">
            Across all four overlap thresholds, composite anomaly recall remains <strong class="text-purple-400">identically 48.37%</strong> across the 130 anomalous patches.
          </p>
          <div class="p-3 rounded bg-[var(--bg-surface-elevated)] font-mono text-[11px] text-purple-300">
            gt_has_anomaly(r) = I( count_nonzero( M_composite[r] > 0 ) > 0 )
          </div>
          <p class="text-xs text-[var(--text-muted)] leading-relaxed">
            Because overall anomaly presence is based strictly on the binary <strong class="text-[var(--text-main)]">any-positive-pixel rule</strong> on the composite mask, category attribution thresholds do not modify the composite ground-truth set or the sampling decisions. This invariance proves that the evaluation framework maintains strict decoupling without cross-contamination.
          </p>
        </div>

      </div>

    </section>

    <!-- ========================================================================= -->
    <!-- TAB 4: APACHE SPARK DISTRIBUTED SCALING -->
    <!-- ========================================================================= -->
    <section id="tab-spark" class="tab-content hidden space-y-6">
      
      <!-- Spark Telemetry Overview Grid -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        
        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)]">
          <span class="text-xs text-[var(--text-muted)] block">Cold JVM Startup</span>
          <span class="text-2xl font-bold font-mono text-amber-400">12.46s</span>
          <span class="text-[11px] text-[var(--text-muted)] block mt-1">One-time Java 17 + Py4J gateway init</span>
        </div>

        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)]">
          <span class="text-xs text-[var(--text-muted)] block">Steady-State Compute</span>
          <span class="text-2xl font-bold font-mono text-cyan-400">14.84s</span>
          <span class="text-[11px] text-[var(--text-muted)] block mt-1">RDD mapPartitions across 640 patches</span>
        </div>

        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)]">
          <span class="text-xs text-[var(--text-muted)] block">Steady Throughput</span>
          <span class="text-2xl font-bold font-mono text-emerald-400">43.13 p/s</span>
          <span class="text-[11px] text-[var(--text-muted)] block mt-1">Patches evaluated per second</span>
        </div>

        <div class="bg-[var(--bg-surface)] p-4 rounded-xl border border-[var(--border-subtle)]">
          <span class="text-xs text-[var(--text-muted)] block">Driver Overhead Delta</span>
          <span class="text-2xl font-bold font-mono text-purple-400">0.071s</span>
          <span class="text-[11px] text-[var(--text-muted)] block mt-1">~0.7% total latency (broadcast + collect)</span>
        </div>

      </div>

      <!-- Scaling Comparison Table -->
      <div class="bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-4">
        <h2 class="text-sm font-bold text-[var(--text-main)] flex items-center justify-between">
          <span>⚡ Worker Thread Core Scaling Benchmark (160 Validation Patches)</span>
          <span class="text-xs font-mono text-[var(--text-muted)]">Single-Machine vs. local[1], local[2], local[4]</span>
        </h2>
        
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="border-b border-[var(--border-subtle)] text-[var(--text-muted)] font-medium">
              <tr>
                <th class="pb-2">Execution Environment</th>
                <th class="pb-2">Cores</th>
                <th class="pb-2">Startup Overhead</th>
                <th class="pb-2">Steady-State Compute</th>
                <th class="pb-2">Total Wall-Clock</th>
                <th class="pb-2">Throughput</th>
                <th class="pb-2">Speedup vs Native</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-[var(--border-subtle)] font-mono">
              <tr class="hover:bg-[var(--bg-surface-elevated)] transition-colors">
                <td class="py-2.5 font-bold text-emerald-400">Single-Machine Native Baseline</td>
                <td class="py-2.5">1 (Native)</td>
                <td class="py-2.5">0.0000s</td>
                <td class="py-2.5">3.0618s</td>
                <td class="py-2.5 text-emerald-400 font-bold">3.0618s</td>
                <td class="py-2.5 text-emerald-400 font-bold">52.26 p/s</td>
                <td class="py-2.5 font-bold text-emerald-400">1.00x (Baseline)</td>
              </tr>
              <tr class="hover:bg-[var(--bg-surface-elevated)] transition-colors">
                <td class="py-2.5 font-bold text-cyan-400">Spark local[1] (Single Worker)</td>
                <td class="py-2.5">1</td>
                <td class="py-2.5 text-amber-400">0.3091s</td>
                <td class="py-2.5">9.6796s</td>
                <td class="py-2.5">10.0622s</td>
                <td class="py-2.5">16.53 p/s</td>
                <td class="py-2.5 text-rose-400">0.30x (IPC Slowdown)</td>
              </tr>
              <tr class="hover:bg-[var(--bg-surface-elevated)] transition-colors">
                <td class="py-2.5 font-bold text-cyan-400">Spark local[2] (Dual Worker)</td>
                <td class="py-2.5">2</td>
                <td class="py-2.5 text-amber-400">0.2367s</td>
                <td class="py-2.5">8.6953s</td>
                <td class="py-2.5">9.0034s</td>
                <td class="py-2.5 font-bold text-cyan-400">18.40 p/s</td>
                <td class="py-2.5 text-rose-400">0.34x (IPC Slowdown)</td>
              </tr>
              <tr class="hover:bg-[var(--bg-surface-elevated)] transition-colors">
                <td class="py-2.5 font-bold text-cyan-400">Spark local[4] (Quad Worker)</td>
                <td class="py-2.5">4</td>
                <td class="py-2.5 text-amber-400">0.2252s</td>
                <td class="py-2.5">10.8656s</td>
                <td class="py-2.5">11.1721s</td>
                <td class="py-2.5">14.73 p/s</td>
                <td class="py-2.5 text-rose-400">0.27x (I/O Contention)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="p-3 rounded-lg bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] text-xs text-[var(--text-muted)] space-y-1">
          <p><strong class="text-[var(--text-main)]">Methodological Scalability Disclosure:</strong> On a single development machine, native vectorized NumPy operations in a single Python thread outperform local Spark execution ($52.26\text{ p/s}$ vs. $14.73 - 18.40\text{ p/s}$). This occurs because local mode incurs Py4J socket serialization, JVM-to-Python IPC, and single-disk file read contention. Apache Spark's true design value is horizontal scaling across multi-node cloud clusters (e.g. Google Cloud Dataproc) for multi-terabyte fleet surveys.</p>
        </div>
      </div>

      <!-- Cloud Multi-Node Extrapolation Calculator -->
      <div class="bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-4">
        <h2 class="text-sm font-bold text-[var(--text-main)] flex items-center justify-between">
          <span>☁️ Enterprise Cloud Cluster Scaling Extrapolation (Google Cloud Dataproc)</span>
          <span class="text-xs text-[var(--text-muted)]">Estimating Horizontal Scale-Out</span>
        </h2>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div class="p-4 rounded-lg bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)]">
            <span class="text-xs text-[var(--text-muted)] block">1 Worker Node (4 Cores)</span>
            <span class="text-xl font-bold font-mono text-[var(--text-main)]">232 sec</span>
            <span class="text-[11px] text-[var(--text-muted)] block mt-1">43.1 p/s (Baseline efficiency)</span>
          </div>

          <div class="p-4 rounded-lg bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)]">
            <span class="text-xs text-[var(--text-muted)] block">4 Worker Nodes (16 Cores)</span>
            <span class="text-xl font-bold font-mono text-cyan-400">65 sec (3.57x)</span>
            <span class="text-[11px] text-[var(--text-muted)] block mt-1">153.8 p/s (89% cluster efficiency)</span>
          </div>

          <div class="p-4 rounded-lg bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)]">
            <span class="text-xs text-[var(--text-muted)] block">8 Worker Nodes (32 Cores)</span>
            <span class="text-xl font-bold font-mono text-emerald-400">36 sec (6.44x)</span>
            <span class="text-[11px] text-[var(--text-muted)] block mt-1">277.8 p/s (81% cluster efficiency)</span>
          </div>

          <div class="p-4 rounded-lg bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)]">
            <span class="text-xs text-[var(--text-muted)] block">16 Worker Nodes (64 Cores)</span>
            <span class="text-xl font-bold font-mono text-purple-400">21 sec (11.0x)</span>
            <span class="text-[11px] text-[var(--text-muted)] block mt-1">476.2 p/s (69% cluster efficiency)</span>
          </div>
        </div>
      </div>

    </section>

    <!-- ========================================================================= -->
    <!-- TAB 5: FLEET CATALOG & PATCH TABLE -->
    <!-- ========================================================================= -->
    <section id="tab-catalog" class="tab-content hidden space-y-6">
      
      <div class="bg-[var(--bg-surface)] p-5 rounded-xl border border-[var(--border-subtle)] space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 class="text-sm font-bold text-[var(--text-main)] flex items-center gap-2">
              <span>📋</span> Fleet Patch Catalog (All 640 Evaluated Patches)
            </h2>
            <p class="text-xs text-[var(--text-muted)]">Search, filter, and inspect metrics for every patch across the 40 flight codes</p>
          </div>

          <!-- Table Search & Filters -->
          <div class="flex flex-wrap items-center gap-3">
            <input type="text" id="patchSearchInput" placeholder="Filter by patch ID, field, or anomaly..." class="bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] rounded-lg px-3 py-1.5 text-xs text-[var(--text-main)] w-64 focus:outline-none focus:border-cyan-400" oninput="filterPatchTable()">
            
            <select id="splitFilter" class="bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] rounded-lg px-2.5 py-1.5 text-xs text-[var(--text-main)]" onchange="filterPatchTable()">
              <option value="all">All Splits (640)</option>
              <option value="train">Train (320)</option>
              <option value="val">Val (160)</option>
              <option value="test">Test (160)</option>
            </select>

            <select id="anomalyFilter" class="bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] rounded-lg px-2.5 py-1.5 text-xs text-[var(--text-main)]" onchange="filterPatchTable()">
              <option value="all">All Patches</option>
              <option value="anomalous">Anomalous Only (130)</option>
              <option value="healthy">Healthy Only (350)</option>
            </select>
          </div>
        </div>

        <!-- Table Viewport -->
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs" id="patchCatalogTable">
            <thead class="border-b border-[var(--border-subtle)] text-[var(--text-muted)] font-medium">
              <tr>
                <th class="pb-2">Patch ID</th>
                <th class="pb-2">Split</th>
                <th class="pb-2">Field ID</th>
                <th class="pb-2">Anomaly Categories</th>
                <th class="pb-2">Valid Regions</th>
                <th class="pb-2">Selected K</th>
                <th class="pb-2">Canopy Reduction</th>
                <th class="pb-2">Anomaly Recall</th>
                <th class="pb-2">Energy Proxy (J)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-[var(--border-subtle)] font-mono" id="patchTableBody">
              <!-- Dynamically rendered via JS -->
            </tbody>
          </table>
        </div>

        <!-- Pagination Controls -->
        <div class="flex items-center justify-between text-xs text-[var(--text-muted)] pt-3 border-t border-[var(--border-subtle)]">
          <span id="tablePageInfo">Showing 1 - 15 of 640 patches</span>
          <div class="flex items-center gap-2">
            <button id="prevPageBtn" class="px-2.5 py-1 rounded bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] disabled:opacity-40" onclick="changePage(-1)">Previous</button>
            <button id="nextPageBtn" class="px-2.5 py-1 rounded bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] disabled:opacity-40" onclick="changePage(1)">Next</button>
          </div>
        </div>
      </div>

    </section>

    <!-- ========================================================================= -->
    <!-- TAB 6: SCIENTIFIC AUDIT DISCLOSURES -->
    <!-- ========================================================================= -->
    <section id="tab-audit" class="tab-content hidden space-y-6">
      
      <div class="bg-[var(--bg-surface)] p-6 rounded-xl border border-[var(--border-subtle)] space-y-6">
        <div>
          <h2 class="text-base font-bold text-[var(--text-main)] flex items-center gap-2">
            <span>🔬</span> Scientific & Methodological Audit Registry
          </h2>
          <p class="text-xs text-[var(--text-muted)]">Verified compliance records, formal disclosures, and mathematical definitions</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          
          <div class="space-y-4">
            <div class="p-4 rounded-xl bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] space-y-2">
              <h3 class="font-bold text-cyan-400">1. Heuristic Spectral Contrast Index (RSDI)</h3>
              <p class="text-[var(--text-muted)] leading-relaxed">
                The index calculated as <code class="text-cyan-300">(NIR - Red) / (NIR + Red + 1e-7)</code> from 8-bit digital numbers is an empirical spectral-contrast heuristic (RSDI). It is explicitly <strong class="text-[var(--text-main)]">not</strong> physically calibrated surface reflectance NDVI. Priority classes (<code class="text-rose-400">HIGH_STRESS</code>, <code class="text-amber-400">MODERATE_STRESS</code>, <code class="text-emerald-400">HEALTHY</code>) are algorithmic sampling priority tiers for drone bandwidth steering, not biological pathology diagnoses.
              </p>
            </div>

            <div class="p-4 rounded-xl bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] space-y-2">
              <h3 class="font-bold text-purple-400">2. Ground-Truth Decoupling & Undefined Category Precision</h3>
              <p class="text-[var(--text-muted)] leading-relaxed">
                Agriculture-Vision anomaly masks were used strictly post-hoc for diagnostic evaluation. The sampling engine operates with zero access to ground truth. Furthermore, because the drone functions as an <strong class="text-[var(--text-main)]">unclassified spatial ranking engine</strong> rather than a multi-class classifier, category-specific false positives and precision are mathematically undefined. Category metrics evaluate qualifying region <strong class="text-[var(--text-main)]">Recall</strong>.
              </p>
            </div>

            <div class="p-4 rounded-xl bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] space-y-2">
              <h3 class="font-bold text-emerald-400">3. Valid Canopy Sampling Accounting</h3>
              <p class="text-[var(--text-muted)] leading-relaxed">
                <code class="text-emerald-300">valid_reduction_pct</code> is strictly the primary sampling efficiency metric: only valid candidate crop canopy regions ($8,920$ total) are evaluated. Non-field and boundary invalid regions ($1,320$ total) are excluded prior to sampling and are never claimed as sampling savings.
              </p>
            </div>
          </div>

          <div class="space-y-4">
            <div class="p-4 rounded-xl bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] space-y-2">
              <h3 class="font-bold text-amber-400">4. Linear Proxy Resource & Energy Modeling</h3>
              <p class="text-[var(--text-muted)] leading-relaxed">
                All computational and RF transmission energy estimates ($834.87\text{ J} \to 335.79\text{ J}$) are theoretical linear computational proxy models based on standard edge specifications ($15\text{ W}$ inference, $2.5\text{ W}$ transmission, $1.0\text{ NCU}$). They are explicitly distinct from physical battery drain measurements.
              </p>
            </div>

            <div class="p-4 rounded-xl bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] space-y-2">
              <h3 class="font-bold text-rose-400">5. Apache Spark Local Mode vs. Multi-Node Clusters</h3>
              <p class="text-[var(--text-muted)] leading-relaxed">
                Running Apache Spark in local mode (<code class="text-rose-300">local[k]</code>) on Windows incurs Py4J socket serialization, JVM-to-Python IPC, and single-disk I/O contention. Vectorized single-thread NumPy is faster for small datasets (~200 MB). Spark's architectural objective is horizontal scaling across multi-node cloud clusters for multi-terabyte fleet surveys.
              </p>
            </div>

            <div class="p-4 rounded-xl bg-[var(--bg-surface-elevated)] border border-[var(--border-subtle)] space-y-2">
              <h3 class="font-bold text-sky-400">6. Cumulative Test Suite Verification Status</h3>
              <p class="text-[var(--text-muted)] leading-relaxed">
                The entire cumulative test suite consists of <strong class="text-emerald-400">111 automated pytest unit & integration tests</strong> spanning adaptive sampling, dataset multi-modality, PySpark distributed evaluation, statistical significance (&alpha; = 0.01), and boundary conditions, with a <strong class="text-emerald-400">100% pass rate</strong> (0 failures, 0 skipped).
              </p>
            </div>
          </div>

        </div>
      </div>

    </section>

  </main>

  <!-- Footer -->
  <footer class="border-t border-[var(--border-subtle)] bg-[var(--bg-surface)] py-4 px-6 text-center text-xs text-[var(--text-muted)]">
    <p>StressSense Autonomous Crop Monitoring Platform &copy; 2026 | Verified under Apache Spark 4.3.0 & Python 3.11</p>
  </footer>

  <!-- Embedded Complete Dataset Payload -->
  <script id="dashboard-payload" type="application/json">
__PAYLOAD_JSON__
  </script>

  <!-- Client-Side Dashboard Interactive Controller -->
  <script>
    // 1. Load Data
    const rawData = JSON.parse(document.getElementById('dashboard-payload').textContent);
    const kpis = rawData.kpis;
    const patchTable = rawData.patch_table;
    const categorySens = rawData.category_sensitivity;
    const sparkScaling = rawData.spark_scaling;
    const repPatches = rawData.representative_patches;

    let currentRepPatchIndex = 0;
    let activeThreshold = "0%";
    let activeLayer = "rsdi";
    let tableCurrentPage = 1;
    const tablePageSize = 15;
    let filteredTableData = [...patchTable];

    // 2. Initialize
    document.addEventListener("DOMContentLoaded", () => {
      initTheme();
      renderRepPatchButtons();
      loadRepPatch(0);
      renderSensitivityTable("0%");
      renderPatchTable();
    });

    // 3. Theme Toggle
    function toggleTheme() {
      const html = document.documentElement;
      const isDark = html.classList.contains("dark");
      if (isDark) {
        html.classList.remove("dark");
        html.classList.add("light");
        document.getElementById("themeIcon").textContent = "🌙";
        localStorage.setItem("st_theme", "light");
      } else {
        html.classList.remove("light");
        html.classList.add("dark");
        document.getElementById("themeIcon").textContent = "☀️";
        localStorage.setItem("st_theme", "dark");
      }
    }

    function initTheme() {
      const saved = localStorage.getItem("st_theme");
      if (saved === "light") {
        document.documentElement.classList.remove("dark");
        document.documentElement.classList.add("light");
        document.getElementById("themeIcon").textContent = "🌙";
      }
    }

    // 4. Tab Navigation
    function switchTab(tabId) {
      document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
      document.querySelectorAll(".tab-btn").forEach(el => {
        el.classList.remove("active");
        el.classList.add("text-[var(--text-muted)]");
      });
      
      const target = document.getElementById(tabId);
      if (target) target.classList.remove("hidden");

      const activeBtn = Array.from(document.querySelectorAll(".tab-btn")).find(b => b.getAttribute("onclick").includes(tabId));
      if (activeBtn) {
        activeBtn.classList.add("active");
        activeBtn.classList.remove("text-[var(--text-muted)]");
      }
    }

    // 5. Representative Patch Buttons & Loading
    function renderRepPatchButtons() {
      const container = document.getElementById("patchButtonContainer");
      container.innerHTML = "";
      repPatches.forEach((p, idx) => {
        const btn = document.createElement("button");
        btn.className = `px-2.5 py-1 rounded text-xs font-medium transition-colors border ${
          idx === currentRepPatchIndex
            ? "bg-cyan-500 text-white border-cyan-400"
            : "bg-[var(--bg-surface-elevated)] text-[var(--text-muted)] border-[var(--border-subtle)] hover:text-[var(--text-main)]"
        }`;
        btn.textContent = p.title.split(" (")[0];
        btn.onclick = () => {
          currentRepPatchIndex = idx;
          renderRepPatchButtons();
          loadRepPatch(idx);
        };
        container.appendChild(btn);
      });
    }

    function loadRepPatch(idx) {
      const p = repPatches[idx];
      document.getElementById("currentPatchTitle").textContent = p.title;
      document.getElementById("currentPatchMeta").textContent = `${p.patch_id} | Split: ${p.split} | Field: ${p.field_id}`;

      // Update images
      updateLayerImages(p);

      // Render 4x4 Grid
      renderSpatialGrid(p);

      // Re-run interactive simulation
      runInteractiveSimulation(p);
    }

    function updateLayerImages(p) {
      const baseImg = document.getElementById("patchBaseImage");
      const gtImg = document.getElementById("patchGtOverlay");

      if (activeLayer === "rsdi") {
        baseImg.src = p.rsdi_b64;
        gtImg.classList.add("hidden");
      } else if (activeLayer === "rgb") {
        baseImg.src = p.rgb_b64;
        gtImg.classList.add("hidden");
      } else if (activeLayer === "nir") {
        baseImg.src = p.nir_b64;
        gtImg.classList.add("hidden");
      } else if (activeLayer === "gt") {
        baseImg.src = p.rgb_b64;
        if (p.gt_b64) {
          gtImg.src = p.gt_b64;
          gtImg.classList.remove("hidden");
        } else {
          gtImg.classList.add("hidden");
        }
      }
    }

    function changeLayer(val) {
      activeLayer = val;
      const labelMap = {
        "rsdi": "RSDI Empirical Contrast Heatmap",
        "rgb": "RGB Aerial Imagery",
        "nir": "Near-Infrared (NIR) Band",
        "gt": "Ground-Truth Anomaly Mask Overlay"
      };
      document.getElementById("activeLayerLabel").textContent = labelMap[val] || val;
      updateLayerImages(repPatches[currentRepPatchIndex]);
    }

    // 6. Spatial 4x4 Grid Rendering
    function renderSpatialGrid(p) {
      const grid = document.getElementById("spatialGridOverlay");
      grid.innerHTML = "";

      p.regions.forEach(r => {
        const cell = document.createElement("div");
        cell.className = "grid-cell relative border border-white/20 flex flex-col justify-between p-1 cursor-pointer transition-all";
        cell.id = `cell-${r.region_id}`;

        // Color coding based on tier and selection
        let bgStyle = "";
        let borderHighlight = "";
        if (!r.is_valid) {
          bgStyle = "background-color: rgba(0, 0, 0, 0.75);";
        } else if (r.is_selected_adaptive) {
          if (r.adaptive_reason === "high_stress_guarantee") {
            bgStyle = "background-color: rgba(220, 38, 38, 0.45);";
            borderHighlight = "box-shadow: inset 0 0 0 2px #ef4444;";
          } else if (r.adaptive_reason === "moderate_stress_priority") {
            bgStyle = "background-color: rgba(245, 158, 11, 0.45);";
            borderHighlight = "box-shadow: inset 0 0 0 2px #f59e0b;";
          } else {
            bgStyle = "background-color: rgba(59, 130, 246, 0.45);";
            borderHighlight = "box-shadow: inset 0 0 0 2px #3b82f6;";
          }
        } else {
          bgStyle = "background-color: rgba(255, 255, 255, 0.05);";
        }

        cell.style = bgStyle + borderHighlight;

        cell.innerHTML = `
          <div class="flex justify-between items-center text-[10px] font-mono font-bold text-white drop-shadow">
            <span>${r.region_id.replace('reg_', '')}</span>
            ${r.is_selected_adaptive ? '<span class="text-xs">✓</span>' : ''}
          </div>
          <div class="text-[9px] font-mono text-white/90 drop-shadow text-right">
            ${r.is_valid ? r.mean_rsdi.toFixed(2) : 'inv'}
          </div>
        `;

        cell.onmouseenter = () => inspectRegion(r);
        grid.appendChild(cell);
      });
    }

    function inspectRegion(r) {
      document.getElementById("inspectedRegionId").textContent = r.region_id;
      document.getElementById("regCoord").textContent = `Row ${r.row}, Col ${r.col}`;
      document.getElementById("regRsdi").textContent = r.is_valid ? r.mean_rsdi.toFixed(4) : "N/A (Invalid)";
      document.getElementById("regTier").textContent = r.stress_tier.toUpperCase();
      document.getElementById("regStatus").textContent = r.is_selected_adaptive ? `SELECTED (${r.adaptive_reason})` : "OMITTED";
      
      const gtText = r.gt_has_anomaly 
        ? `${r.gt_categories.join(', ') || 'Anomaly'} (${(r.gt_fraction * 100).toFixed(1)}% area)`
        : "None (Healthy Canopy)";
      document.getElementById("regGt").textContent = gtText;
    }

    // 7. Interactive Simulation Logic
    function updateInteractiveBudget(val) {
      document.getElementById("budgetSliderVal").textContent = val + "%";
      runInteractiveSimulation(repPatches[currentRepPatchIndex]);
    }

    function updateInteractiveExploration(val) {
      document.getElementById("explorSliderVal").textContent = val + "%";
      runInteractiveSimulation(repPatches[currentRepPatchIndex]);
    }

    function runInteractiveSimulation(p) {
      const budgetRatio = parseFloat(document.getElementById("budgetRatioSlider").value) / 100.0;
      const explorRate = parseFloat(document.getElementById("explorationSlider").value) / 100.0;

      const validRegions = p.regions.filter(r => r.is_valid);
      const validCount = validRegions.length;
      const kBudget = Math.max(1, Math.ceil(validCount * budgetRatio));

      // Classify tiers
      const highStress = validRegions.filter(r => r.stress_tier === "high_stress");
      const modStress = validRegions.filter(r => r.stress_tier === "moderate_stress");
      const healthy = validRegions.filter(r => r.stress_tier === "healthy");

      // Sort moderate by deficit
      modStress.sort((a, b) => b.deficit_score - a.deficit_score);

      const selected = new Set();
      const reasons = {};

      // 1. High stress guarantee
      highStress.forEach(r => {
        selected.add(r.region_id);
        reasons[r.region_id] = "high_stress_guarantee";
      });

      // 2. Exploration
      const nExplor = Math.min(Math.floor(healthy.length * explorRate), Math.max(0, kBudget - selected.size));
      for (let i = 0; i < nExplor; i++) {
        selected.add(healthy[i].region_id);
        reasons[healthy[i].region_id] = "healthy_exploration";
      }

      // 3. Moderate stress
      for (let r of modStress) {
        if (selected.size >= kBudget) break;
        selected.add(r.region_id);
        reasons[r.region_id] = "moderate_stress_priority";
      }

      // 4. Update UI stats
      const canopyRed = validCount > 0 ? (((validCount - selected.size) / validCount) * 100).toFixed(1) : "0.0";
      const highCap = highStress.length > 0 
        ? `${highStress.filter(r => selected.has(r.region_id)).length} / ${highStress.length} (${((highStress.filter(r => selected.has(r.region_id)).length / highStress.length) * 100).toFixed(0)}%)`
        : "None in patch";

      const gtRegions = validRegions.filter(r => r.gt_has_anomaly);
      const gtRecall = gtRegions.length > 0
        ? `${gtRegions.filter(r => selected.has(r.region_id)).length} / ${gtRegions.length} (${((gtRegions.filter(r => selected.has(r.region_id)).length / gtRegions.length) * 100).toFixed(1)}%)`
        : "No anomaly (Healthy)";

      const fullEnergy = (validCount * 52.18).toFixed(1);
      const adaptEnergy = (selected.size * 52.18).toFixed(1);

      document.getElementById("dynValidCount").textContent = `${validCount} of 16 cells`;
      document.getElementById("dynKCount").textContent = `${selected.size} regions`;
      document.getElementById("dynCanopyRed").textContent = `${canopyRed}% savings`;
      document.getElementById("dynHighStressCap").textContent = highCap;
      document.getElementById("dynGtRecall").textContent = gtRecall;
      document.getElementById("dynEnergy").textContent = `${adaptEnergy} J vs ${fullEnergy} J`;

      // Update grid cell borders & indicators
      p.regions.forEach(r => {
        const cell = document.getElementById(`cell-${r.region_id}`);
        if (!cell) return;
        r.is_selected_adaptive = selected.has(r.region_id);
        r.adaptive_reason = reasons[r.region_id] || "not_selected";
        
        let bgStyle = "";
        let borderHighlight = "";
        if (!r.is_valid) {
          bgStyle = "background-color: rgba(0, 0, 0, 0.75);";
        } else if (r.is_selected_adaptive) {
          if (r.adaptive_reason === "high_stress_guarantee") {
            bgStyle = "background-color: rgba(220, 38, 38, 0.45);";
            borderHighlight = "box-shadow: inset 0 0 0 2px #ef4444;";
          } else if (r.adaptive_reason === "moderate_stress_priority") {
            bgStyle = "background-color: rgba(245, 158, 11, 0.45);";
            borderHighlight = "box-shadow: inset 0 0 0 2px #f59e0b;";
          } else {
            bgStyle = "background-color: rgba(59, 130, 246, 0.45);";
            borderHighlight = "box-shadow: inset 0 0 0 2px #3b82f6;";
          }
        } else {
          bgStyle = "background-color: rgba(255, 255, 255, 0.05);";
        }
        cell.style = bgStyle + borderHighlight;
      });
    }

    // 8. Category Sensitivity Table Controller
    function selectThreshold(thresh) {
      activeThreshold = thresh;
      const btns = document.querySelectorAll("#thresholdToggleGroup button");
      btns.forEach(b => {
        if (b.textContent.includes(thresh)) {
          b.className = "px-3 py-1 rounded text-xs font-mono font-medium transition-colors bg-cyan-500 text-white";
        } else {
          b.className = "px-3 py-1 rounded text-xs font-mono font-medium transition-colors text-[var(--text-muted)] hover:text-[var(--text-main)]";
        }
      });
      renderSensitivityTable(thresh);
    }

    function renderSensitivityTable(thresh) {
      const keyMap = { "0%": "thresh_0pct", "5%": "thresh_5pct", "10%": "thresh_10pct", "20%": "thresh_20pct" };
      const tKey = keyMap[thresh] || "thresh_0pct";
      const tData = categorySens.raw_data[tKey].categories;
      const tbody = document.getElementById("sensitivityTableBody");
      tbody.innerHTML = "";

      for (let cat in tData) {
        const c = tData[cat];
        const gain = (c.adaptive_recall_mean - c.equal_recall_mean).toFixed(2);
        const gainSign = gain >= 0 ? `+${gain}%` : `${gain}%`;
        const gainColor = gain >= 0 ? "text-emerald-400" : "text-rose-400";

        const tr = document.createElement("tr");
        tr.className = "hover:bg-[var(--bg-surface-elevated)] transition-colors";
        tr.innerHTML = `
          <td class="py-2 font-bold text-[var(--text-main)]">${cat}</td>
          <td class="py-2 text-[var(--text-muted)]">${c.patch_count}</td>
          <td class="py-2 text-[var(--text-muted)]">${c.qualifying_region_count}</td>
          <td class="py-2 font-bold text-cyan-400">${c.adaptive_recall_mean.toFixed(2)}%</td>
          <td class="py-2 text-purple-400">${c.adaptive_recall_pooled_pct.toFixed(2)}%</td>
          <td class="py-2 text-[var(--text-muted)]">${c.equal_recall_mean.toFixed(2)}%</td>
          <td class="py-2 font-bold ${gainColor}">${gainSign}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    // 9. Patch Table Controller & Pagination
    function filterPatchTable() {
      const q = document.getElementById("patchSearchInput").value.toLowerCase();
      const split = document.getElementById("splitFilter").value;
      const anom = document.getElementById("anomalyFilter").value;

      filteredTableData = patchTable.filter(p => {
        const matchQ = p.patch_id.toLowerCase().includes(q) || p.field_id.toLowerCase().includes(q) || p.categories.toLowerCase().includes(q);
        const matchSplit = split === "all" || p.split === split;
        let matchAnom = true;
        if (anom === "anomalous") matchAnom = p.has_anomaly === true;
        if (anom === "healthy") matchAnom = p.has_anomaly === false;
        return matchQ && matchSplit && matchAnom;
      });

      tableCurrentPage = 1;
      renderPatchTable();
    }

    function renderPatchTable() {
      const tbody = document.getElementById("patchTableBody");
      tbody.innerHTML = "";

      const start = (tableCurrentPage - 1) * tablePageSize;
      const end = Math.min(start + tablePageSize, filteredTableData.length);
      const pageData = filteredTableData.slice(start, end);

      pageData.forEach(p => {
        const tr = document.createElement("tr");
        tr.className = "hover:bg-[var(--bg-surface-elevated)] transition-colors";
        tr.innerHTML = `
          <td class="py-2 text-cyan-400 font-bold">${p.patch_id}</td>
          <td class="py-2 text-[var(--text-muted)]">${p.split}</td>
          <td class="py-2 text-[var(--text-muted)]">${p.field_id}</td>
          <td class="py-2">${p.has_anomaly ? `<span class="px-1.5 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-300 font-sans">${p.categories}</span>` : '<span class="text-[var(--text-muted)]">none</span>'}</td>
          <td class="py-2 text-[var(--text-muted)]">${p.valid_regions}</td>
          <td class="py-2 text-cyan-400">${p.k_selected}</td>
          <td class="py-2 text-emerald-400">${p.valid_reduction_pct}%</td>
          <td class="py-2 text-purple-400">${p.adaptive_recall_pct !== null ? p.adaptive_recall_pct + '%' : 'N/A'}</td>
          <td class="py-2 text-[var(--text-muted)]">${p.energy_adaptive_j} J</td>
        `;
        tbody.appendChild(tr);
      });

      document.getElementById("tablePageInfo").textContent = `Showing ${filteredTableData.length > 0 ? start + 1 : 0} - ${end} of ${filteredTableData.length} patches`;
      document.getElementById("prevPageBtn").disabled = tableCurrentPage <= 1;
      document.getElementById("nextPageBtn").disabled = end >= filteredTableData.length;
    }

    function changePage(delta) {
      tableCurrentPage += delta;
      renderPatchTable();
    }
  </script>
</body>
</html>
"""


class DashboardGenerator:
    """Generates the interactive HTML5 dashboard file for StressSense."""

    def __init__(self, data_builder: Optional[DashboardDataBuilder] = None) -> None:
        """Initialize generator."""
        self.builder = data_builder if data_builder else DashboardDataBuilder()
        self.output_dir = PROJECT_ROOT / "results" / "dashboard"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Generate the complete self-contained HTML dashboard file."""
        target_path = Path(output_path) if output_path else self.output_dir / "index.html"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.builder.build_payload()
        payload_json = json.dumps(payload)

        html_content = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)

        with open(target_path, "w", encoding="utf-8") as fp:
            fp.write(html_content)

        return target_path


def main():
    """CLI generator test routine."""
    generator = DashboardGenerator()
    out = generator.generate_html()
    print(f"Interactive dashboard successfully generated at: {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
