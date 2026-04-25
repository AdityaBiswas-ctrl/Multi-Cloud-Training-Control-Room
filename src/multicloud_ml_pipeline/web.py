from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import load_config
from .flow import run_training_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "examples" / "pipeline-config.json"

PAGE_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Multi-Cloud Trainer</title>
    <style>
      :root {
        --night: #0f172a;
        --slate: #24324d;
        --muted: #5b6b86;
        --mist: #e8eef8;
        --paper: #fcf8ef;
        --panel: rgba(255, 250, 242, 0.72);
        --glass: rgba(255, 255, 255, 0.68);
        --line: rgba(19, 33, 56, 0.12);
        --sky: #2f80ed;
        --teal: #15b8a6;
        --amber: #ffb454;
        --ember: #f97360;
        --violet: #6d5efc;
        --success: #1d9c67;
        --warning: #d97706;
        --danger: #cf334d;
        --shadow: 0 28px 90px rgba(21, 31, 53, 0.14);
      }

      * {
        box-sizing: border-box;
      }

      html {
        scroll-behavior: smooth;
      }

      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Aptos", "Trebuchet MS", "Segoe UI", sans-serif;
        color: var(--night);
        background:
          radial-gradient(circle at 8% 10%, rgba(21, 184, 166, 0.22), transparent 28%),
          radial-gradient(circle at 88% 12%, rgba(109, 94, 252, 0.16), transparent 26%),
          radial-gradient(circle at 50% 100%, rgba(255, 180, 84, 0.18), transparent 38%),
          linear-gradient(135deg, #fff9f2 0%, #eef6ff 52%, #fbfdff 100%);
      }

      body::before,
      body::after {
        content: "";
        position: fixed;
        inset: auto;
        width: 38vw;
        height: 38vw;
        border-radius: 50%;
        z-index: -1;
        filter: blur(60px);
        opacity: 0.32;
      }

      body::before {
        top: -12vw;
        right: -10vw;
        background: rgba(47, 128, 237, 0.26);
      }

      body::after {
        left: -14vw;
        bottom: -12vw;
        background: rgba(249, 115, 96, 0.18);
      }

      .page {
        width: min(1240px, calc(100vw - 32px));
        margin: 24px auto 48px;
      }

      .frame {
        border: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.58), rgba(255, 252, 247, 0.72));
        box-shadow: var(--shadow);
        border-radius: 32px;
        overflow: hidden;
        backdrop-filter: blur(18px);
      }

      .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        padding: 18px 24px;
        border-bottom: 1px solid rgba(19, 33, 56, 0.08);
        background: rgba(255, 255, 255, 0.52);
      }

      .topbar-brand {
        display: flex;
        align-items: center;
        gap: 14px;
      }

      .signal-mark {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background:
          linear-gradient(140deg, rgba(47, 128, 237, 0.94), rgba(21, 184, 166, 0.94));
        position: relative;
        box-shadow: 0 16px 24px rgba(47, 128, 237, 0.24);
      }

      .signal-mark::before,
      .signal-mark::after {
        content: "";
        position: absolute;
        inset: 11px;
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.9);
      }

      .signal-mark::after {
        inset: 6px;
        opacity: 0.35;
      }

      .topbar-title {
        display: grid;
        gap: 2px;
      }

      .kicker,
      .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        color: var(--muted);
      }

      .topbar-title strong {
        font-size: 1rem;
      }

      .topbar-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }

      .chip {
        padding: 10px 14px;
        border-radius: 999px;
        border: 1px solid rgba(19, 33, 56, 0.08);
        background: rgba(255, 255, 255, 0.72);
        color: var(--slate);
        font-size: 0.86rem;
      }

      .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(300px, 0.85fr);
        gap: 24px;
        padding: 28px 24px 12px;
      }

      .hero-copy {
        display: grid;
        gap: 18px;
        align-content: start;
      }

      .headline {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.7rem, 6vw, 5rem);
        line-height: 0.94;
        letter-spacing: -0.05em;
      }

      .headline span {
        display: block;
        color: var(--slate);
      }

      .lead {
        margin: 0;
        max-width: 720px;
        font-size: 1.05rem;
        line-height: 1.75;
        color: var(--muted);
      }

      .cloud-ribbon {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }

      .cloud-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(19, 33, 56, 0.08);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
      }

      .cloud-pill::before {
        content: "";
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--sky), var(--teal));
      }

      .hero-note {
        display: flex;
        flex-wrap: wrap;
        gap: 12px 18px;
        color: var(--slate);
        font-size: 0.95rem;
      }

      .hero-note strong {
        display: block;
        font-size: 1.5rem;
      }

      .hero-note-item {
        min-width: 140px;
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.62);
        border: 1px solid rgba(19, 33, 56, 0.08);
      }

      .hero-visual {
        position: relative;
        min-height: 360px;
        padding: 24px;
        border-radius: 28px;
        overflow: hidden;
        background:
          radial-gradient(circle at 50% 10%, rgba(255,255,255,0.82), rgba(255,255,255,0.16)),
          linear-gradient(160deg, rgba(24, 41, 67, 0.96), rgba(47, 128, 237, 0.82) 56%, rgba(21, 184, 166, 0.78));
        color: #f9fbff;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16);
      }

      .hero-visual::before,
      .hero-visual::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        border: 1px solid rgba(255, 255, 255, 0.18);
      }

      .hero-visual::before {
        inset: 22px;
      }

      .hero-visual::after {
        inset: 68px;
      }

      .radar-title {
        position: relative;
        z-index: 1;
        display: grid;
        gap: 6px;
        max-width: 270px;
      }

      .radar-title h2 {
        margin: 0;
        font-size: 1.5rem;
      }

      .radar-title p {
        margin: 0;
        color: rgba(240, 248, 255, 0.82);
        line-height: 1.6;
      }

      .orbit-node {
        position: absolute;
        z-index: 1;
        display: grid;
        gap: 4px;
        width: 126px;
        padding: 14px 14px 12px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 18px 32px rgba(7, 13, 24, 0.18);
        backdrop-filter: blur(8px);
      }

      .orbit-node strong {
        font-size: 0.98rem;
      }

      .orbit-node span {
        color: rgba(240, 248, 255, 0.82);
        font-size: 0.82rem;
      }

      .orbit-node.aws {
        top: 98px;
        right: 34px;
      }

      .orbit-node.gcp {
        left: 34px;
        bottom: 42px;
      }

      .orbit-node.azure {
        right: 86px;
        bottom: 28px;
      }

      .core-node {
        position: absolute;
        inset: 50% auto auto 50%;
        transform: translate(-50%, -50%);
        width: 148px;
        height: 148px;
        border-radius: 50%;
        background:
          radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.16)),
          linear-gradient(180deg, rgba(255,255,255,0.16), rgba(255,255,255,0.04));
        display: grid;
        place-items: center;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.22);
        box-shadow: 0 22px 40px rgba(7, 13, 24, 0.24);
      }

      .core-node strong {
        display: block;
        font-size: 1.15rem;
      }

      .core-node span {
        display: block;
        font-size: 0.8rem;
        color: rgba(240, 248, 255, 0.78);
      }

      .dashboard {
        display: grid;
        grid-template-columns: 320px minmax(0, 1fr);
        gap: 20px;
        padding: 12px 24px 24px;
      }

      .panel {
        border: 1px solid var(--line);
        border-radius: 24px;
        background: var(--panel);
        padding: 20px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.54);
      }

      .panel-title {
        margin: 0 0 8px;
        font-size: 1.15rem;
      }

      .panel-copy {
        margin: 0;
        color: var(--muted);
        line-height: 1.65;
      }

      .stack {
        display: grid;
        gap: 14px;
      }

      .control-box {
        padding: 16px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(19, 33, 56, 0.08);
      }

      label {
        display: block;
        margin-bottom: 8px;
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
      }

      input {
        width: 100%;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(19, 33, 56, 0.12);
        background: rgba(255, 255, 255, 0.94);
        color: var(--night);
        font: inherit;
      }

      .run-button {
        width: 100%;
        margin-top: 14px;
        padding: 16px 18px;
        border: 0;
        border-radius: 18px;
        font: inherit;
        font-weight: 700;
        color: white;
        cursor: pointer;
        background: linear-gradient(135deg, var(--sky), var(--teal));
        box-shadow: 0 18px 26px rgba(47, 128, 237, 0.22);
        transition: transform 180ms ease, box-shadow 180ms ease, opacity 180ms ease;
      }

      .run-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 24px 34px rgba(21, 184, 166, 0.24);
      }

      .run-button:disabled {
        opacity: 0.8;
        cursor: wait;
        transform: none;
      }

      .status-box {
        margin-top: 14px;
        padding: 12px 14px;
        border-radius: 16px;
        background: rgba(242, 247, 255, 0.82);
        border: 1px solid rgba(19, 33, 56, 0.06);
        color: var(--slate);
        min-height: 52px;
        line-height: 1.5;
      }

      .legend {
        display: grid;
        gap: 10px;
      }

      .legend-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.66);
      }

      .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex: 0 0 auto;
      }

      .legend-row span:last-child {
        color: var(--muted);
      }

      .results-shell {
        display: grid;
        gap: 18px;
      }

      .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
      }

      .summary-card {
        padding: 18px;
        border-radius: 20px;
        background:
          linear-gradient(180deg, rgba(255,255,255,0.86), rgba(241, 246, 255, 0.78));
        border: 1px solid rgba(19, 33, 56, 0.08);
      }

      .summary-card span {
        color: var(--muted);
        font-size: 0.84rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .summary-card strong {
        display: block;
        margin-top: 8px;
        font-size: 1.55rem;
        letter-spacing: -0.03em;
      }

      .jobs {
        display: grid;
        gap: 18px;
      }

      .job-card {
        padding: 20px;
        border-radius: 24px;
        background:
          linear-gradient(180deg, rgba(255,255,255,0.92), rgba(247, 250, 255, 0.9));
        border: 1px solid rgba(19, 33, 56, 0.08);
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
        animation: rise 420ms ease both;
      }

      .job-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 18px;
        margin-bottom: 18px;
      }

      .job-head h3 {
        margin: 6px 0 8px;
        font-size: 1.25rem;
      }

      .job-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }

      .meta-pill,
      .rank-pill,
      .status-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(19, 33, 56, 0.08);
        font-size: 0.82rem;
        background: rgba(255, 255, 255, 0.88);
      }

      .meta-pill.primary {
        background: rgba(47, 128, 237, 0.12);
        color: #184a92;
      }

      .job-cost {
        min-width: 132px;
        padding: 16px;
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(255, 180, 84, 0.18), rgba(255, 255, 255, 0.94));
        border: 1px solid rgba(19, 33, 56, 0.08);
        text-align: right;
      }

      .job-cost span {
        display: block;
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .job-cost strong {
        display: block;
        margin-top: 6px;
        font-size: 1.5rem;
      }

      .job-grid {
        display: grid;
        grid-template-columns: 1.05fr 0.95fr;
        gap: 16px;
      }

      .info-card {
        padding: 16px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(19, 33, 56, 0.08);
      }

      .info-card h4 {
        margin: 0 0 12px;
        font-size: 0.92rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }

      .artifact {
        margin: 0 0 12px;
        font-family: Consolas, "Courier New", monospace;
        font-size: 0.88rem;
        color: var(--slate);
        word-break: break-word;
      }

      .metric-line {
        display: grid;
        gap: 10px;
      }

      .metric-row {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        padding: 10px 0;
        border-bottom: 1px dashed rgba(19, 33, 56, 0.1);
      }

      .metric-row:last-child {
        border-bottom: 0;
        padding-bottom: 0;
      }

      .attempts,
      .route-list {
        display: grid;
        gap: 10px;
      }

      .attempt-item,
      .route-item {
        display: grid;
        gap: 8px;
        padding: 14px 14px 12px;
        border-radius: 16px;
        background: rgba(248, 250, 255, 0.88);
        border: 1px solid rgba(19, 33, 56, 0.08);
      }

      .attempt-top,
      .route-top {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
      }

      .attempt-meta,
      .route-reason {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
      }

      .route-item.selected {
        background: linear-gradient(180deg, rgba(47, 128, 237, 0.12), rgba(21, 184, 166, 0.08));
      }

      .status-pill.completed {
        color: var(--success);
        background: rgba(29, 156, 103, 0.12);
      }

      .status-pill.failed-recoverable {
        color: var(--warning);
        background: rgba(217, 119, 6, 0.12);
      }

      .status-pill.failed-fatal {
        color: var(--danger);
        background: rgba(207, 51, 77, 0.12);
      }

      details {
        border: 1px solid rgba(19, 33, 56, 0.08);
        border-radius: 20px;
        background: rgba(14, 23, 43, 0.95);
        color: #d9e9ff;
        overflow: hidden;
      }

      summary {
        cursor: pointer;
        padding: 16px 18px;
        font-weight: 700;
        letter-spacing: 0.03em;
      }

      pre {
        margin: 0;
        max-height: 360px;
        overflow: auto;
        padding: 0 18px 18px;
        color: #d9e9ff;
        font-family: Consolas, "Courier New", monospace;
        font-size: 0.88rem;
      }

      .empty-state {
        padding: 28px;
        border-radius: 24px;
        border: 1px dashed rgba(19, 33, 56, 0.18);
        background: rgba(255, 255, 255, 0.5);
        text-align: center;
        color: var(--muted);
      }

      @keyframes rise {
        from {
          opacity: 0;
          transform: translateY(14px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @media (max-width: 1080px) {
        .hero,
        .dashboard,
        .job-grid,
        .summary-grid {
          grid-template-columns: 1fr;
        }

        .hero-visual {
          min-height: 320px;
        }
      }

      @media (max-width: 720px) {
        .page {
          width: min(100vw - 20px, 1240px);
          margin-top: 10px;
        }

        .frame {
          border-radius: 24px;
        }

        .topbar,
        .hero,
        .dashboard {
          padding-left: 16px;
          padding-right: 16px;
        }

        .topbar {
          align-items: flex-start;
          flex-direction: column;
        }

        .job-head,
        .attempt-top,
        .route-top {
          flex-direction: column;
          align-items: flex-start;
        }

        .job-cost {
          width: 100%;
          text-align: left;
        }

        .orbit-node.aws,
        .orbit-node.gcp,
        .orbit-node.azure {
          position: relative;
          inset: auto;
          width: 100%;
          margin-top: 12px;
        }

        .hero-visual {
          display: grid;
          gap: 12px;
          align-content: start;
        }

        .core-node {
          position: relative;
          inset: auto;
          transform: none;
          width: 100%;
          height: auto;
          padding: 22px 18px;
          border-radius: 24px;
        }
      }
    </style>
  </head>
  <body>
    <main class="page">
      <div class="frame">
        <header class="topbar">
          <div class="topbar-brand">
            <div class="signal-mark"></div>
            <div class="topbar-title">
              <span class="kicker">Live Orchestration Studio</span>
              <strong>Multi-Cloud Training Control Room</strong>
            </div>
          </div>
          <div class="topbar-badges">
            <span class="chip">Prefect flow orchestration</span>
            <span class="chip">AWS + GCP + Azure failover</span>
            <span class="chip">Cost-aware routing</span>
          </div>
        </header>

        <section class="hero">
          <div class="hero-copy">
            <span class="eyebrow">Train anywhere, recover everywhere</span>
            <h1 class="headline">
              Send the job to the best cloud.
              <span>Let the fallback plan handle the rest.</span>
            </h1>
            <p class="lead">
              This browser UI launches the sample Prefect pipeline, scores each provider by cost,
              health, capacity, and region match, and then promotes the next viable cloud when a
              recoverable training error appears.
            </p>
            <div class="cloud-ribbon">
              <span class="cloud-pill">AWS routing lane</span>
              <span class="cloud-pill">GCP primary candidate</span>
              <span class="cloud-pill">Azure backup runway</span>
            </div>
            <div class="hero-note">
              <div class="hero-note-item">
                <strong>3</strong>
                Clouds scored live
              </div>
              <div class="hero-note-item">
                <strong>2</strong>
                Sample jobs in config
              </div>
              <div class="hero-note-item">
                <strong>Auto</strong>
                Browser run on load
              </div>
            </div>
          </div>

          <aside class="hero-visual">
            <div class="radar-title">
              <span class="eyebrow" style="color: rgba(240,248,255,0.7);">Routing Radar</span>
              <h2>Provider scoring stays visible before and after failover.</h2>
              <p>
                The interface emphasizes route order, cost pressure, and recovery attempts instead
                of hiding everything behind raw logs.
              </p>
            </div>
            <div class="orbit-node aws">
              <strong>AWS</strong>
              <span>Fast warm path, intentional first failure in sample</span>
            </div>
            <div class="orbit-node gcp">
              <strong>GCP</strong>
              <span>Cheapest healthy option for both sample jobs</span>
            </div>
            <div class="orbit-node azure">
              <strong>Azure</strong>
              <span>Available as the third ranked fallback target</span>
            </div>
            <div class="core-node">
              <div>
                <strong>Prefect Flow</strong>
                <span>score -> route -> train -> recover</span>
              </div>
            </div>
          </aside>
        </section>

        <section class="dashboard">
          <aside class="stack">
            <section class="panel">
              <h2 class="panel-title">Launch Deck</h2>
              <p class="panel-copy">
                Use the sample config or point at a different JSON definition. The page auto-runs
                once when it loads so the current tab immediately shows a live execution.
              </p>
              <div class="control-box" style="margin-top: 16px;">
                <label for="configPath">Config path</label>
                <input id="configPath" value="examples/pipeline-config.json">
                <button class="run-button" id="runButton" type="button">Run orchestration</button>
                <div class="status-box" id="status">Preparing the control room.</div>
              </div>
            </section>

            <section class="panel">
              <h2 class="panel-title">Execution Rules</h2>
              <div class="legend">
                <div class="legend-row">
                  <span class="legend-dot" style="background: var(--sky);"></span>
                  <strong>Rank providers</strong>
                  <span>Cost, health, capacity, and region fit feed the route score.</span>
                </div>
                <div class="legend-row">
                  <span class="legend-dot" style="background: var(--amber);"></span>
                  <strong>Recoverable failure</strong>
                  <span>Capacity or transient issues trigger a handoff to the next cloud.</span>
                </div>
                <div class="legend-row">
                  <span class="legend-dot" style="background: var(--success);"></span>
                  <strong>Successful completion</strong>
                  <span>Artifacts, metrics, and final cloud choice are surfaced per job.</span>
                </div>
              </div>
            </section>
          </aside>

          <section class="results-shell">
            <div class="summary-grid" id="summaryGrid">
              <article class="summary-card">
                <span>Jobs</span>
                <strong>0</strong>
              </article>
              <article class="summary-card">
                <span>Total Cost</span>
                <strong>$0.00</strong>
              </article>
              <article class="summary-card">
                <span>Failovers</span>
                <strong>0</strong>
              </article>
              <article class="summary-card">
                <span>Winning Cloud</span>
                <strong>n/a</strong>
              </article>
            </div>

            <div id="jobs" class="jobs">
              <div class="empty-state">
                Run data will appear here with route ladders, attempt history, artifacts, and
                cost summaries.
              </div>
            </div>

            <details open>
              <summary>Raw JSON output</summary>
              <pre id="rawOutput">The page will auto-run the sample pipeline after load.</pre>
            </details>
          </section>
        </section>
      </div>
    </main>

    <script>
      const runButton = document.getElementById("runButton");
      const statusEl = document.getElementById("status");
      const rawOutput = document.getElementById("rawOutput");
      const jobsEl = document.getElementById("jobs");
      const summaryGrid = document.getElementById("summaryGrid");
      const configInput = document.getElementById("configPath");

      function money(value) {
        return `$${Number(value ?? 0).toFixed(2)}`;
      }

      function escapeHtml(value) {
        return String(value ?? "")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
      }

      function statusClass(status) {
        return String(status || "").toLowerCase();
      }

      function countFailovers(jobs) {
        return jobs.reduce((total, job) => {
          const attempts = job.attempts ?? [];
          return total + attempts.filter((attempt) => attempt.status === "failed-recoverable").length;
        }, 0);
      }

      function winningCloud(jobs) {
        if (!jobs.length) {
          return "n/a";
        }

        const counts = new Map();
        for (const job of jobs) {
          const cloud = job.selected_provider || "n/a";
          counts.set(cloud, (counts.get(cloud) || 0) + 1);
        }

        return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
      }

      function renderSummary(payload) {
        const jobs = payload.routed_jobs ?? [];
        const failovers = countFailovers(jobs);
        const winner = winningCloud(jobs);
        summaryGrid.innerHTML = `
          <article class="summary-card">
            <span>Jobs</span>
            <strong>${jobs.length}</strong>
          </article>
          <article class="summary-card">
            <span>Total Cost</span>
            <strong>${money(payload.total_estimated_cost_usd)}</strong>
          </article>
          <article class="summary-card">
            <span>Failovers</span>
            <strong>${failovers}</strong>
          </article>
          <article class="summary-card">
            <span>Winning Cloud</span>
            <strong>${escapeHtml(winner)}</strong>
          </article>
        `;
      }

      function renderJobs(payload) {
        const jobs = payload.routed_jobs ?? [];
        if (!jobs.length) {
          jobsEl.innerHTML = `
            <div class="empty-state">
              No jobs were returned by the current run.
            </div>
          `;
          return;
        }

        jobsEl.innerHTML = jobs.map((job) => {
          const metrics = job.result?.metrics ?? {};
          const accuracy = metrics.accuracy ?? "n/a";
          const f1Score = metrics.f1_score ?? "n/a";
          const attempts = job.attempts ?? [];
          const route = job.route ?? [];

          return `
            <article class="job-card">
              <div class="job-head">
                <div>
                  <span class="eyebrow">Job ${escapeHtml(job.job_id)}</span>
                  <h3>${escapeHtml(job.model_name)} -> ${escapeHtml(job.selected_provider)}</h3>
                  <div class="job-meta">
                    <span class="meta-pill primary">Selected ${escapeHtml(job.selected_provider)}</span>
                    <span class="meta-pill">${attempts.length} attempt(s)</span>
                    <span class="meta-pill">Accuracy ${escapeHtml(accuracy)}</span>
                    <span class="meta-pill">F1 ${escapeHtml(f1Score)}</span>
                  </div>
                </div>
                <div class="job-cost">
                  <span>Estimated spend</span>
                  <strong>${money(job.result?.cost_usd)}</strong>
                </div>
              </div>

              <div class="job-grid">
                <section class="info-card">
                  <h4>Artifact + attempts</h4>
                  <p class="artifact">${escapeHtml(job.result?.artifact_uri ?? "n/a")}</p>
                  <div class="attempts">
                    ${attempts.map((attempt) => `
                      <div class="attempt-item">
                        <div class="attempt-top">
                          <strong>${escapeHtml(attempt.provider)}</strong>
                          <span class="status-pill ${statusClass(attempt.status)}">${escapeHtml(attempt.status)}</span>
                        </div>
                        <div class="attempt-meta">${escapeHtml(attempt.message || "No detail provided.")}</div>
                        <div class="metric-row">
                          <span>Estimated cost</span>
                          <strong>${money(attempt.estimated_cost_usd)}</strong>
                        </div>
                      </div>
                    `).join("")}
                  </div>
                </section>

                <section class="info-card">
                  <h4>Route ladder</h4>
                  <div class="route-list">
                    ${route.map((step) => `
                      <div class="route-item ${step.provider === job.selected_provider ? "selected" : ""}">
                        <div class="route-top">
                          <strong>${escapeHtml(step.provider)}</strong>
                          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                            <span class="rank-pill">Rank ${escapeHtml(step.rank)}</span>
                            <span class="rank-pill">${money(step.estimated_cost_usd)}</span>
                          </div>
                        </div>
                        <div class="route-reason">${escapeHtml(step.reason)}</div>
                      </div>
                    `).join("")}
                  </div>

                  <div class="metric-line" style="margin-top: 14px;">
                    <div class="metric-row">
                      <span>Cloud job id</span>
                      <strong>${escapeHtml(job.result?.cloud_job_id ?? "n/a")}</strong>
                    </div>
                    <div class="metric-row">
                      <span>Status</span>
                      <strong>${escapeHtml(job.result?.status ?? "n/a")}</strong>
                    </div>
                  </div>
                </section>
              </div>
            </article>
          `;
        }).join("");
      }

      async function runPipeline(options = {}) {
        runButton.disabled = true;
        const isAuto = Boolean(options.auto);
        const startedAt = new Date();
        statusEl.textContent = isAuto
          ? "Auto-running the sample pipeline so the page opens with live results."
          : "Running orchestration across the multi-cloud route set.";

        try {
          const response = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ config_path: configInput.value })
          });
          const payload = await response.json();

          if (!response.ok) {
            throw new Error(payload.error || "Request failed");
          }

          rawOutput.textContent = JSON.stringify(payload, null, 2);
          renderSummary(payload);
          renderJobs(payload);

          const finishedAt = new Date();
          statusEl.textContent =
            `Completed ${payload.routed_jobs.length} job(s) at ${finishedAt.toLocaleTimeString()} ` +
            `after starting at ${startedAt.toLocaleTimeString()}.`;
        } catch (error) {
          statusEl.textContent = `Run failed: ${error.message}`;
          rawOutput.textContent = JSON.stringify({ error: error.message }, null, 2);
          jobsEl.innerHTML = `
            <div class="empty-state">
              The pipeline request failed. Adjust the config path and run again.
            </div>
          `;
        } finally {
          runButton.disabled = false;
        }
      }

      runButton.addEventListener("click", () => runPipeline({ auto: false }));
      window.addEventListener("DOMContentLoaded", () => runPipeline({ auto: true }));
    </script>
  </body>
</html>
"""


def resolve_config_path(raw_path: str | None) -> Path:
    if not raw_path:
        return DEFAULT_CONFIG

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


class PipelineHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_html(PAGE_HTML)
            return
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"

        try:
            request_payload = json.loads(raw_body)
            config_path = resolve_config_path(request_payload.get("config_path"))
            policy, providers, jobs = load_config(config_path)
            summary = run_training_pipeline(jobs=jobs, provider_profiles=providers, policy=policy)
            self._send_json(summary.to_dict())
        except Exception as exc:  # pragma: no cover - exercised manually through the UI.
            self._send_json(
                {
                    "error": str(exc),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the multi-cloud training pipeline web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), PipelineHandler)
    print(f"Serving multi-cloud trainer on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
