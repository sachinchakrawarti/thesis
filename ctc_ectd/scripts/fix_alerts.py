# scripts/fix_alerts.py
"""
Apply alert-related fixes across the CTI project:

  1. Add `/api/analyze-demo` endpoint to backend/api.py
  2. Refactor /api/analyze to share a common pipeline (_run_pipeline)
  3. Set NMF vectorizer min_df=1, max_df=0.95 in backend + CLI
  4. Add "Load demo corpus" button + handler in the UI
  5. Register analyzeDemo in ui/config.js

Idempotent: safe to run multiple times.

Usage:
    python scripts/fix_alerts.py
"""

from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_API  = PROJECT_ROOT / "backend" / "api.py"
UI_HTML      = PROJECT_ROOT / "ui" / "index.html"
UI_APP       = PROJECT_ROOT / "ui" / "app.js"
UI_CONFIG    = PROJECT_ROOT / "ui" / "config.js"
CLI_SCRIPT   = PROJECT_ROOT / "scripts" / "analyze_uploads.py"


# ==================================================================
# Helpers
# ==================================================================
def info(msg):  print(f"  ✓ {msg}")
def warn(msg):  print(f"  ! {msg}")
def skip(msg):  print(f"  · {msg}")


def read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ==================================================================
# 1. backend/api.py — add demo endpoint + shared pipeline + min_df fix
# ==================================================================
def patch_backend() -> None:
    print("\n[1/5] Patching backend/api.py ...")
    src = read(BACKEND_API)

    # ---- 1a. vectorizer tuning ----
    old_vec = (
        'vec = TfidfVectorizer(\n'
        '            max_features=5000,\n'
        '            stop_words="english",\n'
        '            ngram_range=(1, 2),\n'
        '            min_df=2,\n'
        '            token_pattern=r"(?u)\\\\b[a-zA-Z][a-zA-Z\\\\-]+\\\\b",\n'
        '        )'
    )
    new_vec = (
        'vec = TfidfVectorizer(\n'
        '            max_features=5000,\n'
        '            stop_words="english",\n'
        '            ngram_range=(1, 2),\n'
        '            min_df=1,\n'
        '            max_df=0.95,\n'
        '            token_pattern=r"(?u)\\\\b[a-zA-Z][a-zA-Z\\\\-]+\\\\b",\n'
        '        )'
    )
    if old_vec in src:
        src = src.replace(old_vec, new_vec)
        info("min_df=1, max_df=0.95 applied to NMF vectorizer")
    elif "min_df=1" in src and "max_df=0.95" in src:
        skip("vectorizer already tuned")
    else:
        # Fallback: regex-based replacement
        src2 = re.sub(
            r"min_df\s*=\s*2",
            "min_df=1,\n            max_df=0.95",
            src,
        )
        if src2 != src:
            src = src2
            info("min_df/max_df updated via regex")
        else:
            warn("could not locate min_df in api.py — check manually")

    # ---- 1b. add demo endpoint if not present ----
    if "/api/analyze-demo" not in src:
        demo_endpoint = '''

DEMO_DIR = PROJECT_ROOT / "uploads"


def _run_pipeline(docs):
    """Shared classification + NMF topic-modeling pipeline."""
    if not docs:
        return {"documents": [], "alerts": [], "topics": None}

    model = train_classifier()
    df = pd.DataFrame(docs)
    df["predicted_category"] = model.predict(df["text"])

    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba(df["text"])
        df["confidence"] = probs.max(axis=1)
    elif hasattr(clf, "decision_function"):
        scores = model.decision_function(df["text"])
        df["confidence"] = np.max(scores, axis=1)
    else:
        df["confidence"] = np.nan

    df["parsed_date"] = df["file_name"].apply(parse_date_from_name)
    df["parsed_date"] = df["parsed_date"].apply(
        lambda d: d.strftime("%Y-%m") if pd.notna(d) else None)

    dated = df.dropna(subset=["parsed_date"]).copy()
    topics_payload = None
    alerts = []

    if len(dated) >= 3:
        dated["text_clean"] = dated["text"].apply(strip_meta_tokens)
        vec = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=r"(?u)\\b[a-zA-Z][a-zA-Z\\-]+\\b",
        )
        M = vec.fit_transform(dated["text_clean"])
        k = min(5, M.shape[0], M.shape[1])

        if k >= 2:
            nmf = NMF(n_components=k, random_state=RANDOM_STATE,
                      init="nndsvda", max_iter=500)
            W = nmf.fit_transform(M)
            for i in range(k):
                dated[f"Topic_{i+1}"] = W[:, i]
            dated["dominant_topic"] = W.argmax(axis=1) + 1

            counts = pd.crosstab(dated["parsed_date"], dated["dominant_topic"])
            shares = counts.div(counts.sum(axis=1), axis=0).fillna(0)
            months = list(shares.index)

            topics_payload = {
                "months": months,
                "series": [
                    {"name": f"Topic {int(t)}",
                     "data": [round(float(shares.loc[m, t]), 3) for m in months]}
                    for t in shares.columns
                ],
            }

            for i in range(1, len(months)):
                pm, cm = months[i-1], months[i]
                for tid in shares.columns:
                    ps  = float(shares.loc[pm, tid])
                    cs  = float(shares.loc[cm, tid])
                    cnt = int(counts.loc[cm, tid])
                    if (cs - ps) >= EMERGING_MIN_SHARE_JUMP and cnt >= EMERGING_MIN_COUNT:
                        alerts.append({
                            "previous_month": pm,
                            "current_month": cm,
                            "topic": int(tid),
                            "previous_share": round(ps, 3),
                            "current_share": round(cs, 3),
                            "increase": round(cs - ps, 3),
                            "current_count": cnt,
                        })

    out = df[["file_name", "word_count", "predicted_category",
              "confidence", "parsed_date"]].copy()
    out["confidence"] = out["confidence"].round(4)
    out.to_csv(ARTIFACTS / "uploaded_classifications.csv", index=False)

    if alerts:
        pd.DataFrame(alerts).to_csv(
            ARTIFACTS / "emerging_threat_alerts.csv", index=False)

    return {
        "documents": out.replace({np.nan: None}).to_dict(orient="records"),
        "alerts": alerts,
        "topics": topics_payload,
    }


@app.post("/api/analyze-demo")
async def analyze_demo():
    """Run the pipeline on every file in uploads/ — for UI demos."""
    if not DEMO_DIR.exists():
        return JSONResponse(
            {"error": f"Demo folder not found: {DEMO_DIR}"},
            status_code=404,
        )
    files = [p for p in sorted(DEMO_DIR.iterdir())
             if p.is_file() and p.suffix.lower() in EXTRACTORS]
    if not files:
        return JSONResponse(
            {"error": f"No demo files in {DEMO_DIR}"},
            status_code=404,
        )

    docs = []
    for p in files:
        try:
            txt = extract_and_clean(p)
        except Exception:
            continue
        docs.append({
            "file_name": p.name,
            "extension": p.suffix.lower(),
            "word_count": len(txt.split()),
            "char_count": len(txt),
            "text": txt,
        })

    return _run_pipeline(docs)

'''
        # Insert before the /api/health route so the code above it stays
        src = src.replace(
            '@app.get("/api/health")',
            demo_endpoint.lstrip("\n") + '\n@app.get("/api/health")',
            1,
        )
        info("added _run_pipeline + /api/analyze-demo endpoint")
    else:
        skip("demo endpoint already present")

    write(BACKEND_API, src)


# ==================================================================
# 2. scripts/analyze_uploads.py — same vectorizer fix
# ==================================================================
def patch_cli() -> None:
    print("\n[2/5] Patching scripts/analyze_uploads.py ...")
    if not CLI_SCRIPT.exists():
        warn("CLI script not found — skipping")
        return

    src = read(CLI_SCRIPT)
    if "min_df=1" in src and "max_df=0.95" in src:
        skip("CLI vectorizer already tuned")
        write(CLI_SCRIPT, src)
        return

    # Add min_df=1 / max_df=0.95 wherever the NMF vectorizer appears
    src2 = src.replace(
        'ngram_range=(1, 2),\n            min_df=2',
        'ngram_range=(1, 2),\n            min_df=1,\n            max_df=0.95',
    )
    if src2 == src:
        src2 = src.replace(
            "ngram_range=(1, 2),",
            "ngram_range=(1, 2),\n        min_df=1,\n        max_df=0.95,",
            1,
        )
    if src2 != src:
        write(CLI_SCRIPT, src2)
        info("CLI vectorizer tuned")
    else:
        warn("could not patch CLI vectorizer — check manually")


# ==================================================================
# 3. ui/config.js — register analyzeDemo endpoint
# ==================================================================
def patch_config() -> None:
    print("\n[3/5] Patching ui/config.js ...")
    if not UI_CONFIG.exists():
        warn("ui/config.js missing — skipping")
        return

    src = read(UI_CONFIG)
    if "analyzeDemo" in src:
        skip("analyzeDemo already registered")
        return

    new = """// Point at the FastAPI backend started by scripts/run_backend.ps1
window.CTI_CONFIG = {
  API_BASE: "http://127.0.0.1:8000",
  ENDPOINTS: {
    health:      "/api/health",
    analyze:     "/api/analyze",
    analyzeDemo: "/api/analyze-demo"
  }
};
"""
    write(UI_CONFIG, new)
    info("analyzeDemo registered")


# ==================================================================
# 4. ui/index.html — add "Load demo corpus" button
# ==================================================================
def patch_html() -> None:
    print("\n[4/5] Patching ui/index.html ...")
    if not UI_HTML.exists():
        warn("ui/index.html missing — skipping")
        return

    src = read(UI_HTML)
    if 'id="demoBtn"' in src:
        skip("demo button already present")
        return

    old = '''<div class="actions">
        <button id="analyzeBtn" class="btn primary" disabled>Analyze</button>
        <button id="resetBtn"   class="btn ghost">Reset</button>
      </div>'''

    new = '''<div class="actions">
        <button id="analyzeBtn" class="btn primary" disabled>Analyze</button>
        <button id="demoBtn"    class="btn ghost">Load demo corpus</button>
        <button id="resetBtn"   class="btn ghost">Reset</button>
      </div>'''

    if old in src:
        src = src.replace(old, new)
        write(UI_HTML, src)
        info("demo button inserted")
    else:
        warn("could not find actions block — check index.html manually")


# ==================================================================
# 5. ui/app.js — wire up demo button
# ==================================================================
def patch_app() -> None:
    print("\n[5/5] Patching ui/app.js ...")
    if not UI_APP.exists():
        warn("ui/app.js missing — skipping")
        return

    src = read(UI_APP)

    # 5a. add demoBtn reference if not present
    if 'const demoBtn' not in src:
        src = src.replace(
            'const resetBtn    = $("#resetBtn");',
            'const resetBtn    = $("#resetBtn");\n'
            '  const demoBtn     = $("#demoBtn");',
        )
        info("added demoBtn DOM reference")

    # 5b. add demoBtn handler before "refreshFileList();" at the end
    if 'demoBtn.addEventListener' not in src:
        handler = '''
  // ---------- demo corpus ----------
  demoBtn.addEventListener("click", async () => {
    demoBtn.disabled = true;
    demoBtn.textContent = "Loading demo\\u2026";
    log("POST /api/analyze-demo \\u2014 running pipeline on uploads/");

    try {
      const r = await fetch(cfg.API_BASE + cfg.ENDPOINTS.analyzeDemo, {
        method: "POST"
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${r.status}`);
      }
      const data = await r.json();
      log(`Demo done: ${data.documents.length} docs, ${data.alerts.length} alerts.`);
      handleResults(data);
    } catch (e) {
      log("Demo failed: " + e.message, "err");
    } finally {
      demoBtn.disabled = false;
      demoBtn.textContent = "Load demo corpus";
    }
  });

  refreshFileList();'''
        src = src.replace("\n  refreshFileList();", handler, 1)
        info("demo button handler attached")
    else:
        skip("demo handler already present")

    write(UI_APP, src)


# ==================================================================
# Main
# ==================================================================
def main() -> int:
    print("=" * 60)
    print("CTI alert fixes")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")

    # Pre-check
    required = [BACKEND_API, UI_HTML, UI_APP, UI_CONFIG]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n❌ Missing required files:")
        for p in missing:
            print(f"   {p}")
        print("\nRun scripts\\setup_ui.py first to create the UI/backend files.")
        return 1

    patch_backend()
    patch_cli()
    patch_config()
    patch_html()
    patch_app()

    print("\n" + "=" * 60)
    print("✅ All patches applied")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Restart the backend (Ctrl+C, then re-run):")
    print("       .\\scripts\\run_backend.ps1")
    print("  2. Hard-reload the browser (Ctrl+Shift+R).")
    print("  3. Click 'Load demo corpus' — the alerts table will fill in.")
    print("\nOr test from CLI:")
    print("  python scripts\\analyze_uploads.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())