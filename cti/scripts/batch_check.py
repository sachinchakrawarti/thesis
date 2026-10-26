"""
Batch-check all PDFs in test_pdfs/ through the analyzer.
Prints a summary table and saves test_pdfs/batch_results.json.

Usage:
    python scripts/batch_check.py
"""
import sys, json
from pathlib import Path

# Make `server` importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.analyzer import CTIAnalyzer

PDF_DIR = Path("test_pdfs")

def main():
    analyzer = CTIAnalyzer()
    analyzer.load()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}/")
        return

    # Load expected verdicts from manifest
    manifest_path = PDF_DIR / "manifest.json"
    expected = {}
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        expected = {m["file"]: m["expected_verdict"] for m in manifest}

    print("=" * 90)
    print(f"{'BATCH ANALYSIS — ' + str(len(pdfs)) + ' PDFs':^90s}")
    print("=" * 90)
    print(f"{'File':<20s} {'Expected':<14s} {'Predicted':<14s} {'Match':<7s} "
          f"{'Relevant':<10s} {'Emerging':<10s} {'Reasons'}")
    print("-" * 90)

    results = []
    correct = 0
    emerging_flags = 0

    for pdf in pdfs:
        result = analyzer.predict_file(str(pdf))
        cls = result.get("classification", {})
        emg = result.get("emerging_detection", {})

        expected_v = expected.get(pdf.name, "?")
        predicted_v = result["verdict"]
        match = "✓" if predicted_v == expected_v else "✗"
        if predicted_v == expected_v:
            correct += 1

        is_emerging = emg.get("emerging_threat", False)
        if is_emerging:
            emerging_flags += 1

        reasons = "; ".join(emg.get("reasons", []))[:40]
        print(f"{pdf.name:<20s} {expected_v:<14s} {predicted_v:<14s} {match:<7s} "
              f"{cls.get('relevant_sentences', 0):<10d} "
              f"{'YES' if is_emerging else 'no':<10s} {reasons}")

        results.append({
            "file": pdf.name,
            "expected": expected_v,
            "predicted": predicted_v,
            "match": predicted_v == expected_v,
            "relevant_sentences": cls.get("relevant_sentences", 0),
            "total_sentences": cls.get("total_sentences", 0),
            "emerging_threat": is_emerging,
            "rule_hits": emg.get("rule_hits", {}),
            "dominant_topic": (emg.get("dominant_topic") or {}).get("topic_id"),
            "novelty_ratio": emg.get("novelty_ratio", 0),
            "reasons": emg.get("reasons", []),
        })

    print("-" * 90)
    print(f"\nSummary:")
    print(f"  Total PDFs        : {len(pdfs)}")
    print(f"  Correct verdicts  : {correct} / {len(pdfs)} "
          f"({correct/len(pdfs)*100:.1f}%)")
    print(f"  Emerging flags    : {emerging_flags}")

    by_pred = {}
    for r in results:
        by_pred[r["predicted"]] = by_pred.get(r["predicted"], 0) + 1
    print(f"\n  Predicted breakdown:")
    for k, v in sorted(by_pred.items()):
        print(f"    {k:<14s}: {v}")

    # Save full results
    out = PDF_DIR / "batch_results.json"
    json.dump({
        "total": len(pdfs),
        "correct": correct,
        "accuracy": round(correct / len(pdfs), 4),
        "emerging_flags": emerging_flags,
        "results": results,
    }, open(out, "w"), indent=2)
    print(f"\n  Full results saved -> {out}")

    print("\n" + "=" * 90)
    print("Emerging-threat PDFs detected:")
    print("=" * 90)
    any_emerging = False
    for r in results:
        if r["emerging_threat"]:
            any_emerging = True
            print(f"  ⚠  {r['file']}  (expected: {r['expected']})")
            print(f"     dominant topic : {r['dominant_topic']}")
            print(f"     novelty ratio  : {r['novelty_ratio']*100:.1f}%")
            print(f"     rule hits      : "
                  f"{[k for k, v in r['rule_hits'].items() if v]}")
            if r["reasons"]:
                print(f"     reasons        :")
                for reason in r["reasons"]:
                    print(f"       - {reason}")
            print()
    if not any_emerging:
        print("  No PDFs were flagged as emerging threats.")


if __name__ == "__main__":
    main()