"""
Generates 50 JSON test reports, runs the analyzer, saves results.
Outputs:
    artifacts/test_reports.json          (50 inputs)
    artifacts/test_results.json          (50 detailed results)
    artifacts/test_results_table.csv     (CSV for thesis)
    artifacts/test_summary.json          (accuracy + confusion matrix, both views)
Run from project root:  python scripts/test_50_reports.py
"""
import json, os, re, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

os.makedirs("artifacts", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 50 test reports
# ─────────────────────────────────────────────────────────────
REPORTS = [
    # 10 KNOWN threat patterns
    {"id": 1, "expected": "known", "category": "ransomware",
     "text": "The CyberVolk ransomware family encrypted files on the victim server. "
             "A ransom note named InvisibleReadMe.txt was placed in each directory. "
             "The operators demanded payment in Bitcoin to a specific wallet address."},
    {"id": 2, "expected": "known", "category": "apt",
     "text": "APT28 used a spear phishing email to deliver a malicious attachment. "
             "The dropper executed a PowerShell script that connected to a command "
             "and control server. The actor used CVE-2023-23397 for initial access."},
    {"id": 3, "expected": "known", "category": "apt",
     "text": "The threat actor deployed the Sandworm CaddyWiper malware against "
             "the OT environment. Industrial control system devices were targeted. "
             "The substation operator observed SCADA anomalies before the wiper triggered."},
    {"id": 4, "expected": "known", "category": "apt",
     "text": "North Korean actors attributed to Lazarus Group targeted cryptocurrency "
             "exchanges. The DPRK campaign used a fake job offer to deliver malware "
             "through a PDF lure. Funds were laundered through mixers."},
    {"id": 5, "expected": "known", "category": "rat",
     "text": "A Remote Access Trojan named AsyncRAT was observed in the wild. "
             "The malware supported keylogging, screen capture, and remote shell. "
             "Command and control traffic was routed over HTTPS to a hardcoded server."},
    {"id": 6, "expected": "known", "category": "ransomware",
     "text": "LockBit 3.0 affiliates targeted healthcare organizations through "
             "exploited VPN credentials. Double extortion was used: files were "
             "encrypted and data was exfiltrated to a leak site."},
    {"id": 7, "expected": "known", "category": "botnet",
     "text": "A Mirai variant infected IoT devices through default credentials. "
             "The botnet launched volumetric DDoS attacks against a financial "
             "institution. The C2 infrastructure rotated domains every 24 hours."},
    {"id": 8, "expected": "known", "category": "phishing",
     "text": "The phishing campaign impersonated Microsoft 365. Victims received "
             "emails requesting credential re-verification. The landing page cloned "
             "the Microsoft login portal and harvested usernames and passwords."},
    {"id": 9, "expected": "known", "category": "supply_chain",
     "text": "The 3CX softphone application was compromised via a supply chain "
             "attack. Multiple versions of the installer contained a malicious DLL. "
             "The incident revealed a double supply chain compromise."},
    {"id": 10, "expected": "known", "category": "ics",
     "text": "Sandworm deployed a new variant of the CaddyWiper malware to further "
             "disrupt the victim's IT environment. The actor gained access to the OT "
             "environment through a hypervisor hosting a SCADA management instance."},

    # 10 EMERGING-like patterns
    {"id": 11, "expected": "emerging", "category": "novel",
     "text": "A previously unknown loader named ShadowForge was observed in the wild. "
             "The malware uses polymorphic encryption and evades endpoint detection by "
             "hooking system calls at runtime. Affiliates advertise the loader on "
             "underground forums as a service."},
    {"id": 12, "expected": "emerging", "category": "novel",
     "text": "Threat actors exploited a newly disclosed vulnerability in Ivanti Connect "
             "Secure tracked as CVE-2024-99999. The exploitation chain involved "
             "authentication bypass followed by remote code execution. No public exploit "
             "code was available at the time of writing."},
    {"id": 13, "expected": "emerging", "category": "novel",
     "text": "Researchers identified a novel technique for bypassing multi-factor "
             "authentication using adversary-in-the-middle proxies. The attack abuses "
             "OAuth device-code flow and fake MFA prompts. Affected organizations "
             "span multiple sectors."},
    {"id": 14, "expected": "emerging", "category": "novel",
     "text": "A new ransomware family called CrimsonVeil was observed encrypting "
             "VMware ESXi hosts. The operators used a leaked ESXiArgs exploit. "
             "The ransom note contained a Tor negotiation portal and a leaked sample "
             "of stolen data as proof."},
    {"id": 15, "expected": "emerging", "category": "novel",
     "text": "An unknown APT cluster tracked as UNC7777 exploited a zero-day in "
             "Fortinet FortiOS to deploy a new backdoor called SilentVault. "
             "The backdoor uses DNS-over-HTTPS for command and control."},
    {"id": 16, "expected": "emerging", "category": "novel",
     "text": "A novel browser-in-the-middle attack targeted cryptocurrency users. "
             "The actors used fake popup windows to capture wallet seed phrases. "
             "The technique was not observed in previous campaigns."},
    {"id": 17, "expected": "emerging", "category": "novel",
     "text": "A new infostealer called ObsidianSpy was distributed through "
             "malvertising. The malware targets browser credential stores and "
             "cryptocurrency wallet extensions. Affiliates use a subscription model."},
    {"id": 18, "expected": "emerging", "category": "novel",
     "text": "An emerging supply chain compromise targeted open-source Python packages "
             "in PyPI. The malicious package executed a reverse shell to an unknown IP "
             "address. Multiple downstream projects were affected."},
    {"id": 19, "expected": "emerging", "category": "novel",
     "text": "A previously unseen wormable variant of a known IoT malware was observed "
             "scanning for a new class of vulnerable routers. The worm uses a "
             "domain-generation algorithm with a novel seed."},
    {"id": 20, "expected": "emerging", "category": "novel",
     "text": "Adversaries exploited a zero-day in a popular VPN appliance vendor. "
             "Post-exploitation activity included lateral movement via WMI and "
             "persistence via scheduled tasks. No patch was available at disclosure."},

    # 10 RELEVANT but generic
    {"id": 21, "expected": "known", "category": "generic",
     "text": "The threat actor used Cobalt Strike to move laterally across the "
             "network. The beacon communicated with the C2 infrastructure over HTTPS. "
             "Credentials were dumped from memory using Mimikatz."},
    {"id": 22, "expected": "known", "category": "generic",
     "text": "The intrusion began with a phishing email containing a malicious "
             "Excel attachment. The document used a macro to download a payload. "
             "The payload was executed from a temp directory."},
    {"id": 23, "expected": "known", "category": "generic",
     "text": "The malware established persistence via a registry run key. "
             "It also created a scheduled task to re-execute itself every hour. "
             "Process injection was used to hide from EDR products."},
    {"id": 24, "expected": "known", "category": "generic",
     "text": "The ransomware encrypted files using AES-256 and exfiltrated data "
             "to a cloud storage bucket. The ransom note contained a Tor URL "
             "and a victim ID for negotiation."},
    {"id": 25, "expected": "known", "category": "generic",
     "text": "The phishing page used HTTPS with a valid certificate issued to a "
             "compromised domain. Users were prompted to enter credentials, which "
             "were then forwarded to the actor's server."},
    {"id": 26, "expected": "known", "category": "generic",
     "text": "The botnet operators used a fast-flux DNS technique to keep the C2 "
             "operational after takedown attempts. The bots polled a list of domains "
             "and updated themselves on command."},
    {"id": 27, "expected": "known", "category": "generic",
     "text": "The attacker exploited a public-facing application to gain initial "
             "access. A web shell was uploaded to the server and used to pivot "
             "internally. The actor used stolen admin credentials for lateral movement."},
    {"id": 28, "expected": "known", "category": "generic",
     "text": "The malware family uses process hollowing to inject into a legitimate "
             "executable. Communication with the C2 occurs over DNS. The malware "
             "supports remote file download and execution."},
    {"id": 29, "expected": "known", "category": "generic",
     "text": "The threat actor used valid accounts to access the cloud tenant. "
             "They enumerated resources and modified role assignments. The activity "
             "was detected via audit logs after two weeks."},
    {"id": 30, "expected": "known", "category": "generic",
     "text": "The phishing infrastructure used a compromised WordPress site as a "
             "redirector. Victims were sent to a fake Microsoft login page. The actor "
             "used the harvested credentials to access Office 365 accounts."},

    # 10 IRRELEVANT
    {"id": 31, "expected": "not_relevant", "category": "irrelevant",
     "text": "The weather forecast for tomorrow shows rain in the afternoon. "
             "Temperatures will drop to 12 degrees. Bring an umbrella if you go out."},
    {"id": 32, "expected": "not_relevant", "category": "irrelevant",
     "text": "Our team celebrated the quarterly success with a team lunch. "
             "Everyone enjoyed the food and the activities. Feedback was positive."},
    {"id": 33, "expected": "not_relevant", "category": "irrelevant",
     "text": "The new restaurant downtown serves excellent pasta and pizza. "
             "The service was friendly and the prices reasonable. Recommended for "
             "a casual dinner."},
    {"id": 34, "expected": "not_relevant", "category": "irrelevant",
     "text": "The quarterly earnings report shows a 12% increase in revenue. "
             "The board approved the dividend. Shareholders will receive payment "
             "in the next quarter."},
    {"id": 35, "expected": "not_relevant", "category": "irrelevant",
     "text": "The music festival features several local bands this year. "
             "Tickets are available online. Food trucks will be present at the venue."},
    {"id": 36, "expected": "not_relevant", "category": "irrelevant",
     "text": "Our company recently relocated to a new office building. "
             "The new space has better lighting and more meeting rooms. "
             "Employees are adjusting well."},
    {"id": 37, "expected": "not_relevant", "category": "irrelevant",
     "text": "The latest smartphone model has a better camera and longer battery "
             "life. It comes in three colors. The price is competitive with similar "
             "devices on the market."},
    {"id": 38, "expected": "not_relevant", "category": "irrelevant",
     "text": "The annual conference will be held in Berlin next spring. "
             "Registration opens in January. Early-bird pricing is available until "
             "the end of the month."},
    {"id": 39, "expected": "not_relevant", "category": "irrelevant",
     "text": "The hiking trail near the lake offers scenic views and is suitable "
             "for beginners. The path is well marked. Bring water and comfortable shoes."},
    {"id": 40, "expected": "not_relevant", "category": "irrelevant",
     "text": "The book club meets every Thursday evening to discuss the current "
             "selection. Everyone is welcome to join. Refreshments are provided."},

    # 10 MIXED / BORDERLINE
    {"id": 41, "expected": "known", "category": "mixed",
     "text": "The security team updated firewall rules to block suspicious outbound "
             "traffic. The new rules blocked an IP range previously associated with "
             "malware distribution. No active compromise was identified."},
    {"id": 42, "expected": "emerging", "category": "mixed",
     "text": "A new TTP was observed involving LOLBins and PowerShell. "
             "The actor leveraged certutil to download payloads from an unusual CDN. "
             "The technique was not previously documented."},
    {"id": 43, "expected": "known", "category": "mixed",
     "text": "The vulnerability management team patched CVE-2024-11111 across "
             "affected systems. The patch addressed a privilege escalation issue "
             "in a widely used library."},
    {"id": 44, "expected": "known", "category": "mixed",
     "text": "The threat intelligence team shared IOCs with industry partners. "
             "The indicators included domain names, IP addresses, and file hashes "
             "tied to a known campaign."},
    {"id": 45, "expected": "emerging", "category": "mixed",
     "text": "An unusual pattern of reconnaissance was detected against web "
             "applications in our environment. The requests used a novel user-agent "
             "and probed endpoints not typically scanned by automated bots."},
    {"id": 46, "expected": "known", "category": "mixed",
     "text": "The user reported receiving a suspicious email with an attachment. "
             "The email claimed to be from a known vendor. Analysis showed the "
             "attachment was a PDF with an embedded link to a phishing page."},
    {"id": 47, "expected": "emerging", "category": "mixed",
     "text": "The honeypot captured new scanning activity targeting a recently "
             "disclosed CVE. The scanning originated from cloud hosting providers "
             "and included attempts to deliver a novel payload."},
    {"id": 48, "expected": "known", "category": "mixed",
     "text": "The security operations center observed anomalous login attempts "
             "from a foreign IP range. The attempts were blocked by conditional access "
             "policies. No successful compromise was detected."},
    {"id": 49, "expected": "known", "category": "mixed",
     "text": "The incident response team isolated the affected endpoint and collected "
             "forensic artifacts. The analysis showed a known malware family. "
             "Remediation steps were completed."},
    {"id": 50, "expected": "emerging", "category": "mixed",
     "text": "A new malware loader was observed in the wild. The loader evades "
             "detection by using sleep obfuscation and indirect syscalls. Affiliates "
             "advertise it on Telegram as a subscription service."},
]

# ─────────────────────────────────────────────────────────────
# Load trained artifacts
# ─────────────────────────────────────────────────────────────
print("Loading artifacts...")
ART = Path("artifacts/models")
vec  = joblib.load(ART / "tfidf_vectorizer.pkl")
clf  = joblib.load(ART / "hybrid_classifier.pkl")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic.load("artifacts/bertopic_model", embedding_model=sbert)
VOCAB = set(json.load(open(ART / "vocab_reference.json")))
TOPIC_BASE = {int(k): float(v) for k, v in json.load(open(ART / "topic_baseline.json")).items()}
print("Artifacts loaded.\n")

def split_sentences(text, min_words=4):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.split()) >= min_words]

def predict(text):
    sents = split_sentences(text)
    if not sents:
        return {
            "verdict": "not_relevant",
            "total_sentences": 0, "relevant_sentences": 0,
            "irrelevant_sentences": 0, "relevant_fraction": 0.0,
            "topic_counts": {}, "dominant_topic": None,
            "emerging_threat": False,
            "novelty_ratio": 0.0,
            "rule_hits": {"outlier": False, "surge": False, "novelty": False},
            "reasons": [],
        }

    emb = sbert.encode(sents, batch_size=32, convert_to_numpy=True)
    X = hstack([vec.transform(sents), normalize(emb)]).tocsr()
    preds = clf.predict(X)
    rel = [s for s, p in zip(sents, preds) if p == 1]
    rel_emb = emb[preds == 1]

    topics = topic_model.transform(rel, rel_emb)[0] if len(rel) > 0 else []
    tc = {}
    for t in topics:
        t = int(t)
        tc[t] = tc.get(t, 0) + 1

    reasons, emerging = [], False
    rule_hits = {"outlier": False, "surge": False, "novelty": False}

    if len(topics) > 0 and -1 in tc and tc[-1] / len(topics) > 0.30:
        emerging = True
        rule_hits["outlier"] = True
        reasons.append(f"{tc[-1]} outlier sentences")

    if len(topics) > 0:
        for tid, c in tc.items():
            if tid == -1:
                continue
            surge = (c / len(topics)) / max(TOPIC_BASE.get(tid, 1e-9), 1e-9)
            if surge > 3.0 and c >= 3:
                emerging = True
                rule_hits["surge"] = True
                reasons.append(f"topic {tid} surge {surge:.1f}x")

    toks = [t for s in rel for t in re.findall(r"\b[a-z]{3,}\b", s.lower())]
    novelty_ratio = 0.0
    if toks:
        novel = sum(1 for t in toks if t not in VOCAB)
        novelty_ratio = novel / len(toks)
        if novelty_ratio > 0.40:
            emerging = True
            rule_hits["novelty"] = True
            reasons.append(f"{novelty_ratio*100:.0f}% novel tokens")

    dom = None
    if tc:
        dom_tid = max(tc, key=tc.get)
        if dom_tid != -1:
            dom = {"topic_id": dom_tid,
                   "top_words": [w for w, _ in topic_model.get_topic(dom_tid)][:8],
                   "count": tc[dom_tid],
                   "share": round(tc[dom_tid] / len(topics), 3)}
            reasons.append(f"dominant topic {dom_tid}: " + ", ".join(dom["top_words"]))

    verdict = "not_relevant" if not rel else ("emerging" if emerging else "known")

    return {
        "verdict": verdict,
        "total_sentences": len(sents),
        "relevant_sentences": len(rel),
        "irrelevant_sentences": len(sents) - len(rel),
        "relevant_fraction": round(len(rel) / len(sents), 3),
        "topic_counts": {str(k): v for k, v in tc.items()},
        "dominant_topic": dom,
        "emerging_threat": emerging,
        "novelty_ratio": round(novelty_ratio, 3),
        "rule_hits": rule_hits,
        "reasons": reasons,
    }

# ─────────────────────────────────────────────────────────────
# Run all 50
# ─────────────────────────────────────────────────────────────
print("Running analyzer on 50 reports...\n")
results = []
for r in REPORTS:
    pred = predict(r["text"])
    results.append({
        "id": r["id"],
        "category": r["category"],
        "expected": r["expected"],
        "predicted": pred["verdict"],
        "match": pred["verdict"] == r["expected"],
        "total_sentences": pred["total_sentences"],
        "relevant_sentences": pred["relevant_sentences"],
        "irrelevant_sentences": pred["irrelevant_sentences"],
        "relevant_fraction": pred["relevant_fraction"],
        "emerging_threat": pred["emerging_threat"],
        "novelty_ratio": pred["novelty_ratio"],
        "dominant_topic": (pred["dominant_topic"] or {}).get("topic_id"),
        "rule_hits": pred["rule_hits"],
        "reasons": "; ".join(pred["reasons"]),
    })

df = pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────
# Save inputs + detailed results + CSV
# ─────────────────────────────────────────────────────────────
json.dump(REPORTS, open("artifacts/test_reports.json", "w"), indent=2)

detailed = []
for r, res in zip(REPORTS, results):
    detailed.append({
        "id": r["id"],
        "category": r["category"],
        "text": r["text"],
        "expected": r["expected"],
        "predicted": res["predicted"],
        "match": res["match"],
        "classification": {
            "total_sentences": res["total_sentences"],
            "relevant_sentences": res["relevant_sentences"],
            "irrelevant_sentences": res["irrelevant_sentences"],
            "relevant_fraction": res["relevant_fraction"],
        },
        "emerging_detection": {
            "emerging_threat": res["emerging_threat"],
            "novelty_ratio": res["novelty_ratio"],
            "rule_hits": res["rule_hits"],
            "dominant_topic": res["dominant_topic"],
        },
        "reasons": res["reasons"],
    })
json.dump(detailed, open("artifacts/test_results.json", "w"), indent=2)

# Remove the dict column from CSV (pandas handles it but keeps it as string)
df_csv = df.copy()
df_csv["rule_hits"] = df_csv["rule_hits"].astype(str)
df_csv.to_csv("artifacts/test_results_table.csv", index=False)

# ─────────────────────────────────────────────────────────────
# Summary + summary JSON
# ─────────────────────────────────────────────────────────────
overall_acc = float(df["match"].mean())

by_category = {
    cat: {
        "accuracy": float(df[df["expected"] == cat]["match"].mean()),
        "correct":  int(df[df["expected"] == cat]["match"].sum()),
        "total":    int((df["expected"] == cat).sum()),
    }
    for cat in df["expected"].unique()
}

confusion = pd.crosstab(df["expected"], df["predicted"])
confusion_dict = {
    exp: {pred: int(confusion.loc[exp, pred]) if pred in confusion.columns else 0
          for pred in ["emerging", "known", "not_relevant"]}
    for exp in confusion.index
}

# emerging-detection specific metrics
tp = int(((df["expected"] == "emerging") & (df["predicted"] == "emerging")).sum())
fp = int(((df["expected"] != "emerging") & (df["predicted"] == "emerging")).sum())
fn = int(((df["expected"] == "emerging") & (df["predicted"] != "emerging")).sum())
precision_e = tp / (tp + fp) if (tp + fp) else 0.0
recall_e    = tp / (tp + fn) if (tp + fn) else 0.0
f1_e        = (2 * precision_e * recall_e / (precision_e + recall_e)) if (precision_e + recall_e) else 0.0

mismatches = df[df["match"] == False][
    ["id", "expected", "predicted", "category", "reasons"]
].to_dict(orient="records")

summary = {
    "total_reports": len(df),
    "overall_accuracy": round(overall_acc, 4),
    "correct": int(df["match"].sum()),
    "by_category": by_category,
    "confusion_matrix": confusion_dict,
    "emerging_metrics": {
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision_e, 4),
        "recall":    round(recall_e, 4),
        "f1":        round(f1_e, 4),
    },
    "mismatches": mismatches,
}
json.dump(summary, open("artifacts/test_summary.json", "w"), indent=2)

# ─────────────────────────────────────────────────────────────
# Print to console
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print(f"{'50-REPORT TEST SUMMARY':^70s}")
print("=" * 70)
print(f"\nOverall verdict accuracy: {overall_acc*100:.1f}% "
      f"({int(df['match'].sum())}/{len(df)} correct)\n")

print("By expected category:")
for cat, m in by_category.items():
    print(f"  {cat:>14s}: {m['accuracy']*100:5.1f}% ({m['correct']}/{m['total']})")

print("\nConfusion matrix (expected × predicted):")
print(confusion.to_string())

print("\nEmerging-threat detection metrics:")
print(f"  True positives  : {tp}")
print(f"  False positives : {fp}")
print(f"  False negatives : {fn}")
print(f"  Precision       : {precision_e*100:.1f}%")
print(f"  Recall          : {recall_e*100:.1f}%")
print(f"  F1              : {f1_e*100:.1f}%")

if mismatches:
    print(f"\n{len(mismatches)} mismatch(es):")
    for m in mismatches:
        print(f"  ID {m['id']:>2d} [{m['category']}]: "
              f"expected={m['expected']}, predicted={m['predicted']}")
else:
    print("\nNo mismatches — all 50 correct.")

print("\nFiles saved:")
print(f"  artifacts/test_reports.json        ({len(REPORTS)} inputs)")
print(f"  artifacts/test_results.json        ({len(detailed)} detailed results)")
print(f"  artifacts/test_results_table.csv   ({len(df)} rows)")
print(f"  artifacts/test_summary.json        (summary + confusion + emerging metrics)")
print("=" * 70)