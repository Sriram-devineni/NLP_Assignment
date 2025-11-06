import os
import re
import pandas as pd
import spacy
from tqdm import tqdm
from collections import Counter

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Paths
DATASET_DIR = "requirements"     # <-- put your extracted dataset folder name here
OUTPUT_CSV = "results.csv"
SUMMARY_CSV = "summary_table.csv"
REPORT_TXT = "final_report.txt"

# Detection helpers
COMPARATIVE_TAGS = {"JJR", "JJS"}
COMPARATIVE_WORDS = {"more", "most", "less", "least", "better", "worse", "best", "worst", "faster", "slower"}
COORD_WORDS = {"and", "or", "as well as", "along with"}

def extract_sentences_from_file(path):
    """Extract text sentences from .txt or .xml file"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if path.endswith(".xml"):
            # Remove XML tags
            content = re.sub(r"<[^>]+>", " ", content)
        # Normalize
        content = re.sub(r"\s+", " ", content)
        # Split into sentences
        doc = nlp(content)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        print("Error reading", path, ":", e)
        return []

def detect_issues(sentence):
    """Detect comparative and non-atomic patterns"""
    doc = nlp(sentence)
    comparative = []
    nonatomic = []

    for token in doc:
        if token.tag_ in COMPARATIVE_TAGS or token.lemma_.lower() in COMPARATIVE_WORDS:
            comparative.append(token.text)
        if token.text.lower() in COORD_WORDS or token.dep_ == "cc":
            nonatomic.append(token.text)

    issues = []
    if comparative:
        issues.append(("Comparative/Superlative", comparative))
    if nonatomic:
        issues.append(("Non-atomic", nonatomic))
    return issues

def analyze_dataset():
    records = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if not f.lower().endswith((".txt", ".xml")):
                continue
            fpath = os.path.join(root, f)
            sentences = extract_sentences_from_file(fpath)
            for idx, sent in enumerate(sentences):
                issues = detect_issues(sent)
                for issue_type, evidence in issues:
                    suggestion = ""
                    if issue_type == "Comparative/Superlative":
                        suggestion = "Replace comparative with measurable metric."
                    elif issue_type == "Non-atomic":
                        suggestion = "Split sentence into multiple independent requirements."
                    records.append({
                        "Document": f,
                        "Sentence_Index": idx,
                        "Sentence": sent,
                        "Issue_Type": issue_type,
                        "Evidence": ", ".join(evidence),
                        "Suggested_Fix": suggestion
                    })
    return pd.DataFrame(records)

def generate_summary(df):
    summary = []
    for issue_type, group in df.groupby("Issue_Type"):
        freq = len(group)
        example = group.iloc[0]["Sentence"]
        summary.append({
            "Issue_Type": issue_type,
            "Frequency": freq,
            "Example": example
        })
    return pd.DataFrame(summary)

def generate_report(df, summary_df):
    total_sentences = len(df)
    comp_count = len(df[df["Issue_Type"] == "Comparative/Superlative"])
    nonatomic_count = len(df[df["Issue_Type"] == "Non-atomic"])

    report = f"""
Assignment 3 – Comparative & Non-atomic Requirement Analysis
============================================================

Dataset: PURE (Ferrari et al., 2017)
Analyzed Folder: {DATASET_DIR}

Summary Statistics
------------------
Total Sentences Analyzed: {total_sentences}
Comparative/Superlative Issues: {comp_count}
Non-atomic Issues: {nonatomic_count}

Breakdown
---------
{summary_df.to_string(index=False)}

Observations
------------
- Comparative issues often include vague words such as "better", "faster", "more efficient".
- Non-atomic sentences frequently use 'and'/'or' joining multiple requirements.

Recommendations
---------------
1. Replace comparatives with measurable criteria.
2. Split non-atomic requirements into atomic, testable statements.
3. Encourage reviewers to validate rewritten requirements for clarity.

Conclusion
----------
This automated analysis provides insight into common structural problems in SRS
documents, helping improve requirement precision and measurability.
"""
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {REPORT_TXT}")

def main():
    print("Analyzing dataset...")
    df = analyze_dataset()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved detailed results to {OUTPUT_CSV}")

    summary_df = generate_summary(df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary table to {SUMMARY_CSV}")

    generate_report(df, summary_df)
    print("Done.")

if __name__ == "__main__":
    main()
