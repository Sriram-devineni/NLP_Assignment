"""
Assignment 3: Comparative & Non-atomic Requirement Analysis (Enhanced with LLM)
Uses POS tagging for stricter detection and Gemini model for intelligent suggestions
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import spacy
from typing import List, Dict, Optional

# ----------------------------
# Load spaCy NLP Model
# ----------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ----------------------------
# Gemini Suggestion Engine
# ----------------------------
class GeminiSuggestionEngine:
    """Gemini integration for intelligent requirement improvement suggestions"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.enabled = False
        self.client = None

        if not self.api_key:
            print("⚠ Gemini API key not found. Using rule-based suggestions only.")
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.enabled = True
            print("✅ Gemini model connected successfully (gemini-2.5-flash).")
        except Exception as e:
            print(f"⚠ Gemini connection failed: {e}")
            self.enabled = False

    def get_suggestion(self, sentence: str, issue_type: str, keyword: str) -> Optional[str]:
        """Generate improvement suggestion using Gemini"""
        if not self.enabled or not self.client:
            return None

        prompt = f"""
You are a software requirements engineering expert.
Analyze the following requirement and provide a concise, measurable, and testable rewrite.

Requirement: "{sentence}"

Issue: Contains {issue_type} term "{keyword}" which may be vague or compound.

Provide:
1. A rewritten, measurable version of the requirement.
2. A short 2-line explanation of how it improves clarity and testability.

Format:
IMPROVED: [rewritten requirement]
EXPLANATION: [brief explanation]
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip() if hasattr(response, "text") else str(response)
        except Exception as e:
            print(f"⚠ Gemini generation failed: {e}")
            return None


# ----------------------------
# Main Analyzer Class
# ----------------------------
class ComparativeNonAtomicAnalyzer:
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini = GeminiSuggestionEngine(gemini_api_key)

        # Comparative & superlative keywords
        self.comparatives = {'better', 'faster', 'slower', 'higher', 'lower', 'more efficient', 'less efficient'}
        self.superlatives = {'best', 'fastest', 'slowest', 'highest', 'lowest', 'optimal'}
        self.coordinators = ['and', 'or', 'as well as', 'along with', 'plus']

        self.results = {'comparatives': [], 'superlatives': [], 'non_atomic': []}

    # ----------------------------
    # Detection Functions
    # ----------------------------
    def detect_comparatives_superlatives(self, text: str, doc_name: str):
        doc = nlp(text)
        for idx, sent in enumerate(doc.sents, 1):
            sent_text = sent.text.strip()
            for token in sent:
                word = token.text.lower()
                bigram = f"{word} {token.nbor(1).text.lower()}" if token.i < len(sent) - 1 else ""
                if word in self.comparatives or bigram in self.comparatives:
                    self.results['comparatives'].append({
                        'document': doc_name, 'sentence_num': idx,
                        'sentence': sent_text, 'keyword': bigram if bigram in self.comparatives else word,
                        'type': 'comparative'
                    })
                elif word in self.superlatives or bigram in self.superlatives:
                    self.results['superlatives'].append({
                        'document': doc_name, 'sentence_num': idx,
                        'sentence': sent_text, 'keyword': bigram if bigram in self.superlatives else word,
                        'type': 'superlative'
                    })

    def detect_non_atomic(self, text: str, doc_name: str):
        doc = nlp(text)
        for idx, sent in enumerate(doc.sents, 1):
            sent_text = sent.text.strip().lower()
            if any(m in sent_text for m in ['shall', 'must', 'should', 'will']):
                for coord in self.coordinators:
                    if re.search(rf"\b{coord}\b", sent_text):
                        self.results['non_atomic'].append({
                            'document': doc_name,
                            'sentence_num': idx,
                            'sentence': sent.text.strip(),
                            'coordinator': coord
                        })
                        break

    # ----------------------------
    # Suggestion Generation
    # ----------------------------
    def suggest_improvements(self, item: Dict, item_type: str) -> List[str]:
        suggestions = []
        gemini_output = self.gemini.get_suggestion(item['sentence'], item_type, item.get('keyword', ''))
        if gemini_output:
            suggestions.append("=== Gemini Suggestion ===")
            suggestions.append(gemini_output)
            suggestions.append("==========================\n")

        # Add rule-based fallback
        sent = item['sentence']
        if item_type in ['comparative', 'superlative']:
            suggestions.append(f"Original: {sent}")
            suggestions.append("Suggestion: Replace vague terms with measurable criteria, e.g.,")
            suggestions.append(" - 'shall respond within 2 seconds'")
            suggestions.append(" - 'shall process 1000 transactions per second'")
        else:
            suggestions.append(f"Original: {sent}")
            suggestions.append("Suggestion: Split this into separate, atomic requirements.")
            suggestions.append("Each should express one distinct functionality.")
        return suggestions

    # ----------------------------
    # Directory & Reporting
    # ----------------------------
    def analyze_directory(self, directory_path: str):
        path = Path(directory_path)
        if not path.exists():
            print(f"❌ Directory {directory_path} not found.")
            return

        files = list(path.rglob("*.txt")) + list(path.rglob("*.srs")) + list(path.rglob("*.req"))
        print(f"Found {len(files)} files to analyze.")
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                self.detect_comparatives_superlatives(text, file_path.name)
                #self.detect_non_atomic(text, file_path.name)

    def generate_report(self, output_file='assignment3_report.txt'):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ASSIGNMENT 3: Comparative & Non-Atomic Requirement Analysis\n")
            f.write("=" * 80 + "\n")
            f.write(f"Gemini Suggestions: {'Enabled' if self.gemini.enabled else 'Disabled (rule-based only)'}\n\n")

            for category in ['comparatives', 'superlatives', 'non_atomic']:
                if self.results[category]:
                    f.write(f"\n{category.upper()} DETECTED\n{'-' * 80}\n")
                    for idx, item in enumerate(self.results[category], 1):
                        f.write(f"{idx}. Sentence: {item['sentence']}\n")
                        if 'keyword' in item:
                            f.write(f"   Keyword: {item['keyword']}\n")
                        suggestions = self.suggest_improvements(item, category[:-1])
                        for s in suggestions:
                            f.write(f"   {s}\n")
                        f.write("\n")

        print(f"\n✅ Report generated: {output_file}")


# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 80)
    print("Assignment 3: Comparative & Non-Atomic Requirement Analysis")
    print("=" * 80)

    # Load Gemini key (env or manual)
    api_key = "AIzaSyAF-OsmWpu9YyImRUh1k366opCq9U9d1_o"

    analyzer = ComparativeNonAtomicAnalyzer(api_key)

    dataset_path = input("\nEnter the path to the dataset folder (default: ./PURE): ").strip() or "./PURE"
    analyzer.analyze_directory(dataset_path)
    analyzer.generate_report()


if __name__ == "__main__":
    main()
