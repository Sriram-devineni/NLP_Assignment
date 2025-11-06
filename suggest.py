#!/usr/bin/env python3
"""
assignment3_merged.py

Merged script for:
- POS-based comparative & superlative detection
- Handling quantified comparatives like "more than 1 ms" or ">= 1ms"
- Non-atomic requirement detection (coordinators + multiple action verbs)
- Optional Gemini suggestions (if API configured)
- Console summary table + detailed report + JSON output

Usage: python assignment3_merged.py
"""

import re
import os
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional
import sys

# ----------------------------
# spaCy setup
# ----------------------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # Attempt to download model and load again
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print("Error: could not load or download spaCy model 'en_core_web_sm'.")
        print("Please install spaCy and the model manually: pip install spacy && python -m spacy download en_core_web_sm")
        raise e

# ----------------------------
# Gemini suggestion engine (optional)
# ----------------------------
class GeminiSuggestionEngine:
    """Optional Gemini integration for intelligent suggestions (best-effort)."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.enabled = False
        self.client = None
        self.client_style = None
        if not self.api_key:
            # Not configured
            return

        # Try 'google.genai' style client first, then legacy.
        try:
            from google import genai
            try:
                self.client = genai.Client(api_key=self.api_key)
            except TypeError:
                self.client = genai.Client()
                try:
                    genai.configure(api_key=self.api_key)
                except Exception:
                    pass
            self.client_style = 'genai_client'
            self.enabled = True
            return
        except Exception:
            pass

        try:
            import google.generativeai as genai_old
            try:
                genai_old.configure(api_key=self.api_key)
            except Exception:
                pass
            self.client = genai_old
            self.client_style = 'generativeai_legacy'
            self.enabled = True
            return
        except Exception:
            pass

        # If reached here, Gemini not available
        self.enabled = False

    def get_suggestion(self, sentence: str, issue_type: str, keyword: str) -> Optional[str]:
        if not self.enabled or not self.client:
            return None

        prompt = f"""You are a requirements engineering expert. Improve the following requirement.

Requirement: "{sentence}"
Issue: Contains {issue_type} term "{keyword}" which may be vague or compound.

Provide:
1. IMPROVED: [rewritten requirement - specific & measurable]
2. EXPLANATION: [2-line explanation of why it's better]
Keep it concise.
"""
        try:
            if self.client_style == 'genai_client':
                response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                # Try to extract text
                if hasattr(response, "text") and response.text:
                    return response.text.strip()
                # fallback shapes
                try:
                    return response.output[0].content[0].text.strip()
                except Exception:
                    return str(response).strip()
            elif self.client_style == 'generativeai_legacy':
                try:
                    resp = self.client.generate_text(prompt)
                    return getattr(resp, "text", str(resp)).strip()
                except Exception:
                    try:
                        model = self.client.GenerativeModel("text-bison-001")
                        resp = model.generate_content(prompt)
                        return getattr(resp, "text", str(resp)).strip()
                    except Exception:
                        return None
        except Exception as e:
            # On any exception, disable to avoid repeated failures
            self.enabled = False
            return None

# ----------------------------
# Analyzer
# ----------------------------
class ComparativeNonAtomicAnalyzer:
    def __init__(self, gemini_api_key: Optional[str] = None):
        # keyword sets (expand as needed)
        self.comparatives = {
            'better', 'worse', 'faster', 'slower', 'larger', 'smaller',
            'higher', 'lower', 'greater', 'lesser', 'easier', 'harder',
            'stronger', 'weaker', 'cheaper', 'costlier', 'simpler',
            'more efficient', 'less efficient', 'more reliable', 'less reliable',
            'more', 'less', 'more secure', 'less secure'
        }
        self.superlatives = {
            'best', 'worst', 'fastest', 'slowest', 'largest', 'smallest',
            'highest', 'lowest', 'greatest', 'least', 'easiest',
            'hardest', 'strongest', 'weakest', 'cheapest', 'optimal',
            'optimum', 'maximum', 'minimum', 'most efficient', 'most reliable',
            'most', 'least', 'optimal'
        }

        self.coordinators = ['and', 'or', 'as well as', 'along with', 'plus']
        self.results = {'comparatives': [], 'superlatives': [], 'non_atomic': []}
        self.gemini = GeminiSuggestionEngine(gemini_api_key)

        # regex patterns for quantified comparatives (including units like ms, s, seconds, MB, %)
        number = r'(?:\d+(?:\.\d+)?)'
        unit = r'(?:ms|s|sec|secs|seconds|minutes|mins|hours|hr|hrs|mb|gb|kb|%|percent|bps|tps|requests/s|rps)'
        # allow patterns like 'more than 1 ms', 'more than 1ms', '> 1ms', '>= 1 ms', 'within 500ms'
        self.quantified_patterns = [
            re.compile(rf'\b(?:more|less|greater|fewer|higher|lower)\s+than\s+{number}\s*{unit}\b', re.I),
            re.compile(rf'\b(?:more|less|greater|fewer|higher|lower)\s+than\s+{number}\b', re.I),
            re.compile(rf'\b(?:>|<|>=|<=)\s*{number}\s*{unit}\b'),
            re.compile(rf'\b(?:>|<|>=|<=)\s*{number}\b'),
            re.compile(rf'\bwithin\s+{number}\s*{unit}\b', re.I),
            re.compile(rf'\bat\s+(?:least|most)\s+{number}\s*{unit}\b', re.I),
            re.compile(rf'\bup to\s+{number}\s*{unit}\b', re.I),
            re.compile(rf'\bno\s+(?:more|less)\s+than\s+{number}\s*{unit}\b', re.I),
            re.compile(rf'\bexceed\w*\s+{number}\s*{unit}\b', re.I),
            re.compile(rf'\b{number}\s*{unit}\b', re.I),  # simple numeric unit like '1ms' or '500 ms'
        ]

    def is_quantified_comparative(self, sent_text: str) -> bool:
        """Return True if sentence contains quantified comparative expressions."""
        for p in self.quantified_patterns:
            if p.search(sent_text):
                return True
        return False

    def is_valid_comparative_context(self, token, sent) -> bool:
        """
        Return True if comparative usage is problematic (i.e., vague) and should be flagged.
        If quantified forms are present, treat as acceptable -> return False.
        """
        sent_text = sent.text.lower()
        if self.is_quantified_comparative(sent_text):
            return False  # acceptable because numeric threshold/unit present

        if token.tag_ not in ['JJR', 'RBR', 'JJS', 'RBS']:
            if token.text.lower() in ['more', 'most', 'less', 'least']:
                try:
                    next_tok = token.nbor(1)
                    if next_tok.pos_ in ['ADJ', 'ADV']:
                        # no quantification -> vague
                        return True
                except Exception:
                    return False
            return False

        # token is comparative/superlative POS (JJR/JJS etc.)
        # if sentence contains baseline indicators like 'than version' but no numbers -> vague
        baseline_indicators = ['than the', 'than version', 'than previous', 'than current', 'than existing', 'compared to', 'than']
        has_baseline = any(ind in sent_text for ind in baseline_indicators)
        if has_baseline and not self.is_quantified_comparative(sent_text):
            return True

        # otherwise, no baseline and no quantification -> vague
        return True

    def is_valid_superlative_context(self, token, sent) -> bool:
        sent_text = sent.text.lower()
        # if numeric thresholds present -> acceptable
        if self.is_quantified_comparative(sent_text):
            return False

        # very often superlatives are vague; check for numeric qualifiers
        acceptable_patterns = [
            r'maximum\s+(of\s+)?\d+',
            r'minimum\s+(of\s+)?\d+',
            r'at\s+(most|least)\s+\d+'
        ]
        for pat in acceptable_patterns:
            if re.search(pat, sent_text):
                return False

        # POS checks (JJS / RBS)
        if token.tag_ in ['JJS', 'RBS']:
            return True

        # 'most'/'least' + adj/adverb
        if token.text.lower() in ['most', 'least']:
            try:
                nxt = token.nbor(1)
                if nxt.pos_ in ['ADJ', 'ADV']:
                    return True
            except Exception:
                pass

        # words like 'optimal', 'best' are considered vague
        if token.text.lower() in ['optimal', 'optimum', 'best', 'worst']:
            return True

        return False

    def detect_comparatives_superlatives(self, text: str, doc_name: str) -> None:
        doc = nlp(text)
        for sent_idx, sent in enumerate(doc.sents, 1):
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            for token in sent:
                token_lower = token.text.lower()
                # check bigram (multiword) like 'more efficient'
                bigram = None
                try:
                    if token.i < len(sent) - 1:
                        bigram = f"{token.text.lower()} {token.nbor(1).text.lower()}"
                except Exception:
                    bigram = None

                # comparatives
                if token_lower in self.comparatives or (bigram and bigram in self.comparatives):
                    keyword = bigram if (bigram and bigram in self.comparatives) else token_lower
                    if self.is_valid_comparative_context(token, sent):
                        self.results['comparatives'].append({
                            'document': doc_name,
                            'sentence_num': sent_idx,
                            'sentence': sent_text,
                            'keyword': keyword,
                            'type': 'comparative',
                            'pos_tag': token.tag_
                        })
                        break

                # superlatives
                if token_lower in self.superlatives or (bigram and bigram in self.superlatives):
                    keyword = bigram if (bigram and bigram in self.superlatives) else token_lower
                    if self.is_valid_superlative_context(token, sent):
                        self.results['superlatives'].append({
                            'document': doc_name,
                            'sentence_num': sent_idx,
                            'sentence': sent_text,
                            'keyword': keyword,
                            'type': 'superlative',
                            'pos_tag': token.tag_
                        })
                        break

    def detect_non_atomic(self, text: str, doc_name: str) -> None:
        """Detect non-atomic requirements using coordinators + verb counting"""
        doc = nlp(text)
        for sent_idx, sent in enumerate(doc.sents, 1):
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            modal_verbs = ['shall', 'must', 'should', 'will', 'required', 'needs to', 'shall not']
            has_modal = any(m in sent_lower for m in modal_verbs)
            if not has_modal:
                continue

            for coord in self.coordinators:
                pattern = r'\b' + re.escape(coord) + r'\b'
                matches = list(re.finditer(pattern, sent_lower))
                if matches:
                    # count action verbs (exclude auxiliaries)
                    action_verbs = [t for t in sent if t.pos_ == 'VERB' and t.dep_ not in ('aux', 'auxpass')]
                    # Also include gerunds as verbs (VBG)
                    # If there are 2+ action verbs or multiple coordinators -> non-atomic
                    if len(action_verbs) >= 2 or len(matches) >= 2:
                        self.results['non_atomic'].append({
                            'document': doc_name,
                            'sentence_num': sent_idx,
                            'sentence': sent_text,
                            'coordinator': coord,
                            'coordinator_count': len(matches),
                            'verb_count': len(action_verbs)
                        })
                        break

    def get_gemini_suggestion(self, item: Dict, item_type: str) -> Optional[str]:
        if not self.gemini.enabled:
            return None
        return self.gemini.get_suggestion(item['sentence'], item_type, item.get('keyword', item.get('coordinator', '')))

    def suggest_improvements(self, item: Dict, item_type: str) -> List[str]:
        suggestions = []
        gem = self.get_gemini_suggestion(item, item_type)
        if gem:
            suggestions.append("=== AI-POWERED SUGGESTION (Gemini) ===")
            suggestions.append(gem)
            suggestions.append("\n=== RULE-BASED SUGGESTIONS ===")

        if item_type in ['comparative', 'superlative']:
            sent = item['sentence']
            keyword = item.get('keyword', '')
            suggestions.append(f"Original: {sent}")
            suggestions.append(f"Issue: Contains {item_type} term '{keyword}' (POS: {item.get('pos_tag', 'N/A')}) which may be vague.")
            suggestions.append("Suggestions:")
            suggestions.append(" - Replace with specific numeric thresholds, e.g., 'shall respond within 2 seconds' or 'shall process 1000 TPS'.")
            suggestions.append(" - If comparing to baseline, state the baseline and metric (e.g., '30% faster than v1.2 in 95th percentile latency').")
            suggestions.append(" - Use units (ms, s, %), sample sizes, percentiles, or standards to make testable.")
        else:  # non_atomic
            sent = item['sentence']
            coord = item.get('coordinator', '')
            suggestions.append(f"Original: {sent}")
            suggestions.append(f"Issue: Non-atomic requirement using coordinator '{coord}' ({item.get('coordinator_count', 0)} times).")
            suggestions.append(f"Contains {item.get('verb_count', 0)} action verbs.")
            suggestions.append("Suggestions:")
            # attempt naive split
            parts = re.split(r'\b' + re.escape(coord) + r'\b', sent)
            modal_match = re.search(r'\b(shall|must|should|will)\b', sent.lower())
            modal = modal_match.group(0) if modal_match else 'shall'
            if len(parts) >= 2:
                for i, p in enumerate(parts, 1):
                    p = p.strip(' ,.;')
                    if i == 1:
                        suggestions.append(f" - REQ-X.{i}: {p}")
                    else:
                        if not re.search(r'\b(shall|must|should|will)\b', p.lower()):
                            suggestions.append(f" - REQ-X.{i}: The system {modal} {p}")
                        else:
                            suggestions.append(f" - REQ-X.{i}: {p}")
            else:
                suggestions.append(" - Split into separate, atomic requirements. Each should be independently testable.")
        return suggestions

    # ----------------------------
    # Directory analysis & outputs
    # ----------------------------
    def analyze_directory(self, directory_path: str) -> None:
        path = Path(directory_path)
        if not path.exists():
            print(f"Error: Directory '{directory_path}' not found.")
            return

        file_patterns = ['*.txt', '*.srs', '*.req', '*.md']
        files = []
        for pat in file_patterns:
            files.extend(path.rglob(pat))

        if not files:
            print(f"No requirement files found in '{directory_path}'.")
            return

        print(f"Found {len(files)} files to analyze.")
        for fpath in files:
            try:
                content = fpath.read_text(encoding='utf-8', errors='ignore')
                doc_name = fpath.name
                self.detect_comparatives_superlatives(content, doc_name)
                self.detect_non_atomic(content, doc_name)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

    def generate_summary_table(self) -> None:
        print("\n" + "=" * 100)
        print("SUMMARY TABLE")
        print("=" * 100)
        print(f"{'Bad Smell Type':<30} {'Frequency':<12} {'Description':<52}")
        print("-" * 100)
        print(f"{'Comparative Terms':<30} {len(self.results['comparatives']):<12} {'Vague relative comparisons (POS-validated)'}")
        print(f"{'Superlative Terms':<30} {len(self.results['superlatives']):<12} {'Unmeasurable extremes (POS-validated)'}")
        print(f"{'Non-Atomic Requirements':<30} {len(self.results['non_atomic']):<12} {'Multiple requirements joined with coordinators/verbs'}")
        print("=" * 100)

    def generate_report(self, output_file: str = 'report.txt') -> None:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ASSIGNMENT 3: COMPARATIVE & NON-ATOMIC REQUIREMENT ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Comparative Terms Found: {len(self.results['comparatives'])}\n")
            f.write(f"Total Superlative Terms Found: {len(self.results['superlatives'])}\n")
            f.write(f"Total Non-Atomic Requirements Found: {len(self.results['non_atomic'])}\n")
            f.write(f"AI Suggestions: {'Enabled (Gemini)' if self.gemini.enabled else 'Disabled (rule-based only)'}\n\n")

            if self.results['comparatives']:
                f.write("\n" + "=" * 72 + "\nCOMPARATIVE TERMS\n" + "=" * 72 + "\n")
                for idx, item in enumerate(self.results['comparatives'], 1):
                    f.write(f"\n{idx}. Document: {item['document']}\n")
                    f.write(f"   Sentence #{item['sentence_num']}\n")
                    f.write(f"   Keyword: '{item['keyword']}' (POS Tag: {item.get('pos_tag','N/A')})\n")
                    f.write(f"   Sentence: {item['sentence']}\n")
                    f.write("   Suggestions:\n")
                    for s in self.suggest_improvements(item, 'comparative'):
                        f.write(f"     {s}\n")
                    f.write("-" * 72 + "\n")

            if self.results['superlatives']:
                f.write("\n" + "=" * 72 + "\nSUPERLATIVE TERMS\n" + "=" * 72 + "\n")
                for idx, item in enumerate(self.results['superlatives'], 1):
                    f.write(f"\n{idx}. Document: {item['document']}\n")
                    f.write(f"   Sentence #{item['sentence_num']}\n")
                    f.write(f"   Keyword: '{item['keyword']}' (POS Tag: {item.get('pos_tag','N/A')})\n")
                    f.write(f"   Sentence: {item['sentence']}\n")
                    f.write("   Suggestions:\n")
                    for s in self.suggest_improvements(item, 'superlative'):
                        f.write(f"     {s}\n")
                    f.write("-" * 72 + "\n")

            if self.results['non_atomic']:
                f.write("\n" + "=" * 72 + "\nNON-ATOMIC REQUIREMENTS\n" + "=" * 72 + "\n")
                for idx, item in enumerate(self.results['non_atomic'], 1):
                    f.write(f"\n{idx}. Document: {item['document']}\n")
                    f.write(f"   Sentence #{item['sentence_num']}\n")
                    f.write(f"   Coordinator: '{item['coordinator']}' (count: {item['coordinator_count']})\n")
                    f.write(f"   Action Verbs: {item['verb_count']}\n")
                    f.write(f"   Sentence: {item['sentence']}\n")
                    f.write("   Suggestions:\n")
                    for s in self.suggest_improvements(item, 'non_atomic'):
                        f.write(f"     {s}\n")
                    f.write("-" * 72 + "\n")

            # Frequency analysis
            f.write("\n" + "=" * 72 + "\nFREQUENCY ANALYSIS\n" + "=" * 72 + "\n")
            comp_freq = defaultdict(int)
            for it in self.results['comparatives']:
                comp_freq[it['keyword']] += 1
            sup_freq = defaultdict(int)
            for it in self.results['superlatives']:
                sup_freq[it['keyword']] += 1
            coord_freq = defaultdict(int)
            for it in self.results['non_atomic']:
                coord_freq[it['coordinator']] += 1

            if comp_freq:
                f.write("\nComparative Terms Frequency:\n")
                for k, v in sorted(comp_freq.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {k}: {v}\n")
            if sup_freq:
                f.write("\nSuperlative Terms Frequency:\n")
                for k, v in sorted(sup_freq.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {k}: {v}\n")
            if coord_freq:
                f.write("\nCoordinator Frequency:\n")
                for k, v in sorted(coord_freq.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {k}: {v}\n")

        print(f"\nReport written to: {output_file}")

    def generate_json_output(self, output_file: str = 'assignment3_results.json') -> None:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved to: {output_file}")

# ----------------------------
# Main CLI
# ----------------------------
def main():
    print("=" * 80)
    print("Assignment 3: Comparative & Non-Atomic Requirement Analysis (Merged)")
    print("=" * 80)
    # Ask the user whether to enable AI-powered suggestions
    ai_choice = input("Enable AI-powered suggestions? (y/N): ").strip().lower()
    ai_enabled = ai_choice in ('y', 'yes')

    
    if ai_enabled:

        # Initialize analyzer (pass None if AI disabled)
        api_key = "AIzaSyAF-OsmWpu9YyImRUh1k366opCq9U9d1_o"

        analyzer = ComparativeNonAtomicAnalyzer(api_key)
    else:
        analyzer = ComparativeNonAtomicAnalyzer()

    dataset_path = input("\nEnter the path to the dataset folder (default: ./PURE): ").strip() or "./PURE"

    analyzer.analyze_directory(dataset_path)
    analyzer.generate_summary_table()

    # Choose output filenames depending on whether AI suggestions were used
    if ai_enabled and analyzer.gemini.enabled:
        report_file = 'report_with_ai.txt'
        json_file = 'results_with_ai.json'
    else:
        report_file = 'report_rulebased.txt'
        json_file = 'results_rulebased.json'

    analyzer.generate_report(output_file=report_file)
    #analyzer.generate_json_output(output_file=json_file)

    print("\nDone. Files produced (if findings were present):")
    print(f" - {report_file}")
    #print(f" - {json_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
