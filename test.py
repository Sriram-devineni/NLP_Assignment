"""
Assignment 3: Comparative & Non-atomic Requirement Analysis (Enhanced)
Uses POS tagging for stricter detection and Gemini API for intelligent suggestions
"""

import re
import os
import json
from pathlib import Path
from collections import defaultdict
import spacy
from typing import List, Dict, Tuple, Optional

# Load spaCy model for NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


class GeminiSuggestionEngine:
    """Optional Gemini API integration for intelligent suggestions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        self.enabled = False
        self.client = None
        self.client_style = None
        # A small list of likely models to try in order. Adjust if you know a model name.
        self.preferred_models = [
            'gemini-2.5-flash',
            'gemini-2.1',
            'gemini-1.0',
            'text-bison-001'
        ]
        
        if self.api_key:
            # Prefer the new client-style import if available: `from google import genai`
            try:
                # This matches the snippet you posted: `from google import genai; client = genai.Client()`
                from google import genai
                # Try creating a client. Some versions accept api_key in constructor, some read env vars.
                try:
                    self.client = genai.Client(api_key=self.api_key)
                except TypeError:
                    # fallback: construct without param and hope the library reads environment or configure helper
                    self.client = genai.Client()
                    # try configure if available on the module
                    try:
                        genai.configure(api_key=self.api_key)  # best-effort
                    except Exception:
                        pass

                self.client_style = 'genai_client'
                self.enabled = True
                print("✓ Gemini API (genai.Client) enabled for intelligent suggestions")
            except ImportError:
                # Fall back to the older google.generativeai package if present (best-effort)
                try:
                    import google.generativeai as genai_old
                    # Try to configure the older package
                    try:
                        genai_old.configure(api_key=self.api_key)
                    except Exception:
                        pass
                    # Keep a reference to the module for later calls
                    self.client = genai_old
                    self.client_style = 'generativeai_legacy'
                    self.enabled = True
                    print("✓ Gemini API (google.generativeai) enabled for intelligent suggestions")
                except ImportError:
                    print("⚠ google-generativeai not installed. Run: pip install google-generativeai")
                except Exception as e:
                    print(f"⚠ Gemini API setup failed (legacy): {e}")
        else:
            print("ℹ Gemini API not configured (optional). Using rule-based suggestions.")
    
    def get_suggestion(self, sentence: str, issue_type: str, keyword: str) -> str:
        """Get AI-powered improvement suggestion"""
        if not self.enabled:
            return None

        prompt = f"""You are a requirements engineering expert. Analyze this requirement and provide a specific, measurable improvement.

Original Requirement: "{sentence}"

Issue: Contains {issue_type} term "{keyword}" which is vague and unmeasurable.

Provide:
1. A specific rewritten requirement with measurable criteria
2. Brief explanation of the improvement (2-3 lines max)

Format your response as:
IMPROVED: [rewritten requirement]
EXPLANATION: [brief explanation]

Keep it concise and focus on making it testable and measurable."""

        last_exc = None

        # If we have the newer client-style API, try a small set of likely models
        if self.client_style == 'genai_client' and self.client is not None:
            for model_name in self.preferred_models:
                try:
                    # user-provided example: client.models.generate_content(model="gemini-2.5-flash", contents="...")
                    response = self.client.models.generate_content(model=model_name, contents=prompt)
                    # Try common response shapes
                    text = None
                    if hasattr(response, 'text') and response.text:
                        text = response.text
                    else:
                        # some responses expose output -> content -> text
                        try:
                            text = response.output[0].content[0].text
                        except Exception:
                            try:
                                # fallback to candidates
                                text = response.candidates[0].content[0].text
                            except Exception:
                                text = str(response)

                    if text:
                        return text.strip()
                except Exception as e:
                    last_exc = e
                    # try next model in the list
                    continue

            # If none of the models worked, give a helpful diagnostic
            print("  Warning: Gemini API generation failed for preferred models.")
            if last_exc is not None:
                print(f"    Last error: {last_exc}")
            print("    Tip: list available models with: client.models.list() or consult the SDK docs to find supported model names.")
            # disable to avoid repeated failing calls
            self.enabled = False
            return None

        # Legacy google.generativeai usage (best-effort)
        if self.client_style == 'generativeai_legacy' and self.client is not None:
            try:
                # Older API shapes varied; try a few common call patterns
                try:
                    # Some older libs provide a 'generate_text' helper
                    response = self.client.generate_text(prompt)
                    text = getattr(response, 'text', None) or str(response)
                    return text.strip()
                except Exception:
                    # Try pattern used earlier in file (GenerativeModel)
                    try:
                        model = self.client.GenerativeModel('text-bison-001')
                        response = model.generate_content(prompt)
                        return getattr(response, 'text', str(response)).strip()
                    except Exception as e:
                        print(f"  Warning (legacy genai) generation failed: {e}")
                        self.enabled = False
                        return None
            except Exception as e:
                print(f"  Warning: Gemini API call failed: {e}")
                self.enabled = False
                return None

        # If none of the above branches matched, return None
        return None


class ComparativeNonAtomicAnalyzer:
    def __init__(self, gemini_api_key: Optional[str] = None):
        # Comparative and superlative keywords with POS context
        self.comparatives = {
            'better', 'worse', 'faster', 'slower', 'larger', 'smaller',
            'higher', 'lower', 'greater', 'lesser', 'easier', 'harder',
            'stronger', 'weaker', 'cheaper', 'costlier', 'simpler',
            'more efficient', 'less efficient', 'more reliable', 'less reliable'
        }
        
        self.superlatives = {
            'best', 'worst', 'fastest', 'slowest', 'largest', 'smallest',
            'highest', 'lowest', 'greatest', 'least', 'easiest',
            'hardest', 'strongest', 'weakest', 'cheapest', 'optimal',
            'optimum', 'maximum', 'minimum', 'most efficient', 'most reliable'
        }
        
        # Coordinators that may indicate non-atomic requirements
        self.coordinators = ['and', 'or', 'as well as', 'along with', 'plus']
        
        self.results = {
            'comparatives': [],
            'superlatives': [],
            'non_atomic': []
        }
        
        # Initialize Gemini for suggestions
        self.gemini = GeminiSuggestionEngine(gemini_api_key)
        
    def is_quantified_comparative(self, sent) -> bool:
        """
        Check if comparative is quantified (e.g., "more than 5", "less than 10")
        These are acceptable as they have specific thresholds
        """
        # Patterns that indicate quantified comparatives (acceptable)
        quantified_patterns = [
            r'\b(more|less|greater|fewer|higher|lower)\s+than\s+\d+',
            r'\b(more|less|greater|fewer|higher|lower)\s+than\s+\w+\s+(seconds?|minutes?|hours?|days?|MB|GB|KB)',
            r'\bat\s+(least|most)\s+\d+',
            r'\bno\s+(more|less)\s+than\s+\d+',
            r'\bup\s+to\s+\d+',
            r'\bexceed\w*\s+\d+',
            r'\bwithin\s+\d+',
        ]
        
        sent_text = sent.text.lower()
        for pattern in quantified_patterns:
            if re.search(pattern, sent_text):
                return True
        return False
    
    def is_valid_comparative_context(self, token, sent) -> bool:
        """
        Use POS tagging to verify if comparative is used in vague context
        Returns True if it's a problematic comparative (vague usage)
        """
        # If comparative is quantified, it's acceptable
        if self.is_quantified_comparative(sent):
            return False
        
        # Check if it's actually an adjective/adverb comparative (JJR, RBR)
        if token.tag_ not in ['JJR', 'RBR', 'JJS', 'RBS']:
            # Check if it's part of "more/most + adjective" construction
            if token.text.lower() in ['more', 'most', 'less', 'least']:
                # Check next token
                next_token = token.nbor(1) if token.i < len(sent) - 1 else None
                if next_token and next_token.pos_ in ['ADJ', 'ADV']:
                    # This is "more efficient" type - check if quantified
                    if not self.is_quantified_comparative(sent):
                        return True
            return False
        
        # Check for specific baseline comparisons (acceptable)
        # e.g., "faster than version 1.0", "better than competitor X"
        baseline_indicators = ['than the', 'than version', 'than previous', 
                              'than current', 'than existing', 'compared to']
        sent_text = sent.text.lower()
        
        # If there's a specific baseline but no quantification, still vague
        has_baseline = any(indicator in sent_text for indicator in baseline_indicators)
        if has_baseline:
            # Check if it also has quantification
            if self.is_quantified_comparative(sent):
                return False  # Has both baseline and quantification - good!
            # Has baseline but no quantification - still vague
            return True
        
        # No baseline, no quantification - definitely vague
        return True
    
    def is_valid_superlative_context(self, token, sent) -> bool:
        """
        Check if superlative is used in vague context
        Returns True if it's problematic
        """
        # Superlatives are almost always vague unless they refer to specific metrics
        sent_text = sent.text.lower()
        
        # Check if it's in a technical/specific context that makes it acceptable
        acceptable_contexts = [
            r'maximum\s+(of\s+)?\d+',  # "maximum of 100"
            r'minimum\s+(of\s+)?\d+',  # "minimum of 5"
            r'at\s+(most|least)\s+\d+',
        ]
        
        for pattern in acceptable_contexts:
            if re.search(pattern, sent_text):
                return False
        
        # Check POS tag for superlative adjectives/adverbs (JJS, RBS)
        if token.tag_ in ['JJS', 'RBS']:
            return True
        
        # Check for "most/least + adjective" construction
        if token.text.lower() in ['most', 'least']:
            next_token = token.nbor(1) if token.i < len(sent) - 1 else None
            if next_token and next_token.pos_ in ['ADJ', 'ADV']:
                return True
        
        # Words like "optimal", "best", "worst" are always vague
        if token.text.lower() in ['optimal', 'optimum', 'best', 'worst']:
            return True
        
        return False
    
    def detect_comparatives_superlatives(self, text: str, doc_name: str) -> None:
        """Detect comparative and superlative terms with POS validation"""
        doc = nlp(text)
        
        for sent_idx, sent in enumerate(doc.sents, 1):
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            # Check each token for comparatives/superlatives
            for token in sent:
                token_lower = token.text.lower()
                
                # Check for multi-word expressions first
                bigram = f"{token.text.lower()} {token.nbor(1).text.lower()}" if token.i < len(sent) - 1 else ""
                
                # Check comparatives (with POS validation)
                if token_lower in self.comparatives or bigram in self.comparatives:
                    keyword = bigram if bigram in self.comparatives else token_lower
                    if self.is_valid_comparative_context(token, sent):
                        self.results['comparatives'].append({
                            'document': doc_name,
                            'sentence_num': sent_idx,
                            'sentence': sent_text,
                            'keyword': keyword,
                            'type': 'comparative',
                            'pos_tag': token.tag_
                        })
                        break  # Only record once per sentence
                
                # Check superlatives (with POS validation)
                if token_lower in self.superlatives or bigram in self.superlatives:
                    keyword = bigram if bigram in self.superlatives else token_lower
                    if self.is_valid_superlative_context(token, sent):
                        self.results['superlatives'].append({
                            'document': doc_name,
                            'sentence_num': sent_idx,
                            'sentence': sent_text,
                            'keyword': keyword,
                            'type': 'superlative',
                            'pos_tag': token.tag_
                        })
                        break  # Only record once per sentence
    
    def detect_non_atomic(self, text: str, doc_name: str) -> None:
        """Detect non-atomic requirements with coordinators"""
        doc = nlp(text)
        
        for sent_idx, sent in enumerate(doc.sents, 1):
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            # Check if sentence contains modal verbs indicating requirements
            modal_verbs = ['shall', 'must', 'should', 'will', 'required', 'needs to']
            has_modal = any(modal in sent_lower for modal in modal_verbs)
            
            if has_modal:
                # Check for coordinators
                for coord in self.coordinators:
                    pattern = r'\b' + re.escape(coord) + r'\b'
                    matches = list(re.finditer(pattern, sent_lower))
                    
                    if matches:
                        # Count action verbs (not auxiliary/modal verbs)
                        action_verbs = [token for token in sent 
                                      if token.pos_ == 'VERB' 
                                      and token.dep_ not in ['aux', 'auxpass']
                                      and token.text.lower() not in modal_verbs]
                        
                        # Flag if multiple action verbs with coordinator
                        if len(action_verbs) >= 2 or len(matches) >= 2:
                            self.results['non_atomic'].append({
                                'document': doc_name,
                                'sentence_num': sent_idx,
                                'sentence': sent_text,
                                'coordinator': coord,
                                'coordinator_count': len(matches),
                                'verb_count': len(action_verbs)
                            })
                            break  # Only record once per sentence
    
    def get_gemini_suggestion(self, item: Dict, item_type: str) -> Optional[str]:
        """Get AI-powered suggestion from Gemini"""
        if not self.gemini.enabled:
            return None
        
        return self.gemini.get_suggestion(
            item['sentence'],
            item_type,
            item.get('keyword') or item.get('coordinator', '')
        )
    
    def suggest_improvements(self, item: Dict, item_type: str) -> List[str]:
        """Generate improvement suggestions (with optional Gemini enhancement)"""
        suggestions = []
        
        # Try Gemini first if available
        gemini_suggestion = self.get_gemini_suggestion(item, item_type)
        
        if gemini_suggestion:
            suggestions.append("=== AI-POWERED SUGGESTION (Gemini) ===")
            suggestions.append(gemini_suggestion)
            suggestions.append("\n=== RULE-BASED SUGGESTIONS ===")
        
        # Always include rule-based suggestions as backup/comparison
        if item_type in ['comparative', 'superlative']:
            sent = item['sentence']
            keyword = item['keyword']
            
            suggestions.append(f"Original: {sent}")
            suggestions.append(f"\nIssue: Contains {item_type} term '{keyword}' (POS: {item.get('pos_tag', 'N/A')}) which is vague and unmeasurable.")
            suggestions.append("\nSuggestions:")
            suggestions.append("1. Replace with specific, measurable criteria:")
            
            if keyword in ['faster', 'fast', 'fastest', 'slower', 'slowest']:
                suggestions.append("   - 'shall respond within 2 seconds'")
                suggestions.append("   - 'shall process requests in less than 100ms'")
                suggestions.append("   - 'shall achieve 95th percentile latency below 500ms'")
            elif keyword in ['better', 'best', 'optimal', 'optimum', 'worse', 'worst']:
                suggestions.append("   - 'shall achieve 99.9% uptime'")
                suggestions.append("   - 'shall meet performance benchmark of 1000 TPS'")
                suggestions.append("   - 'shall score above 8.0 on usability testing'")
            elif keyword in ['larger', 'smaller', 'largest', 'smallest', 'bigger', 'biggest']:
                suggestions.append("   - 'shall support files up to 10MB'")
                suggestions.append("   - 'shall occupy no more than 500MB of memory'")
                suggestions.append("   - 'shall scale to 10,000 concurrent users'")
            elif keyword in ['more', 'most', 'less', 'least']:
                suggestions.append("   - 'shall provide at least 5 configuration options'")
                suggestions.append("   - 'shall reduce error rate to below 0.1%'")
                suggestions.append("   - 'shall support a minimum of 3 authentication methods'")
            elif keyword in ['higher', 'lower', 'highest', 'lowest']:
                suggestions.append("   - 'shall maintain CPU usage below 70%'")
                suggestions.append("   - 'shall achieve throughput of at least 10,000 requests/second'")
            elif keyword in ['more efficient', 'most efficient', 'efficient']:
                suggestions.append("   - 'shall reduce processing time to under 500ms'")
                suggestions.append("   - 'shall decrease memory usage by 30% compared to baseline'")
            elif keyword in ['more reliable', 'most reliable', 'reliable']:
                suggestions.append("   - 'shall achieve 99.99% availability'")
                suggestions.append("   - 'shall have MTBF of at least 10,000 hours'")
            elif keyword in ['maximum', 'minimum']:
                suggestions.append("   - Specify the exact numeric threshold")
                suggestions.append("   - 'shall support maximum of 1000 concurrent connections'")
            else:
                suggestions.append("   - Define specific metrics and thresholds")
                suggestions.append("   - Use quantifiable measures instead of relative terms")
                suggestions.append("   - Reference specific standards or benchmarks")
        
        elif item_type == 'non_atomic':
            sent = item['sentence']
            coord = item['coordinator']
            
            suggestions.append(f"Original: {sent}")
            suggestions.append(f"\nIssue: Non-atomic requirement using coordinator '{coord}' ({item['coordinator_count']} times).")
            suggestions.append(f"Contains {item['verb_count']} action verbs - multiple requirements combined.")
            suggestions.append("\nSuggestions:")
            suggestions.append("1. Split into separate, atomic requirements:")
            
            # Attempt to split the requirement intelligently
            parts = re.split(r'\b' + re.escape(coord) + r'\b', sent)
            
            if len(parts) >= 2:
                # Extract the modal verb to reuse
                modal_match = re.search(r'\b(shall|must|should|will)\b', sent.lower())
                modal = modal_match.group(0) if modal_match else 'shall'
                
                for i, part in enumerate(parts, 1):
                    part = part.strip()
                    if i == 1:
                        suggestions.append(f"   REQ-X.{i}: {part}")
                    else:
                        # Try to make it a complete sentence
                        if not any(m in part.lower() for m in ['shall', 'must', 'should', 'will']):
                            suggestions.append(f"   REQ-X.{i}: The system {modal} {part}")
                        else:
                            suggestions.append(f"   REQ-X.{i}: {part}")
            else:
                suggestions.append("   - Identify each distinct functional requirement")
                suggestions.append("   - Create separate requirement statements for each capability")
                suggestions.append("   - Assign unique, traceable requirement IDs")
                suggestions.append("   - Each requirement should be independently testable")
        
        return suggestions
    
    def analyze_directory(self, directory_path: str) -> None:
        """Analyze all text files in directory"""
        path = Path(directory_path)
        
        if not path.exists():
            print(f"Error: Directory {directory_path} not found")
            return
        
        # Find all text and requirements files
        file_patterns = ['*.txt', '*.srs', '*.req', '*.md']
        files = []
        for pattern in file_patterns:
            files.extend(path.rglob(pattern))
        
        if not files:
            print(f"No requirement files found in {directory_path}")
            return
        
        print(f"Found {len(files)} files to analyze\n")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    doc_name = file_path.name
                    
                    print(f"Analyzing: {doc_name}")
                    self.detect_comparatives_superlatives(content, doc_name)
                    self.detect_non_atomic(content, doc_name)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    def generate_report(self, output_file: str = 'assignment3_report.txt') -> None:
        """Generate detailed analysis report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ASSIGNMENT 3: COMPARATIVE & NON-ATOMIC REQUIREMENT ANALYSIS\n")
            f.write("Enhanced with POS Tagging & AI Suggestions\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Comparative Terms Found: {len(self.results['comparatives'])}\n")
            f.write(f"Total Superlative Terms Found: {len(self.results['superlatives'])}\n")
            f.write(f"Total Non-Atomic Requirements Found: {len(self.results['non_atomic'])}\n")
            f.write(f"AI Suggestions: {'Enabled (Gemini)' if self.gemini.enabled else 'Disabled (rule-based only)'}\n\n")
            
            # Comparative findings
            if self.results['comparatives']:
                f.write("\n" + "=" * 80 + "\n")
                f.write("COMPARATIVE TERMS DETECTED\n")
                f.write("=" * 80 + "\n\n")
                
                for idx, item in enumerate(self.results['comparatives'], 1):
                    f.write(f"\n{idx}. Document: {item['document']}\n")
                    f.write(f"   Sentence #{item['sentence_num']}\n")
                    f.write(f"   Keyword: '{item['keyword']}' (POS Tag: {item.get('pos_tag', 'N/A')})\n")
                    f.write(f"   Sentence: {item['sentence']}\n")
                    f.write("\n   " + "-" * 76 + "\n")
                    suggestions = self.suggest_improvements(item, 'comparative')
                    f.write("   " + "\n   ".join(suggestions) + "\n")
                    f.write("   " + "-" * 76 + "\n")
            
            # Superlative findings
            if self.results['superlatives']:
                f.write("\n" + "=" * 80 + "\n")
                f.write("SUPERLATIVE TERMS DETECTED\n")
                f.write("=" * 80 + "\n\n")
                
                for idx, item in enumerate(self.results['superlatives'], 1):
                    f.write(f"\n{idx}. Document: {item['document']}\n")
                    f.write(f"   Sentence #{item['sentence_num']}\n")
                    f.write(f"   Keyword: '{item['keyword']}' (POS Tag: {item.get('pos_tag', 'N/A')})\n")
                    f.write(f"   Sentence: {item['sentence']}\n")
                    f.write("\n   " + "-" * 76 + "\n")
                    suggestions = self.suggest_improvements(item, 'superlative')
                    f.write("   " + "\n   ".join(suggestions) + "\n")
                    f.write("   " + "-" * 76 + "\n")
            
            # Non-atomic findings
            if self.results['non_atomic']:
                f.write("\n" + "=" * 80 + "\n")
                f.write("NON-ATOMIC REQUIREMENTS DETECTED\n")
                f.write("=" * 80 + "\n\n")
                
                for idx, item in enumerate(self.results['non_atomic'], 1):
                    f.write(f"\n{idx}. Document: {item['document']}\n")
                    f.write(f"   Sentence #{item['sentence_num']}\n")
                    f.write(f"   Coordinator: '{item['coordinator']}' (appears {item['coordinator_count']} times)\n")
                    f.write(f"   Action Verbs: {item['verb_count']}\n")
                    f.write(f"   Sentence: {item['sentence']}\n")
                    f.write("\n   " + "-" * 76 + "\n")
                    suggestions = self.suggest_improvements(item, 'non_atomic')
                    f.write("   " + "\n   ".join(suggestions) + "\n")
                    f.write("   " + "-" * 76 + "\n")
            
            # Frequency analysis
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("FREQUENCY ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            # Count by keyword
            comp_freq = defaultdict(int)
            for item in self.results['comparatives']:
                comp_freq[item['keyword']] += 1
            
            sup_freq = defaultdict(int)
            for item in self.results['superlatives']:
                sup_freq[item['keyword']] += 1
            
            coord_freq = defaultdict(int)
            for item in self.results['non_atomic']:
                coord_freq[item['coordinator']] += 1
            
            if comp_freq:
                f.write("Comparative Terms Frequency:\n")
                for keyword, count in sorted(comp_freq.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {keyword}: {count}\n")
                f.write("\n")
            
            if sup_freq:
                f.write("Superlative Terms Frequency:\n")
                for keyword, count in sorted(sup_freq.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {keyword}: {count}\n")
                f.write("\n")
            
            if coord_freq:
                f.write("Coordinator Frequency in Non-Atomic Requirements:\n")
                for coord, count in sorted(coord_freq.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {coord}: {count}\n")
        
        print(f"\n✓ Report generated: {output_file}")
    
    def generate_json_output(self, output_file: str = 'assignment3_results.json') -> None:
        """Generate JSON output for programmatic access"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON results saved: {output_file}")
    
    def generate_summary_table(self) -> None:
        """Generate summary table for the report"""
        print("\n" + "=" * 100)
        print("SUMMARY TABLE")
        print("=" * 100)
        print(f"{'Bad Smell Type':<30} {'Frequency':<15} {'Description':<55}")
        print("-" * 100)
        print(f"{'Comparative Terms':<30} {len(self.results['comparatives']):<15} {'Vague relative comparisons (validated with POS)':<55}")
        print(f"{'Superlative Terms':<30} {len(self.results['superlatives']):<15} {'Unmeasurable extremes (validated with POS)':<55}")
        print(f"{'Non-Atomic Requirements':<30} {len(self.results['non_atomic']):<15} {'Multiple requirements with coordinators (verb-counted)':<55}")
        print("=" * 100)


def main():
    """Main execution function"""
    print("=" * 80)
    print("Assignment 3: Comparative & Non-atomic Requirement Analysis")
    print("Enhanced with POS Tagging & Optional AI Suggestions")
    print("=" * 80)
    
    # Check for Gemini API key
    gemini_key = input("\nEnter Gemini API key (or press Enter to skip AI suggestions): ").strip()
    if not gemini_key:
        gemini_key = os.environ.get('GEMINI_API_KEY')
    
    if gemini_key:
        print("\n📝 Note: Gemini API will be used for intelligent suggestions")
        print("   This may take longer but provides better quality suggestions")
    else:
        print("\n📝 Using rule-based suggestions only (faster)")
    
    # Initialize analyzer
    analyzer = ComparativeNonAtomicAnalyzer(gemini_api_key=gemini_key)
    
    # Get dataset path from user
    dataset_path = input("\nEnter the path to PURE dataset directory (or press Enter for './PURE'): ").strip()
    if not dataset_path:
        dataset_path = './PURE'
    
    # Analyze documents
    print("\n" + "=" * 80)
    print("Starting analysis with POS tagging...")
    print("=" * 80 + "\n")
    analyzer.analyze_directory(dataset_path)
    
    # Generate outputs
    print("\n" + "=" * 80)
    print("Generating reports...")
    print("=" * 80 + "\n")
    analyzer.generate_summary_table()
    analyzer.generate_report()
    analyzer.generate_json_output()
    
    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)
    print("\nOutput files:")
    print("  - assignment3_report.txt (Detailed report with POS tags)")
    print("  - assignment3_results.json (JSON data)")
    if analyzer.gemini.enabled:
        print("\n✓ Reports include AI-powered suggestions from Gemini")
    print("=" * 80)


if __name__ == "__main__":
    main()