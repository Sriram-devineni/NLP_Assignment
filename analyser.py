"""
Assignment 3: Comparative & Non-atomic Requirement Analysis
Detects superlatives/comparatives and non-atomic requirements in SRS documents
"""

import re
import os
import json
from pathlib import Path
from collections import defaultdict
import spacy
from typing import List, Dict, Tuple

# Load spaCy model for NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


class ComparativeNonAtomicAnalyzer:
    def __init__(self):
        # Comparative and superlative keywords
        self.comparatives = [
            'better', 'worse', 'faster', 'slower', 'larger', 'smaller',
            'higher', 'lower', 'greater', 'lesser', 'more', 'less',
            'easier', 'harder', 'stronger', 'weaker', 'cheaper', 'costlier'
        ]
        
        self.superlatives = [
            'best', 'worst', 'fastest', 'slowest', 'largest', 'smallest',
            'highest', 'lowest', 'greatest', 'least', 'most', 'easiest',
            'hardest', 'strongest', 'weakest', 'cheapest', 'optimal',
            'maximum', 'minimum', 'optimum'
        ]
        
        # Coordinators that may indicate non-atomic requirements
        self.coordinators = ['and', 'or', 'as well as', 'along with', 'plus']
        
        self.results = {
            'comparatives': [],
            'superlatives': [],
            'non_atomic': []
        }
        
    def detect_comparatives_superlatives(self, text: str, doc_name: str) -> None:
        """Detect comparative and superlative terms in text"""
        doc = nlp(text)
        
        for sent_idx, sent in enumerate(doc.sents, 1):
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            # Check for comparatives
            for comp in self.comparatives:
                if re.search(r'\b' + comp + r'\b', sent_lower):
                    self.results['comparatives'].append({
                        'document': doc_name,
                        'sentence_num': sent_idx,
                        'sentence': sent_text,
                        'keyword': comp,
                        'type': 'comparative'
                    })
            
            # Check for superlatives
            for sup in self.superlatives:
                if re.search(r'\b' + sup + r'\b', sent_lower):
                    self.results['superlatives'].append({
                        'document': doc_name,
                        'sentence_num': sent_idx,
                        'sentence': sent_text,
                        'keyword': sup,
                        'type': 'superlative'
                    })
    
    def detect_non_atomic(self, text: str, doc_name: str) -> None:
        """Detect non-atomic requirements with coordinators"""
        doc = nlp(text)
        
        for sent_idx, sent in enumerate(doc.sents, 1):
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            # Check if sentence contains modal verbs (shall, must, should, will)
            # which indicate requirements
            modal_verbs = ['shall', 'must', 'should', 'will', 'required', 'needs to']
            has_modal = any(modal in sent_lower for modal in modal_verbs)
            
            if has_modal:
                # Check for coordinators
                for coord in self.coordinators:
                    pattern = r'\b' + re.escape(coord) + r'\b'
                    if re.search(pattern, sent_lower):
                        # Count verbs to estimate number of actions
                        verbs = [token for token in sent if token.pos_ == 'VERB']
                        
                        if len(verbs) >= 2 or coord in ['or', 'and']:
                            self.results['non_atomic'].append({
                                'document': doc_name,
                                'sentence_num': sent_idx,
                                'sentence': sent_text,
                                'coordinator': coord,
                                'verb_count': len(verbs)
                            })
                            break
    
    def suggest_improvements(self, item: Dict, item_type: str) -> List[str]:
        """Suggest improvements for detected issues"""
        suggestions = []
        
        if item_type in ['comparative', 'superlative']:
            sent = item['sentence']
            keyword = item['keyword']
            
            suggestions.append(f"Original: {sent}")
            suggestions.append(f"\nIssue: Contains {item_type} term '{keyword}' which is vague and unmeasurable.")
            suggestions.append("\nSuggestions:")
            suggestions.append("1. Replace with specific, measurable criteria:")
            
            if keyword in ['faster', 'fast', 'fastest']:
                suggestions.append("   - 'shall respond within 2 seconds'")
                suggestions.append("   - 'shall process requests in less than 100ms'")
            elif keyword in ['better', 'best', 'optimal']:
                suggestions.append("   - 'shall achieve 99.9% uptime'")
                suggestions.append("   - 'shall meet performance benchmark of X operations/second'")
            elif keyword in ['larger', 'smaller', 'largest', 'smallest']:
                suggestions.append("   - 'shall support files up to 10MB'")
                suggestions.append("   - 'shall occupy no more than 500MB of memory'")
            elif keyword in ['more', 'most', 'less', 'least']:
                suggestions.append("   - Specify exact quantities or percentages")
                suggestions.append("   - 'shall provide at least 5 options'")
            else:
                suggestions.append("   - Define specific metrics and thresholds")
                suggestions.append("   - Use quantifiable measures instead of relative terms")
        
        elif item_type == 'non_atomic':
            sent = item['sentence']
            coord = item['coordinator']
            
            suggestions.append(f"Original: {sent}")
            suggestions.append(f"\nIssue: Non-atomic requirement using coordinator '{coord}'.")
            suggestions.append("Multiple requirements combined into one sentence.")
            suggestions.append("\nSuggestions:")
            suggestions.append("1. Split into separate, atomic requirements:")
            
            # Attempt to split the requirement
            parts = re.split(r'\b' + re.escape(coord) + r'\b', sent, maxsplit=1)
            if len(parts) == 2:
                # Extract the modal verb to reuse
                modal_match = re.search(r'\b(shall|must|should|will)\b', sent.lower())
                modal = modal_match.group(0) if modal_match else 'shall'
                
                suggestions.append(f"   REQ-X.1: {parts[0].strip()}")
                suggestions.append(f"   REQ-X.2: The system {modal} {parts[1].strip()}")
            else:
                suggestions.append("   - Identify each distinct requirement")
                suggestions.append("   - Create separate requirement statements for each")
                suggestions.append("   - Assign unique requirement IDs")
        
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
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Comparative Terms Found: {len(self.results['comparatives'])}\n")
            f.write(f"Total Superlative Terms Found: {len(self.results['superlatives'])}\n")
            f.write(f"Total Non-Atomic Requirements Found: {len(self.results['non_atomic'])}\n\n")
            
            # Comparative findings
            if self.results['comparatives']:
                f.write("\n" + "=" * 80 + "\n")
                f.write("COMPARATIVE TERMS DETECTED\n")
                f.write("=" * 80 + "\n\n")
                
                for idx, item in enumerate(self.results['comparatives'], 1):
                    f.write(f"\n{idx}. Document: {item['document']}\n")
                    f.write(f"   Sentence #{item['sentence_num']}\n")
                    f.write(f"   Keyword: '{item['keyword']}'\n")
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
                    f.write(f"   Keyword: '{item['keyword']}'\n")
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
                    f.write(f"   Coordinator: '{item['coordinator']}'\n")
                    f.write(f"   Verb Count: {item['verb_count']}\n")
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
        
        print(f"\nReport generated: {output_file}")
    
    def generate_json_output(self, output_file: str = 'assignment3_results.json') -> None:
        """Generate JSON output for programmatic access"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved: {output_file}")
    
    def generate_summary_table(self) -> None:
        """Generate summary table for the report"""
        print("\n" + "=" * 100)
        print("SUMMARY TABLE")
        print("=" * 100)
        print(f"{'Bad Smell Type':<30} {'Frequency':<15} {'Description':<55}")
        print("-" * 100)
        print(f"{'Comparative Terms':<30} {len(self.results['comparatives']):<15} {'Vague relative comparisons (better, faster, etc.)':<55}")
        print(f"{'Superlative Terms':<30} {len(self.results['superlatives']):<15} {'Unmeasurable extremes (best, fastest, optimal)':<55}")
        print(f"{'Non-Atomic Requirements':<30} {len(self.results['non_atomic']):<15} {'Multiple requirements combined with coordinators':<55}")
        print("=" * 100)


def main():
    """Main execution function"""
    print("Assignment 3: Comparative & Non-atomic Requirement Analysis")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ComparativeNonAtomicAnalyzer()
    
    # Get dataset path from user
    dataset_path = input("\nEnter the path to PURE dataset directory (or press Enter for './PURE'): ").strip()
    if not dataset_path:
        dataset_path = './PURE'
    
    # Analyze documents
    print("\nStarting analysis...\n")
    analyzer.analyze_directory(dataset_path)
    
    # Generate outputs
    print("\nGenerating reports...\n")
    analyzer.generate_summary_table()
    analyzer.generate_report()
    analyzer.generate_json_output()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("Output files:")
    print("  - assignment3_report.txt (Detailed report)")
    print("  - assignment3_results.json (JSON data)")
    print("=" * 80)


if __name__ == "__main__":
    main()