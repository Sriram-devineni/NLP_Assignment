# Comparative and Non-Atomic Requirement Analysis (Enhanced with API Integration)

## Overview
This project performs automated analysis of Software Requirements Specification (SRS) documents to identify and improve low-quality requirements. It detects comparative, superlative, and non-atomic statements that may reduce clarity, testability, and atomicity. The system uses linguistic analysis through Part-of-Speech (POS) tagging and integrates an external API for intelligent suggestion generation.

## Objectives
- Detect comparative and superlative terms that make requirements vague or subjective.  
- Identify non-atomic requirements containing multiple actions in a single statement.  
- Generate refined, measurable, and testable alternatives to improve requirement quality.  

## Requirements
- Python 3.8 or above  
- Required Python packages:
  ```bash
  pip install spacy google-genai python-dotenv
  python -m spacy download en_core_web_sm

## Gemini API Setup

To enable intelligent, context-aware suggestions, the system integrates with the Gemini API. Follow the steps below to configure it properly.

### Step 1: Get your Gemini API key
1. Visit [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Go to **Get API key** under the **API Keys** section.
4. Copy your unique API key.

### Step 2: Set your API key
In your terminal, run:
```bash
export api_key="your_api_key_here"
```

## How to Run

Follow the steps below to set up and run the project.

### step 1:Clone the repository
```bash
git clone https://github.com/Sriram-devineni/NLP_Assignment.git
cd NLP_ASSGN
```

### Step 2: Install Dependencies
Before running the program, ensure that all required Python libraries are installed.

```bash
pip install spacy google-genai python-dotenv
python -m spacy download en_core_web_sm
```

### step 3: Run the script
```bash
python3 suggest.py
```


