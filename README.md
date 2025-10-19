# LLM-Interactions

A collection of LLM interaction patterns and utilities.

## Voting Bots

This repo runs three LLMs in parallel, has each one vote on the others' answers, and returns the majority-selected response.

### Current Features
- Parallel execution of LLM queries
- Cross-feeding responses between models for voting
- Majority vote selection
- Stub clients for testing without API keys

### Quick Start
```bash
python3 voting_bots.py
```

### Structure
- `voting_bots.py` - Core voting logic and LLM interaction patterns
- `utils.py` - Shared utilities
- `.env` - Environment variables (API keys)

### Coming Soon
- Production LLM client integration
- Improved error handling
- CLI interface
- Tests

### Usage Notes
Currently uses stub LLM clients for demonstration. To use with real LLMs, replace the `CLIENTS` and `MODELS` dictionaries with your API client instances.
Just for the first version second version would be uploaded in the next 8 hours and would use 
Claude Sonnet 3.5, GPT 4.1 and Gemini 2.0 flash lite concurrently.
