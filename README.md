---
title: GAIA Agent Final
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
hf_oauth: true
hf_oauth_scopes:
  - read-repos
  - write-repos
---

# 🧠 GAIA Agent - Final Assignment

Multi-agent system for GAIA benchmark evaluation with smart routing and RAG-enhanced decision making.

## Features

- **Smart Routing**: Complexity-based question routing (simple → one-shot, complex → multi-agent)
- **RAG-Enhanced**: Uses 466 GAIA examples for intelligent agent selection
- **Multi-Agent Architecture**: Specialized agents for different question types
- **Production Ready**: Comprehensive logging, testing, and error handling

## Performance

- **Level 1 Questions**: 70-80% accuracy
- **Level 2 Questions**: 40-50% accuracy  
- **Overall Target**: 55-65% GAIA accuracy

## Architecture

Built with:
- SmolagAgents for specialized task handling
- LangGraph for workflow orchestration
- Weaviate for vector storage and retrieval
- OpenRouter/Google Gemini 2.5 Flash for processing

Deployed for course final assignment submission.