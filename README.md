---
title: GAIA Agent
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.36.2
app_file: app.py
hf_oauth: true
---

# 🧠 GAIA Agent

**Multi-agent system for GAIA benchmark evaluation**

## 🌟 Features

- **🤖 Langgraph based workflow**: Complexity detection routes simple questions to direct LLM processing and complex questions to specialized agents
- **📁 Embedded smolagents agent system for direct file i/o and delegation to specialized agents**: Handles all 17 GAIA file types including Excel, PDFs, images, audio, and more  
- **🔍 RAG-Enhanced Decision Making**: Uses 165 GAIA examples to guide agent selection and strategy

## 🏗️ Architecture

### Core Components
- **GAIAAgent**: Main orchestrator with LangGraph workflow
- **Smart Routing System**: Complexity-based strategy selection
- **Specialized Agents**: Data analyst, web researcher, multimodal content processor, general assistant
- **RAG System**: Example-driven decision making

### Performance
- **Target Accuracy**: 50-60% on GAIA benchmark
- **Smart Routing**: Optimizes cost and speed
- **Multi-Provider Support**: Google, OpenRouter with automatic fallback

## 🚀 Built for HF Agents Course Final Assignment

This implementation demonstrates advanced multi-agent architecture principles with production-ready features including comprehensive testing, intelligent routing, and robust error handling.
