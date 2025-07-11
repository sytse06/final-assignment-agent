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
- **📁 Embedded smolagents agent system for direct file i/o and delegation to specialized agents**: Via python i/o different file types like Excel, PDFs, images, audio, and more  
- **🔍 Vision browser tool and many more**: Makes snapshots of web pages to capture context  

## 🏗️ Architecture

### Core Components
- **GAIAAgent**: Main orchestrator with LangGraph workflow
- **LLM based complexity check**: Complexity-based strategy selection
- **Specialized Agents**: Coordinator agent with data analyst, web researcher, content processor as specialists
- **RAG System**: Example-driven decision making to improve context

### Performance
- **Target Accuracy**: 50-60% on GAIA benchmark
- **Smart Routing**: Optimizes cost and speed
- **Multi-Provider Support**: OpenRouter, Google with automatic fallback

## 🚀 Built for HF Agents Course Final Assignment
