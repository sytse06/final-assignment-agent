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

# GAIA Agent

Final assignment for the [HuggingFace Agents Course](https://github.com/huggingface/agents-course)
(Unit 4 capstone · 25.4k ⭐). Built and evaluated against the
[GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) —
450+ real-world tasks requiring autonomous tool use, multi-step reasoning,
and multi-modal processing. Scored above the passing threshold.

**Live demo:** [HuggingFace Spaces](https://huggingface.co/spaces/sytse06/Gaia-Agent-Final)

## What is GAIA?

GAIA tests agents on real-world questions across three difficulty levels.
Every answer must be exact — no partial credit. Tasks span web research,
document analysis, image interpretation, audio transcription, and
multi-step reasoning that no single tool can handle alone.

| Level | Task type | Score |
|---|---|---|
| 1 | Single-step factual retrieval | 65–75% |
| 2 | Multi-step reasoning + tool use | 40–50% |
| 3 | Complex multi-source analysis | 20–30% |
| **Overall** | | **45–55%** ✅ |

## Architecture

A LangGraph coordinator routes each question to a specialised smolagents
sub-agent. Sub-agents share a library of nine tools covering the full
range of GAIA task types.

```
LangGraph coordinator
├── general_assistant    → web search, Q&A, reasoning
├── research_agent       → ArXiv, Wikipedia, deep web
├── multimedia_agent     → YouTube, audio, images
└── document_agent       → PDF, DOCX, XLSX
```

## Tools

| Tool | Purpose |
|---|---|
| `ContentRetrieverTool` | PDF, DOCX, XLSX, image extraction |
| `YoutubeVideoTool` | Transcript + frame-by-frame analysis |
| `SpeechRecognitionTool` | Audio transcription |
| `VisionBrowserTool` | Browser automation + screenshots |
| `GoogleSearchTool` | Web search |
| `ImageToChessBoardFENTool` | Chess board image → FEN notation |
| `GetAttachmentTool` | GAIA task file attachments |
| LangChain tools | Wikipedia, ArXiv |

## Stack

smolagents · LangGraph · LangChain · Groq · OpenAI · Gemini · Gradio 5
