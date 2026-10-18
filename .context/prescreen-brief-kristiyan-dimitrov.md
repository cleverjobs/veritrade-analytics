# Real-Time Financial News Analysis Service

## Overview
VeriTrade Analytics is a fintech firm providing high-speed market intelligence to day traders and quantitative analysts. Our analysts currently waste precious minutes manually reading breaking news articles, press releases, and earnings reports to gauge market sentiment. In a world where milliseconds matter, this manual process introduces critical delays, leading to missed opportunities. To solve this, VeriTrade needs a tool that can ingest a news article and provide an instant, AI-powered summary and sentiment analysis.

Your task is to build a prototype for this internal tool. You will create a web service that accepts a URL to a public financial news article. The service should then fetch the article's content, use a Large Language Model (LLM) to generate a concise summary and a sentiment analysis (e.g., Bullish, Bearish, Neutral), and return this information via an API. The core business constraint is speed; the system must be designed for low latency to be useful for traders making time-sensitive decisions. You will also build a simple frontend interface to interact with this service.

A successful submission will demonstrate thoughtful system design focused on performance, effective use of an LLM for a specific task, and clear communication of your design trade-offs. The frontend UI does not need to be visually polished; a functional interface that clearly presents the workflow and results is sufficient.

## Deliverables
- A public GitHub repository or a single public GitHub gist URL containing your complete solution.
- A backend service written in Python that exposes an API to analyze a given news article URL.
- A basic frontend web application (e.g., React, Next.js, or plain HTML/CSS/JS) that consumes the backend API and displays the results.
- A `README.md` file explaining your design decisions, trade-offs (especially regarding latency optimization), instructions on how to run the application and any tests, and a brief note on which AI tools you used.
- The conversation log from any AI assistants used during development (e.g., VS Code Copilot Chat export, Claude conversation history) committed to the repository.

## Suggested tools / libraries
- Backend Framework: FastAPI, Flask
- LLM Integration: OpenAI API, Anthropic API, Groq, or a local model via Ollama
- Web Scraping: BeautifulSoup, requests
- Frontend Framework: React + Vite, Next.js, or plain HTML/CSS/JavaScript
- API Client: `fetch` API, Axios
- Containerization: Docker (optional)

## On AI assistants & follow-up
- We expect and encourage the use of AI assistants. Your goal is to deliver a great solution, not to memorize APIs.
- Please commit the conversation log from your AI assistant (e.g., Copilot Chat export, Claude history) to your repository and briefly describe its role in your README.
- Be prepared to walk through your code and justify your design decisions, especially regarding performance and architectural trade-offs, in the follow-up interview.
- The focus is on your design judgment, problem decomposition, and verification, not on writing every line of code from scratch.

## How to submit
Reply to this email with **one** public GitHub repository or gist URL containing your solution.

<!-- AI-EVALUATOR:PRESCREEN-META {"language":"Python","dueDays":10,"timeBudgetHours":8} -->