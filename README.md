# Black LLab
Black LLab is a completely open source LLM orchestration framework designed for local and cloud environments. It provides enterprise grade memory management, intelligent routing, and advanced retrieval augmented generation. The architecture is built to maximize context efficiency while minimizing API costs and latency.

# Core Architecture
Prompt Evaluation and Dynamic Routing
The system does not treat all prompts equally. Instead, it utilizes a multi layered pipeline to evaluate and route queries intelligently.

Speculative Decoding: As the user types, a lightweight model predicts the remainder of the query and fetches web context in the background. This results in zero latency data retrieval by the time the user submits the prompt.

Contextual Reformulation: The router analyzes the raw prompt against recent chat history to resolve pronouns into standalone queries; this ensures external tools and search engines receive complete context.

Complexity Grading: A background router grades each prompt on a 1 to 100 scale to determine if the request requires coding capabilities, vision, or live search.

Dynamic Routing: Based on the complexity score, the system directs the task to the most efficient model. Simple tasks are handled locally, while highly complex analytical requests trigger larger frontier models or System 2 agentic clusters.

# Context and Memory Management
Context windows are treated as a strict resource to prevent token bloat and AI hallucination.

XML Isolation: Retrieved facts, web search results, and memory are isolated within strict XML boundaries. This prevents the model from confusing retrieved data with user instructions.

Rolling Summarization: The chat maintains a maximum threshold of messages. Once reached, a background task summarizes the oldest interactions while preserving the exact text of the most recent messages.

Adaptive Personas: The system dynamically injects system overrides based on prompt complexity. For example, it will seamlessly apply a software architect persona for coding tasks.

Prompt Caching: For massive context blocks, the system applies ephemeral cache control tags to freeze context in memory and drastically reduce subsequent API costs.

# The Multi Stage RAG Pipeline
The retrieval augmented generation pipeline blends semantic search with graph relations and agentic reasoning.

Tabular and OCR Extraction: Document ingestion explicitly detects and reconstructs tables into strict Markdown. If a page lacks text, it defaults to a local optical character recognition engine to read the images.

Global Context DNA: A fast router reads the beginning of a document to generate a global summary. This summary is prepended to every overlapping chunk to ensure isolated paragraphs preserve their broader meaning.

HyDE and RRF Retrieval: The system generates a hypothetical perfect answer to the user prompt. It queries the local vector database using both the real query and the hypothetical answer, fusing the results via reciprocal rank fusion.

Cross Encoder Re ranking: Retrieved chunks are aggressively re ranked using a local cross encoder. The surviving chunks are ordered using the lost in the middle technique, which places the highest scoring data at the extreme edges of the context window where models pay the most attention.

Graph Retrieval: In the background, the system extracts entity relationship triples and maps them to a local graph. During chat, it matches the query to graph nodes and injects relational neighbors into the context window.

# System Requirements
To run Black LLab effectively, your host machine requires the following infrastructure:

Python 3.10+

Docker: Required for spawning secure OpenClaw agent sandboxes and web search containers.

Ollama: Required for running local background models and local chat capabilities.

# Installation and Setup
1. Clone the Repository

Bash
git clone https://github.com/yourusername/black-llab.git
cd black-llab
2. Install Python Dependencies
It is highly recommended to use a virtual environment.

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
3. Docker and Web Search Initialization
The system relies on Docker to isolate the OpenClaw agent and to run the local meta search engine. Please ensure Docker is installed and running on your system. For detailed instructions on setting up the OpenClaw Docker environment, please refer to the official documentation at https://docs.openclaw.ai/install/docker.

# To initialize the local SearxNG web search container, run the following commands in your terminal. The second and third commands are critical; they explicitly unlock the JSON API format required by the Black LLab router.

Bash
docker run -d --name searxng -p 8080:8080 -e SEARXNG_BASE_URL=http://127.0.0.1:8080 searxng/searxng
sleep 3
docker exec searxng sed -i 's/- html/- html\n    - json/g' /etc/searxng/settings.yml
docker restart searxng
4. Initialize Local Models
Ensure Ollama is installed and running on your system, then pull the required routing and execution models:

Bash
ollama pull ministral-3:3b
ollama pull qwen3.5:9b
5. Environment Configuration
Create a .env file in the root directory. Use the following template to configure your access tokens and API keys:

Ini, TOML
# Server Configuration
PORT=8000
HOST=127.0.0.1
NEXUS_TOKEN=your_secure_authentication_token

# Local API Endpoints
OLLAMA_API_BASE=http://localhost:11434/api

# Local Model Definitions
ROUTER_MODEL=ministral-3:3b
GIANT_MODEL=qwen3.5:9b

# Cloud API Keys (Optional but recommended for dynamic routing)
XIAOMI_API_KEY=your_xiaomi_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
ZHIPU_API_KEY=your_zhipu_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
6. Launch the Application
Start the FastAPI server. The system will automatically bootstrap the local vector database, initialize the knowledge graph, and manage the Docker sandboxes.

Bash
python main.py
Access the user interface by navigating to http://localhost:8000 in your web browser. You will be prompted to enter your NEXUS_TOKEN upon your first visit.
