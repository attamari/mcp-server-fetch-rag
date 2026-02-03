# MCP Server Fetch RAG

A Model Context Protocol (MCP) server that fetches web content and returns only relevant chunks using RAG (Retrieval-Augmented Generation).

GitHub: https://github.com/attamari/mcp-server-fetch-rag

## Features

- **Semantic Chunking**: Splits content into meaningful chunks based on sentence similarity
- **Query-based Filtering**: Returns chunks most relevant to your query
- **Centrality Scoring**: When no query is provided, returns chunks most representative of the document
- **Multilingual Support**: Uses `paraphrase-multilingual-MiniLM-L12-v2` for multilingual support
- **Context Efficient**: Filters out irrelevant content to reduce token usage

## Installation

### From GitHub (recommended)

```bash
# Install via uv
uv pip install git+https://github.com/attamari/mcp-server-fetch-rag.git

# Or install via pip
pip install git+https://github.com/attamari/mcp-server-fetch-rag.git
```

### From source

```bash
git clone https://github.com/attamari/mcp-server-fetch-rag.git
cd mcp-server-fetch-rag
uv pip install -e .
```

## Usage

### As MCP Server

Add to your MCP client configuration:

**Using uvx with GitHub (recommended):**

```json
{
  "mcpServers": {
    "fetch-rag": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/attamari/mcp-server-fetch-rag.git", "mcp-server-fetch-rag"]
    }
  }
}
```

**Using uv:**

```json
{
  "mcpServers": {
    "fetch-rag": {
      "command": "uv",
      "args": ["--from", "git+https://github.com/attamari/mcp-server-fetch-rag.git", "run", "mcp-server-fetch-rag"]
    }
  }
}
```

**Using python (after installation):**

```json
{
  "mcpServers": {
    "fetch-rag": {
      "command": "python",
      "args": ["-m", "mcp_server_fetch_rag"]
    }
  }
}
```

### Tool: `fetch_rag`

Fetches a URL and returns relevant content chunks.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | Yes | - | URL to fetch |
| `query` | string | No | null | Search query for relevance filtering |
| `max_chunks` | int | No | 7 | Maximum number of chunks to return |

**Examples:**

```python
# With query - returns chunks relevant to the query
fetch_rag(url="https://example.com/docs", query="authentication")

# Without query - returns most representative chunks
fetch_rag(url="https://example.com/docs")

# Limit number of chunks returned
fetch_rag(url="https://example.com/docs", query="API", max_chunks=10)
```

## How It Works

1. **Fetch**: Downloads and extracts content from URL (using trafilatura for HTML, pypdfium2 for PDF)
2. **Sentence Split**: Splits into sentences (using wtpsplit, 85+ languages)
3. **Embed**: Generates embeddings (using paraphrase-multilingual-MiniLM-L12-v2 via FastEmbed)
4. **Semantic Chunking**: Groups sentences by similarity
5. **Score**: Calculates relevance (query similarity or centrality)
6. **Filter & Limit**: Returns top chunks based on score and max_chunks parameter

## Models Used

- **Sentence Segmentation**: [wtpsplit](https://github.com/segment-any-text/wtpsplit) (sat-3l-sm)
- **Embeddings**: [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) via FastEmbed (ONNX)
- **Content Extraction**: [trafilatura](https://trafilatura.readthedocs.io/) for HTML, [pypdfium2](https://pypdfium2.readthedocs.io/) for PDF

## Advanced Features

- **GPU Acceleration**: Automatically detects and uses available GPU providers (CUDA, DirectML, ROCm, OpenVINO)
- **PDF Support**: Extracts text content from PDF documents
- **Robots.txt Compliance**: Respects robots.txt by default (can be disabled with `--ignore-robots-txt`)
- **Proxy Support**: Configure proxy with `--proxy-url` parameter
- **Custom User-Agent**: Set custom User-Agent with `--user-agent` parameter

## License

MIT License - See [LICENSE](LICENSE) file for details

Based on [MCP Fetch Server](https://github.com/modelcontextprotocol/servers) from Anthropic, PBC.
