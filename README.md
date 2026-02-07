# MCP Server Fetch RAG

A Model Context Protocol (MCP) server that fetches web content and returns relevant chunks using RAG (Retrieval-Augmented Generation).

GitHub: https://github.com/attamari/mcp-server-fetch-rag

## Features

- **Semantic Chunking**: Groups sentences into coherent chunks based on embedding similarity
- **Query-based Scoring**: Late Interaction with Power Mean aggregation for precise relevance scoring
- **LexRank Scoring**: Graph-based centrality scoring when no query is provided
- **LexRank Backfill**: Supplements query results with high-centrality chunks when needed
- **Percentile Filtering**: Dynamic threshold based on score distribution
- **Multilingual Support**: Uses `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages)
- **PDF Support**: Extracts text from PDF documents
- **GPU Acceleration**: Auto-detects CUDA, DirectML, ROCm, OpenVINO providers
- **Context Efficient**: Filters out irrelevant content to reduce token usage

## Usage

### MCP Client Configuration

Add to your MCP client configuration (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "fetch-rag": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/attamari/mcp-server-fetch-rag",
        "mcp-server-fetch-rag"
      ]
    }
  }
}
```

With CLI options:

```json
{
  "mcpServers": {
    "fetch-rag": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/attamari/mcp-server-fetch-rag",
        "mcp-server-fetch-rag",
        "--ignore-robots-txt",
        "--user-agent", "your-custom-user-agent"
      ]
    }
  }
}
```

### CLI Options

| Option | Description |
|---|---|
| `--user-agent` | Custom User-Agent string (overrides default MCP UA) |
| `--ignore-robots-txt` | Ignore robots.txt restrictions |
| `--proxy-url` | Proxy URL for HTTP requests |

### Tool: fetch_rag

Fetches a URL and returns relevant content chunks.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `url` | string | Yes | - | URL to fetch |
| `query` | string | No | null | Search query for relevance filtering |
| `max_chunks` | int | No | 10 | Maximum number of chunks to return |

## How It Works

1. **Fetch**: Downloads content from URL (HTML via trafilatura, PDF via pypdfium2)
2. **Split**: Segments text into sentences using wtpsplit (sat-3l-sm, 85+ languages)
3. **Embed**: Generates L2-normalized embeddings (paraphrase-multilingual-MiniLM-L12-v2 via FastEmbed/ONNX)
4. **Chunk**: Groups adjacent sentences by embedding similarity into semantic chunks
5. **Score**:
   - With query: Late Interaction — sentence-level query similarity aggregated via Power Mean
   - Without query: LexRank — sentence-level graph centrality aggregated via Power Mean
6. **Filter**: Applies percentile-based dynamic threshold (P30)
7. **Backfill**: When query scoring yields insufficient chunks, supplements with high-centrality LexRank chunks (P30 filtered)
8. **Return**: Top chunks sorted in original document order

## License

MIT License — See [LICENSE](LICENSE) for details.
