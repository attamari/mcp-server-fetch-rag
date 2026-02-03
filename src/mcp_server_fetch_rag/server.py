import gc
import os
from typing import Annotated
from urllib.parse import urlparse, urlunparse

import numpy as np
import onnxruntime as ort
import pypdfium2 as pdfium
import trafilatura
from fastembed import TextEmbedding
from mcp.shared.exceptions import McpError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ErrorData,
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    TextContent,
    Tool,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from protego import Protego
from pydantic import BaseModel, Field, AnyUrl
from wtpsplit_lite import SaT

# Constants
DEFAULT_USER_AGENT_AUTONOMOUS = "ModelContextProtocol/1.0 (Autonomous; +https://github.com/modelcontextprotocol/servers)"
DEFAULT_USER_AGENT_MANUAL = "ModelContextProtocol/1.0 (User-Specified; +https://github.com/modelcontextprotocol/servers)"

MIN_CHUNK_CHARS = 50
MAX_CHUNK_CHARS = 3000
SIMILARITY_THRESHOLD = 0.38
MIN_SCORE = 0.38
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedding_model: TextEmbedding | None = None
_sentence_splitter: SaT | None = None


def _get_onnx_providers() -> list[str]:
    """Get available ONNX providers in priority order.

    Can be overridden via MCP_FETCH_RAG_PROVIDERS environment variable.
    """
    env_providers = os.environ.get("MCP_FETCH_RAG_PROVIDERS")
    if env_providers:
        return [p.strip() for p in env_providers.split(",")]

    available = ort.get_available_providers()
    preferred = [
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "DmlExecutionProvider",
        "ROCMExecutionProvider",
        "OpenVINOExecutionProvider",
        "DnnlExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in preferred if p in available]


def get_embedding_model() -> TextEmbedding:
    """Lazy load embedding model with auto-detected providers."""
    global _embedding_model
    if _embedding_model is None:
        providers = _get_onnx_providers()
        _embedding_model = TextEmbedding(EMBEDDING_MODEL, providers=providers)
    return _embedding_model


def get_sentence_splitter() -> SaT:
    """Lazy load sentence splitter with auto-detected providers."""
    global _sentence_splitter
    if _sentence_splitter is None:
        providers = _get_onnx_providers()
        _sentence_splitter = SaT("sat-3l-sm", ort_providers=providers)
    return _sentence_splitter


def reset_models() -> None:
    """Reset models to release ONNX runtime memory arena."""
    global _embedding_model, _sentence_splitter
    _embedding_model = None
    _sentence_splitter = None
    gc.collect()


def extract_content_from_html(html: str) -> str:
    """Extract and convert HTML content to Markdown format."""
    content = trafilatura.extract(
        html,
        include_links=True,
        include_formatting=True,
        include_tables=True,
        output_format="markdown",
        deduplicate=True,
        favor_precision=True,
    )
    if not content:
        return "<error>Page failed to be simplified from HTML</error>"
    return content


def extract_content_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes."""
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        text_parts = []
        for page in pdf:
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            if text.strip():
                text_parts.append(text.strip())
        pdf.close()
        if not text_parts:
            return "<error>No text content found in PDF</error>"
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"<error>Failed to extract PDF content: {e!r}</error>"


def get_robots_txt_url(url: str) -> str:
    """Get the robots.txt URL for a given website URL."""
    parsed = urlparse(url)
    robots_url = urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    return robots_url


async def check_may_autonomously_fetch_url(
    url: str, user_agent: str, proxy_url: str | None = None
) -> None:
    """Check if the URL can be fetched according to robots.txt."""
    from httpx import AsyncClient, HTTPError

    robot_txt_url = get_robots_txt_url(url)

    async with AsyncClient(proxy=proxy_url) as client:
        try:
            response = await client.get(
                robot_txt_url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            )
        except HTTPError:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Failed to fetch robots.txt {robot_txt_url} due to a connection issue",
                )
            )
        if response.status_code in (401, 403):
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"When fetching robots.txt ({robot_txt_url}), received status {response.status_code} so assuming that autonomous fetching is not allowed",
                )
            )
        elif 400 <= response.status_code < 500:
            return
        robot_txt = response.text

    processed_robot_txt = "\n".join(
        line for line in robot_txt.splitlines() if not line.strip().startswith("#")
    )
    robot_parser = Protego.parse(processed_robot_txt)
    if not robot_parser.can_fetch(str(url), user_agent):
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"The site's robots.txt ({robot_txt_url}) specifies that autonomous fetching of this page is not allowed.",
            )
        )


async def fetch_url(
    url: str, user_agent: str, proxy_url: str | None = None
) -> tuple[str, str]:
    """Fetch the URL and return the content."""
    from httpx import AsyncClient, HTTPError

    async with AsyncClient(proxy=proxy_url) as client:
        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
                timeout=30,
            )
        except HTTPError as e:
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}")
            )
        if response.status_code >= 400:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Failed to fetch {url} - status code {response.status_code}",
                )
            )

    content_type = response.headers.get("content-type", "")

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        return extract_content_from_pdf(response.content), ""

    page_raw = response.text
    is_page_html = (
        "<html" in page_raw[:100] or "text/html" in content_type or not content_type
    )

    if is_page_html:
        return extract_content_from_html(page_raw), ""

    return (
        page_raw,
        f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
    )


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using wtpsplit."""
    splitter = get_sentence_splitter()
    sentences = splitter.split(text)
    return [s.strip() for s in sentences if s.strip()]


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed texts using the configured embedding model."""
    model = get_embedding_model()

    # E5 models require "query:" / "passage:" prefixes
    if "e5" in EMBEDDING_MODEL.lower():
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]

    embeddings = list(model.embed(texts))
    return np.array(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

    return np.dot(a_norm, b_norm.T)


def semantic_chunking(
    sentences: list[str],
    embeddings: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[list[str], np.ndarray]:
    """Perform semantic chunking based on sentence similarity."""
    if len(sentences) == 0:
        return [], np.array([])

    if len(sentences) == 1:
        return sentences, embeddings

    chunks = []
    chunk_embeddings = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_indices = [0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])[0, 0]
        current_chunk_text = " ".join(current_chunk_sentences)

        should_split = sim < threshold
        if len(current_chunk_text) + len(sentences[i]) > MAX_CHUNK_CHARS:
            should_split = True

        if should_split and len(current_chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append(current_chunk_text)
            chunk_emb = embeddings[current_chunk_indices].mean(axis=0)
            chunk_embeddings.append(chunk_emb)

            current_chunk_sentences = [sentences[i]]
            current_chunk_indices = [i]
        else:
            current_chunk_sentences.append(sentences[i])
            current_chunk_indices.append(i)

    if current_chunk_sentences:
        current_chunk_text = " ".join(current_chunk_sentences)
        chunks.append(current_chunk_text)
        chunk_emb = embeddings[current_chunk_indices].mean(axis=0)
        chunk_embeddings.append(chunk_emb)

    return chunks, np.array(chunk_embeddings)


def calculate_centrality_scores(chunk_embeddings: np.ndarray) -> np.ndarray:
    """Calculate centrality scores (similarity to document centroid)."""
    if len(chunk_embeddings) == 0:
        return np.array([])

    centroid = chunk_embeddings.mean(axis=0)
    scores = cosine_similarity(centroid, chunk_embeddings)[0]
    return scores


def calculate_query_scores(
    query: str, chunk_embeddings: np.ndarray
) -> np.ndarray:
    """Calculate similarity scores between query and chunks."""
    if len(chunk_embeddings) == 0:
        return np.array([])

    query_embedding = embed_texts([query], is_query=True)[0]
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    return scores


def process_content_with_rag(
    content: str,
    query: str | None,
    min_score: float,
    similarity_threshold: float,
    max_chunks: int | None = None,
) -> list[str]:
    """Process content with RAG: split, embed, chunk, score, filter, and limit."""
    try:
        sentences = split_sentences(content)
        if not sentences:
            return []

        sentence_embeddings = embed_texts(sentences, is_query=False)
        chunks, chunk_embeddings = semantic_chunking(sentences, sentence_embeddings, similarity_threshold)
        del sentence_embeddings

        if not chunks:
            return []

        if query:
            scores = calculate_query_scores(query, chunk_embeddings)
        else:
            scores = calculate_centrality_scores(chunk_embeddings)

        scored_chunks = [
            (i, chunk, score)
            for i, (chunk, score) in enumerate(zip(chunks, scores))
            if score >= min_score
        ]

        # Supplement with centrality-based chunks if needed
        effective_max = max_chunks if max_chunks is not None else len(chunks)
        if len(scored_chunks) < effective_max:
            centrality_scores = calculate_centrality_scores(chunk_embeddings)
            included_indices = {i for i, _, _ in scored_chunks}
            remaining_chunks = [
                (i, chunk, centrality_scores[i])
                for i, chunk in enumerate(chunks)
                if i not in included_indices
            ]
            remaining_chunks.sort(key=lambda x: x[2], reverse=True)
            needed = effective_max - len(scored_chunks)
            scored_chunks.extend(remaining_chunks[:needed])

        if max_chunks is not None and len(scored_chunks) > max_chunks:
            scored_chunks.sort(key=lambda x: x[2], reverse=True)
            scored_chunks = scored_chunks[:max_chunks]

        scored_chunks.sort(key=lambda x: x[0])
        return [chunk for _, chunk, _ in scored_chunks]
    finally:
        reset_models()


class FetchRag(BaseModel):
    """Parameters for fetching a URL with RAG processing."""

    url: Annotated[AnyUrl, Field(description="URL to fetch")]
    query: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional search query. If provided, returns chunks relevant to the query. If not provided, returns chunks with high centrality (representative of the document).",
        ),
    ]
    max_chunks: Annotated[
        int | None,
        Field(
            default=7,
            description="Maximum number of chunks to return. If None, returns all chunks above min_score. Chunks are sorted by relevance score.",
        ),
    ]


async def serve(
    custom_user_agent: str | None = None,
    ignore_robots_txt: bool = False,
    proxy_url: str | None = None,
) -> None:
    """Run the fetch-rag MCP server."""
    server = Server("mcp-fetch-rag")
    user_agent_autonomous = custom_user_agent or DEFAULT_USER_AGENT_AUTONOMOUS
    user_agent_manual = custom_user_agent or DEFAULT_USER_AGENT_MANUAL

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="fetch_rag",
                description="""Fetches a URL and returns only the most relevant content.

- With query: returns content matching the query
- Without query: returns most representative content

Use for focused retrieval instead of fetching entire pages.""",
                inputSchema=FetchRag.model_json_schema(),
            )
        ]

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return [
            Prompt(
                name="fetch_rag",
                description="Fetch a URL and extract relevant content chunks",
                arguments=[
                    PromptArgument(
                        name="url", description="URL to fetch", required=True
                    ),
                    PromptArgument(
                        name="query",
                        description="Optional search query for relevance filtering",
                        required=False,
                    ),
                ],
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            args = FetchRag(**arguments)
        except ValueError as e:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))

        url = str(args.url)
        if not url:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

        if not ignore_robots_txt:
            await check_may_autonomously_fetch_url(
                url, user_agent_autonomous, proxy_url
            )

        content, prefix = await fetch_url(
            url, user_agent_autonomous, proxy_url=proxy_url
        )

        if content.startswith("<error>"):
            return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]

        chunks = process_content_with_rag(
            content, args.query, MIN_SCORE, SIMILARITY_THRESHOLD, args.max_chunks
        )

        if not chunks:
            return [
                TextContent(
                    type="text",
                    text=f"No relevant content found for {url}" + (f" with query '{args.query}'" if args.query else ""),
                )
            ]

        result = f"Relevant content from {url}:\n\n"
        result += "\n\n---\n\n".join(chunks)

        return [TextContent(type="text", text=result)]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
        if not arguments or "url" not in arguments:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

        url = arguments["url"]
        query = arguments.get("query")

        try:
            content, prefix = await fetch_url(url, user_agent_manual, proxy_url=proxy_url)
        except McpError as e:
            return GetPromptResult(
                description=f"Failed to fetch {url}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=str(e)),
                    )
                ],
            )

        chunks = process_content_with_rag(content, query, MIN_SCORE, SIMILARITY_THRESHOLD)

        if not chunks:
            result = f"No relevant content found for {url}"
        else:
            result = f"Relevant content from {url}:\n\n"
            result += "\n\n---\n\n".join(chunks)

        return GetPromptResult(
            description=f"Relevant content from {url}",
            messages=[
                PromptMessage(role="user", content=TextContent(type="text", text=result))
            ],
        )

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)
