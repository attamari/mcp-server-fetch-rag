import asyncio
import gc
import os
from typing import Annotated
from urllib.parse import urlparse, urlunparse

import httpx
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
DEFAULT_USER_AGENT_AUTONOMOUS = "ModelContextProtocol/1.0 (Autonomous; +https://github.com/anthropics/mcp-fetch-rag)"
DEFAULT_USER_AGENT_MANUAL = "ModelContextProtocol/1.0 (User-Specified; +https://github.com/anthropics/mcp-fetch-rag)"
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Priority": "u=0, i",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

MIN_CHUNK_CHARS = 50
MAX_CHUNK_CHARS = 3000
SIMILARITY_THRESHOLD = 0.38
PERCENTILE_THRESHOLD = 30
POWER_MEAN_P = 3.0
LEXRANK_THRESHOLD = 0.1
LEXRANK_DAMPING = 0.85
LEXRANK_MAX_ITERATIONS = 100
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


def _prefetch_models() -> None:
    """Prefetch models in background on server startup."""
    try:
        get_embedding_model()
        get_sentence_splitter()
    except Exception:
        pass


def extract_content_from_html(html: str) -> str:
    """Extract and convert HTML content to Markdown format."""
    content = trafilatura.extract(
        html,
        include_links=False,
        include_comments=False,
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
            stripped = text.strip()
            if stripped:
                text_parts.append(stripped)
        pdf.close()
        if not text_parts:
            return "<error>No text content found in PDF</error>"
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"<error>Failed to extract PDF content: {e!r}</error>"


async def check_may_autonomously_fetch_url(
    url: str, user_agent: str, proxy_url: str | None = None
) -> None:
    """Check if the URL can be fetched according to robots.txt."""
    parsed = urlparse(url)
    robot_txt_url = urlunparse(
        (parsed.scheme, parsed.netloc, "/robots.txt", "", "", "")
    )

    async with httpx.AsyncClient(proxy=proxy_url) as client:
        try:
            response = await client.get(
                robot_txt_url,
                follow_redirects=True,
                headers={"User-Agent": user_agent, **DEFAULT_HEADERS},
                timeout=30,
            )
        except httpx.HTTPError:
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

    robot_parser = Protego.parse(robot_txt)
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
    async with httpx.AsyncClient(proxy=proxy_url) as client:
        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": user_agent, **DEFAULT_HEADERS},
                timeout=30,
            )
        except httpx.HTTPError as e:
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
    stripped = [s.strip() for s in sentences]
    return [s for s in stripped if s]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using the configured embedding model."""
    model = get_embedding_model()
    result = np.array(list(model.embed(texts)))
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result /= norms
    return result


def semantic_chunking(
    sentences: list[str],
    embeddings: np.ndarray,
) -> tuple[list[str], list[list[int]]]:
    """Perform semantic chunking based on sentence similarity."""
    if not sentences:
        return [], []

    if len(sentences) == 1:
        return sentences, [[0]]

    adj_similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)

    chunks = []
    chunk_sentence_indices = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_indices = [0]
    current_chunk_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = adj_similarities[i - 1]
        next_len = len(sentences[i])

        should_split = sim < SIMILARITY_THRESHOLD
        if current_chunk_len + next_len > MAX_CHUNK_CHARS:
            should_split = True

        if should_split and current_chunk_len >= MIN_CHUNK_CHARS:
            chunks.append(" ".join(current_chunk_sentences))
            chunk_sentence_indices.append(current_chunk_indices.copy())

            current_chunk_sentences = [sentences[i]]
            current_chunk_indices = [i]
            current_chunk_len = next_len
        else:
            current_chunk_sentences.append(sentences[i])
            current_chunk_indices.append(i)
            current_chunk_len += 1 + next_len

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        chunk_sentence_indices.append(current_chunk_indices)

    return chunks, chunk_sentence_indices


def aggregate_power_mean_chunks(
    scores: np.ndarray,
    chunk_sentence_indices: list[list[int]],
) -> np.ndarray:
    """Vectorized power mean aggregation over contiguous chunk groups."""
    if not chunk_sentence_indices:
        return np.array([])
    starts = np.array([indices[0] for indices in chunk_sentence_indices])
    lengths = np.array([len(indices) for indices in chunk_sentence_indices])
    powered = np.power(np.maximum(scores, 0.0), POWER_MEAN_P)
    sums = np.add.reduceat(powered, starts)
    return np.power(sums / lengths, 1.0 / POWER_MEAN_P)


def calculate_lexrank_scores(
    sentence_embeddings: np.ndarray,
    chunk_sentence_indices: list[list[int]],
) -> np.ndarray:
    """Calculate LexRank scores at sentence level, then aggregate to chunk level."""
    if len(chunk_sentence_indices) == 0:
        return np.array([])
    n_sentences = len(sentence_embeddings)
    if n_sentences == 0:
        return np.array([])
    if n_sentences == 1:
        return np.array([1.0])
    similarity_matrix = sentence_embeddings @ sentence_embeddings.T
    np.fill_diagonal(similarity_matrix, 0)
    similarity_matrix[similarity_matrix < LEXRANK_THRESHOLD] = 0
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.where(
        row_sums > 0,
        similarity_matrix / row_sums,
        1.0 / n_sentences,
    )
    sentence_scores = np.ones(n_sentences) / n_sentences
    for _ in range(LEXRANK_MAX_ITERATIONS):
        prev_scores = sentence_scores.copy()
        sentence_scores = (
            (1 - LEXRANK_DAMPING) / n_sentences
            + LEXRANK_DAMPING * transition_matrix.T @ sentence_scores
        )
        if np.allclose(sentence_scores, prev_scores, atol=1e-6):
            break
    return aggregate_power_mean_chunks(sentence_scores, chunk_sentence_indices)


def calculate_query_scores(
    query: str,
    sentence_embeddings: np.ndarray,
    chunk_sentence_indices: list[list[int]],
) -> np.ndarray:
    """Calculate similarity scores using late interaction (power mean of sentence scores)."""
    if len(chunk_sentence_indices) == 0:
        return np.array([])
    query_embedding = embed_texts([query])[0]
    all_scores = query_embedding @ sentence_embeddings.T
    return aggregate_power_mean_chunks(all_scores, chunk_sentence_indices)


def process_content_with_rag(
    content: str,
    query: str | None,
    max_chunks: int | None = None,
) -> list[str]:
    """Process content with RAG: split, embed, chunk, score, filter, and limit."""
    try:
        sentences = split_sentences(content)
        if not sentences:
            return []

        sentence_embeddings = embed_texts(sentences)
        chunks, chunk_sentence_indices = semantic_chunking(
            sentences, sentence_embeddings
        )

        if not chunks:
            return []

        if query:
            scores = calculate_query_scores(
                query, sentence_embeddings, chunk_sentence_indices
            )
            lexrank_scores = None
        else:
            lexrank_scores = calculate_lexrank_scores(
                sentence_embeddings, chunk_sentence_indices
            )
            scores = lexrank_scores

        threshold = max(float(np.percentile(scores, PERCENTILE_THRESHOLD)), 0.0)

        scored_chunks = [
            (i, chunk, score)
            for i, (chunk, score) in enumerate(zip(chunks, scores))
            if score >= threshold
        ]
        passed_indices = {i for i, _, _ in scored_chunks}

        effective_max = max_chunks if max_chunks is not None else len(chunks)
        if len(scored_chunks) < effective_max and query:
            if lexrank_scores is None:
                lexrank_scores = calculate_lexrank_scores(
                    sentence_embeddings, chunk_sentence_indices
                )
            lr_threshold = max(
                float(np.percentile(lexrank_scores, PERCENTILE_THRESHOLD)), 0.0
            )
            remaining_chunks = [
                (i, chunk, lexrank_scores[i])
                for i, chunk in enumerate(chunks)
                if i not in passed_indices and lexrank_scores[i] >= lr_threshold
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


async def _run_fetch_pipeline(
    url: str,
    query: str | None,
    user_agent: str,
    proxy_url: str | None,
    check_robots: bool,
    max_chunks: int | None = None,
) -> str:
    """Fetch URL, run RAG pipeline, return formatted text."""
    if check_robots:
        await check_may_autonomously_fetch_url(url, user_agent, proxy_url)

    content, prefix = await fetch_url(url, user_agent, proxy_url=proxy_url)

    if content.startswith("<error>"):
        return f"{prefix}Content from {url}:\n{content}"

    chunks = process_content_with_rag(content, query, max_chunks)

    if not chunks:
        msg = f"No relevant content found for {url}"
        if query:
            msg += f" with query '{query}'"
        return msg

    return f"{prefix}Content from {url}:\n\n" + "\n\n---\n\n".join(chunks)


class FetchRag(BaseModel):
    """Parameters for fetching a URL with RAG processing."""

    url: Annotated[AnyUrl, Field(description="URL to fetch")]
    query: Annotated[
        str | None,
        Field(
            default=None,
            description="Query to find relevant content",
        ),
    ]
    max_chunks: Annotated[
        int,
        Field(
            default=10,
            description="Maximum number of chunks to return (default: 10)",
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

    asyncio.create_task(asyncio.to_thread(_prefetch_models))

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="fetch_rag",
                description="Fetch URL and return relevant content. Optionally provide a query to find specific information.",
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
                        description="Query to find relevant content",
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
        try:
            text = await _run_fetch_pipeline(
                url,
                args.query,
                user_agent_autonomous,
                proxy_url,
                check_robots=not ignore_robots_txt,
                max_chunks=args.max_chunks,
            )
        except McpError as e:
            text = str(e)
        except Exception as e:
            text = f"Failed to process content from {url}: {e!r}"
        return [TextContent(type="text", text=text)]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
        if not arguments or "url" not in arguments:
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message="URL is required")
            )
        url = arguments["url"]
        query = arguments.get("query")
        try:
            text = await _run_fetch_pipeline(
                url, query, user_agent_manual, proxy_url, check_robots=False
            )
        except McpError as e:
            text = str(e)
        except Exception as e:
            text = f"Failed to process content from {url}: {e!r}"
        return GetPromptResult(
            description=f"Content from {url}",
            messages=[
                PromptMessage(
                    role="user", content=TextContent(type="text", text=text)
                )
            ],
        )

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)
