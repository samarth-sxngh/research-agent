import logging
import os
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from src.document_processing.doc_processor import DocumentChunk

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Domains known to block requests — use playwright for these
JS_HEAVY_DOMAINS = {
    "medium.com", "towardsdatascience.com", "betterprogramming.pub",
    "javascript.plainenglish.io", "levelup.gitconnected.com",
    "substack.com", "bloomberg.com", "wsj.com", "ft.com"
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class WebPageData:
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class WebScraper:
    def __init__(self, api_key: str = None):
        # api_key kept for signature compat — unused
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        logger.info("WebScraper initialized with requests+BeautifulSoup (no API key)")

    def scrape_url(
        self,
        url: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        wait_for_results: int = 30
    ) -> List[DocumentChunk]:
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL format: {url}")

        logger.info(f"Scraping URL: {url}")

        try:
            domain = urlparse(url).netloc.lstrip("www.")
            use_playwright = PLAYWRIGHT_AVAILABLE and domain in JS_HEAVY_DOMAINS

            if use_playwright:
                logger.info(f"Using playwright for: {domain}")
                page_data = self._fetch_with_playwright(url, timeout=wait_for_results)
            else:
                page_data = self._fetch_and_parse(url, timeout=wait_for_results)

            # Auto-fallback to playwright if requests got nothing
            if (not page_data.success or not page_data.content.strip()) and PLAYWRIGHT_AVAILABLE and not use_playwright:
                logger.info(f"Requests failed, retrying with playwright: {url}")
                page_data = self._fetch_with_playwright(url, timeout=wait_for_results)

            chunks = self._create_chunks_from_web_content(page_data, chunk_size, chunk_overlap)
            logger.info(f"Successfully scraped {url}: {len(chunks)} chunks created")
            return chunks

        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            raise

    def _fetch_with_playwright(self, url: str, timeout: int = 30) -> WebPageData:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=HEADERS["User-Agent"])
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                page.wait_for_selector("body", timeout=10000)
                html = page.content()
                browser.close()
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script","style","nav","footer","header","aside","noscript"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            desc_tag = soup.find("meta", attrs={"name": "description"})
            description = desc_tag.get("content", "") if desc_tag else ""
            main = (soup.find("article") or soup.find("main") or soup.find("body"))
            content = self._html_to_text(main or soup)
            domain = urlparse(url).netloc
            return WebPageData(
                url=url, title=title or f"Web Page - {domain}", content=content,
                metadata={"scraped_at": datetime.now().isoformat(), "original_url": url,
                    "title": title, "description": description, "word_count": len(content.split()),
                    "character_count": len(content), "domain": domain, "scraper": "playwright"},
                success=bool(content.strip())
            )
        except Exception as e:
            logger.error(f"Playwright error for {url}: {str(e)}")
            return WebPageData(url=url, title=f"Error - {urlparse(url).netloc}", content="",
                metadata={"error": str(e), "scraped_at": datetime.now().isoformat()},
                success=False, error=str(e))

    def _fetch_and_parse(self, url: str, timeout: int = 30) -> WebPageData:
        try:
            resp = self.session.get(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove boilerplate tags
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "noscript", "iframe", "ads"]):
                tag.decompose()

            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            description = ""
            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag:
                description = desc_tag.get("content", "")

            # Try to extract main content area first
            main = (
                soup.find("article")
                or soup.find("main")
                or soup.find(id=re.compile(r"content|main|article", re.I))
                or soup.find(class_=re.compile(r"content|main|article|post", re.I))
                or soup.find("body")
            )

            # Convert to markdown-like plain text
            content = self._html_to_text(main or soup)

            domain = urlparse(url).netloc
            metadata = {
                "scraped_at": datetime.now().isoformat(),
                "original_url": url,
                "title": title,
                "description": description,
                "language": resp.headers.get("Content-Language", "en"),
                "word_count": len(content.split()),
                "character_count": len(content),
                "domain": domain,
                "status_code": resp.status_code,
            }

            return WebPageData(
                url=url,
                title=title or f"Web Page - {domain}",
                content=content,
                metadata=metadata,
                success=True
            )

        except requests.HTTPError as e:
            logger.error(f"HTTP error for {url}: {str(e)}")
            return WebPageData(
                url=url,
                title=f"Error - {urlparse(url).netloc}",
                content="",
                metadata={"error": str(e), "scraped_at": datetime.now().isoformat()},
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return WebPageData(
                url=url,
                title=f"Error - {urlparse(url).netloc}",
                content="",
                metadata={"error": str(e), "scraped_at": datetime.now().isoformat()},
                success=False,
                error=str(e)
            )

    def _html_to_text(self, element) -> str:
        lines = []
        for tag in element.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "td", "th"]
        ):
            text = tag.get_text(separator=" ", strip=True)
            if not text:
                continue
            name = tag.name
            if name in ("h1", "h2"):
                lines.append(f"\n# {text}\n")
            elif name in ("h3", "h4"):
                lines.append(f"\n## {text}\n")
            elif name in ("h5", "h6"):
                lines.append(f"\n### {text}\n")
            elif name == "li":
                lines.append(f"- {text}")
            elif name in ("blockquote",):
                lines.append(f"> {text}")
            else:
                lines.append(text)

        content = "\n".join(lines)
        # Collapse 3+ blank lines to 2
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    def _create_chunks_from_web_content(
        self,
        page_data: WebPageData,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[DocumentChunk]:
        if not page_data.success or not page_data.content.strip():
            logger.warning(f"No content to process for {page_data.url}")
            return []

        chunks = []
        content = page_data.content
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            if end < len(content):
                last_double_newline = content.rfind("\n\n", start, end)
                if last_double_newline > start + chunk_size * 0.3:
                    end = last_double_newline + 2
                else:
                    last_period = content.rfind(".", start, end)
                    if last_period > start + chunk_size * 0.5:
                        end = last_period + 1

            chunk_text = content[start:end].strip()
            if chunk_text:
                chunk_metadata = page_data.metadata.copy()
                chunk_metadata.update({
                    "chunk_character_start": start,
                    "chunk_character_end": end - 1,
                    "url_fragment": f"{page_data.url}#chunk-{chunk_index}"
                })
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source_file=page_data.title,
                    source_type="web",
                    page_number=None,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end - 1,
                    metadata=chunk_metadata
                ))
                chunk_index += 1

            start = max(start + chunk_size - chunk_overlap, end)

        return chunks

    def batch_scrape_urls(
        self,
        urls: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        delay_between_requests: float = 1.0
    ) -> List[List[DocumentChunk]]:
        all_chunks = []
        for i, url in enumerate(urls):
            try:
                chunks = self.scrape_url(url, chunk_size, chunk_overlap)
                all_chunks.append(chunks)
                logger.info(f"Scraped {url}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
                all_chunks.append([])

            if i < len(urls) - 1:
                time.sleep(delay_between_requests)

        total = sum(len(c) for c in all_chunks)
        logger.info(f"Batch complete: {total} total chunks from {len(urls)} URLs")
        return all_chunks

    def get_url_preview(self, url: str) -> Dict[str, Any]:
        try:
            page_data = self._fetch_and_parse(url, timeout=15)
            content = page_data.content
            return {
                "url": url,
                "title": page_data.title,
                "description": page_data.metadata.get("description", ""),
                "word_count": page_data.metadata.get("word_count", 0),
                "character_count": page_data.metadata.get("character_count", 0),
                "domain": page_data.metadata.get("domain", ""),
                "content_preview": content[:500] + "..." if len(content) > 500 else content,
                "language": page_data.metadata.get("language", "unknown")
            }
        except Exception as e:
            logger.error(f"Error getting URL preview: {str(e)}")
            return {"error": str(e)}

    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


if __name__ == "__main__":
    scraper = WebScraper()

    try:
        test_url = "https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag"
        preview = scraper.get_url_preview(test_url)
        print(f"URL Preview: {preview}")

        chunks = scraper.scrape_url(test_url)
        print(f"\nGenerated {len(chunks)} chunks")

        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Content: {chunk.content[:200]}...")
            print(f"Source: {chunk.source_file}")

    except Exception as e:
        print(f"Error: {e}")