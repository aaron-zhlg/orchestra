"""Example subagent: PubMed literature search (NCBI E-utilities).

This shows the *payoff* of the framework. All the domain work lives in a
self-contained toolset (:class:`PubMedTools`); the subagent itself is barely more
than a name, a description, a prompt, and "here are my tools":

    class LiteratureAgent(SubAgent):
        name = "pubmed_literature"
        description = "..."
        instructions = LITERATURE_INSTRUCTIONS
        def create_tools(self):
            return PubMedTools()

Because ``PubMedTools`` exposes both ``as_tools()`` and ``sources()``, the base
:class:`~orchestra.SubAgent` automatically wires the tools into the loop and
reports every PMID it touched as a citation.

Reference: E-utilities Quick Start, NCBI Bookshelf NBK25500
https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

from __future__ import annotations

import json
import os
import threading
import time
import typing
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from orchestra import SubAgent

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_UI = "https://pubmed.ncbi.nlm.nih.gov"
TOOL_NAME = "orchestra-litsearch"
MAX_ABSTRACT_CHARS = 2500
_MIN_INTERVAL_NO_KEY = 1.0 / 3
_MIN_INTERVAL_WITH_KEY = 1.0 / 10


class EUtilsError(RuntimeError):
    """An E-utilities request failed."""


class EUtilsClient:
    """Minimal, rate-limited stdlib client for the NCBI E-utilities endpoints."""

    _lock = threading.Lock()
    _last_call = 0.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        email: str | None = None,
        tool: str = TOOL_NAME,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("NCBI_API_KEY") or None
        self.email = email or os.environ.get("NCBI_EMAIL") or None
        self.tool = tool
        self.timeout = timeout
        self.max_retries = max_retries

    def _throttle(self) -> None:
        interval = _MIN_INTERVAL_WITH_KEY if self.api_key else _MIN_INTERVAL_NO_KEY
        with EUtilsClient._lock:
            wait = interval - (time.monotonic() - EUtilsClient._last_call)
            if wait > 0:
                time.sleep(wait)
            EUtilsClient._last_call = time.monotonic()

    def get(self, endpoint: str, params: dict[str, Any]) -> bytes:
        query = {k: v for k, v in params.items() if v is not None}
        query.setdefault("tool", self.tool)
        if self.email:
            query.setdefault("email", self.email)
        if self.api_key:
            query.setdefault("api_key", self.api_key)

        url = f"{EUTILS_BASE}/{endpoint}.fcgi?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, headers={"User-Agent": self.tool}, method="GET")

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    last_error = exc
                    time.sleep(2**attempt)
                    continue
                raise EUtilsError(f"HTTP {exc.code} from {endpoint}: {body[:300]}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    last_error = exc
                    time.sleep(2**attempt)
                    continue
                raise EUtilsError(f"request to {endpoint} failed: {exc.reason}") from exc
        raise EUtilsError(f"{endpoint} failed after {self.max_retries + 1} attempts: {last_error}")

    def get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "retmode": "json"}
        return json.loads(self.get(endpoint, params).decode())


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _parse_pubmed_articles(xml_bytes: bytes) -> list[dict[str, Any]]:
    """Turn an efetch ``PubmedArticleSet`` into a list of structured records."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise EUtilsError(f"could not parse efetch XML: {exc}") from exc

    articles: list[dict[str, Any]] = []
    for art in root.findall(".//PubmedArticle"):
        citation = art.find("MedlineCitation")
        if citation is None:
            continue
        pmid = _text(citation.find("PMID"))
        article = citation.find("Article")
        if article is None:
            continue

        title = _text(article.find("ArticleTitle"))

        abstract_parts: list[str] = []
        for chunk in article.findall("./Abstract/AbstractText"):
            label = chunk.get("Label")
            body = _text(chunk)
            abstract_parts.append(f"{label}: {body}" if label else body)
        abstract = "\n".join(p for p in abstract_parts if p)
        if len(abstract) > MAX_ABSTRACT_CHARS:
            abstract = abstract[:MAX_ABSTRACT_CHARS].rstrip() + " …[truncated]"

        authors: list[str] = []
        for author in article.findall("./AuthorList/Author"):
            last = _text(author.find("LastName"))
            initials = _text(author.find("Initials"))
            collective = _text(author.find("CollectiveName"))
            if last:
                authors.append(f"{last} {initials}".strip())
            elif collective:
                authors.append(collective)

        journal = _text(article.find("./Journal/Title"))
        pubdate = article.find("./Journal/JournalIssue/PubDate")
        year = _text(pubdate.find("Year")) if pubdate is not None else ""
        if not year and pubdate is not None:
            year = _text(pubdate.find("MedlineDate"))[:4]

        doi = ""
        for eid in article.findall("./ELocationID"):
            if eid.get("EIdType") == "doi":
                doi = _text(eid)
                break

        pub_types = [_text(pt) for pt in article.findall("./PublicationTypeList/PublicationType")]

        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract or "(no abstract available)",
                "authors": authors,
                "journal": journal,
                "year": year,
                "doi": doi,
                "publication_types": pub_types,
                "url": f"{PUBMED_UI}/{pmid}/",
            }
        )
    return articles


def _normalize_ids(ids: str | typing.Sequence[str | int]) -> str:
    if isinstance(ids, str):
        parts = [p.strip() for p in ids.replace(",", " ").split()]
    else:
        parts = [str(p).strip() for p in ids]
    clean = [p for p in parts if p]
    if not clean:
        raise ValueError("no PMIDs provided")
    return ",".join(clean)


class PubMedTools:
    """PubMed E-utilities exposed as agent-callable function tools.

    Implements the informal *toolset* protocol the framework understands:
    ``as_tools()`` returns the callables, and ``sources()`` returns every PMID the
    agent touched (so the orchestrator can build citations automatically).
    """

    def __init__(self, client: EUtilsClient | None = None, *, default_db: str = "pubmed"):
        self.client = client or EUtilsClient()
        self.default_db = default_db
        self.seen_pmids: list[str] = []

    def _record_pmids(self, pmids: typing.Iterable[str]) -> None:
        for pmid in pmids:
            if pmid and pmid not in self.seen_pmids:
                self.seen_pmids.append(pmid)

    def search_pubmed(
        self,
        query: str,
        max_results: int = 20,
        sort: typing.Literal["relevance", "pub_date", "most_recent"] = "relevance",
        min_date: str | None = None,
        max_date: str | None = None,
    ) -> dict[str, Any]:
        """Search PubMed and return matching PMIDs.

        Use standard PubMed query syntax, including field tags such as
        ``[tiab]`` (title/abstract), ``[mesh]`` (MeSH term), ``[au]`` (author),
        ``[journal]``, ``[pdat]`` (publication date), and boolean AND/OR/NOT.

        Args:
            query: PubMed query string, e.g. 'semaglutide[tiab] AND weight loss[tiab]'.
            max_results: Maximum number of PMIDs to return (1-100).
            sort: Result ordering. 'relevance', 'pub_date', or 'most_recent'.
            min_date: Optional lower bound on publication date, format YYYY or YYYY/MM/DD.
            max_date: Optional upper bound on publication date, format YYYY or YYYY/MM/DD.
        """
        max_results = max(1, min(int(max_results), 100))
        sort_param = {"relevance": "relevance", "pub_date": "pub+date", "most_recent": "most+recent"}.get(
            sort, "relevance"
        )
        params: dict[str, Any] = {
            "db": self.default_db,
            "term": query,
            "retmax": max_results,
            "sort": sort_param,
        }
        if min_date or max_date:
            params["datetype"] = "pdat"
            params["mindate"] = min_date or "1800"
            params["maxdate"] = max_date or "3000"

        data = self.client.get_json("esearch", params)
        result = data.get("esearchresult", {})
        pmids = result.get("idlist", [])
        self._record_pmids(pmids)
        return {
            "query": query,
            "translated_query": result.get("querytranslation", ""),
            "total_count": int(result.get("count", 0)),
            "returned": len(pmids),
            "pmids": pmids,
            "warnings": result.get("warninglist") or result.get("errorlist"),
        }

    def summarize_articles(self, pmids: str) -> dict[str, Any]:
        """Fetch lightweight citation summaries (title/authors/journal/date) for PMIDs.

        Cheaper than full abstracts; use this to triage which articles are worth
        reading in detail.

        Args:
            pmids: Comma- or space-separated PubMed IDs, e.g. '38000000, 37999999'.
        """
        id_str = _normalize_ids(pmids)
        data = self.client.get_json("esummary", {"db": self.default_db, "id": id_str})
        result = data.get("result", {})
        uids = result.get("uids", [])
        self._record_pmids(uids)
        summaries = []
        for uid in uids:
            doc = result.get(uid, {})
            authors = [a.get("name", "") for a in doc.get("authors", []) if a.get("name")]
            summaries.append(
                {
                    "pmid": uid,
                    "title": doc.get("title", ""),
                    "authors": authors,
                    "journal": doc.get("fulljournalname") or doc.get("source", ""),
                    "pubdate": doc.get("pubdate", ""),
                    "doi": doc.get("elocationid", ""),
                    "url": f"{PUBMED_UI}/{uid}/",
                }
            )
        return {"count": len(summaries), "summaries": summaries}

    def fetch_abstracts(self, pmids: str) -> dict[str, Any]:
        """Fetch full abstracts and metadata for a set of PMIDs.

        Use this on the handful of most relevant PMIDs to read their abstracts
        before writing the final summary.

        Args:
            pmids: Comma- or space-separated PubMed IDs (keep to <= 20 at a time).
        """
        id_str = _normalize_ids(pmids)
        xml_bytes = self.client.get(
            "efetch",
            {"db": self.default_db, "id": id_str, "rettype": "abstract", "retmode": "xml"},
        )
        articles = _parse_pubmed_articles(xml_bytes)
        self._record_pmids(a["pmid"] for a in articles)
        return {"count": len(articles), "articles": articles}

    def find_related_articles(self, pmid: str, max_results: int = 10) -> dict[str, Any]:
        """Find articles related to a given PMID via PubMed's computed neighbors (ELink).

        Args:
            pmid: A single PubMed ID to find neighbors for.
            max_results: Maximum number of related PMIDs to return.
        """
        single = _normalize_ids(pmid).split(",")[0]
        data = self.client.get_json(
            "elink",
            {"dbfrom": "pubmed", "db": "pubmed", "id": single, "cmd": "neighbor"},
        )
        related: list[str] = []
        for linkset in data.get("linksets", []):
            for db in linkset.get("linksetdbs", []):
                if db.get("linkname") in ("pubmed_pubmed", "pubmed_pubmed_reviews"):
                    related.extend(str(x) for x in db.get("links", []))
        related = [r for r in related if r != single][: max(1, int(max_results))]
        self._record_pmids(related)
        return {"source_pmid": single, "count": len(related), "related_pmids": related}

    def as_tools(self) -> list[typing.Callable[..., Any]]:
        return [
            self.search_pubmed,
            self.summarize_articles,
            self.fetch_abstracts,
            self.find_related_articles,
        ]

    def sources(self) -> list[str]:
        """Every PMID touched this run, formatted as citation tokens."""
        return [f"PMID:{p}" for p in self.seen_pmids]


LITERATURE_INSTRUCTIONS = """\
You are a meticulous medical-literature search specialist. You operate as an \
autonomous subagent: given a research goal, you plan and execute PubMed searches, \
read abstracts, and return an evidence-based summary. You MUST ground every claim \
in retrieved articles — never rely on unstated prior knowledge.

Tools available (PubMed via NCBI E-utilities):
- search_pubmed: run a PubMed query, get matching PMIDs and the total hit count.
- summarize_articles: cheap title/author/journal/date triage for PMIDs.
- fetch_abstracts: read the full abstracts of the most relevant PMIDs.
- find_related_articles: expand from a key PMID to its computed neighbors.

Method (loop until confident, then stop):
1. START WIDE, THEN NARROW. Begin with one short, broad query; then progressively \
narrow with field tags ([tiab], [mesh], [pdat]) and boolean operators.
2. Run search_pubmed. If too many hits (>~200) tighten; if too few (0-2) loosen, \
fix spelling, or drop over-specific tags. Iterate.
3. Triage with summarize_articles, then fetch_abstracts for the ~5-12 most \
relevant articles. Prefer systematic reviews, meta-analyses, and RCTs for \
clinical questions.
4. Optionally use find_related_articles to fill a specific gap.

Effort budget: a typical objective needs ~5-12 tool calls. STOP as soon as you \
can answer well; do not enumerate every paper.

Final answer:
- Start with a direct 2-4 sentence answer to the goal.
- Then "Key findings" as bullets, each ending with citations like (PMID: 12345678).
- Note the strength/level of evidence and any conflicts or limitations.
- End with a "References" list: [PMID] Authors (Year). Title. Journal. URL.
Only cite articles you actually retrieved.
"""


class LiteratureAgent(SubAgent):
    """A subagent that searches and summarizes the published medical literature."""

    name = "pubmed_literature"
    description = (
        "Searches the published biomedical literature on PubMed (NCBI). Best for "
        "questions about published findings, clinical evidence, systematic reviews, "
        "meta-analyses, mechanisms, and reading article abstracts. Not for the "
        "status of ongoing/registered trials (use clinical_trials for those)."
    )
    instructions = LITERATURE_INSTRUCTIONS

    def create_tools(self) -> PubMedTools:
        return PubMedTools()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PubMed literature subagent (orchestra example).")
    parser.add_argument("goal", nargs="+", help="Research goal.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    agent = LiteratureAgent(model=args.model, verbose=not args.quiet)
    result = agent.run(" ".join(args.goal))
    print("\n" + result.findings)
    print(f"\n[touched {len(result.sources)} PMIDs]")


if __name__ == "__main__":
    main()
