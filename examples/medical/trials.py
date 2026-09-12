"""Example subagent: ClinicalTrials.gov search (REST API v2).

A second specialized worker, alongside :mod:`examples.medical.literature`. Same
pattern: a self-contained toolset (:class:`ClinicalTrialsTools`) plus a tiny
:class:`~orchestra.SubAgent` subclass that just names it, describes it, and points
its prompt at those tools.

Reference: ClinicalTrials.gov REST API v2, https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import json
import typing
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from orchestra import SubAgent

API_BASE = "https://clinicaltrials.gov/api/v2"
STUDY_UI = "https://clinicaltrials.gov/study"


class ClinicalTrialsError(RuntimeError):
    """A ClinicalTrials.gov request failed."""


class ClinicalTrialsClient:
    """Minimal stdlib client for the ClinicalTrials.gov v2 REST API."""

    def __init__(self, *, timeout: float = 30.0, max_retries: int = 2, user_agent: str = "orchestra-trials"):
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent

    def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
        url = f"{API_BASE}/{path}?{query}"
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json"}, method="GET"
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    last_error = exc
                    continue
                raise ClinicalTrialsError(f"HTTP {exc.code}: {body[:300]}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    last_error = exc
                    continue
                raise ClinicalTrialsError(f"request failed: {exc.reason}") from exc
        raise ClinicalTrialsError(f"failed after {self.max_retries + 1} attempts: {last_error}")


def _study_digest(study: dict[str, Any]) -> dict[str, Any]:
    """Flatten a v2 study record into the handful of fields agents care about."""
    ps = study.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    status = ps.get("statusModule", {})
    design = ps.get("designModule", {})
    conditions = ps.get("conditionsModule", {})
    arms = ps.get("armsInterventionsModule", {})
    desc = ps.get("descriptionModule", {})
    sponsor = ps.get("sponsorCollaboratorsModule", {})
    outcomes = ps.get("outcomesModule", {})

    nct_id = ident.get("nctId", "")
    interventions = [
        f"{i.get('type', '')}: {i.get('name', '')}".strip(": ")
        for i in arms.get("interventions", [])
    ]
    primary_outcomes = [o.get("measure", "") for o in outcomes.get("primaryOutcomes", [])]
    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle", ""),
        "status": status.get("overallStatus", ""),
        "phases": design.get("phases", []),
        "study_type": design.get("studyType", ""),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "conditions": conditions.get("conditions", []),
        "interventions": interventions,
        "primary_outcomes": primary_outcomes,
        "lead_sponsor": (sponsor.get("leadSponsor") or {}).get("name", ""),
        "start_date": (status.get("startDateStruct") or {}).get("date", ""),
        "completion_date": (status.get("completionDateStruct") or {}).get("date", ""),
        "brief_summary": desc.get("briefSummary", ""),
        "url": f"{STUDY_UI}/{nct_id}" if nct_id else "",
    }


class ClinicalTrialsTools:
    """ClinicalTrials.gov endpoints exposed as agent-callable tools.

    Implements the toolset protocol: ``as_tools()`` + ``sources()`` (every NCT id
    touched, so the orchestrator can build citations automatically).
    """

    def __init__(self, client: ClinicalTrialsClient | None = None):
        self.client = client or ClinicalTrialsClient()
        self.seen_ncts: list[str] = []

    def _record(self, ncts: typing.Iterable[str]) -> None:
        for n in ncts:
            if n and n not in self.seen_ncts:
                self.seen_ncts.append(n)

    def search_trials(
        self,
        query: str,
        condition: str | None = None,
        intervention: str | None = None,
        status: str | None = None,
        phase: str | None = None,
        max_results: int = 15,
    ) -> dict[str, Any]:
        """Search ClinicalTrials.gov for studies matching the criteria.

        Args:
            query: Free-text query, e.g. 'semaglutide obesity cardiovascular'.
            condition: Optional disease/condition filter, e.g. 'obesity'.
            intervention: Optional intervention/drug filter, e.g. 'semaglutide'.
            status: Optional recruitment status filter, e.g. 'RECRUITING',
                'COMPLETED', 'ACTIVE_NOT_RECRUITING', 'TERMINATED'.
            phase: Optional phase filter, e.g. 'PHASE3'.
            max_results: Maximum number of studies to return (1-50).
        """
        max_results = max(1, min(int(max_results), 50))
        params: dict[str, Any] = {
            "query.term": query or None,
            "query.cond": condition,
            "query.intr": intervention,
            "pageSize": max_results,
            "countTotal": "true",
        }
        filters: list[str] = []
        if status:
            filters.append(f"AREA[OverallStatus]{status.upper()}")
        if phase:
            filters.append(f"AREA[Phase]{phase.upper()}")
        if filters:
            params["filter.advanced"] = " AND ".join(filters)

        data = self.client.get("studies", params)
        studies = [_study_digest(s) for s in data.get("studies", [])]
        self._record(s["nct_id"] for s in studies)
        return {
            "query": query,
            "total_count": data.get("totalCount"),
            "returned": len(studies),
            "studies": studies,
        }

    def get_trial(self, nct_id: str) -> dict[str, Any]:
        """Fetch the full digest for a single trial by its NCT identifier.

        Args:
            nct_id: A ClinicalTrials.gov identifier, e.g. 'NCT03548935'.
        """
        nct = nct_id.strip().upper()
        data = self.client.get(f"studies/{nct}", {})
        self._record([nct])
        return _study_digest(data)

    def as_tools(self) -> list[typing.Callable[..., Any]]:
        return [self.search_trials, self.get_trial]

    def sources(self) -> list[str]:
        """Every NCT id touched this run (already usable as a citation token)."""
        return list(self.seen_ncts)


TRIALS_INSTRUCTIONS = """\
You are a clinical-trials research specialist operating as an autonomous \
subagent. Given an objective, you search ClinicalTrials.gov and report on the \
relevant registered studies, grounding every statement in retrieved trials.

Tools:
- search_trials: find studies by free text, condition, intervention, status, or phase.
- get_trial: pull full details for a specific NCT id.

Method:
1. START WIDE, THEN NARROW. Begin with one broad query, inspect the hit count, \
then narrow with condition/intervention/phase/status filters as needed.
2. Prefer interventional Phase 3/4 and completed studies when the objective is \
about efficacy; include recruiting trials when asked about the pipeline.
3. Read brief summaries and primary outcomes of the most relevant trials.

Effort budget: a typical objective needs ~3-10 tool calls. STOP once you can \
answer — do not enumerate every registered study.

Final answer:
- A direct answer to the objective, then bullets of key trials each ending with \
its identifier like (NCT03548935).
- Note status, phase, enrollment, sponsor, and primary outcomes where relevant.
- End with a "Trials" list: [NCT id] Title — status, phase. URL.
Only cite trials you actually retrieved.
"""


class ClinicalTrialsAgent(SubAgent):
    """A subagent that searches and summarizes registered clinical trials."""

    name = "clinical_trials"
    description = (
        "Searches ClinicalTrials.gov for registered clinical trials. Best for "
        "questions about ongoing/completed studies, trial phases, interventions "
        "under investigation, recruitment status, enrollment, sponsors, and "
        "primary outcome measures. Not for published results/abstracts (use "
        "pubmed_literature for those)."
    )
    instructions = TRIALS_INSTRUCTIONS

    def create_tools(self) -> ClinicalTrialsTools:
        return ClinicalTrialsTools()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ClinicalTrials.gov subagent (orchestra example).")
    parser.add_argument("objective", nargs="+", help="Search objective.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    agent = ClinicalTrialsAgent(model=args.model, verbose=not args.quiet)
    result = agent.run(" ".join(args.objective))
    print("\n" + result.findings)
    print(f"\n[touched {len(result.sources)} trials]")


if __name__ == "__main__":
    main()
