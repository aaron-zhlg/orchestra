"""A two-subagent medical research system built on orchestra.

* :class:`~examples.medical.literature.LiteratureAgent` — PubMed literature.
* :class:`~examples.medical.trials.ClinicalTrialsAgent`  — ClinicalTrials.gov.

Wire them under the orchestrator::

    from orchestra import Orchestrator
    from examples.medical import LiteratureAgent, ClinicalTrialsAgent

    lead = Orchestrator([LiteratureAgent, ClinicalTrialsAgent])
    report = lead.run("Do GLP-1 agonists reduce MACE in type 2 diabetes?")
    print(report.answer)
"""

from examples.medical.literature import LiteratureAgent, PubMedTools
from examples.medical.trials import ClinicalTrialsAgent, ClinicalTrialsTools

__all__ = [
    "LiteratureAgent",
    "PubMedTools",
    "ClinicalTrialsAgent",
    "ClinicalTrialsTools",
]
