from __future__ import annotations

from abc import ABC, abstractmethod

from parsing_agent.config import WorkflowConfig
from parsing_agent.models import DocumentSource, EvaluationMetrics, JudgeResult, ParseCandidate, RepairAction


class ParserAdapter(ABC):
    name: str

    @abstractmethod
    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        raise NotImplementedError


class CandidateEvaluator(ABC):
    @abstractmethod
    def evaluate(self, source: DocumentSource, candidate: ParseCandidate) -> EvaluationMetrics:
        raise NotImplementedError


class CandidateJudge(ABC):
    @abstractmethod
    def judge(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
    ) -> JudgeResult:
        raise NotImplementedError


class CandidateRepairer(ABC):
    @abstractmethod
    def repair(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
    ) -> tuple[ParseCandidate, list[RepairAction]]:
        raise NotImplementedError
