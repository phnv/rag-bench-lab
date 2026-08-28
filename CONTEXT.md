# RAG Bench Lab — Core Guidelines and Context

## Agent Guidelines & Non-Goals
This project evaluates RAG approaches (retrieval independently from generation) using a single plug-pluggable RAG pipeline.
**Strict Non-Goals (DO NOT IMPLEMENT):**
- GraphRAG
- Agentic RAG (LangGraph, tool-calling agents)
- MLflow experiment tracking
- Multiple generation models
- Prompt engineering optimization
- Custom knowledge bases
- Hyperparameter grid search
- Production deployment

All work must align with the architectural decisions in `docs/adr/`.

---

## Glossary

A domain vocabulary for the RAG Benchmark Lab project. Implementation details live elsewhere.

---

## Core Entities

**Experiment** — A single, named execution of one Recipe against one Scenario. Produces a set of Metrics. Immutable once run; results are stored in the Results Store.

**Recipe** — A fully specified configuration of a RAG pipeline: one choice per stage (Query Transformation, Retrieval, Fusion, Reranking, Context Selection, Generation). Not every stage must be filled — `null` is a valid choice for optional stages.

**Scenario** — One of the four benchmark experiment conditions defined for this project: Technical × Single-hop, Technical × Multi-hop, Wikipedia × Single-hop, Wikipedia × Multi-hop. Each Scenario maps to a specific RagWorkbench dataset.

**Pipeline** — The ordered sequence of stages that transforms a Question into a generated Answer with retrieved Context. Not every Recipe activates every stage.

**Stage** — One discrete, swappable slot in the Pipeline. Stages: Query Transformation → Retrieval → Fusion → Reranking → Context Selection → Generation.

**Baseline** — An Experiment using the simplest valid Recipe for a stage. The retrieval baseline ladder is: No RAG → BM25 → Dense → Hybrid → Hybrid + Reranker.

**Recipe Matrix** — The enumerated design space of all technically valid Stage combinations. A feasibility map, not an execution plan.

**Evaluation Decision Tree** — The controlled funnel that selects which Recipes to execute in full, driven by empirical results from simpler Experiments.

---

## Pipeline Stages

**Query Transformation** — The strategy applied to the raw Question before retrieval. Options: Original (passthrough), Multi-query, HyDE, Step-back, Decomposition.

**Retrieval** — The mechanism that fetches candidate documents from a Corpus. Options: BM25 (lexical), Dense (vector/semantic), Hybrid (BM25 + Dense concurrently).

**Fusion** — The algorithm that merges multiple ranked candidate lists into a single list. Option: RRF (Reciprocal Rank Fusion). Only relevant when Retrieval produces multiple lists (Hybrid) or Query Transformation produces multiple queries.

**Reranking** — Post-retrieval re-scoring of a candidate pool to improve precision. Options: None, Cross-encoder.

**Context Selection** — The strategy for constructing the final context window from ranked candidates. Options: Top-K, Adjacent-chunk expansion, Parent-context retrieval, Deduplication, Compression.

**Generation** — The LLM call that produces the final Answer from the Context and Question. Fixed across all comparable Experiments; not an experimental variable.

---

## Evaluation

**Retrieval Metric** — A metric that scores retrieval quality without invoking an LLM. Computed against benchmark relevance judgments. Examples: Recall@K, MRR, nDCG, MAP.

**RAG Metric** — A metric that scores end-to-end RAG quality; requires a generated Answer. Examples: faithfulness, context precision, context recall, answer relevance, answer correctness.

**Engineering Metric** — A non-quality metric measuring cost, latency, or resource use. Examples: end-to-end latency, token usage, estimated API cost.

**Retrieval@K** — The fraction of relevant documents retrieved in the top-K results. Also called Recall@K in this codebase (consistent with RagWorkbench terminology).

**MRR** — Mean Reciprocal Rank: the average of the reciprocal of the rank of the first relevant document.

**Faithfulness** — A RAG Metric measuring whether the generated Answer is grounded in the retrieved Context (not hallucinated).

---

## Infrastructure

**RagWorkbench** — The IBM open-source Python framework (`ragworkbench` on PyPI) used as the benchmark and experiment foundation. Provides: DataLoader, IngestPipeline, InferencePipeline, Evaluator, Experiment, caching, and Unitxt integration.

**Unitxt** — The evaluation metrics library integrated into RagWorkbench. Provides Retrieval@K, MRR, MAP, BERT Score, LLM-as-a-Judge, and others.

**Ragas** — An independent RAG-specific evaluation library used alongside Unitxt for faithfulness, context precision, context recall, and answer relevance metrics.

**LiteLLM** — The LLM proxy layer used by RagWorkbench for model-agnostic LLM calls and cost tracking.

**Results Store** — The persistent store for all Experiment outputs: retrieved candidates, generated answers, Metric values, and engineering metrics. Implemented as filesystem JSON/CSV (via RagWorkbench caching + project output layer).

**Cache** — The intermediate-artifact caching layer. Organized by pipeline stage: Dataset → Query Transformation → Retrieval → Reranking → Context → Generation → Evaluation. Prevents redundant API calls across Experiment runs.

---

## Dataset Concepts

**Benchmark** — The QA dataset that provides Questions, Ground Truth answers, and (where available) relevance judgments. Loaded via RagWorkbench DataLoader.

**Corpus** — The document collection indexed by the Ingest Pipeline. Retrieved from during inference.

**Development Subset** — ~30 deterministically sampled Questions per Scenario, used during pipeline development and debugging.

**Evaluation Subset** — ~100 deterministically sampled, frozen Questions per Scenario, used for all comparative Experiments. Must not change once comparison begins.

**Multi-hop** — A QA task requiring evidence from more than one document or multiple reasoning steps to answer correctly.

**Single-hop** — A QA task answerable from a single retrieved document or passage.

---

## Topology

**Single-hop topology** — `query → single retrieval operation → context → answer`

**Multi-hop topology** — `query → decomposition/iteration → multiple retrieval operations → evidence aggregation → context → answer`

**Decomposition** — A Query Transformation that splits a compound Question into sub-questions, each retrieved independently, then aggregated. Primarily relevant to Multi-hop topology.

**Iterative Retrieval** — A Multi-hop strategy where the LLM identifies missing evidence after an initial retrieval and triggers a second retrieval round.
