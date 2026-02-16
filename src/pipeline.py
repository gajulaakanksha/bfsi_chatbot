"""3-Tier Response Pipeline orchestrated by LangGraph.

State graph:
  START  ->  guardrail_check
  guardrail_check  ->  dataset_match  (if valid)
  guardrail_check  ->  END            (if rejected)
  dataset_match    ->  END            (if match found)
  dataset_match    ->  slm_generate   (if no match)
  slm_generate     ->  rag_augment    (if RAG threshold crossed)
  slm_generate     ->  post_process   (otherwise)
  rag_augment      ->  post_process
  post_process     ->  END
"""
import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.5"))


# ── State schema ──────────────────────────────────────────────────────
class PipelineState(TypedDict):
    query: str
    response: str
    tier_used: str          # "dataset" | "slm" | "rag" | "guardrail"
    dataset_score: float
    rag_score: float
    rag_context: str
    is_valid: bool
    rejection_reason: str


# ── Pipeline Builder ──────────────────────────────────────────────────
class BFSIPipeline:
    """Build and run the 3-tier LangGraph pipeline."""

    def __init__(self, dataset_matcher, slm_engine, rag_engine, guardrails):
        self.dataset_matcher = dataset_matcher
        self.slm_engine = slm_engine
        self.rag_engine = rag_engine
        self.guardrails = guardrails
        self.graph = self._build_graph()

    # ── Node functions ────────────────────────────────────────────────
    def _guardrail_check(self, state: PipelineState) -> PipelineState:
        is_valid, reason = self.guardrails.check_query(state["query"])
        state["is_valid"] = is_valid
        state["rejection_reason"] = reason
        if not is_valid:
            state["response"] = reason
            state["tier_used"] = "guardrail"
        return state

    def _dataset_match(self, state: PipelineState) -> PipelineState:
        answer, score = self.dataset_matcher.search(state["query"])
        state["dataset_score"] = score
        if answer is not None:
            state["response"] = answer
            state["tier_used"] = "dataset"
        return state

    def _slm_generate(self, state: PipelineState) -> PipelineState:
        # Heuristic: Skip RAG for creative/generative tasks to avoid context constraining the output
        creative_prefixes = ("write", "draft", "compose", "generate", "suggest", "create")
        is_creative_task = state["query"].lower().strip().startswith(creative_prefixes)

        # Try RAG retrieval first to see if we should augment (unless it's a creative task)
        if self.rag_engine is not None and not is_creative_task:
            chunks = self.rag_engine.retrieve(state["query"], k=3)
            if chunks and chunks[0]["score"] <= RAG_RELEVANCE_THRESHOLD:
                # Low distance = high relevance in ChromaDB
                context = self.rag_engine.get_context_string(state["query"])
                state["rag_context"] = context
                state["rag_score"] = chunks[0]["score"]
                response = self.slm_engine.generate(
                    state["query"], rag_context=context
                )
                state["response"] = response
                state["tier_used"] = "rag"
                return state

        # Pure SLM generation (no RAG context)
        response = self.slm_engine.generate(state["query"])
        state["response"] = response
        state["tier_used"] = "slm"
        return state

    def _post_process(self, state: PipelineState) -> PipelineState:
        state["response"] = self.guardrails.sanitise_response(state["response"])
        return state

    # ── Routing functions ─────────────────────────────────────────────
    def _route_after_guardrail(self, state: PipelineState) -> str:
        return "dataset_match" if state["is_valid"] else "end"

    def _route_after_dataset(self, state: PipelineState) -> str:
        return "end" if state.get("tier_used") == "dataset" else "slm_generate"

    # ── Graph construction ────────────────────────────────────────────
    def _build_graph(self):
        builder = StateGraph(PipelineState)

        # Add nodes
        builder.add_node("guardrail_check", self._guardrail_check)
        builder.add_node("dataset_match", self._dataset_match)
        builder.add_node("slm_generate", self._slm_generate)
        builder.add_node("post_process", self._post_process)

        # Set entry point
        builder.set_entry_point("guardrail_check")

        # Conditional edges
        builder.add_conditional_edges(
            "guardrail_check",
            self._route_after_guardrail,
            {"dataset_match": "dataset_match", "end": END},
        )
        builder.add_conditional_edges(
            "dataset_match",
            self._route_after_dataset,
            {"slm_generate": "slm_generate", "end": END},
        )

        # Direct edges
        builder.add_edge("slm_generate", "post_process")
        builder.add_edge("post_process", END)

        return builder.compile()

    # ── Public API ────────────────────────────────────────────────────
    def run(self, query: str) -> dict:
        """Execute the pipeline and return the final state."""
        initial_state: PipelineState = {
            "query": query,
            "response": "",
            "tier_used": "",
            "dataset_score": 0.0,
            "rag_score": 0.0,
            "rag_context": "",
            "is_valid": True,
            "rejection_reason": "",
        }
        result = self.graph.invoke(initial_state)
        return result
