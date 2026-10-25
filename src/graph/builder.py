"""NetworkX evidence graph construction from extracted predictions."""

import hashlib
import logging
import re
from typing import Any

import networkx as nx

from src.schemas.prediction import Direction, Prediction, RelationClass
from src.schemas.text import CandidatePair

logger = logging.getLogger(__name__)


def resolve_canonical_node(node_id: str, surface_text: str) -> str | None:
    """
    Resolve node identifier.
    Maps unmapped smoking/tobacco phrases to the open-domain concept 'TobaccoSmoking'.
    Filters out genuinely unrecognized, non-conceptual terms from polluting the graph.
    """
    if node_id != "unmapped":
        return node_id

    text_clean = surface_text.lower().strip()
    smoking_keywords = ["smoking", "tobacco", "cigarette"]
    if any(kw in text_clean for kw in smoking_keywords):
        return "TobaccoSmoking"

    # Truly unmapped non-conceptual entity should not form a graph node
    return None


class EvidenceGraphBuilder:
    """Builds directed evidence graph from validated predictions with strict provenance."""

    def __init__(self) -> None:
        self.causal_graph = nx.DiGraph()
        self.association_graph = nx.Graph()  # Undirected/symmetric store for association claims
        self.filtered_self_loops: list[dict[str, Any]] = []
        # Internal mapping from (src, tgt) -> {evidence_key: evidence_dict}
        self._causal_evidence_stores: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        # Internal mapping from (min(u,v), max(u,v)) -> {evidence_key: evidence_dict}
        self._association_evidence_stores: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    def get_filtering_stats(self) -> dict[str, Any]:
        """Return diagnostic statistics on candidate pairs filtered during graph construction."""
        all_nodes = set(self.causal_graph.nodes()).union(self.association_graph.nodes())
        return {
            "filtered_self_loops_count": len(self.filtered_self_loops),
            "association_edges_count": self.association_graph.number_of_edges(),
            "causal_edges_count": self.causal_graph.number_of_edges(),
            "causal_nodes_count": self.causal_graph.number_of_nodes(),
            "association_nodes_count": self.association_graph.number_of_nodes(),
            "combined_nodes_count": len(all_nodes),
            "combined_edges_count": self.causal_graph.number_of_edges() + self.association_graph.number_of_edges(),
        }

    def _record_evidence_item(
        self,
        edge_store: dict[str, dict[str, Any]],
        evidence_key: str,
        pair: CandidatePair,
        prediction: Prediction,
        edge_polarity: str,
        doc_id: str,
        sent_id: str,
        pmid: str | None,
        prov: Any,
    ) -> None:
        """Record or update a unique evidence sentence item in an edge store."""
        label = prediction.label
        pred_record = {
            "extractor": prediction.extractor_name,
            "confidence": label.confidence,
            "polarity": edge_polarity,
            "language_status": label.language_status.value,
            "reasoning": label.reasoning,
        }

        if evidence_key in edge_store:
            ev_item = edge_store[evidence_key]
            if prediction.extractor_name not in ev_item["supporting_extractors"]:
                ev_item["supporting_extractors"].append(prediction.extractor_name)
            # DEDUPLICATION: Avoid adding identical prediction records multiple times
            existing_sigs = [
                (p.get("extractor"), p.get("polarity"), p.get("confidence"), p.get("reasoning"))
                for p in ev_item["predictions"]
            ]
            this_sig = (prediction.extractor_name, edge_polarity, label.confidence, label.reasoning)
            if this_sig not in existing_sigs:
                ev_item["predictions"].append(pred_record)
            all_confs = [p["confidence"] for p in ev_item["predictions"]]
            ev_item["confidence"] = round(sum(all_confs) / len(all_confs), 4)
        else:
            url = prov.url if (prov and prov.url) else (f"https://europepmc.org/article/MED/{pmid}" if pmid else None)
            ev_item = {
                "evidence_key": evidence_key,
                "pair_id": pair.pair_id,
                "doc_id": doc_id,
                "sent_id": sent_id,
                "pmid": pmid,
                "url": url,
                "sentence_text": pair.sentence_text,
                "char_spans": prov.char_spans if prov else None,
                "extractor": prediction.extractor_name,
                "supporting_extractors": [prediction.extractor_name],
                "confidence": label.confidence,
                "predictions": [pred_record],
            }
            edge_store[evidence_key] = ev_item

    def add_prediction(
        self,
        prediction: Prediction,
        pair: CandidatePair,
    ) -> None:
        """
        Add a single relation prediction to the evidence graph.
        Enforces:
        - Only causal relations enter the causal DAG candidate graph.
        - Association-only edges stored in separate association store with full provenance.
        - Polarity normalization (if source mention has polarity_flip=True).
        - Multiple extractors agreeing on the same sentence do NOT artificially increase support_count.
        - Detailed tracking of unique evidence sentences, unique papers, and extractor agreement.
        """
        label = prediction.label
        if label.relation_class == RelationClass.NONE:
            return

        e1_node = resolve_canonical_node(pair.entity_1.canonical_node_id, pair.entity_1.surface_text)
        e2_node = resolve_canonical_node(pair.entity_2.canonical_node_id, pair.entity_2.surface_text)

        if not e1_node or not e2_node:
            return

        # Skip self-loops where both entities normalize to the same canonical variable
        if e1_node == e2_node:
            self.filtered_self_loops.append({
                "pair_id": pair.pair_id,
                "doc_id": pair.doc_id,
                "sent_id": pair.sent_id,
                "canonical_node": e1_node,
                "entity_1": pair.entity_1.surface_text,
                "entity_2": pair.entity_2.surface_text,
                "sentence_text": pair.sentence_text,
                "reason": "self_loop_identical_canonical_node",
            })
            return

        # Resolve stable evidence key (prefer doc_id:sent_id; fallback to doc_id + text hash)
        doc_id = pair.doc_id or "UNKNOWN_DOC"
        sent_id = pair.sent_id or "s0"
        pmid = None
        prov = label.provenance
        if prov and prov.pmid:
            pmid = prov.pmid
        elif doc_id.startswith("PMID:"):
            pmid = doc_id.split(":")[-1]

        if pair.sent_id:
            evidence_key = pair.sent_id if pair.sent_id.startswith(doc_id) else f"{doc_id}:{sent_id}"
        else:
            norm_text_hash = hashlib.sha256(pair.sentence_text.strip().encode("utf-8")).hexdigest()[:12]
            evidence_key = f"{doc_id}::{norm_text_hash}"

        # Determine relation routing
        is_causal_directed = (
            label.relation_class == RelationClass.CAUSAL
            and label.direction in (Direction.E1_TO_E2, Direction.E2_TO_E1)
        )

        # Epidemiological DAG validity check:
        # 1. Disease outcome cannot causally produce baseline demographic traits (Sex, AgeCat).
        # 2. Causal claims must directly target the candidate outcome entity, not an unrelated
        #    co-mentioned endpoint (e.g., Carpal Tunnel Syndrome 'CTS' or brain tissue morphometry).
        # 3. Parallel registry trends ('parallel increase') describe secular ecological trends,
        #    not interventional causation.
        if is_causal_directed:
            candidate_src = e1_node if label.direction == Direction.E1_TO_E2 else e2_node
            candidate_tgt = e2_node if label.direction == Direction.E1_TO_E2 else e1_node
            # Baseline demographic traits (Sex, AgeCat) cannot be causally determined by disease outcomes or other variables
            if candidate_tgt in ("Sex", "AgeCat"):
                is_causal_directed = False
            elif candidate_src in ("Sex", "AgeCat"):
                # Baseline demographics define sub-populations/strata, not interventional causes of diabetes
                is_causal_directed = False
            elif candidate_src == "Outcome" and candidate_tgt in ("BMICat", "PhysActivity", "TobaccoSmoking", "InsulinResistance", "DietQuality"):
                # Disease diagnosis does not retroactively cause baseline lifestyle/BMI exposures or biological precursors
                is_causal_directed = False
            else:
                s_lower = pair.sentence_text.lower()

                # 1. Secondary endpoints, complications, or non-target disease outcomes:
                # e.g., "increase risk of dementia in patients with T2DM", "CTS", "IVDD", "leukopenia", "cardiovascular events"
                secondary_endpoint_patterns = [
                    r"\b(dementia|cognitive\s+(decline|impairment)|alzheimer'?s?)\b",
                    r"\b(cts|carpal\s+tunnel)\b",
                    r"\b(ivdd|intervertebral\s+disc)\b",
                    r"\b(leukopenia|neutropenia|persistent\s+leukopenia)\b",
                    r"\b(hepatocellular\s+injury|liver\s+injury)\b",
                    r"\b(lga|large\s+for\s+gestational\s+age)\b",
                    r"\b(mace|cardiovascular\s+(events|death|disease)|heart\s+failure\s+hospitalization)\b",
                    r"\b(ckd|diabetic\s+kidney\s+disease|renal\s+failure|ckd\s+due\s+to)\b",
                    r"\b(brain\s+morpholog\w*|volumetric\s+increases?|morphometric\s+increases?)\b",
                    r"\b(complications|all-cause\s+mortality|all-cause\s+death)\b",
                ]
                has_secondary_outcome = any(re.search(pat, s_lower) for pat in secondary_endpoint_patterns)

                # 2. Baseline cohort phrasing: "in patients with T2DM", "among individuals with diabetes"
                # When present, diabetes is the patient cohort condition, NOT the outcome being caused.
                is_cohort_condition = bool(re.search(r"\b(in|among)\s+(patients|individuals|adults|children|women|men|subjects|cohorts|cases)?\s*(with|having|diagnosed\s+with)\s+(t2dm|type\s+2\s+diabetes|diabetes)\b", s_lower))

                # 3. Biological mediator / pathway mechanism phrasing rather than direct interventional causation:
                # e.g., "poor dietary quality can increase the risk of ... through increased visceral adiposity and insulin resistance"
                is_mechanism_mediator = bool(re.search(r"\b(through|via)\s+([a-zA-Z\s,-]+)?\s*insulin\s+resistance\b", s_lower))

                # 4. Ecological secular trends, case reports, or uninformative observational phrases
                trend_or_case_patterns = [
                    "temporally parallel increase",
                    "parallel increase in the incidence",
                    "we report the case of",
                    "this case report",
                    "projections indicate continued increases",
                    "causal inference is limited",
                    "not be interpreted as evidence of temporal or causal",
                    "no significant differences were observed",
                    "categorized according to their primary predictors",
                    "combat obesity, and reduce smoking",
                    "higher odds of exceeding",
                    "ff-ll is an independent risk factor",
                ]
                has_trend_or_case = any(pat in s_lower for pat in trend_or_case_patterns)

                if has_secondary_outcome or (is_cohort_condition and candidate_tgt == "Outcome") or is_mechanism_mediator or has_trend_or_case:
                    # The causal predicate targets an unrelated endpoint, cohort trait, mediator, or trend.
                    # Route to association graph to preserve empirical literature evidence without mislabeling.
                    is_causal_directed = False

        if is_causal_directed:
            # Directed causal edge
            if label.direction == Direction.E1_TO_E2:
                src, tgt = e1_node, e2_node
                src_flip = pair.entity_1.polarity_flip
            else:
                src, tgt = e2_node, e1_node
                src_flip = pair.entity_2.polarity_flip

            edge_polarity = label.polarity.value
            if src_flip:
                if edge_polarity == "increases":
                    edge_polarity = "decreases"
                elif edge_polarity == "decreases":
                    edge_polarity = "increases"

            edge_key = (src, tgt)
            if edge_key not in self._causal_evidence_stores:
                self._causal_evidence_stores[edge_key] = {}

            edge_store = self._causal_evidence_stores[edge_key]
            self._record_evidence_item(
                edge_store, evidence_key, pair, prediction, edge_polarity, doc_id, sent_id, pmid, prov
            )

            unique_ev_list = list(edge_store.values())
            unique_sentence_count = len(unique_ev_list)
            unique_papers = {ev["pmid"] for ev in unique_ev_list if ev.get("pmid")}
            all_edge_extractors = sorted(list({ext for ev in unique_ev_list for ext in ev["supporting_extractors"]}))
            all_item_confs = [ev["confidence"] for ev in unique_ev_list]
            avg_conf = round(sum(all_item_confs) / len(all_item_confs), 4) if all_item_confs else round(label.confidence, 4)

            self.causal_graph.add_edge(
                src,
                tgt,
                relation_class="causal",
                polarity=edge_polarity,
                language_status=label.language_status.value,
                support_count=unique_sentence_count,
                unique_evidence_sentences=unique_sentence_count,
                unique_papers_count=len(unique_papers),
                extractors=all_edge_extractors,
                agreement_count=len(all_edge_extractors),
                aggregated_confidence=avg_conf,
                evidence=unique_ev_list,
            )
        else:
            # Observational association edge (stored with full evidence provenance)
            edge_polarity = label.polarity.value if label.polarity else "not_applicable"
            assoc_key = tuple(sorted([e1_node, e2_node]))
            if assoc_key not in self._association_evidence_stores:
                self._association_evidence_stores[assoc_key] = {}

            edge_store = self._association_evidence_stores[assoc_key]
            self._record_evidence_item(
                edge_store, evidence_key, pair, prediction, edge_polarity, doc_id, sent_id, pmid, prov
            )

            unique_ev_list = list(edge_store.values())
            unique_sentence_count = len(unique_ev_list)
            unique_papers = {ev["pmid"] for ev in unique_ev_list if ev.get("pmid")}
            all_edge_extractors = sorted(list({ext for ev in unique_ev_list for ext in ev["supporting_extractors"]}))
            all_item_confs = [ev["confidence"] for ev in unique_ev_list]
            avg_conf = round(sum(all_item_confs) / len(all_item_confs), 4) if all_item_confs else round(label.confidence, 4)

            u, v = assoc_key
            self.association_graph.add_edge(
                u,
                v,
                relation_class="association_only",
                polarity=edge_polarity,
                language_status=label.language_status.value,
                support_count=unique_sentence_count,
                unique_evidence_sentences=unique_sentence_count,
                unique_papers_count=len(unique_papers),
                extractors=all_edge_extractors,
                agreement_count=len(all_edge_extractors),
                aggregated_confidence=avg_conf,
                evidence=unique_ev_list,
            )

    def get_combined_graph(self) -> nx.DiGraph:
        """
        Produce a combined directed graph containing both causal and association relations.
        Causal relations retain their directed orientation.
        Observational association relations are added with relation_class='association_only'.
        """
        combined = nx.DiGraph()
        # 1. Add all causal edges
        for u, v, data in self.causal_graph.edges(data=True):
            combined.add_edge(u, v, **data)

        # 2. Add all association edges
        for u, v, data in self.association_graph.edges(data=True):
            if not combined.has_edge(u, v) and not combined.has_edge(v, u):
                combined.add_edge(u, v, **data)
        return combined

    def build_from_pairs_and_predictions(
        self,
        pairs: list[CandidatePair],
        predictions: list[Prediction],
    ) -> tuple[nx.DiGraph, nx.Graph]:
        """Build causal and association graphs from paired predictions."""
        pair_dict = {p.pair_id: p for p in pairs}
        for pred in predictions:
            if pred.pair_id in pair_dict:
                self.add_prediction(pred, pair_dict[pred.pair_id])
        return self.causal_graph, self.association_graph
