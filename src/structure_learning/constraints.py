from pgmpy.causal_discovery import ExpertKnowledge

# ============================================================
# ACTG175 VARIABLES BY TEMPORAL STAGE
# ============================================================

# Baseline / pre-treatment variables.
BASELINE_VARIABLES = [
    "age",
    "wtkg",
    "hemo",
    "homo",
    "drugs",
    "karnof",
    "oprior",
    "z30",
    "preanti",
    "race",
    "gender",
    "strat",
    "symptom",
    "cd40",
    "cd80",
]

# Treatment assignment.
TREATMENT_VARIABLES = [
    "trt",
]

# Final outcome.
OUTCOME_VARIABLES = [
    "label",
]


# ============================================================
# TEMPORAL ORDER
# ============================================================

TEMPORAL_ORDER = [
    BASELINE_VARIABLES,
    TREATMENT_VARIABLES,
    OUTCOME_VARIABLES,
]


# ============================================================
# EXPLICITLY FORBIDDEN EDGES
# ============================================================
#
# Temporal ordering already prevents later -> earlier edges.
#
# This explicit prohibition documents the most important
# treatment/outcome restriction:
#
#     label -> trt
#
# is not allowed.
# ============================================================

FORBIDDEN_EDGES = [
    ("label", "trt"),
]


# ============================================================
# REQUIRED EDGES
# ============================================================
#
# We intentionally do NOT force:
#
#     trt -> label
#
# because structure learning should determine whether the
# dependency is supported by the data.
# ============================================================

REQUIRED_EDGES = []


# ============================================================
# BUILD EXPERT KNOWLEDGE
# ============================================================

def build_expert_knowledge() -> ExpertKnowledge:
    """
    Build ACTG175 domain knowledge for causal discovery.

    Temporal structure:

        baseline -> treatment -> outcome

    The ordering restricts impossible directions without
    forcing arbitrary causal relationships.
    """

    return ExpertKnowledge(
        forbidden_edges=FORBIDDEN_EDGES,
        required_edges=REQUIRED_EDGES,
        temporal_order=TEMPORAL_ORDER,
    )