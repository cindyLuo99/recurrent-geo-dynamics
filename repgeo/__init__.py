"""repgeo: representational-geometry analysis of recurrent vision models.

Module map — one rule per module:

    config.py     model registry: names, colors, plot order, layer conventions
    data.py       stimulus images -> dataloader (used by extraction)
    loading.py    saved activation files -> the features dict
    geometry.py   raw tensor(s) in -> numbers out (the metrics themselves)
    analysis.py   features dict in -> tables / summaries out
    stats.py      anything that returns a p-value
    untrained.py  multi-seed untrained (random-weight) controls
    plotting.py   figures

Most-used functions are re-exported here, so notebooks can simply do:

    from repgeo import load_features, geometry_summary_table, plotting

See notebooks/reproduce_paper.ipynb for the guided walkthrough and
notebooks/analyze_your_own_model.ipynb to apply this to your own model.
"""

__version__ = "0.1.0"

from .loading import (                                              # noqa: F401
    load_features, load_convrnn_states, sanity_check,
)
from .analysis import (                                             # noqa: F401
    accuracy_table, compute_cluster_sizes_and_rdms, build_prototype_rdms,
    compute_effective_dims, between_prototype_summary, geometry_summary_table,
)
from .stats import (                                                # noqa: F401
    run_local_lme_tests, run_global_permutation_tests,
    local_cluster_lme, between_separation_permutation,
)
from .geometry import (                                             # noqa: F401
    compute_prototypes, exemplar_proto_dists, rdm_cosine,
    nearest_centroid_accuracy, pair_geometry_metrics,
)
from . import config, plotting                                      # noqa: F401
