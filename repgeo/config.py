"""Shared plotting/analysis configuration: model order, colors, display names."""

DEFAULT_COLORS = {
    'base_LRM': '#8E0B0D',
    'mod_LRM': '#E41A1C',

    'base_LRA': '#1B7F1B',
    'mod_LRA': '#2ECC71',

    'base_COR': '#1F77B4',
    'mod_COR': '#1E90FF',

    'base_CONV': '#B33B72',
    'mod_CONV': '#BF6B99',

    'base_BL': '#4C1D95',
    'mod_BL': '#7C3AED',

    # neutral references (ordered darkest -> lightest)
    'ResNet50': '#F4A300',
    'ResNet50-Robust': '#FAD679',
    'ResNet101': '#E19703',
    'VGG16': '#404040',
    'vanilla': '#928B8B',   # AlexNet
    'B': '#564A65',         # rcnn-sat feedforward control
    'CLIP': '#0097A7',
}

MODEL_COLOR_MAP = {
    "LRM3":      {"baseline": DEFAULT_COLORS['base_LRM'], "after": DEFAULT_COLORS['mod_LRM']},
    "LRA3":      {"baseline": DEFAULT_COLORS['base_LRA'], "after": DEFAULT_COLORS['mod_LRA']},
    "CORnet-RT": {"baseline": DEFAULT_COLORS['base_COR'], "after": DEFAULT_COLORS['mod_COR']},
    "ConvRNN":   {"baseline": DEFAULT_COLORS['base_CONV'], "after": DEFAULT_COLORS['mod_CONV']},
    "BL":        {"baseline": DEFAULT_COLORS['base_BL'], "after": DEFAULT_COLORS['mod_BL']},
    "AlexNet":   {"baseline": DEFAULT_COLORS['vanilla']},
    "VGG16":     {"baseline": DEFAULT_COLORS['VGG16']},
    "ResNet50":  {"baseline": DEFAULT_COLORS['ResNet50']},
    "ResNet50-Robust": {"baseline": DEFAULT_COLORS['ResNet50-Robust']},
    "ResNet101": {"baseline": DEFAULT_COLORS['ResNet101']},
    "B":         {"baseline": DEFAULT_COLORS['B']},
    "CLIP":      {"baseline": DEFAULT_COLORS['CLIP']},
}

MODEL_ORDER = [
    "LRM3", "LRA3", "BL", "CORnet-RT", "ConvRNN",
    "AlexNet", "B", "VGG16", "ResNet50", "ResNet101",
    "ResNet50-Robust", "CLIP",
]

STATE_ORDER = {"baseline": 0, "after": 1}

DISPLAY_NAMES = {
    "AlexNet baseline":         "AlexNet",
    "VGG16 baseline":           "VGG-16",
    "ResNet50 baseline":        "ResNet-50",
    "ResNet50-Robust baseline": "ResNet-50 Robust",
    "ResNet101 baseline":       "ResNet-101",
    "B baseline":               "B",
    "CLIP baseline":            "CLIP",
    "ConvRNN baseline":         "ConvRNN (t=16)",
    "ConvRNN after":            "ConvRNN (t=17)",
    "BL baseline":              "BL (t=1)",
    "BL after":                 "BL (t=8)",
    "LRM3 baseline":            "LRM (pass 1)",
    "LRM3 after":               "LRM (pass 3)",
    "LRA3 baseline":            "LRA (pass 1)",
    "LRA3 after":               "LRA (pass 3)",
    "CORnet-RT baseline":       "CORnet-RT (t=4)",
    "CORnet-RT after":          "CORnet-RT (t=5)",
}

# Models with a recurrent "baseline -> after" comparison
RECURRENT_MODELS = ["LRM3", "LRA3", "BL", "CORnet-RT", "ConvRNN"]

# Representation used for each model's headline baseline->after comparison.
# CORnet-RT uses penultimate features: its decoder is trained only on the
# final timestep, so its baseline-step logits are not a meaningful decision
# readout (see paper, "Activation extraction and analysis layer").
# To analyze your own model at a non-default layer, add an entry here, e.g.
#   config.REP_OVERRIDES["MyModel"] = "penult"
DEFAULT_REP = "logits"
REP_OVERRIDES = {"CORnet-RT": "penult"}


def pick_color(name: str):
    """Color for a 'Model state' tag (e.g. 'LRM3 baseline', 'AlexNet')."""
    parts = name.split()
    model = parts[0]
    state = parts[1] if len(parts) > 1 else "baseline"
    return MODEL_COLOR_MAP.get(model, {}).get(state)


def ordered_model_names(names):
    """Sort tags so plots share a consistent model/state order."""
    def _key(name: str):
        parts = name.split()
        model = parts[0]
        state = parts[1] if len(parts) > 1 else "baseline"
        m_idx = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
        s_idx = STATE_ORDER.get(state, len(STATE_ORDER))
        return (m_idx, s_idx, name)
    return sorted(names, key=_key)
