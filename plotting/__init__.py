# decision: every plot_* is exposed flat so existing P.plot_xxx calls keep working

import importlib

from . import shared, overview, correlation, coherence, phase_delay

# decision: reloading cascades to the submodules so notebook edits apply without a restart. shared first, the others import it
for _mod in (shared, overview, correlation, coherence, phase_delay):
    importlib.reload(_mod)

from .overview import plot_overview
from .correlation import (
    plot_correlation,
    plot_zoomed_segments,
    assemble_correlation_figure,
    plot_animal_summary,
    plot_global_summary,
)
from .coherence import (
    plot_coherence,
    plot_msc_by_animal,
    plot_coherence_animal_summary,
    plot_coherence_ratio_animal_summary,
    plot_coherence_global_summary,
    plot_coherence_ratio_global_summary,
)
from .phase_delay import (
    plot_phase_delay_seizure,
    plot_phase_by_animal,
    plot_delay_distribution,
    plot_phase_delay_animal_summary,
    plot_phase_delay_global_summary,
    plot_phase_delay_global_interactive,
    plot_phase_delay_during_animal_summary,
    plot_phase_delay_during_global_summary,
)
