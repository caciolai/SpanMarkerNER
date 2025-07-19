import logging

from span_marker.optim.utils import spread_sample
from span_marker.trainer import Trainer as SpanMarkerTrainer

logger = logging.getLogger(__name__)


class Trainer(SpanMarkerTrainer):
    spread_sample_fn = staticmethod(spread_sample)
