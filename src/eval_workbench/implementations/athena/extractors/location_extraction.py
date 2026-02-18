import json
import logging
from typing import Any

from axion.dataset import DatasetItem

from eval_workbench.shared.extractors import ExtractorHelpers
from eval_workbench.shared.langfuse.trace import Trace

logger = logging.getLogger(__name__)


class LocationExtractionExtractor(ExtractorHelpers[Trace]):
    """OOP extractor for Athena location-extraction traces."""

    def extract(self, source: Trace) -> DatasetItem:
        """
        Extract a DatasetItem from an Athena "location-extraction" trace step.

        Adds `product_initiate` (from the rendered Quote JSON) into DatasetItem.dataset_metadata.
        """
        trace = source
        step = self.safe_get(trace, 'location-extraction', None)
        if step is None:
            # Fallback to underscore variant (some callers may use it)
            step = self.safe_get(trace, 'location_extraction', None)

        quote_raw = self.safe_get(step, 'variables.quote', None) if step else None
        quote: dict[str, Any] = {}
        if isinstance(quote_raw, str) and quote_raw.strip():
            try:
                quote = json.loads(quote_raw)
            except json.JSONDecodeError:
                logger.warning('Failed to JSON-decode location-extraction variables.quote')
        elif isinstance(quote_raw, dict):
            quote = quote_raw

        quote_locator = (
            self.safe_get(quote, 'locator', None)
            or self.safe_get(step, 'span.input.quote_locator', None)
            or self.safe_get(trace, 'id', 'unknown')
        )

        product_initiate = self.safe_get(quote, 'element.data.product_initiate', None)

        # Trace metadata (existing) + augmentation
        trace_metadata: dict[str, Any] = {}
        meta = self.safe_get(trace, 'metadata', None)
        if meta is not None:
            plain_meta = self.to_plain_dict(meta)
            if isinstance(plain_meta, dict):
                trace_metadata = plain_meta

        if product_initiate is not None:
            trace_metadata['product_initiate'] = product_initiate

        selected_observation = self.select_step_generation(step) if step else None
        span_for_meta = self.select_step_span(step) if step else None

        latency = self.safe_get(span_for_meta, 'latency', None)
        if latency is None:
            latency = self.safe_get(selected_observation, 'latency', None)

        trace_id = str(self.safe_get(trace, 'id', ''))
        observation_id = (
            self.safe_get(selected_observation, 'id', '')
            or self.safe_get(span_for_meta, 'id', '')
            or ''
        )

        # Best-effort actual output capture (keep raw for downstream parsing)
        actual_output = self.safe_get(selected_observation, 'output', None)
        if isinstance(actual_output, (dict, list)):
            actual_output = json.dumps(actual_output)
        if actual_output is None:
            actual_output = ''

        return DatasetItem(
            id=str(quote_locator),
            query=f'Extract and classify all addresses for {quote_locator}',
            expected_output=None,
            acceptance_criteria=None,
            additional_input={
                # Keep this small; full quote is already recoverable via trace prompt variables.
                'quote_locator': quote_locator,
            },
            dataset_metadata=json.dumps(trace_metadata),
            actual_output=actual_output,
            latency=latency,
            trace_id=trace_id,
            observation_id=observation_id,
            additional_output={
                'product_initiate': product_initiate,
            },
        )


_LOCATION_EXTRACTOR = LocationExtractionExtractor()


def extract_location_extraction(trace: Trace) -> DatasetItem:
    """Backward-compatible function wrapper for LocationExtractionExtractor."""
    return _LOCATION_EXTRACTOR.extract(trace)

