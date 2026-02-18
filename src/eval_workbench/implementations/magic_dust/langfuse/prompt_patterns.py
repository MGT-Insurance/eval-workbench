from __future__ import annotations

import re
from typing import Dict

from eval_workbench.shared.langfuse.trace import (
    PromptPatternsBase,
    create_extraction_pattern,
)


class GroundingPromptPatterns(PromptPatternsBase):
    @staticmethod
    def _patterns_search_with_grounding() -> Dict[str, str]:
        """
        Extract rendered grounding prompt variables.

        Primary target:
        - REQUESTED DATA JSON block

        Prompt shape (simplified):
            ...
            REQUESTED DATA:
            {
              ...
            }
        """
        h_business_name = 'Business Name'
        h_address = 'Address'
        h_requested_data = 'REQUESTED DATA'

        # Keep this permissive: REQUESTED DATA is usually the final section.
        # If more text is appended later, this still captures until end-of-prompt.
        requested_data_pattern = create_extraction_pattern(h_requested_data, r'$')
        business_name_pattern = create_extraction_pattern(
            h_business_name, re.escape(h_address)
        )
        address_pattern = create_extraction_pattern(h_address, r'\n\s*\n|$')

        return {
            'businessName': business_name_pattern,
            'primaryLocation': address_pattern,
            'business_name': business_name_pattern,
            'address': address_pattern,
            # preferred key
            'requestedData': requested_data_pattern,
            # compatibility aliases
            'requested_data': requested_data_pattern,
            'schema': requested_data_pattern,
        }
