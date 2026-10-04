"""Gemma 4's normalized metadata accounting, never a full weight load."""

import re

from .base import normalize_routed_inventory


class Gemma4Adapter:
    family = 'gemma4'

    def normalize(self, descriptor, inventory, identity):
        if descriptor.family != self.family or descriptor.architecture != 'gemma4-moe':
            raise ValueError('wrong family/architecture for Gemma 4 adapter')
        return normalize_routed_inventory(descriptor, inventory, identity)

    def normalize_profile(self, profile, *, profile_id):
        if profile.get('diagnostic_synchronization') is not True:
            raise ValueError('profile requires diagnostic synchronization')
        rows = []
        for record in profile.get('records', []):
            match = re.fullmatch(r'ffn_moe_gate_up-(\d+)', record.get('first_node', ''))
            if match:
                rows.append({'layer_id': int(match.group(1)), 'total_us': record['total_us'],
                             'backend': record['backend'], 'profile_id': profile_id})
        if not rows:
            raise ValueError('profile contains no routed expert splits')
        return rows
