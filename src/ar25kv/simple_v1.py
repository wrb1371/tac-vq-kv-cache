"""Factorial simplification of the frozen budget-gated representation."""
from ar24kv.pressure_switch_v1 import PressureSwitch
from ar24kv.attribution import AdditiveMarginals

class ConditionalBudget(PressureSwitch):
    """No encoding or cost change; G/probe restrictions are explicit policies."""
    pass

class SeparableBudget(AdditiveMarginals, PressureSwitch):
    """Original observer, separable target only during budget-gated planning."""
    pass
