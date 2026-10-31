"""Instrumentation wrappers for runtime-layout implementations (never timed)."""
from .capture import Capture
from .dynamic import DynamicAllocated, DynamicExactAllocated, DynamicGain, DynamicExactGain
class CaptureDynamic(Capture,DynamicAllocated):pass
class CaptureDynamicExact(Capture,DynamicExactAllocated):pass
class CaptureDynamicGain(Capture,DynamicGain):pass
class CaptureDynamicExactGain(Capture,DynamicExactGain):pass
