from dataclasses import dataclass

@dataclass
class WheelDimensions:
    rim_radius: float = 8.0
    tire_width: float = 4.0
    tire_radius: float = 10.0
    hub_radius: float = 2.0
    spoke_count: int = 8
    spoke_width: float = 1.8
