"""
World Environment
Defines the simulation world (city, airspace, terrain)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


@dataclass
class WorldConfig:
    """World configuration"""
    name: str = "default"
    size: Tuple[float, float, float] = (1000, 1000, 500)  # meters
    ground_type: str = "flat"  # flat, terrain, city
    weather: str = "clear"
    time_of_day: float = 12.0  # 24h format


@dataclass 
class Zone:
    """Defines a zone in the world (no-fly, restricted, etc)"""
    name: str
    zone_type: str  # "no_fly", "restricted", "free"
    bounds: Tuple[float, float, float, float, float, float]  # min_x, min_y, min_z, max_x, max_y, max_z
    active: bool = True


@dataclass
class Obstacle:
    """Static obstacle in the world"""
    name: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    obstacle_type: str = "building"
    body_id: Optional[int] = None


class World:
    """
    Simulation world environment
    Manages terrain, obstacles, zones, weather
    """
    
    def __init__(self, config: Optional[WorldConfig] = None):
        self.config = config or WorldConfig()
        self.obstacles: Dict[str, Obstacle] = {}
        self.zones: Dict[str, Zone] = {}
        self.physics_client = None
        
    def initialize(self, physics_client: int):
        """Initialize world in physics engine"""
        self.physics_client = physics_client
        
        # Create boundary walls (invisible)
        self._create_boundaries()
        
        print(f"✅ World '{self.config.name}' initialized ({self.config.size}m)")
        return self
    
    def _create_boundaries(self):
        """Create world boundary walls"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            return
            
        sx, sy, sz = self.config.size
        
        # Create invisible boundary planes (optional, for containment)
        # For now, we just track bounds in software
        pass
    
    def add_obstacle(self, 
                     name: str,
                     position: Tuple[float, float, float],
                     size: Tuple[float, float, float],
                     obstacle_type: str = "building") -> Obstacle:
        """Add a static obstacle to the world"""
        obstacle = Obstacle(
            name=name,
            position=position,
            size=size,
            obstacle_type=obstacle_type
        )
        
        if PYBULLET_AVAILABLE and self.physics_client is not None:
            # Create collision shape
            half_extents = [s/2 for s in size]
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents
            )
            
            # Create visual shape
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.5, 0.5, 0.5, 1]  # Gray
            )
            
            # Create body
            obstacle.body_id = p.createMultiBody(
                baseMass=0,  # Static
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position
            )
        
        self.obstacles[name] = obstacle
        return obstacle
    
    def add_zone(self,
                 name: str,
                 zone_type: str,
                 bounds: Tuple[float, float, float, float, float, float]) -> Zone:
        """Add a zone (no-fly, restricted, etc)"""
        zone = Zone(name=name, zone_type=zone_type, bounds=bounds)
        self.zones[name] = zone
        return zone
    
    def check_zone(self, position: Tuple[float, float, float]) -> List[Zone]:
        """Check which zones a position is in"""
        zones_at_position = []
        x, y, z = position
        
        for zone in self.zones.values():
            if not zone.active:
                continue
            min_x, min_y, min_z, max_x, max_y, max_z = zone.bounds
            if (min_x <= x <= max_x and 
                min_y <= y <= max_y and 
                min_z <= z <= max_z):
                zones_at_position.append(zone)
        
        return zones_at_position
    
    def is_position_valid(self, position: Tuple[float, float, float]) -> Tuple[bool, str]:
        """Check if position is valid (in bounds, not in no-fly zone)"""
        x, y, z = position
        sx, sy, sz = self.config.size
        
        # Check world bounds
        if not (0 <= x <= sx and 0 <= y <= sy and 0 <= z <= sz):
            return False, "Out of world bounds"
        
        # Check zones
        zones = self.check_zone(position)
        for zone in zones:
            if zone.zone_type == "no_fly":
                return False, f"In no-fly zone: {zone.name}"
        
        return True, "OK"
    
    def raycast(self, 
                start: Tuple[float, float, float],
                end: Tuple[float, float, float]) -> Optional[Dict]:
        """Cast a ray and return hit info"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            return None
        
        result = p.rayTest(start, end)
        if result[0][0] != -1:  # Hit something
            return {
                "hit": True,
                "body_id": result[0][0],
                "position": result[0][3],
                "normal": result[0][4],
                "distance": np.linalg.norm(np.array(result[0][3]) - np.array(start))
            }
        return {"hit": False}
    
    def get_state(self) -> Dict:
        """Get world state"""
        return {
            "name": self.config.name,
            "size": self.config.size,
            "obstacles": len(self.obstacles),
            "zones": len(self.zones),
            "weather": self.config.weather,
            "time_of_day": self.config.time_of_day
        }


class CityWorld(World):
    """Pre-configured city environment"""
    
    def __init__(self, city_name: str = "test_city"):
        super().__init__(WorldConfig(
            name=city_name,
            size=(2000, 2000, 500),
            ground_type="city"
        ))
    
    def generate_city_grid(self, 
                           blocks_x: int = 5,
                           blocks_y: int = 5,
                           block_size: float = 200,
                           building_height_range: Tuple[float, float] = (20, 100)):
        """Generate a simple city grid"""
        import random
        
        for i in range(blocks_x):
            for j in range(blocks_y):
                # Building in each block
                x = i * block_size + block_size / 2
                y = j * block_size + block_size / 2
                height = random.uniform(*building_height_range)
                
                self.add_obstacle(
                    name=f"building_{i}_{j}",
                    position=(x, y, height / 2),
                    size=(block_size * 0.6, block_size * 0.6, height),
                    obstacle_type="building"
                )
        
        # Add no-fly zones (airports, etc)
        self.add_zone(
            name="airport",
            zone_type="no_fly",
            bounds=(0, 0, 0, 400, 400, 500)
        )
        
        print(f"✅ Generated city grid: {blocks_x}x{blocks_y} blocks")


class AirspaceWorld(World):
    """Pre-configured airspace for drone testing"""
    
    def __init__(self):
        super().__init__(WorldConfig(
            name="airspace",
            size=(5000, 5000, 1000),
            ground_type="flat"
        ))
        
        # Define altitude layers
        self.add_zone("low_altitude", "free", (0, 0, 0, 5000, 5000, 120))
        self.add_zone("controlled", "restricted", (0, 0, 120, 5000, 5000, 500))
        self.add_zone("high_altitude", "restricted", (0, 0, 500, 5000, 5000, 1000))
