"""
NIS-CITY Integration Template
Complete example of integrating NIS Protocol into smart city IoT system
"""

import asyncio
from nis_protocol import NISCore
from nis_protocol.plugins import CityPlugin

class SmartCitySystem:
    """Example smart city management system using NIS Protocol"""
    
    def __init__(self, city_config: dict = None):
        self.city_config = city_config or {
            'city_name': 'Example City',
            'population': 500000,
            'area_km2': 150,
            'timezone': 'America/Los_Angeles'
        }
        
        self.nis = NISCore()
        self.city_plugin = CityPlugin(config={
            'city': self.city_config,
            'iot_endpoints': {
                'traffic': 'http://traffic-api.city.gov',
                'energy': 'http://energy-api.city.gov',
                'environment': 'http://env-api.city.gov'
            },
            'enable_predictions': True,
            'data_retention_days': 90
        })
        
    async def initialize(self):
        """Initialize NIS Protocol and register city plugin"""
        await self.nis.initialize()
        await self.nis.register_plugin(self.city_plugin)
        print(f"✅ Smart City System initialized for {self.city_config['city_name']}")
        
    async def optimize_traffic(self, traffic_data: dict):
        """Optimize city traffic flow"""
        query = f"""
        Analyze traffic data: {traffic_data}
        Optimize traffic light timing and suggest route adjustments
        """
        result = await self.nis.process_autonomously(query)
        return result
        
    async def energy_management(self, consumption: dict):
        """Manage city energy consumption"""
        query = f"""
        Current energy consumption: {consumption}
        Optimize distribution and predict peak demand
        """
        result = await self.nis.process_autonomously(query)
        return result
        
    async def waste_optimization(self, collection_data: dict):
        """Optimize waste collection routes"""
        query = f"Optimize waste collection based on: {collection_data}"
        result = await self.nis.process_autonomously(query)
        return result
        
    async def environmental_monitoring(self, sensors: dict):
        """Monitor environmental conditions"""
        query = f"Analyze environmental sensor data: {sensors}"
        result = await self.nis.process_autonomously(query)
        return result
        
    async def public_safety(self, incidents: list):
        """Coordinate public safety response"""
        query = f"Analyze incidents and coordinate emergency response: {incidents}"
        result = await self.nis.process_autonomously(query)
        return result


async def main():
    # Example 1: Initialize system
    city = SmartCitySystem({
        'city_name': 'San Francisco',
        'population': 873965,
        'area_km2': 121,
        'timezone': 'America/Los_Angeles'
    })
    await city.initialize()
    
    # Example 2: Traffic optimization
    traffic_data = {
        'congestion_zones': ['Downtown', 'Mission District'],
        'avg_speed_mph': 15,
        'incident_count': 3,
        'time_of_day': '17:30'
    }
    traffic = await city.optimize_traffic(traffic_data)
    print(f"Traffic optimization: {traffic}")
    
    # Example 3: Energy management
    consumption = {
        'current_mw': 450,
        'predicted_peak_mw': 520,
        'renewable_percent': 35,
        'grid_load': 'high'
    }
    energy = await city.energy_management(consumption)
    print(f"Energy management: {energy}")
    
    # Example 4: Waste collection optimization
    collection_data = {
        'bins': [
            {'id': 'BIN001', 'fill_level': 85, 'location': [37.7749, -122.4194]},
            {'id': 'BIN002', 'fill_level': 92, 'location': [37.7750, -122.4195]},
            {'id': 'BIN003', 'fill_level': 45, 'location': [37.7751, -122.4196]}
        ],
        'trucks_available': 2
    }
    waste = await city.waste_optimization(collection_data)
    print(f"Waste optimization: {waste}")
    
    # Example 5: Environmental monitoring
    sensors = {
        'air_quality_index': 65,
        'pm25': 35,
        'temperature_f': 72,
        'humidity_percent': 55,
        'noise_level_db': 60
    }
    environment = await city.environmental_monitoring(sensors)
    print(f"Environmental analysis: {environment}")
    
    # Example 6: Public safety coordination
    incidents = [
        {'type': 'fire', 'location': [37.7749, -122.4194], 'severity': 'high'},
        {'type': 'accident', 'location': [37.7750, -122.4195], 'severity': 'medium'}
    ]
    safety = await city.public_safety(incidents)
    print(f"Public safety response: {safety}")


if __name__ == "__main__":
    asyncio.run(main())

