"""
NIS-AUTO Integration Template
Complete example of integrating NIS Protocol into vehicle diagnostics system
"""

import asyncio
from nis_protocol import NISCore
from nis_protocol.plugins import AutoPlugin

class VehicleDiagnosticSystem:
    """Example vehicle diagnostic system using NIS Protocol"""
    
    def __init__(self, vehicle_info: dict = None):
        self.vehicle_info = vehicle_info or {
            'make': 'Tesla',
            'model': 'Model 3',
            'year': 2023,
            'vin': 'EXAMPLE123456789'
        }
        
        self.nis = NISCore()
        self.auto_plugin = AutoPlugin(config={
            'vehicle': self.vehicle_info,
            'obd_port': '/dev/ttyUSB0',
            'enable_predictive_maintenance': True,
            'data_logging': True
        })
        
    async def initialize(self):
        """Initialize NIS Protocol and register auto plugin"""
        await self.nis.initialize()
        await self.nis.register_plugin(self.auto_plugin)
        print(f"✅ Vehicle Diagnostic System initialized for {self.vehicle_info['make']} {self.vehicle_info['model']}")
        
    async def diagnose_error_code(self, dtc_code: str):
        """Diagnose OBD-II error code"""
        query = f"Diagnose error code {dtc_code} for {self.vehicle_info['make']} {self.vehicle_info['model']}"
        result = await self.nis.process_autonomously(query)
        return result
        
    async def predictive_maintenance(self, mileage: int, last_service: dict):
        """Predict maintenance needs"""
        query = f"""
        Vehicle: {self.vehicle_info['make']} {self.vehicle_info['model']}
        Current mileage: {mileage}
        Last service: {last_service}
        Predict maintenance needs and schedule
        """
        result = await self.nis.process_autonomously(query)
        return result
        
    async def performance_analysis(self, telemetry: dict):
        """Analyze vehicle performance"""
        query = f"Analyze vehicle performance metrics: {telemetry}"
        result = await self.nis.process_autonomously(query)
        return result
        
    async def fuel_optimization(self, driving_data: dict):
        """Optimize fuel consumption"""
        query = f"Analyze driving patterns and suggest fuel optimization: {driving_data}"
        result = await self.nis.process_autonomously(query)
        return result


async def main():
    # Example 1: Initialize system
    vehicle = VehicleDiagnosticSystem({
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2022,
        'vin': 'JTM1234567890'
    })
    await vehicle.initialize()
    
    # Example 2: Diagnose error code
    diagnosis = await vehicle.diagnose_error_code('P0420')
    print(f"Diagnosis: {diagnosis}")
    
    # Example 3: Predictive maintenance
    maintenance = await vehicle.predictive_maintenance(
        mileage=45000,
        last_service={
            'date': '2024-06-15',
            'type': 'oil_change',
            'mileage': 40000
        }
    )
    print(f"Maintenance prediction: {maintenance}")
    
    # Example 4: Performance analysis
    telemetry = {
        'avg_speed': 65,
        'fuel_efficiency': 28.5,
        'engine_temp': 195,
        'rpm_avg': 2500,
        'acceleration_events': 45
    }
    performance = await vehicle.performance_analysis(telemetry)
    print(f"Performance: {performance}")
    
    # Example 5: Fuel optimization
    driving_data = {
        'avg_speed': 70,
        'acceleration_rate': 'moderate',
        'braking_frequency': 'high',
        'idle_time_percent': 15
    }
    optimization = await vehicle.fuel_optimization(driving_data)
    print(f"Fuel optimization: {optimization}")


if __name__ == "__main__":
    asyncio.run(main())

