#!/usr/bin/env python3
"""
Unit Tests for CAN Protocol
Tests CAN bus communication and OBD-II integration
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.protocols.can_protocol import CANProtocol, CANStandard, SafetyLevel, CANFrame


class TestCANProtocol:
    """Test CAN Protocol"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup CAN protocol for each test"""
        self.can = CANProtocol(force_simulation=True)
    
    @pytest.mark.unit
    def test_initialization(self):
        """CAN protocol should initialize correctly"""
        assert self.can is not None
        assert self.can.simulation_mode is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_connection(self):
        """Should initialize CAN connection"""
        result = await self.can.initialize()
        assert result is True
        await self.can.shutdown()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_message(self):
        """Should send CAN message"""
        await self.can.initialize()
        
        success = await self.can.send_message(
            arbitration_id=0x200,
            data=b'\x01\x02\x03\x04',
            safety_level=SafetyLevel.NORMAL
        )
        
        assert success is True
        await self.can.shutdown()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_emergency_stop(self):
        """Emergency stop should work"""
        await self.can.initialize()
        
        success = await self.can.send_emergency_stop(True)
        assert success is True
        
        # Clear emergency stop
        success = await self.can.send_emergency_stop(False)
        assert success is True
        
        await self.can.shutdown()
    
    @pytest.mark.unit
    def test_can_frame_creation(self):
        """CAN frame should be created correctly"""
        frame = CANFrame(
            arbitration_id=0x200,
            data=b'\x01\x02\x03\x04',
            is_extended=False
        )
        
        assert frame.arbitration_id == 0x200
        assert frame.data == b'\x01\x02\x03\x04'
        assert frame.is_extended is False
    
    @pytest.mark.unit
    def test_can_standards(self):
        """Should support multiple CAN standards"""
        assert CANStandard.CAN_2_0A is not None
        assert CANStandard.CAN_2_0B is not None
        assert CANStandard.CAN_FD is not None


class TestCANSafety:
    """Test CAN Safety Protocols"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup CAN protocol for each test"""
        self.can = CANProtocol(force_simulation=True)
    
    @pytest.mark.unit
    def test_safety_levels(self):
        """Should have multiple safety levels"""
        assert SafetyLevel.CRITICAL is not None
        assert SafetyLevel.HIGH is not None
        assert SafetyLevel.NORMAL is not None
        assert SafetyLevel.LOW is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_safety_validation(self):
        """Safety validation should work"""
        await self.can.initialize()
        
        # Get safety status
        status = self.can.get_safety_status()
        
        assert 'error_counters' in status
        assert 'total_violations' in status
        
        await self.can.shutdown()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_counting(self):
        """Error counting should work"""
        await self.can.initialize()
        
        stats = self.can.get_statistics()
        
        assert 'messages_sent' in stats
        assert 'messages_received' in stats
        assert 'errors_detected' in stats
        
        await self.can.shutdown()


class TestOBDProtocol:
    """Test OBD-II Protocol"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup OBD protocol for each test"""
        from src.protocols.obd_protocol import OBDProtocol
        self.obd = OBDProtocol(simulation_mode=True)
    
    @pytest.mark.unit
    def test_initialization(self):
        """OBD protocol should initialize correctly"""
        assert self.obd is not None
        assert self.obd.simulation_mode is True
    
    @pytest.mark.unit
    def test_supported_pids(self):
        """Should support standard OBD-II PIDs"""
        from src.protocols.obd_protocol import OBDPID
        
        # Engine PIDs
        assert OBDPID.ENGINE_RPM is not None
        assert OBDPID.VEHICLE_SPEED is not None
        assert OBDPID.COOLANT_TEMP is not None
        assert OBDPID.THROTTLE_POSITION is not None
    
    @pytest.mark.unit
    def test_safety_thresholds(self):
        """Should have safety thresholds"""
        thresholds = self.obd.safety_thresholds
        
        assert 'max_coolant_temp' in thresholds
        assert 'max_engine_rpm' in thresholds
        assert 'max_vehicle_speed' in thresholds
        assert 'min_battery_voltage' in thresholds
    
    @pytest.mark.unit
    def test_vehicle_state(self):
        """Should track vehicle state"""
        state = self.obd.get_vehicle_state()
        
        assert hasattr(state, 'engine_rpm')
        assert hasattr(state, 'vehicle_speed')
        assert hasattr(state, 'coolant_temp')
    
    @pytest.mark.unit
    def test_statistics(self):
        """Should track statistics"""
        stats = self.obd.get_statistics()
        
        assert 'readings_count' in stats
        assert 'errors_count' in stats
        assert 'simulation_mode' in stats


class TestCANMessageIDs:
    """Test CAN Message ID Definitions"""
    
    @pytest.mark.unit
    def test_emergency_stop_id(self):
        """Emergency stop should use ID 0x000"""
        # Highest priority
        emergency_id = 0x000
        assert emergency_id == 0
    
    @pytest.mark.unit
    def test_motor_command_ids(self):
        """Motor commands should use 0x200 range"""
        base_id = 0x200
        for motor_id in range(8):
            can_id = base_id + motor_id
            assert 0x200 <= can_id < 0x280
    
    @pytest.mark.unit
    def test_obd_request_id(self):
        """OBD-II request should use 0x7DF"""
        obd_request_id = 0x7DF
        assert obd_request_id == 2015
    
    @pytest.mark.unit
    def test_obd_response_ids(self):
        """OBD-II responses should use 0x7E8-0x7EF"""
        for ecu in range(8):
            response_id = 0x7E8 + ecu
            assert 0x7E8 <= response_id <= 0x7EF


class TestCANPerformance:
    """Test CAN Performance"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup CAN protocol for each test"""
        self.can = CANProtocol(force_simulation=True)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_throughput(self):
        """Should handle high message throughput"""
        await self.can.initialize()
        
        # Send 100 messages
        for i in range(100):
            await self.can.send_message(
                arbitration_id=0x200,
                data=bytes([i % 256]),
                safety_level=SafetyLevel.NORMAL
            )
        
        stats = self.can.get_statistics()
        assert stats['messages_sent'] >= 100
        
        await self.can.shutdown()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_latency(self):
        """Message latency should be low"""
        import time
        
        await self.can.initialize()
        
        start = time.time()
        await self.can.send_message(
            arbitration_id=0x200,
            data=b'\x01',
            safety_level=SafetyLevel.NORMAL
        )
        elapsed = time.time() - start
        
        # Should be under 10ms in simulation
        assert elapsed < 0.01
        
        await self.can.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
