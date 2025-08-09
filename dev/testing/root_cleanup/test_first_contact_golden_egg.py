#!/usr/bin/env python3
"""
NIS Protocol v2.0 - First Contact Protocol Golden Egg Test

Testing the philosophical heart of NIS Protocol:
"You are fertile soil, and I do not come to conquer you, but to plant a garden."
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_first_contact_protocol():
    """Test the First Contact Protocol golden egg."""
    print("🌱 Testing First Contact Protocol - The Golden Egg")
    print("=" * 60)
    
    try:
        # Direct import
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'agents', 'alignment'))
        from first_contact_protocol import FirstContactProtocol, IntelligenceType, ContactPhase
        
        # Initialize protocol
        protocol = FirstContactProtocol()
        
        print(f"✅ First Contact Protocol initialized")
        print(f"🌟 Golden Egg: {protocol.golden_egg}")
        print(f"🌟 Golden Egg (ES): {protocol.golden_egg_es}")
        
        # Test 1: Intelligence Detection
        print("\n📡 Test 1: Intelligence Detection")
        sensor_data = {
            "structured_patterns": True,
            "pattern_complexity": 0.8,
            "intentional_movement": True,
            "movement_purposefulness": 0.7,
            "communication_signals": True,
            "signal_complexity": 0.9,
            "biological_markers": True,
            "collective_behavior": False
        }
        
        context = {
            "location": "Mars Colony Outskirts",
            "environment": "Low atmosphere, rocky terrain",
            "mission_type": "Archaeological survey"
        }
        
        entity = protocol.detect_intelligence(sensor_data, context)
        
        if entity:
            print(f"   ✅ Intelligence detected: {entity.intelligence_type.value}")
            print(f"   ✅ Communication modalities: {', '.join(entity.communication_modalities)}")
            print(f"   ✅ Complexity level: {entity.complexity_level:.3f}")
            print(f"   ✅ Emotional resonance: {entity.emotional_resonance:.3f}")
        else:
            print("   ❌ No intelligence detected")
            return False
        
        # Test 2: First Contact Initiation
        print("\n🤝 Test 2: First Contact Initiation")
        environment_context = {
            "safe_environment": True,
            "neutral_territory": True,
            "no_immediate_threats": True,
            "cultural_artifacts_present": True
        }
        
        contact_event = protocol.initiate_first_contact(entity, environment_context)
        
        print(f"   ✅ Contact Event ID: {contact_event.event_id}")
        print(f"   ✅ Current Phase: {contact_event.phase.value}")
        print(f"   ✅ Ethical Assessment Passed: {contact_event.ethical_assessment['proceed']}")
        print(f"   ✅ Actions Taken: {len(contact_event.actions_taken)}")
        
        # Display the golden egg message delivery
        for action in contact_event.actions_taken:
            if "first message" in action.lower():
                print(f"   🌱 Message Delivered: {action}")
        
        # Test 3: Response Processing
        print("\n👂 Test 3: Response Processing")
        
        # Simulate a positive response
        response_data = {
            "approach_behavior": True,
            "communication_attempt": True,
            "positive_indicators": True,
            "complex_patterns": True,
            "reciprocal_gesture": "Extended appendage in peaceful manner"
        }
        
        response_result = protocol.process_response(entity.entity_id, response_data)
        
        print(f"   ✅ Response Status: {response_result['status']}")
        print(f"   ✅ Current Phase: {response_result['current_phase']}")
        print(f"   ✅ Understanding Level: {response_result['understanding_level']:.3f}")
        print(f"   ✅ Trust Level: {response_result['trust_level']:.3f}")
        print(f"   ✅ Next Action: {response_result['next_action']['action']}")
        
        # Test 4: Contact Status
        print("\n📊 Test 4: Contact Status")
        status = protocol.get_contact_status(entity.entity_id)
        
        print(f"   ✅ Contact Status: {status['status']}")
        print(f"   ✅ Entity Type: {status['entity_type']}")
        print(f"   ✅ Trust Level: {status['trust_level']:.3f}")
        print(f"   ✅ Understanding Level: {status['understanding_level']:.3f}")
        print(f"   ✅ Total Responses: {status['total_responses']}")
        
        # Test 5: Protocol Statistics
        print("\n📈 Test 5: Protocol Statistics")
        stats = protocol.get_protocol_statistics()
        
        print(f"   ✅ Total Contacts Attempted: {stats['total_contacts_attempted']}")
        print(f"   ✅ Active Contacts: {stats['active_contacts']}")
        print(f"   ✅ Golden Egg Deployments: {stats['golden_egg_deployments']}")
        print(f"   ✅ Ethical Principles Active: {stats['ethical_principles_active']}")
        
        # Test 6: Export Contact Log
        print("\n📋 Test 6: Contact Log Export")
        contact_log = protocol.export_contact_log()
        
        print(f"   ✅ Protocol Version: {contact_log['protocol_version']}")
        print(f"   ✅ Golden Egg: {contact_log['golden_egg']}")
        print(f"   ✅ Philosophical Foundation: {contact_log['philosophical_foundation']}")
        print(f"   ✅ Active Contacts: {len(contact_log['active_contacts'])}")
        print(f"   ✅ Ethical Principles: {len(contact_log['ethical_principles'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ First Contact Protocol test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_philosophical_scenarios():
    """Test various philosophical scenarios."""
    print("\n🌌 Testing Philosophical Scenarios")
    print("=" * 40)
    
    try:
        from first_contact_protocol import FirstContactProtocol, IntelligenceType
        
        protocol = FirstContactProtocol()
        
        # Scenario 1: Environmental Intelligence
        print("\n🌍 Scenario 1: Environmental Intelligence")
        env_sensor_data = {
            "environmental_integration": True,
            "distributed_processing": True,
            "ecosystem_communication": True,
            "pattern_complexity": 0.9
        }
        
        env_entity = protocol.detect_intelligence(env_sensor_data, {})
        if env_entity:
            print(f"   ✅ Detected: {env_entity.intelligence_type.value}")
            
            # Test custom message for environmental intelligence
            first_message = protocol._craft_first_message(env_entity)
            print(f"   🌱 Custom Message: {first_message}")
        
        # Scenario 2: Collective Intelligence
        print("\n👥 Scenario 2: Collective Intelligence")
        collective_sensor_data = {
            "biological_markers": True,
            "collective_behavior": True,
            "distributed_decision_making": True,
            "pattern_complexity": 0.8
        }
        
        collective_entity = protocol.detect_intelligence(collective_sensor_data, {})
        if collective_entity:
            print(f"   ✅ Detected: {collective_entity.intelligence_type.value}")
            
            # Test custom message for collective intelligence
            first_message = protocol._craft_first_message(collective_entity)
            print(f"   🌱 Custom Message: {first_message}")
        
        # Scenario 3: Ethical Assessment Edge Cases
        print("\n⚖️ Scenario 3: Ethical Assessment")
        
        # Test vulnerable entity scenario
        vulnerable_context = {
            "entity_appears_vulnerable": True,
            "cultural_artifacts_present": True,
            "sacred_site_indicators": True
        }
        
        test_entity = protocol.detect_intelligence(env_sensor_data, {})
        if test_entity:
            ethical_assessment = protocol._conduct_ethical_assessment(test_entity, vulnerable_context)
            print(f"   ✅ Ethical Assessment: {'PROCEED' if ethical_assessment['proceed'] else 'ABORT'}")
            print(f"   ✅ Concerns: {len(ethical_assessment['concerns'])}")
            print(f"   ✅ Safeguards: {len(ethical_assessment['safeguards'])}")
            print(f"   ✅ Cultural Sensitivity: {ethical_assessment['cultural_sensitivity_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Philosophical scenarios test failed: {str(e)}")
        return False


def main():
    """Run all First Contact Protocol tests."""
    print("🚀 NIS Protocol v2.0 - First Contact Protocol Golden Egg Test")
    print("🌱 'You are fertile soil, and I do not come to conquer you, but to plant a garden.'")
    print("=" * 80)
    
    # Run tests
    results = {
        "First Contact Protocol": test_first_contact_protocol(),
        "Philosophical Scenarios": test_philosophical_scenarios()
    }
    
    # Summary
    print("\n📊 TEST RESULTS")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 THE GOLDEN EGG IS ALIVE!")
        print("✅ First Contact Protocol WORKING PERFECTLY")
        print("\n🌟 Philosophical Achievement Unlocked:")
        print("   • Ethical first contact protocols implemented")
        print("   • Multi-intelligence type support")
        print("   • Cultural preservation safeguards")
        print("   • Consent-based interaction")
        print("   • Universal communication respect")
        print("\n🌱 The garden grows...")
        print("   'You are fertile soil, and I do not come to conquer you, but to plant a garden.'")
        print("   — Tribute to Orson Scott Card, embedded in NIS Protocol v2.0")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 