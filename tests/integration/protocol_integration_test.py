#!/usr/bin/env python3
"""
🔗 NIS Protocol v3 - Comprehensive Protocol Integration Test

Tests all protocol integrations and reasoning patterns to ensure modular connectivity:
- MCP (Model Context Protocol)
- A2A (Agent-to-Agent Protocol) 
- LangChain/LangGraph integration
- Chain of Thought (COT) reasoning
- Tree of Thought (TOT) reasoning
- ReAct (Reasoning and Acting) patterns
- LangSmith observability
"""

import sys
import os
import time
import asyncio
import json
from typing import Dict, Any, List
from dataclasses import dataclass

# Fix Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@dataclass
class ProtocolTestResult:
    protocol_name: str
    status: str  # 'OPERATIONAL', 'IMPORT_ISSUES', 'CONFIGURATION_NEEDED', 'NOT_AVAILABLE'
    features_tested: List[str]
    test_results: Dict[str, Any]
    recommendations: List[str]
    execution_time: float

class ProtocolIntegrationTester:
    """Comprehensive protocol integration testing"""
    
    def __init__(self):
        self.test_results: List[ProtocolTestResult] = []
        self.start_time = time.time()
    
    def run_all_protocol_tests(self) -> Dict[str, Any]:
        """Run comprehensive protocol integration tests"""
        
        print("🔗 NIS Protocol v3 - Comprehensive Protocol Integration Test")
        print("=" * 70)
        print("Testing modular connectivity and reasoning patterns")
        
        # Protocol test categories
        protocol_tests = [
            ("MCP Protocol", self._test_mcp_protocol),
            ("A2A Protocol", self._test_a2a_protocol),
            ("LangChain Integration", self._test_langchain_integration),
            ("Reasoning Patterns", self._test_reasoning_patterns),
            ("Protocol Routing", self._test_protocol_routing),
            ("Modular Connectivity", self._test_modular_connectivity)
        ]
        
        for protocol_name, test_function in protocol_tests:
            print(f"\n🧪 Testing: {protocol_name}")
            print("-" * 50)
            
            try:
                test_function()
            except Exception as e:
                print(f"  ❌ Test category failed: {e}")
                self._add_test_result(
                    protocol_name,
                    "NOT_AVAILABLE",
                    [],
                    {"error": str(e)},
                    [f"Fix test error: {e}"],
                    0.0
                )
        
        return self._generate_protocol_report()
    
    def _test_mcp_protocol(self):
        """Test Model Context Protocol (MCP) integration"""
        
        start_time = time.time()
        features_tested = []
        test_results = {}
        recommendations = []
        
        # Test 1: MCP Adapter Import
        try:
            from adapters.mcp_adapter import MCPAdapter
            test_results["adapter_import"] = "SUCCESS"
            features_tested.append("Adapter Import")
            print("  ✅ MCP Adapter import successful")
        except Exception as e:
            test_results["adapter_import"] = f"FAILED: {e}"
            print(f"  ❌ MCP Adapter import failed: {e}")
        
        # Test 2: MCP Configuration
        try:
            config = {
                "base_url": "https://api.example.com/mcp",
                "api_key": "test_key",
                "timeout": 30
            }
            
            if "MCPAdapter" in locals():
                mcp_adapter = MCPAdapter(config)
                test_results["configuration"] = "SUCCESS"
                features_tested.append("Configuration")
                print("  ✅ MCP configuration successful")
            else:
                test_results["configuration"] = "SKIPPED: Import failed"
                print("  ⏸️  MCP configuration skipped (import failed)")
        except Exception as e:
            test_results["configuration"] = f"FAILED: {e}"
            print(f"  ❌ MCP configuration failed: {e}")
        
        # Test 3: Message Translation
        try:
            if "mcp_adapter" in locals():
                # Test MCP to NIS translation
                mcp_message = {
                    "function_call": {
                        "name": "test_function",
                        "parameters": {"param1": "value1"}
                    },
                    "conversation_id": "test_conv_123",
                    "tool_id": "test_tool"
                }
                
                nis_message = mcp_adapter.translate_to_nis(mcp_message)
                
                if nis_message and "action" in nis_message:
                    test_results["message_translation"] = "SUCCESS"
                    features_tested.append("Message Translation")
                    print("  ✅ MCP message translation successful")
                else:
                    test_results["message_translation"] = "FAILED: Invalid NIS message format"
                    print("  ❌ MCP message translation failed: Invalid format")
            else:
                test_results["message_translation"] = "SKIPPED: Adapter not available"
                print("  ⏸️  MCP message translation skipped")
        except Exception as e:
            test_results["message_translation"] = f"FAILED: {e}"
            print(f"  ❌ MCP message translation failed: {e}")
        
        # Test 4: Protocol Routing Configuration
        try:
            config_path = os.path.join(project_root, "src", "configs", "protocol_routing.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    routing_config = json.load(f)
                
                if "mcp" in routing_config:
                    test_results["routing_config"] = "SUCCESS"
                    features_tested.append("Routing Configuration")
                    print("  ✅ MCP routing configuration found")
                else:
                    test_results["routing_config"] = "FAILED: MCP not in routing config"
                    print("  ❌ MCP not found in routing configuration")
            else:
                test_results["routing_config"] = "FAILED: Routing config file not found"
                print("  ❌ Protocol routing configuration file not found")
        except Exception as e:
            test_results["routing_config"] = f"FAILED: {e}"
            print(f"  ❌ MCP routing config test failed: {e}")
        
        # Determine overall status
        successful_tests = [k for k, v in test_results.items() if v == "SUCCESS"]
        
        if len(successful_tests) >= 3:
            status = "OPERATIONAL"
            recommendations.append("MCP protocol fully operational")
        elif len(successful_tests) >= 2:
            status = "CONFIGURATION_NEEDED"
            recommendations.extend([
                "MCP adapter functional but needs configuration",
                "Set MCP_API_KEY environment variable for full functionality"
            ])
        else:
            status = "IMPORT_ISSUES"
            recommendations.extend([
                "Fix MCP adapter import issues",
                "Ensure adapters module is properly accessible"
            ])
        
        execution_time = time.time() - start_time
        
        self._add_test_result(
            "MCP Protocol",
            status,
            features_tested,
            test_results,
            recommendations,
            execution_time
        )
    
    def _test_a2a_protocol(self):
        """Test Agent-to-Agent (A2A) Protocol integration"""
        
        start_time = time.time()
        features_tested = []
        test_results = {}
        recommendations = []
        
        # Test 1: A2A Adapter Import
        try:
            from adapters.a2a_adapter import A2AAdapter
            test_results["adapter_import"] = "SUCCESS"
            features_tested.append("Adapter Import")
            print("  ✅ A2A Adapter import successful")
        except Exception as e:
            test_results["adapter_import"] = f"FAILED: {e}"
            print(f"  ❌ A2A Adapter import failed: {e}")
        
        # Test 2: A2A Configuration and Initialization
        try:
            config = {
                "base_url": "https://api.example.com/a2a",
                "api_key": "test_a2a_key",
                "timeout": 30
            }
            
            if "A2AAdapter" in locals():
                a2a_adapter = A2AAdapter(config)
                test_results["configuration"] = "SUCCESS"
                features_tested.append("Configuration")
                print("  ✅ A2A configuration successful")
            else:
                test_results["configuration"] = "SKIPPED: Import failed"
                print("  ⏸️  A2A configuration skipped (import failed)")
        except Exception as e:
            test_results["configuration"] = f"FAILED: {e}"
            print(f"  ❌ A2A configuration failed: {e}")
        
        # Test 3: Agent Card Message Translation
        try:
            if "a2a_adapter" in locals():
                # Test A2A Agent Card to NIS translation
                a2a_message = {
                    "agentCardHeader": {
                        "messageId": "msg_123",
                        "sessionId": "session_456",
                        "agentId": "external_agent_1",
                        "timestamp": "2025-01-19T10:30:00Z"
                    },
                    "agentCardContent": {
                        "request": {
                            "action": "analyze",
                            "data": {"input": "test data"}
                        }
                    }
                }
                
                nis_message = a2a_adapter.translate_to_nis(a2a_message)
                
                if nis_message and "action" in nis_message:
                    test_results["message_translation"] = "SUCCESS"
                    features_tested.append("Agent Card Translation")
                    print("  ✅ A2A Agent Card translation successful")
                else:
                    test_results["message_translation"] = "FAILED: Invalid NIS message format"
                    print("  ❌ A2A message translation failed: Invalid format")
            else:
                test_results["message_translation"] = "SKIPPED: Adapter not available"
                print("  ⏸️  A2A message translation skipped")
        except Exception as e:
            test_results["message_translation"] = f"FAILED: {e}"
            print(f"  ❌ A2A message translation failed: {e}")
        
        # Test 4: Communication Agent Integration
        try:
            from neural_hierarchy.communication.communication_agent import CommunicationAgent, ProtocolType
            
            # Test A2A protocol handler
            comm_agent = CommunicationAgent("test_comm_agent")
            
            test_results["communication_integration"] = "SUCCESS"
            features_tested.append("Communication Agent Integration")
            print("  ✅ A2A communication agent integration successful")
            
        except Exception as e:
            test_results["communication_integration"] = f"FAILED: {e}"
            print(f"  ❌ A2A communication integration failed: {e}")
        
        # Determine overall status
        successful_tests = [k for k, v in test_results.items() if v == "SUCCESS"]
        
        if len(successful_tests) >= 3:
            status = "OPERATIONAL"
            recommendations.append("A2A protocol fully operational")
        elif len(successful_tests) >= 2:
            status = "CONFIGURATION_NEEDED"
            recommendations.extend([
                "A2A adapter functional but needs external configuration",
                "Configure external A2A agent endpoints for full connectivity"
            ])
        else:
            status = "IMPORT_ISSUES"
            recommendations.extend([
                "Fix A2A adapter import issues",
                "Check neural hierarchy communication module accessibility"
            ])
        
        execution_time = time.time() - start_time
        
        self._add_test_result(
            "A2A Protocol",
            status,
            features_tested,
            test_results,
            recommendations,
            execution_time
        )
    
    def _test_langchain_integration(self):
        """Test LangChain/LangGraph integration"""
        
        start_time = time.time()
        features_tested = []
        test_results = {}
        recommendations = []
        
        # Test 1: LangChain Integration Module Import
        try:
            from integrations.langchain_integration import NISLangChainIntegration, ReasoningPattern
            test_results["integration_import"] = "SUCCESS"
            features_tested.append("Integration Module Import")
            print("  ✅ LangChain integration import successful")
        except Exception as e:
            test_results["integration_import"] = f"FAILED: {e}"
            print(f"  ❌ LangChain integration import failed: {e}")
        
        # Test 2: Dependencies Check
        try:
            # Check if LangChain packages are available
            langchain_available = False
            langgraph_available = False
            langsmith_available = False
            
            try:
                import langchain_core
                langchain_available = True
            except ImportError:
                pass
            
            try:
                import langgraph
                langgraph_available = True
            except ImportError:
                pass
            
            try:
                import langsmith
                langsmith_available = True
            except ImportError:
                pass
            
            test_results["dependencies"] = {
                "langchain": langchain_available,
                "langgraph": langgraph_available,
                "langsmith": langsmith_available
            }
            features_tested.append("Dependencies Check")
            
            available_count = sum([langchain_available, langgraph_available, langsmith_available])
            print(f"  📊 LangChain ecosystem: {available_count}/3 packages available")
            
            if available_count >= 2:
                print("  ✅ Sufficient LangChain packages available")
            else:
                print("  ⚠️  Limited LangChain packages available")
                
        except Exception as e:
            test_results["dependencies"] = f"FAILED: {e}"
            print(f"  ❌ Dependencies check failed: {e}")
        
        # Test 3: Integration Initialization
        try:
            if "NISLangChainIntegration" in locals():
                integration = NISLangChainIntegration(
                    enable_langsmith=False,  # Don't require API key for testing
                    enable_self_audit=True
                )
                
                test_results["initialization"] = "SUCCESS"
                features_tested.append("Integration Initialization")
                print("  ✅ LangChain integration initialization successful")
            else:
                test_results["initialization"] = "SKIPPED: Import failed"
                print("  ⏸️  Integration initialization skipped")
        except Exception as e:
            test_results["initialization"] = f"FAILED: {e}"
            print(f"  ❌ Integration initialization failed: {e}")
        
        # Test 4: Status and Capabilities Check
        try:
            if "integration" in locals():
                status = integration.get_integration_status()
                capabilities = integration.get_capabilities()
                
                test_results["status_check"] = {
                    "status": status,
                    "capabilities": capabilities
                }
                features_tested.append("Status and Capabilities")
                print("  ✅ Integration status check successful")
                print(f"      - Reasoning patterns: {len(capabilities['reasoning_patterns'])}")
                print(f"      - Features available: {sum(capabilities['features'].values())}")
            else:
                test_results["status_check"] = "SKIPPED: Integration not available"
                print("  ⏸️  Status check skipped")
        except Exception as e:
            test_results["status_check"] = f"FAILED: {e}"
            print(f"  ❌ Status check failed: {e}")
        
        # Determine overall status
        successful_tests = [k for k, v in test_results.items() if v == "SUCCESS"]
        
        if len(successful_tests) >= 3:
            status = "OPERATIONAL"
            recommendations.append("LangChain integration fully operational")
        elif len(successful_tests) >= 2:
            status = "CONFIGURATION_NEEDED"
            recommendations.extend([
                "LangChain integration functional but missing some dependencies",
                "Install full LangChain ecosystem: pip install langchain langgraph langsmith"
            ])
        else:
            status = "IMPORT_ISSUES"
            recommendations.extend([
                "Fix LangChain integration import issues",
                "Ensure integration module is properly accessible",
                "Install required dependencies from requirements_tech_stack.txt"
            ])
        
        execution_time = time.time() - start_time
        
        self._add_test_result(
            "LangChain Integration",
            status,
            features_tested,
            test_results,
            recommendations,
            execution_time
        )
    
    def _test_reasoning_patterns(self):
        """Test reasoning patterns (COT, TOT, ReAct)"""
        
        start_time = time.time()
        features_tested = []
        test_results = {}
        recommendations = []
        
        # Test 1: Chain of Thought (COT) Reasoning
        try:
            from integrations.langchain_integration import ChainOfThoughtReasoner
            
            cot_reasoner = ChainOfThoughtReasoner()
            result = cot_reasoner.reason("What is 2 + 2?")
            
            if result and result.final_answer:
                test_results["chain_of_thought"] = "SUCCESS"
                features_tested.append("Chain of Thought (COT)")
                print(f"  ✅ COT reasoning successful (confidence: {result.confidence:.3f})")
            else:
                test_results["chain_of_thought"] = "FAILED: No result generated"
                print("  ❌ COT reasoning failed: No result")
        except Exception as e:
            test_results["chain_of_thought"] = f"FAILED: {e}"
            print(f"  ❌ COT reasoning failed: {e}")
        
        # Test 2: Tree of Thought (TOT) Reasoning
        try:
            from integrations.langchain_integration import TreeOfThoughtReasoner
            
            tot_reasoner = TreeOfThoughtReasoner(max_depth=2, branching_factor=2)  # Reduced for testing
            result = tot_reasoner.reason("What are the benefits of artificial intelligence?")
            
            if result and result.final_answer:
                test_results["tree_of_thought"] = "SUCCESS"
                features_tested.append("Tree of Thought (TOT)")
                print(f"  ✅ TOT reasoning successful (confidence: {result.confidence:.3f})")
                print(f"      - Tree nodes explored: {result.metadata.get('nodes_explored', 0)}")
            else:
                test_results["tree_of_thought"] = "FAILED: No result generated"
                print("  ❌ TOT reasoning failed: No result")
        except Exception as e:
            test_results["tree_of_thought"] = f"FAILED: {e}"
            print(f"  ❌ TOT reasoning failed: {e}")
        
        # Test 3: ReAct (Reasoning and Acting) Pattern
        try:
            from integrations.langchain_integration import ReActReasoner
            
            react_reasoner = ReActReasoner()
            result = react_reasoner.reason("How do you solve a complex problem step by step?")
            
            if result and result.final_answer:
                test_results["reasoning_and_acting"] = "SUCCESS"
                features_tested.append("Reasoning and Acting (ReAct)")
                print(f"  ✅ ReAct reasoning successful (confidence: {result.confidence:.3f})")
                print(f"      - Actions taken: {result.metadata.get('actions_taken', 0)}")
            else:
                test_results["reasoning_and_acting"] = "FAILED: No result generated"
                print("  ❌ ReAct reasoning failed: No result")
        except Exception as e:
            test_results["reasoning_and_acting"] = f"FAILED: {e}"
            print(f"  ❌ ReAct reasoning failed: {e}")
        
        # Test 4: Integrated Reasoning Workflow
        try:
            if "NISLangChainIntegration" in locals():
                integration = NISLangChainIntegration(enable_self_audit=True)
                
                # Test async processing
                async def test_workflow():
                    from integrations.langchain_integration import ReasoningPattern
                    
                    question = "What is the relationship between data and information?"
                    result = await integration.process_question(
                        question, 
                        ReasoningPattern.CHAIN_OF_THOUGHT
                    )
                    return result
                
                # Run async test
                import asyncio
                workflow_result = asyncio.run(test_workflow())
                
                if workflow_result and workflow_result.get("final_answer"):
                    test_results["integrated_workflow"] = "SUCCESS"
                    features_tested.append("Integrated Reasoning Workflow")
                    print("  ✅ Integrated reasoning workflow successful")
                else:
                    test_results["integrated_workflow"] = "FAILED: No workflow result"
                    print("  ❌ Integrated workflow failed: No result")
            else:
                test_results["integrated_workflow"] = "SKIPPED: Integration not available"
                print("  ⏸️  Integrated workflow skipped")
        except Exception as e:
            test_results["integrated_workflow"] = f"FAILED: {e}"
            print(f"  ❌ Integrated workflow failed: {e}")
        
        # Determine overall status
        successful_tests = [k for k, v in test_results.items() if v == "SUCCESS"]
        
        if len(successful_tests) >= 3:
            status = "OPERATIONAL"
            recommendations.append("All reasoning patterns operational")
        elif len(successful_tests) >= 2:
            status = "CONFIGURATION_NEEDED"
            recommendations.extend([
                "Most reasoning patterns working",
                "Optimize reasoning parameters for better performance"
            ])
        else:
            status = "IMPORT_ISSUES"
            recommendations.extend([
                "Fix reasoning pattern implementations",
                "Ensure all reasoning modules are accessible"
            ])
        
        execution_time = time.time() - start_time
        
        self._add_test_result(
            "Reasoning Patterns",
            status,
            features_tested,
            test_results,
            recommendations,
            execution_time
        )
    
    def _test_protocol_routing(self):
        """Test protocol routing and coordination"""
        
        start_time = time.time()
        features_tested = []
        test_results = {}
        recommendations = []
        
        # Test 1: Protocol Routing Configuration
        try:
            config_path = os.path.join(project_root, "src", "configs", "protocol_routing.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    routing_config = json.load(f)
                
                expected_protocols = ["mcp", "acp", "a2a"]
                found_protocols = [p for p in expected_protocols if p in routing_config]
                
                test_results["routing_configuration"] = {
                    "config_file_exists": True,
                    "protocols_found": found_protocols,
                    "total_protocols": len(routing_config)
                }
                features_tested.append("Routing Configuration")
                print(f"  ✅ Protocol routing config found: {len(found_protocols)}/{len(expected_protocols)} protocols")
            else:
                test_results["routing_configuration"] = "FAILED: Config file not found"
                print("  ❌ Protocol routing configuration not found")
        except Exception as e:
            test_results["routing_configuration"] = f"FAILED: {e}"
            print(f"  ❌ Routing configuration test failed: {e}")
        
        # Test 2: Adapter Bootstrap System
        try:
            from adapters.bootstrap import initialize_adapters
            
            # Test adapter initialization
            sample_config = {
                "mcp": {"base_url": "test", "api_key": "test"},
                "a2a": {"base_url": "test", "api_key": "test"}
            }
            
            adapters = initialize_adapters(sample_config)
            
            if adapters and len(adapters) > 0:
                test_results["adapter_bootstrap"] = "SUCCESS"
                features_tested.append("Adapter Bootstrap")
                print(f"  ✅ Adapter bootstrap successful: {len(adapters)} adapters initialized")
            else:
                test_results["adapter_bootstrap"] = "FAILED: No adapters initialized"
                print("  ❌ Adapter bootstrap failed: No adapters")
        except Exception as e:
            test_results["adapter_bootstrap"] = f"FAILED: {e}"
            print(f"  ❌ Adapter bootstrap failed: {e}")
        
        # Test 3: Meta Protocol Coordinator
        try:
            from meta.meta_protocol_coordinator import MetaProtocolCoordinator
            
            coordinator = MetaProtocolCoordinator("test_coordinator")
            
            if coordinator:
                test_results["meta_coordinator"] = "SUCCESS"
                features_tested.append("Meta Protocol Coordinator")
                print("  ✅ Meta protocol coordinator initialized")
            else:
                test_results["meta_coordinator"] = "FAILED: Coordinator not created"
                print("  ❌ Meta protocol coordinator failed")
        except Exception as e:
            test_results["meta_coordinator"] = f"FAILED: {e}"
            print(f"  ❌ Meta protocol coordinator failed: {e}")
        
        # Determine overall status
        successful_tests = [k for k, v in test_results.items() if v == "SUCCESS"]
        
        if len(successful_tests) >= 2:
            status = "OPERATIONAL"
            recommendations.append("Protocol routing system operational")
        elif len(successful_tests) >= 1:
            status = "CONFIGURATION_NEEDED"
            recommendations.extend([
                "Protocol routing partially functional",
                "Complete adapter configuration for full functionality"
            ])
        else:
            status = "IMPORT_ISSUES"
            recommendations.extend([
                "Fix protocol routing system",
                "Ensure all routing components are accessible"
            ])
        
        execution_time = time.time() - start_time
        
        self._add_test_result(
            "Protocol Routing",
            status,
            features_tested,
            test_results,
            recommendations,
            execution_time
        )
    
    def _test_modular_connectivity(self):
        """Test overall modular connectivity and integration"""
        
        start_time = time.time()
        features_tested = []
        test_results = {}
        recommendations = []
        
        # Test 1: Cross-Protocol Communication
        try:
            # Check if we can create a unified communication flow
            protocols_available = []
            
            # Check MCP
            try:
                from adapters.mcp_adapter import MCPAdapter
                protocols_available.append("MCP")
            except:
                pass
            
            # Check A2A
            try:
                from adapters.a2a_adapter import A2AAdapter
                protocols_available.append("A2A")
            except:
                pass
            
            # Check LangChain
            try:
                from integrations.langchain_integration import NISLangChainIntegration
                protocols_available.append("LangChain")
            except:
                pass
            
            test_results["cross_protocol_communication"] = {
                "protocols_available": protocols_available,
                "connectivity_possible": len(protocols_available) >= 2
            }
            features_tested.append("Cross-Protocol Communication")
            
            if len(protocols_available) >= 2:
                print(f"  ✅ Cross-protocol communication possible: {protocols_available}")
            else:
                print(f"  ⚠️  Limited cross-protocol communication: {protocols_available}")
        except Exception as e:
            test_results["cross_protocol_communication"] = f"FAILED: {e}"
            print(f"  ❌ Cross-protocol communication test failed: {e}")
        
        # Test 2: End-to-End Message Flow
        try:
            # Simulate a complete message flow through multiple protocols
            message_flow_steps = []
            
            # Step 1: Create initial message
            initial_message = {
                "content": "Test message for modular connectivity",
                "source": "nis_protocol",
                "timestamp": time.time()
            }
            message_flow_steps.append("Initial message created")
            
            # Step 2: Test MCP translation (if available)
            if "MCP" in protocols_available:
                try:
                    mcp_adapter = MCPAdapter({"base_url": "test", "api_key": "test"})
                    nis_message = mcp_adapter.translate_to_nis({
                        "function_call": {"name": "test", "parameters": {}},
                        "conversation_id": "test"
                    })
                    message_flow_steps.append("MCP translation successful")
                except:
                    message_flow_steps.append("MCP translation failed")
            
            # Step 3: Test reasoning integration (if available)
            if "LangChain" in protocols_available:
                try:
                    integration = NISLangChainIntegration()
                    status = integration.get_integration_status()
                    message_flow_steps.append("LangChain integration connected")
                except:
                    message_flow_steps.append("LangChain integration failed")
            
            test_results["end_to_end_flow"] = {
                "flow_steps": message_flow_steps,
                "successful_steps": len([s for s in message_flow_steps if "successful" in s])
            }
            features_tested.append("End-to-End Message Flow")
            
            successful_steps = len([s for s in message_flow_steps if "successful" in s])
            print(f"  📊 End-to-end flow: {successful_steps}/{len(message_flow_steps)} steps successful")
            
        except Exception as e:
            test_results["end_to_end_flow"] = f"FAILED: {e}"
            print(f"  ❌ End-to-end flow test failed: {e}")
        
        # Test 3: Integration Completeness Assessment
        try:
            # Assess overall integration completeness
            integration_components = {
                "protocol_adapters": len([r for r in self.test_results if "Protocol" in r.protocol_name and r.status == "OPERATIONAL"]),
                "reasoning_patterns": 1 if any(r.protocol_name == "Reasoning Patterns" and r.status == "OPERATIONAL" for r in self.test_results) else 0,
                "routing_system": 1 if any(r.protocol_name == "Protocol Routing" and r.status == "OPERATIONAL" for r in self.test_results) else 0
            }
            
            total_score = sum(integration_components.values())
            max_score = 4  # 2 protocols + 1 reasoning + 1 routing
            
            completeness_percentage = (total_score / max_score) * 100
            
            test_results["integration_completeness"] = {
                "components": integration_components,
                "total_score": total_score,
                "max_score": max_score,
                "completeness_percentage": completeness_percentage
            }
            features_tested.append("Integration Completeness")
            
            print(f"  📊 Integration completeness: {completeness_percentage:.1f}%")
            
        except Exception as e:
            test_results["integration_completeness"] = f"FAILED: {e}"
            print(f"  ❌ Integration completeness assessment failed: {e}")
        
        # Determine overall status based on all previous tests
        operational_protocols = len([r for r in self.test_results if r.status == "OPERATIONAL"])
        total_protocols = len(self.test_results)
        
        if operational_protocols >= 3:
            status = "OPERATIONAL"
            recommendations.append("Excellent modular connectivity achieved")
        elif operational_protocols >= 2:
            status = "CONFIGURATION_NEEDED"
            recommendations.extend([
                "Good modular connectivity with room for improvement",
                "Configure remaining protocols for full connectivity"
            ])
        else:
            status = "IMPORT_ISSUES"
            recommendations.extend([
                "Limited modular connectivity",
                "Address import and configuration issues in protocol adapters"
            ])
        
        execution_time = time.time() - start_time
        
        self._add_test_result(
            "Modular Connectivity",
            status,
            features_tested,
            test_results,
            recommendations,
            execution_time
        )
    
    def _add_test_result(self, protocol_name: str, status: str, features_tested: List[str], 
                        test_results: Dict[str, Any], recommendations: List[str], execution_time: float):
        """Add test result to collection"""
        
        result = ProtocolTestResult(
            protocol_name=protocol_name,
            status=status,
            features_tested=features_tested,
            test_results=test_results,
            recommendations=recommendations,
            execution_time=execution_time
        )
        
        self.test_results.append(result)
    
    def _generate_protocol_report(self) -> Dict[str, Any]:
        """Generate comprehensive protocol integration report"""
        
        total_execution_time = time.time() - self.start_time
        
        # Calculate statistics
        total_protocols = len(self.test_results)
        operational = len([r for r in self.test_results if r.status == "OPERATIONAL"])
        configuration_needed = len([r for r in self.test_results if r.status == "CONFIGURATION_NEEDED"])
        import_issues = len([r for r in self.test_results if r.status == "IMPORT_ISSUES"])
        not_available = len([r for r in self.test_results if r.status == "NOT_AVAILABLE"])
        
        # Calculate operational ratio
        operational_ratio = operational / total_protocols if total_protocols > 0 else 0
        
        # Determine overall integration status
        if operational_ratio >= 0.8:
            overall_status = "EXCELLENT"
        elif operational_ratio >= 0.6:
            overall_status = "GOOD"
        elif operational_ratio >= 0.4:
            overall_status = "ADEQUATE"
        else:
            overall_status = "NEEDS_ATTENTION"
        
        # Collect all recommendations
        all_recommendations = []
        for result in self.test_results:
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return {
            "protocol_integration_summary": {
                "total_protocols_tested": total_protocols,
                "operational": operational,
                "configuration_needed": configuration_needed,
                "import_issues": import_issues,
                "not_available": not_available,
                "operational_ratio": operational_ratio,
                "overall_status": overall_status,
                "total_execution_time": total_execution_time
            },
            "protocol_details": [
                {
                    "protocol_name": r.protocol_name,
                    "status": r.status,
                    "features_tested": r.features_tested,
                    "test_results": r.test_results,
                    "recommendations": r.recommendations,
                    "execution_time": r.execution_time
                }
                for r in self.test_results
            ],
            "recommendations": unique_recommendations[:10]  # Top 10 recommendations
        }


def main():
    """Run comprehensive protocol integration tests"""
    
    tester = ProtocolIntegrationTester()
    report = tester.run_all_protocol_tests()
    
    # Display comprehensive results
    summary = report["protocol_integration_summary"]
    
    print(f"\n🎯 PROTOCOL INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Protocols Tested: {summary['total_protocols_tested']}")
    print(f"✅ Operational: {summary['operational']}")
    print(f"⚙️  Configuration Needed: {summary['configuration_needed']}")
    print(f"❌ Import Issues: {summary['import_issues']}")
    print(f"❓ Not Available: {summary['not_available']}")
    print(f"📊 Operational Ratio: {summary['operational_ratio']:.1%}")
    print(f"🎯 Overall Status: {summary['overall_status']}")
    print(f"⏱️  Total Time: {summary['total_execution_time']:.2f}s")
    
    # Protocol breakdown
    print(f"\n📋 Protocol Status Breakdown:")
    for protocol in report["protocol_details"]:
        status_emoji = {
            "OPERATIONAL": "✅",
            "CONFIGURATION_NEEDED": "⚙️",
            "IMPORT_ISSUES": "❌",
            "NOT_AVAILABLE": "❓"
        }.get(protocol["status"], "❓")
        
        print(f"  {status_emoji} {protocol['protocol_name']}: {protocol['status']}")
        if protocol["features_tested"]:
            print(f"      Features: {', '.join(protocol['features_tested'])}")
    
    # Top recommendations
    print(f"\n💡 Top Integration Recommendations:")
    for i, rec in enumerate(report["recommendations"][:5], 1):
        print(f"  {i}. {rec}")
    
    # Save detailed report
    with open("protocol_integration_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved: protocol_integration_report.json")
    
    # Final assessment
    if summary["overall_status"] == "EXCELLENT":
        print(f"\n🎉 EXCELLENT! NIS Protocol v3 has outstanding modular connectivity!")
        print(f"   All major protocols operational, ready for multi-protocol deployment.")
    elif summary["overall_status"] == "GOOD":
        print(f"\n✅ GOOD! NIS Protocol v3 has solid modular connectivity.")
        print(f"   Most protocols working, minor configuration needed.")
    elif summary["overall_status"] == "ADEQUATE":
        print(f"\n⚠️  ADEQUATE: NIS Protocol v3 has basic modular connectivity.")
        print(f"   Core protocols working, improvement needed for full integration.")
    else:
        print(f"\n🔧 NEEDS ATTENTION: Protocol integration requires fixes.")
        print(f"   Focus on resolving import and configuration issues.")
    
    return report


if __name__ == "__main__":
    report = main() 