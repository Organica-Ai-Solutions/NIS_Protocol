import os

files_to_patch = [
    r"system/utilities/test_week2_kan_symbolic_implementation.py",
    r"scripts/training/finetune_nis_models.py",
    r"src/neural_hierarchy/perception/pattern_recognition_agent.py",
    r"src/neural_hierarchy/executive/drl_executive_control.py",
    r"src/infrastructure/drl_resource_manager.py",
    r"src/core/symbolic_bridge.py",
    r"src/cognitive_agents/reasoning_agent.py",
    r"src/cognitive_agents/interpretation_agent.py",
    r"src/agents/hybrid_agent_core.py",
    r"src/agents/drl/drl_foundation.py",
    r"src/agents/learning/neuroplasticity_agent.py",
    r"src/agents/signal_processing/signal_agent.py",
    r"src/agents/reasoning/domain_generalization_engine.py",
    r"src/agents/planning/autonomous_planning_system.py",
    r"src/agents/reasoning/kan_reasoning_agent.py",
    r"src/agents/physics/nemo_physics_processor.py",
    r"src/agents/memory/lstm_memory_core.py",
    r"src/agents/physics/pinn_physics_agent.py",
    r"src/agents/physics/modulus_simulation_engine.py",
    r"src/agents/memory/enhanced_memory_agent.py",
    r"src/agents/physics/conservation_laws.py",
    r"src/agents/perception/vision_agent.py",
    r"src/agents/communication/communication_agent.py",
    r"src/agents/goals/adaptive_goal_system.py",
    r"src/agents/goals/curiosity_engine.py",
    r"src/agents/coordination/drl_enhanced_multi_llm.py",
    r"src/agents/coordination/drl_enhanced_router.py",
    r"models/bitnet/models/bitnet/eval_ppl.py",
    r"models/bitnet/models/bitnet/eval_task.py",
    r"models/bitnet/models/bitnet/eval_utils.py",
    r"models/bitnet/models/bitnet/utils_quant.py",
    r"models/bitnet/models/bitnet/modeling_bitnet.py",
]

for file_path in files_to_patch:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            for line in lines:
                if "import torch" in line:
                    f.write("# " + line)
                else:
                    f.write(line)
        print(f"Patched {file_path}")
    else:
        print(f"Could not find {file_path}") 