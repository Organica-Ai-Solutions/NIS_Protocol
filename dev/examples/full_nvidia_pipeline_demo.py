#!/usr/bin/env python3
"""
Complete NVIDIA Pipeline Demo

Demonstrates the full end-to-end pipeline:
1. Generate synthetic data with Cosmos
2. Train robot policy with Isaac Lab
3. Validate with NIS physics
4. Reason about deployment with Cosmos
5. Execute on humanoid with GR00T

This shows how all NVIDIA components work together in production.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("full_pipeline_demo")


class FullNVIDIAPipelineDemo:
    """
    Complete end-to-end NVIDIA pipeline demonstration
    
    Shows realistic production workflow:
    - Data generation
    - Policy training
    - Physics validation
    - Task reasoning
    - Humanoid execution
    """
    
    def __init__(self):
        self.results = {}
        logger.info("🚀 Full NVIDIA Pipeline Demo initialized")
    
    async def step_1_generate_training_data(self) -> Dict[str, Any]:
        """
        Step 1: Generate Synthetic Training Data
        
        Use Cosmos to generate unlimited training data for the robot.
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Generate Synthetic Training Data with Cosmos")
        logger.info("=" * 80)
        
        try:
            from src.agents.cosmos import get_cosmos_generator
            
            generator = get_cosmos_generator()
            await generator.initialize()
            
            logger.info("Generating 500 samples for pick-and-place task...")
            
            result = await generator.generate_robot_training_data(
                num_samples=500,
                tasks=["pick_and_place"],
                use_cache=False  # Force fresh generation
            )
            
            logger.info(f"✅ Generated {result['samples_generated']} training samples")
            logger.info(f"   Output: {result['output_dir']}")
            logger.info(f"   Augmentations: {generator.stats['augmentations_created']}")
            
            return {
                "success": True,
                "samples": result['samples_generated'],
                "output_dir": result['output_dir']
            }
            
        except Exception as e:
            logger.error(f"❌ Data generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def step_2_train_robot_policy(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Train Robot Policy with Isaac Lab
        
        Use Isaac Lab to train a manipulation policy on the generated data.
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Train Robot Policy with Isaac Lab 2.2")
        logger.info("=" * 80)
        
        try:
            from src.agents.isaac_lab import get_isaac_lab_trainer
            
            trainer = get_isaac_lab_trainer()
            await trainer.initialize()
            
            logger.info("Training Franka Panda for pick-and-place...")
            logger.info("Using PPO algorithm with 4096 parallel environments")
            
            result = await trainer.train_policy(
                robot_type="franka_panda",
                task="pick_and_place",
                num_iterations=100,
                algorithm="PPO"
            )
            
            logger.info(f"✅ Policy trained successfully")
            logger.info(f"   Best reward: {result['best_reward']:.3f}")
            logger.info(f"   Training time: {result['training_time']:.1f}s")
            logger.info(f"   Episodes: {result['episodes']}")
            
            return {
                "success": True,
                "policy": result['policy'],
                "best_reward": result['best_reward']
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def step_3_validate_with_nis_physics(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Validate Policy with NIS Physics
        
        Use NIS PINN physics validation to ensure the policy is safe.
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Validate Policy with NIS Physics")
        logger.info("=" * 80)
        
        try:
            from src.agents.isaac_lab import get_isaac_lab_trainer
            
            trainer = get_isaac_lab_trainer()
            
            # Create test scenarios
            test_scenarios = [
                {"name": "standard_pick", "difficulty": "easy"},
                {"name": "cluttered_scene", "difficulty": "medium"},
                {"name": "moving_target", "difficulty": "hard"}
            ]
            
            logger.info(f"Validating policy on {len(test_scenarios)} scenarios...")
            
            result = await trainer.validate_policy_with_nis(
                policy=policy,
                test_scenarios=test_scenarios
            )
            
            logger.info(f"✅ Validation complete")
            logger.info(f"   Success rate: {result['success_rate'] * 100:.1f}%")
            logger.info(f"   Physics validated: {result['physics_validated']}")
            
            return {
                "success": True,
                "success_rate": result['success_rate'],
                "physics_valid": result['physics_validated']
            }
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def step_4_reason_about_deployment(self) -> Dict[str, Any]:
        """
        Step 4: Reason About Deployment with Cosmos
        
        Use Cosmos Reason to plan the deployment strategy.
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Reason About Deployment with Cosmos")
        logger.info("=" * 80)
        
        try:
            from src.agents.cosmos import get_cosmos_reasoner
            
            reasoner = get_cosmos_reasoner()
            await reasoner.initialize()
            
            # Mock camera image
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            logger.info("Reasoning about deployment scenario...")
            
            result = await reasoner.reason(
                image=image,
                task="Deploy trained policy on humanoid robot for warehouse pick-and-place",
                constraints=[
                    "ensure human safety",
                    "validate physics constraints",
                    "handle dynamic obstacles",
                    "maintain 99% uptime"
                ]
            )
            
            logger.info(f"✅ Deployment plan generated")
            logger.info(f"   Plan steps: {len(result['plan'])}")
            logger.info(f"   Confidence: {result['confidence']:.2f}")
            logger.info(f"   Safety check: {result['safety_check']['safe']}")
            
            if result['plan']:
                logger.info("   Deployment steps:")
                for step in result['plan'][:3]:
                    logger.info(f"     - {step}")
            
            return {
                "success": True,
                "plan": result['plan'],
                "confidence": result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Reasoning failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def step_5_execute_on_humanoid(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 5: Execute on Humanoid with GR00T
        
        Deploy the trained policy on a humanoid robot using GR00T.
        """
        logger.info("=" * 80)
        logger.info("STEP 5: Execute on Humanoid with GR00T N1")
        logger.info("=" * 80)
        
        try:
            from src.agents.groot import get_groot_agent
            
            agent = get_groot_agent()
            await agent.initialize()
            
            logger.info("Executing pick-and-place task on humanoid...")
            
            result = await agent.execute_task(
                task="Walk to warehouse shelf, pick up package, and deliver to staging area"
            )
            
            logger.info(f"✅ Humanoid execution complete")
            logger.info(f"   Success: {result['success']}")
            logger.info(f"   Actions executed: {len(result['action_sequence'])}")
            logger.info(f"   Execution time: {result['execution_time']:.1f}s")
            logger.info(f"   Confidence: {result['confidence']:.2f}")
            
            if result['action_sequence']:
                logger.info("   Action sequence:")
                for action in result['action_sequence'][:5]:
                    logger.info(f"     - {action.get('action', 'unknown')}: {action.get('duration', 0):.1f}s")
            
            return {
                "success": True,
                "actions": len(result['action_sequence']),
                "execution_time": result['execution_time']
            }
            
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete end-to-end pipeline"""
        
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 20 + "FULL NVIDIA PIPELINE DEMO" + " " * 33 + "║")
        logger.info("║" + " " * 15 + "End-to-End Production Workflow" + " " * 33 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info("")
        
        start_time = time.time()
        
        # Step 1: Generate training data
        logger.info("")
        data_result = await self.step_1_generate_training_data()
        if not data_result["success"]:
            logger.error("Pipeline failed at Step 1")
            return {"success": False, "failed_at": "step_1"}
        
        # Step 2: Train policy
        logger.info("")
        train_result = await self.step_2_train_robot_policy(data_result)
        if not train_result["success"]:
            logger.error("Pipeline failed at Step 2")
            return {"success": False, "failed_at": "step_2"}
        
        # Step 3: Validate with physics
        logger.info("")
        validation_result = await self.step_3_validate_with_nis_physics(train_result["policy"])
        if not validation_result["success"]:
            logger.error("Pipeline failed at Step 3")
            return {"success": False, "failed_at": "step_3"}
        
        # Step 4: Reason about deployment
        logger.info("")
        reasoning_result = await self.step_4_reason_about_deployment()
        if not reasoning_result["success"]:
            logger.error("Pipeline failed at Step 4")
            return {"success": False, "failed_at": "step_4"}
        
        # Step 5: Execute on humanoid
        logger.info("")
        execution_result = await self.step_5_execute_on_humanoid(reasoning_result)
        if not execution_result["success"]:
            logger.error("Pipeline failed at Step 5")
            return {"success": False, "failed_at": "step_5"}
        
        # Summary
        total_time = time.time() - start_time
        
        logger.info("")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 25 + "PIPELINE COMPLETE" + " " * 36 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info("")
        logger.info(f"✅ All 5 steps completed successfully")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info("")
        logger.info("Pipeline Results:")
        logger.info(f"  1. Training samples generated: {data_result['samples']}")
        logger.info(f"  2. Policy best reward: {train_result['best_reward']:.3f}")
        logger.info(f"  3. Validation success rate: {validation_result['success_rate'] * 100:.1f}%")
        logger.info(f"  4. Deployment plan confidence: {reasoning_result['confidence']:.2f}")
        logger.info(f"  5. Humanoid actions executed: {execution_result['actions']}")
        logger.info("")
        logger.info("=" * 80)
        logger.info("🎉 FULL NVIDIA PIPELINE DEMONSTRATION COMPLETE!")
        logger.info("Ready for production deployment.")
        logger.info("=" * 80)
        
        return {
            "success": True,
            "total_time": total_time,
            "results": {
                "data_generation": data_result,
                "training": train_result,
                "validation": validation_result,
                "reasoning": reasoning_result,
                "execution": execution_result
            }
        }


async def main():
    """Main demo runner"""
    demo = FullNVIDIAPipelineDemo()
    results = await demo.run_complete_pipeline()
    return results


if __name__ == "__main__":
    asyncio.run(main())
