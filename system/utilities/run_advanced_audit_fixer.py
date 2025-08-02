
import asyncio
import logging
import json
from src.agents.action.audit_fixing_agent import AuditFixingActionAgent

async def main():
    """
    Initializes and runs the AuditFixingActionAgent to systematically
    correct integrity violations in the project.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create the advanced agent
    fixer_agent = AuditFixingActionAgent()
    
    # Define the directories to scan and fix
    # Focusing on the directories with the most issues
    target_dirs = ['src', 'system/docs']
    
    logging.info(f"Starting advanced audit fixing session for: {target_dirs}")
    
    # Start a fixing session
    session_id = await fixer_agent.start_audit_fixing_session(target_directories=target_dirs)
    
    # Get and print the session report
    report = fixer_agent.get_session_report(session_id)
    
    logging.info("Advanced audit fixing session complete. Summary:")
    logging.info(json.dumps(report, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
