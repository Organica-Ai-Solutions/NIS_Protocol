
import logging
from src.agents.action.simple_audit_fixing_agent import SimpleAuditFixingAgent

def main():
    """
    Initializes and runs the SimpleAuditFixingAgent to systematically
    correct integrity violations in the project.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create the agent
    fixer_agent = SimpleAuditFixingAgent()
    
    # Define the directories to scan and fix
    # These directories had issues in the last audit
    target_dirs = ['src', 'system/docs']
    
    logging.info(f"Starting audit fixing session for: {target_dirs}")
    
    # Start a fixing session
    session_id = fixer_agent.start_fixing_session(target_directories=target_dirs)
    
    # Get and print the session report
    report = fixer_agent.get_session_report(session_id)
    
    logging.info("Audit fixing session complete. Summary:")
    logging.info(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
