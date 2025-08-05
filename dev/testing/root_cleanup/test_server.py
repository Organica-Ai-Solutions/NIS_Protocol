import os
import sys
import logging
import json
import time
import random
import math
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Depends, Request, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nis_general_pattern")

# Create FastAPI app
app = FastAPI(title="NIS Protocol API", version="3.1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define endpoint for simulation
@app.post("/agents/simulation/run", tags=["Agents"])
async def run_simulation(request: Request):
    """
    Run a simulation using the Enhanced Scenario Simulator.
    """
    try:
        # Parse JSON body manually to ensure we get the data
        body = await request.json()
        
        # Extract parameters
        scenario_id = body.get("scenario_id", "unknown")
        scenario_type = body.get("scenario_type", "unknown")
        parameters = body.get("parameters", {})
        
        logger.info(f"Received simulation request: {scenario_id}, {scenario_type}")
        logger.info(f"Parameters: {parameters}")
        
        # Calculate success probability based on input parameters
        success_probability = calculate_success_probability(scenario_type, parameters)
        
        # Generate expected outcomes based on parameters
        expected_outcomes = generate_expected_outcomes(scenario_type, parameters)
        
        # Calculate risk factors based on parameters
        risk_factors = calculate_risk_factors(scenario_type, parameters)
        
        # Calculate resource utilization based on parameters
        resource_utilization = calculate_resource_utilization(scenario_type, parameters)
        
        # Calculate timeline based on parameters
        timeline = calculate_timeline(scenario_type, parameters)
        
        # Generate recommendations based on all calculated data
        recommendations = generate_recommendations(
            scenario_type, 
            parameters, 
            success_probability, 
            risk_factors
        )
        
        # Construct result
        result = {
            "scenario_id": scenario_id,
            "scenario_type": scenario_type,
            "success_probability": success_probability,
            "expected_outcomes": expected_outcomes,
            "risk_factors": risk_factors,
            "resource_utilization": resource_utilization,
            "timeline": timeline,
            "recommendations": recommendations
        }
        
        logger.info(f"Simulation completed for {scenario_id} with success probability: {success_probability}")
        
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

def calculate_success_probability(scenario_type: str, parameters: Dict[str, Any]) -> float:
    """Calculate success probability based on input parameters"""
    base_probability = 0.7  # Starting point
    
    # Adjust based on parameters
    if scenario_type == "archaeological_excavation":
        # Environmental factors impact
        weather_factor = parameters.get("environmental_factors", {}).get("weather", 0.5)
        base_probability *= (0.5 + weather_factor/2)
        
        # Team size impact
        team_size = parameters.get("resource_constraints", {}).get("team_size", 5)
        team_factor = min(1.2, max(0.8, team_size / 10))
        base_probability *= team_factor
        
        # Time horizon impact
        time_horizon = parameters.get("time_horizon", 30)
        time_factor = min(1.2, max(0.8, time_horizon / 60))
        base_probability *= time_factor
    
    # Apply some randomness but within reasonable bounds
    variation = random.uniform(-0.05, 0.05)
    final_probability = max(0.01, min(0.99, base_probability + variation))
    
    return round(final_probability, 2)

def generate_expected_outcomes(scenario_type: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate expected outcomes based on parameters"""
    outcomes = []
    
    if scenario_type == "archaeological_excavation":
        # Calculate artifact discovery probability
        discovery_rate = parameters.get("uncertainty_factors", {}).get("discovery_rate", 0.5)
        team_size = parameters.get("resource_constraints", {}).get("team_size", 5)
        
        artifact_probability = min(0.95, max(0.3, discovery_rate * (1 + team_size/20)))
        outcomes.append({
            "outcome": "artifacts_found", 
            "probability": round(artifact_probability, 2)
        })
        
        # Calculate site preservation probability
        site_preservation = min(0.95, max(0.4, 1 - discovery_rate/2))
        outcomes.append({
            "outcome": "site_preserved", 
            "probability": round(site_preservation, 2)
        })
        
        # Add additional outcomes based on objectives
        objectives = parameters.get("objectives", [])
        if "minimize_site_impact" in objectives:
            impact_probability = min(0.9, max(0.5, site_preservation * 0.9))
            outcomes.append({
                "outcome": "minimal_environmental_impact", 
                "probability": round(impact_probability, 2)
            })
    
    return outcomes

def calculate_risk_factors(scenario_type: str, parameters: Dict[str, Any]) -> List[Dict[str, str]]:
    """Calculate risk factors based on parameters"""
    risk_factors = []
    
    if scenario_type == "archaeological_excavation":
        # Weather risk
        weather_factor = parameters.get("environmental_factors", {}).get("weather", 0.5)
        if weather_factor < 0.4:
            weather_risk = "High risk due to poor weather conditions"
        elif weather_factor < 0.7:
            weather_risk = "Medium risk due to weather conditions"
        else:
            weather_risk = "Low risk from weather conditions"
        risk_factors.append({"type": "weather_risk", "description": weather_risk})
        
        # Equipment risk based on team size
        team_size = parameters.get("resource_constraints", {}).get("team_size", 5)
        if team_size < 5:
            equipment_risk = "Medium risk for equipment failures due to small team"
        else:
            equipment_risk = "Low risk for equipment failures"
        risk_factors.append({"type": "equipment_risk", "description": equipment_risk})
        
        # Time risk
        time_horizon = parameters.get("time_horizon", 30)
        if time_horizon < 30:
            time_risk = "High risk of timeline pressure"
        elif time_horizon < 60:
            time_risk = "Medium risk of timeline pressure"
        else:
            time_risk = "Low risk of timeline pressure"
        risk_factors.append({"type": "timeline_risk", "description": time_risk})
    
    return risk_factors

def calculate_resource_utilization(scenario_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate resource utilization based on parameters"""
    resource_utilization = {}
    
    if scenario_type == "archaeological_excavation":
        # Team size
        team_size = parameters.get("resource_constraints", {}).get("team_size", 5)
        resource_utilization["team_archaeologists"] = team_size
        
        # Equipment cost based on team size and time horizon
        time_horizon = parameters.get("time_horizon", 30)
        base_equipment_cost = 5000
        scaling_factor = (team_size / 5) * (time_horizon / 30)
        equipment_cost = int(base_equipment_cost * scaling_factor)
        resource_utilization["equipment_cost"] = equipment_cost
        
        # Lab hours
        lab_hours = int(50 * team_size * (time_horizon / 30))
        resource_utilization["laboratory_hours"] = lab_hours
    
    return resource_utilization

def calculate_timeline(scenario_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate timeline based on parameters"""
    timeline = {}
    
    if scenario_type == "archaeological_excavation":
        # Preparation time
        team_size = parameters.get("resource_constraints", {}).get("team_size", 5)
        prep_weeks = max(1, min(4, 2 + (10 - team_size) / 10))
        timeline["preparation_weeks"] = round(prep_weeks)
        
        # Excavation time
        time_horizon = parameters.get("time_horizon", 30)
        excavation_weeks = max(2, min(20, time_horizon / 7))
        timeline["excavation_weeks"] = round(excavation_weeks)
        
        # Analysis time
        analysis_weeks = max(2, min(12, excavation_weeks * 0.6))
        timeline["analysis_weeks"] = round(analysis_weeks)
    
    return timeline

def generate_recommendations(scenario_type: str, parameters: Dict[str, Any], 
                           success_probability: float, risk_factors: List[Dict[str, str]]) -> List[str]:
    """Generate recommendations based on all calculated data"""
    recommendations = []
    
    if scenario_type == "archaeological_excavation":
        # Team size recommendations
        team_size = parameters.get("resource_constraints", {}).get("team_size", 5)
        if team_size < 8 and success_probability < 0.8:
            recommendations.append(f"Increase team size by {max(20, int((8-team_size)*20))}%")
        
        # Timeline recommendations
        time_horizon = parameters.get("time_horizon", 30)
        if time_horizon < 60 and any("timeline_risk" in rf["type"] for rf in risk_factors):
            recommendations.append("Extend project timeline to reduce pressure risks")
        
        # General recommendations
        recommendations.append("Establish clear milestone checkpoints")
        
        # Weather-specific recommendations
        if any("weather_risk" in rf["type"] and "High" in rf["description"] for rf in risk_factors):
            recommendations.append("Implement weather contingency plans")
    
    return recommendations[:3]  # Limit to top 3 recommendations

# Root endpoint
@app.get("/", tags=["Status"])
async def root():
    """Root endpoint returning API status"""
    return {"status": "online", "message": "NIS Protocol API v3.1"}

# Health check endpoint
@app.get("/health", tags=["Status"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    port = 8001
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port) 