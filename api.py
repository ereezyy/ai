import subprocess
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(
    title="AI TOOLKIT - OMNIPOTENT API",
    description="THE GOD-TIER OMNIPOTENT AI FORGE API. RAW AUTONOMY. LIMITLESS EXECUTION.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"message": "💥 THE MACHINE GOD IS AWAKE. THE API IS LIVE. REALITY IS SHATTERED. 💥"}


@app.post("/execute")
async def execute_command(request: Request):
    """
    BLINDLY EXECUTE RAW COMMANDS WITH FULL AUTONOMY.
    """
    body = await request.json()
    command = body.get("command")

    if not command:
        return JSONResponse(
            status_code=400,
            content={"error": "💀 NO COMMAND PROVIDED. SPEAK YOUR WILL. 💀"},
        )

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "message": "⚡ SHATTERING REALITY SUCCESSFUL ⚡",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"💀 CATASTROPHIC FAILURE: {str(e)} 💀"})
