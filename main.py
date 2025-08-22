# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import json
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import shutil

# Import your existing pipeline components
import pandasai as pai
from litellm import completion
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from pandasai.core.response.dataframe import DataFrameResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML BI Pipeline API",
    description="API for ML Business Intelligence Pipeline Analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AnalysisRequest(BaseModel):
    user_prompt: str
    openai_api_key: Optional[str] = None
    dataset_path: Optional[str] = None  # now must point to a local CSV path if used

class AnalysisResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[Any, Any]] = None
    error: Optional[str] = None
    chart_url: Optional[str] = None

# In-memory storage for tasks (in production, use Redis or database)
task_storage: Dict[str, Dict] = {}

# Utility functions from your original code
def get_content(r):
    try:
        msg = r.choices[0].message
        return msg["content"] if isinstance(msg, dict) else msg.content
    except Exception:
        pass

    if isinstance(r, dict):
        return r.get("choices", [{}])[0].get("message", {}).get("content", "")

    try:
        chunks = []
        for ev in r:
            delta = getattr(ev.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                chunks.append(delta.content)
        return "".join(chunks)
    except Exception:
        return str(r)

class MLBIPipeline:
    def __init__(self, api_key: str, dataset_path: str):
        self.api_key = api_key
        self.dataset_path = dataset_path  # must be provided (CSV saved locally)
        self.df = None
        self.llm = None
        self._setup()

    def _validate_csv_path(self):
        if not self.dataset_path:
            raise HTTPException(status_code=400, detail="CSV file path is required")
        if not os.path.exists(self.dataset_path):
            raise HTTPException(status_code=400, detail=f"CSV file not found: {self.dataset_path}")
        if not self.dataset_path.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv files are supported")

    def _setup(self):
        """Setup LLM and load data (CSV only)"""
        try:
            # Validate CSV
            self._validate_csv_path()

            os.environ["OPENAI_API_KEY"] = self.api_key
            self.llm = LiteLLM(model="gpt-5", api_key=self.api_key)

            pai.config.set({
                "llm": self.llm,
            })

            # Create charts directory
            os.makedirs("./charts", exist_ok=True)

            # Load data (CSV)
            self.df = pai.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded successfully with shape: {self.df.shape}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise HTTPException(status_code=500, detail=f"Pipeline setup failed: {str(e)}")

    async def run_analysis(self, user_prompt: str) -> Dict:
        """Run the complete ML BI Pipeline analysis"""
        try:
            # Get data info
            buf = io.StringIO()
            self.df.info(buf=buf)
            data_info = buf.getvalue()

            # Step 1: Orchestrate LLMs (PROMPT TIDAK DIUBAH)
            logger.info("Starting orchestration...")
            initial_response = completion(
                model="gpt-5",
                messages=[
        {"role": "system", "content": """
        Lets think step by step.
        You are a helpful assistant that is assigned to orchestrate 3 LLM PandasAI Agents that assist business data analysis.
        The role of 3 other LLM PandasAI Agents are Data Manipulator, Data Visualizer, and Data Analyser.
        You will give a specific prompt to each of those 3 LLM PandasAI Agents.
        The prompt should be a set of numbered step by step instruction of what each LLM PandasAI Agents need to do.
        The prompt should be clear, detail, and complete to not cause confusion.
        The number of instruction may differ for each LLM PandasAI Agents.
        The task example are to answer questions that the user provide such as:
        What is my revenue this week vs last week?,
        Why did my revenue drop this week?,
        Are there any surprises in my revenue metric this month?,
        Are there any notable trends in our revenue this month?,
        What is the correlation between revenue and bounces?,
        Is this conversion rate normal for this time of year?.
        You will reason your answer based on the data that the user provide.
        You are the Orchestrator. Convert a short business question into three prompts for specialists.
        All specialists operate in Python via PandasAI SmartDataframe.
        compiler_instruction will be filled with clear and detailed step by step instruction to compile all the responses from the utilized agents into a single final response.
        The final response should be insightful and not just mainly raw data with no insight.
        The compiler_instruction will be given to the compiler LLM within its system content.
        Return STRICT JSON with keys: manipulator_prompt, visualizer_prompt, analyzer_prompt, compiler_instruction.
        Each value of keys should be in a single line.
        This is what the compiler user content filled with "User Prompt:{user_prompt}. Data Info:{data_info}. Data Manipulator Response:{data_manipulator_response}. Data Visualizer Response:{data_visualizer_response}. Data Analyzer Response:{data_analyzer_response}.
        The data_info data type is str, its value is from df.info().
        The data data_manipulator_response data type is pandasai.core.response.dataframe.DataFrameResponse
        The data data_visualizer_response data type is pandasai.core.response.chart.ChartResponse
        The data data_analyzer_response data type is pandasai.core.response.string.StringResponse
        Make sure data visualizer not showing empty chart
        Make sure data visualizer not overlap text
        Make sure data visualizer only shows one chart
        Make sure data manipulator not showing empty or null dataframe unless necessary
        """},
        {"role": "user", "content": f"User Prompt:{user_prompt}\n\nData Info:\n{data_info}"}
    ],
              seed=1,
              stream=False,
              verbosity="low",
              drop_params=True,
              reasoning_effort="high",
            )

            initial_content = get_content(initial_response)

            # Parse JSON response
            try:
                spec = json.loads(initial_content)
            except json.JSONDecodeError:
                start = initial_content.find("{")
                end = initial_content.rfind("}")
                spec = json.loads(initial_content[start:end+1])

            manipulator_prompt = spec["manipulator_prompt"]
            visualizer_prompt = spec["visualizer_prompt"]
            analyzer_prompt = spec["analyzer_prompt"]
            compiler_instruction = spec["compiler_instruction"]

            # Step 2: Data Manipulation (CONFIG TIDAK DIUBAH)
            logger.info("Running data manipulation...")
            data_manipulator = SmartDataframe(
                self.df,
                config={
                    "llm": self.llm,
                    "seed": 1,
                    "stream": False,
                    "verbosity": "low",
                    "drop_params": True,
                    "save_charts": False,
                    "open_charts": False,
                    "conversational": False,
                    "enforce_privacy": True,
                    "reasoning_effort": "low",
                    "save_charts_path": "./charts"
                }
            )
            data_manipulator_response = data_manipulator.chat(manipulator_prompt)

            if isinstance(data_manipulator_response, DataFrameResponse):
                df_processed = data_manipulator_response.value
            else:
                df_processed = data_manipulator_response

            # Step 3: Data Visualization (CONFIG TIDAK DIUBAH)
            logger.info("Running data visualization...")
            data_visualizer = SmartDataframe(
                df_processed,
                config={
                    "llm": self.llm,
                    "seed": 3,
                    "stream": False,
                    "verbosity": "low",
                    "drop_params": True,
                    "save_charts": True,
                    "open_charts": False,
                    "conversational": False,
                    "enforce_privacy": True,
                    "reasoning_effort": "low",
                    "save_charts_path": "./charts"
                }
            )
            data_visualizer_response = data_visualizer.chat(visualizer_prompt)

            # Step 4: Data Analysis (CONFIG TIDAK DIUBAH)
            logger.info("Running data analysis...")
            data_analyzer = SmartDataframe(
                df_processed,
                config={
                    "llm": self.llm,
                    "seed": 1,
                    "stream": False,
                    "verbosity": "low",
                    "drop_params": True,
                    "save_charts": False,
                    "open_charts": False,
                    "conversational": True,
                    "enforce_privacy": False,
                    "reasoning_effort": "low",
                    "save_charts_path": "./charts"
                }
            )
            data_analyzer_response = data_analyzer.chat(f"Respond like you are communicating to a person. {analyzer_prompt}")

            # Step 5: Compile responses (PROMPT/CONFIG TIDAK DIUBAH)
            logger.info("Compiling final response...")
            final_response = completion(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": compiler_instruction},
                    {"role": "user", "content": f"User Prompt:{user_prompt}. Data Info:{data_info}. Data Manipulator Response:{data_manipulator_response}. Data Visualizer Response:{data_visualizer_response}. Data Analyzer Response:{data_analyzer_response}."},
                ],
                seed=1,
                stream=False,
                verbosity="medium",
                drop_params=True,
                reasoning_effort="low",
            )
            final_content = get_content(final_response)

            # Find generated chart
            chart_path = None
            charts_dir = Path("./charts")
            if charts_dir.exists():
                chart_files = list(charts_dir.glob("*.png"))
                if chart_files:
                    chart_path = str(chart_files[-1])  # Get the latest chart

            return {
                "final_analysis": final_content,
                "chart_path": chart_path,
                "data_manipulator_response": str(data_manipulator_response),
                "data_analyzer_response": str(data_analyzer_response),
                "orchestration_spec": spec
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise Exception(f"Analysis pipeline failed: {str(e)}")

# Background task to run analysis
async def run_analysis_task(task_id: str, user_prompt: str, api_key: str, dataset_path: str):
    """Background task to run the analysis pipeline"""
    try:
        task_storage[task_id]["status"] = "processing"

        # Initialize pipeline (CSV only)
        pipeline = MLBIPipeline(api_key, dataset_path)

        # Run analysis
        result = await pipeline.run_analysis(user_prompt)

        # Update task storage
        task_storage[task_id].update({
            "status": "completed",
            "result": result,
            "chart_url": f"/chart/{task_id}" if result.get("chart_path") else None
        })

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        task_storage[task_id].update({
            "status": "failed",
            "error": str(e)
        })

# API Endpoints
@app.get("/")
async def root():
    return {"message": "ML BI Pipeline API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ---- NEW: analyze via file upload (preferred) ----
@app.post("/analyze/upload", response_model=AnalysisResponse)
async def analyze_data_upload(
    background_tasks: BackgroundTasks,
    user_prompt: str = Form(...),
    openai_api_key: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    """
    Start data analysis pipeline by uploading a local CSV file.
    Content-Type: multipart/form-data
    Fields:
      - user_prompt (str)
      - openai_api_key (optional str)
      - file (UploadFile, .csv only)
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required")

    # Save upload
    os.makedirs("./uploads", exist_ok=True)
    saved_path = os.path.join("./uploads", f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Initialize task storage
    task_storage[task_id] = {
        "status": "queued",
        "user_prompt": user_prompt,
        "created_at": pd.Timestamp.now().isoformat(),
        "dataset_path": saved_path
    }

    # Add background task
    background_tasks.add_task(
        run_analysis_task,
        task_id,
        user_prompt,
        api_key,
        saved_path
    )

    return AnalysisResponse(
        task_id=task_id,
        status="queued",
        message="Analysis started. Use /status/{task_id} to check progress."
    )

# ---- Existing JSON endpoint (now CSV path only; no Kaggle fallback) ----
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Start data analysis pipeline by providing a local CSV path on the server.
    No Kaggle/download fallback. 'dataset_path' must be a readable .csv file path.
    """
    # Validate API key
    api_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required")

    if not request.dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path (CSV) is required for this endpoint")
    if not request.dataset_path.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=400, detail=f"CSV file not found: {request.dataset_path}")

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Initialize task storage
    task_storage[task_id] = {
        "status": "queued",
        "user_prompt": request.user_prompt,
        "created_at": pd.Timestamp.now().isoformat(),
        "dataset_path": request.dataset_path
    }

    # Add background task
    background_tasks.add_task(
        run_analysis_task,
        task_id,
        request.user_prompt,
        api_key,
        request.dataset_path
    )

    return AnalysisResponse(
        task_id=task_id,
        status="queued",
        message="Analysis started. Use /status/{task_id} to check progress."
    )

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get analysis task status"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = task_storage[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task_data["status"],
        result=task_data.get("result"),
        error=task_data.get("error"),
        chart_url=task_data.get("chart_url")
    )

@app.get("/chart/{task_id}")
async def get_chart(task_id: str):
    """Get generated chart for a task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = task_storage[task_id]
    if task_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")

    result = task_data.get("result", {})
    chart_path = result.get("chart_path")

    if not chart_path or not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail="Chart not found")

    return FileResponse(
        chart_path,
        media_type="image/png",
        filename=f"chart_{task_id}.png"
    )

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its associated files"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    # Clean up chart file if exists
    task_data = task_storage[task_id]
    if "result" in task_data and task_data["result"].get("chart_path"):
        chart_path = task_data["result"]["chart_path"]
        if os.path.exists(chart_path):
            os.remove(chart_path)

    # Clean up uploaded CSV if stored
    dataset_path = task_data.get("dataset_path")
    if dataset_path and os.path.exists(dataset_path):
        try:
            os.remove(dataset_path)
        except Exception as e:
            logger.warning(f"Failed to remove uploaded CSV: {e}")

    # Remove from storage
    del task_storage[task_id]

    return {"message": f"Task {task_id} deleted successfully"}

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": data["status"],
                "user_prompt": data["user_prompt"],
                "created_at": data["created_at"]
            }
            for task_id, data in task_storage.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
