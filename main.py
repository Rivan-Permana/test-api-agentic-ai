# main.py — versi perbaikan (upload lokal → GCS → analisis → status & chart)

import io
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pandasai as pai
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google.cloud import firestore, storage
from litellm import completion
from pandasai import SmartDataframe
from pandasai.core.response.dataframe import DataFrameResponse
from pandasai_litellm.litellm import LiteLLM
from pydantic import BaseModel

# --- KONFIGURASI ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi Klien Google Cloud
try:
    storage_client = storage.Client()
    firestore_client = firestore.Client()
    BUCKET_NAME = "ml-bi-pipeline-uploads-2025"
    logger.info("Successfully initialized Google Cloud clients.")
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud clients: {e}")
    storage_client = None
    firestore_client = None

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="ML BI Pipeline API",
    description="API for ML Business Intelligence Pipeline Analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL DATA (PYDANTIC SCHEMAS) ---
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


# --- UTIL LLM ---
def get_content(r):
    """Ambil konten dari response litellm (support non-stream dan stream)."""
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


# --- KELAS LOGIKA INTI (PIPELINE) ---
class MLBIPipeline:
    def __init__(self, api_key: str, dataset_path: str):
        self.api_key = api_key
        self.dataset_path = dataset_path
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
        try:
            self._validate_csv_path()
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.llm = LiteLLM(model="gpt-5", api_key=self.api_key)
            pai.config.set({"llm": self.llm})
            os.makedirs("./charts", exist_ok=True)
            self.df = pai.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded successfully with shape: {self.df.shape}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise HTTPException(status_code=500, detail=f"Pipeline setup failed: {str(e)}")

    async def run_analysis(self, user_prompt: str) -> dict:
        try:
            buf = io.StringIO()
            self.df.info(buf=buf)
            data_info = buf.getvalue()

            system_prompt = """
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
            You are the Orchestrator. Convert a short business question into three prompts for specialists.
            All specialists operate in Python via PandasAI SmartDataframe.
            Return STRICT JSON with keys: manipulator_prompt, visualizer_prompt, analyzer_prompt, compiler_instruction.
            Each value should be in a single line.
            """
            initial_response = completion(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Prompt:{user_prompt}\n\nData Info:\n{data_info}"},
                ],
                seed=1,
                stream=False,
                verbosity="low",
                drop_params=True,
                reasoning_effort="high",
            )
            initial_content = get_content(initial_response)
            try:
                spec = json.loads(initial_content)
            except json.JSONDecodeError:
                start = initial_content.find("{")
                end = initial_content.rfind("}")
                spec = json.loads(initial_content[start : end + 1])

            manipulator_prompt = spec["manipulator_prompt"]
            visualizer_prompt = spec["visualizer_prompt"]
            analyzer_prompt = spec["analyzer_prompt"]
            compiler_instruction = spec["compiler_instruction"]

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
                    "save_charts_path": "./charts",
                },
            )
            data_manipulator_response = data_manipulator.chat(manipulator_prompt)
            if isinstance(data_manipulator_response, DataFrameResponse):
                df_processed = data_manipulator_response.value
            else:
                df_processed = data_manipulator_response

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
                    "save_charts_path": "./charts",
                },
            )
            data_visualizer_response = data_visualizer.chat(visualizer_prompt)

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
                    "save_charts_path": "./charts",
                },
            )
            data_analyzer_response = data_analyzer.chat(
                f"Respond like you are communicating to a person. {analyzer_prompt}"
            )

            logger.info("Compiling final response...")
            final_response = completion(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": compiler_instruction},
                    {
                        "role": "user",
                        "content": (
                            f"User Prompt:{user_prompt}. Data Info:{data_info}. "
                            f"Data Manipulator Response:{data_manipulator_response}. "
                            f"Data Visualizer Response:{data_visualizer_response}. "
                            f"Data Analyzer Response:{data_analyzer_response}."
                        ),
                    },
                ],
                seed=1,
                stream=False,
                verbosity="medium",
                drop_params=True,
                reasoning_effort="low",
            )
            final_content = get_content(final_response)

            # ambil chart terakhir (jika ada)
            chart_path = None
            charts_dir = Path("./charts")
            if charts_dir.exists():
                chart_files = list(charts_dir.glob("*.png"))
                if chart_files:
                    chart_path = str(sorted(chart_files, key=lambda p: p.stat().st_mtime)[-1])

            return {
                "final_analysis": final_content,
                "chart_path": chart_path,
                "data_manipulator_response": str(data_manipulator_response),
                "data_analyzer_response": str(data_analyzer_response),
                "orchestration_spec": spec,
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise Exception(f"Analysis pipeline failed: {str(e)}")


# --- LOGIKA BACKGROUND TASK ---
async def run_analysis_task(task_id: str, user_prompt: str, api_key: str, gcs_path: str):
    """Background task untuk menjalankan pipeline analisis dari file di GCS."""
    if storage_client is None or firestore_client is None:
        raise RuntimeError("GCP clients are not initialized")

    task_ref = firestore_client.collection("tasks").document(task_id)
    local_csv_path = ""
    try:
        task_ref.update({"status": "processing"})

        temp_dir = Path("./temp_data")
        temp_dir.mkdir(exist_ok=True)
        local_csv_path = temp_dir / f"{task_id}.csv"

        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = gcs_path.replace(f"gs://{BUCKET_NAME}/", "")
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_csv_path))
        logger.info(f"Successfully downloaded {gcs_path} to {local_csv_path}")

        pipeline = MLBIPipeline(api_key, str(local_csv_path))
        result = await pipeline.run_analysis(user_prompt)

        # jika ada chart, salin ke nama khusus task: ./charts/{task_id}.png
        chart_url = None
        if result.get("chart_path") and os.path.exists(result["chart_path"]):
            charts_dir = Path("./charts")
            charts_dir.mkdir(exist_ok=True)
            dest = charts_dir / f"{task_id}.png"
            try:
                shutil.copy(result["chart_path"], dest)
                chart_url = f"/chart/{task_id}"
            except Exception as e:
                logger.warning(f"Failed to copy chart for task {task_id}: {e}")

        task_ref.update(
            {
                "status": "completed",
                "result": result,
                "chart_url": chart_url,
            }
        )
        logger.info(f"Task {task_id} completed successfully.")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        task_ref.update({"status": "failed", "error": str(e)})
    finally:
        if local_csv_path and os.path.exists(local_csv_path):
            os.remove(local_csv_path)
            logger.info(f"Cleaned up local file: {local_csv_path}")


# --- API ENDPOINTS ---
@app.get("/")
async def root():
    return {"message": "ML BI Pipeline API is running!", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/analyze/upload", response_model=AnalysisResponse)
async def analyze_data_upload(
    background_tasks: BackgroundTasks,
    user_prompt: str = Form(...),
    openai_api_key: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    # Validasi fleksibel: izinkan content-type umum CSV ATAU nama berakhiran .csv
    allowed_types = {"text/csv", "application/vnd.ms-excel", "application/octet-stream"}
    if (file.content_type not in allowed_types) and (not file.filename.lower().endswith(".csv")):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required")

    if storage_client is None or firestore_client is None:
        raise HTTPException(status_code=500, detail="GCP clients not initialized")

    task_id = str(uuid.uuid4())

    # Paksa nama simpanan .csv untuk konsistensi
    orig_name = Path(file.filename or "upload").name
    if not orig_name.lower().endswith(".csv"):
        orig_name = f"{orig_name}.csv"
    gcs_filename = f"uploads/{task_id}_{orig_name}"

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_filename)
        blob.upload_from_file(file.file)
        gcs_path = f"gs://{BUCKET_NAME}/{gcs_filename}"
        logger.info(f"File uploaded to {gcs_path}")
    except Exception as e:
        logger.error(f"Failed to upload file to GCS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not upload file to storage.")

    # Catat task & jalankan background job
    task_ref = firestore_client.collection("tasks").document(task_id)
    task_ref.set(
        {
            "status": "queued",
            "user_prompt": user_prompt,
            "created_at": pd.Timestamp.now().isoformat(),
            "dataset_path": gcs_path,
        }
    )

    background_tasks.add_task(run_analysis_task, task_id, user_prompt, api_key, gcs_path)

    return AnalysisResponse(
        task_id=task_id,
        status="queued",
        message="Analysis started. Use /status/{task_id} to check progress.",
    )


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get analysis task status from Firestore"""
    if firestore_client is None:
        raise HTTPException(status_code=500, detail="Firestore not initialized")

    task_ref = firestore_client.collection("tasks").document(task_id)
    task_doc = task_ref.get()

    if not task_doc.exists:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = task_doc.to_dict()
    return TaskStatus(
        task_id=task_id,
        status=task_data.get("status"),
        result=task_data.get("result"),
        error=task_data.get("error"),
        chart_url=task_data.get("chart_url"),
    )


@app.get("/tasks")
async def list_tasks():
    """List all tasks from Firestore."""
    if firestore_client is None:
        raise HTTPException(status_code=500, detail="Firestore not initialized")

    tasks_ref = firestore_client.collection("tasks").stream()
    tasks_list = []
    for task in tasks_ref:
        task_data = task.to_dict()
        tasks_list.append(
            {
                "task_id": task.id,
                "status": task_data.get("status"),
                "user_prompt": task_data.get("user_prompt"),
                "created_at": task_data.get("created_at"),
            }
        )
    return {"tasks": tasks_list}


@app.get("/chart/{task_id}")
async def get_chart(task_id: str):
    """Serve chart PNG yang sudah disalin ke ./charts/{task_id}.png"""
    charts_dir = Path("./charts")
    file_path = charts_dir / f"{task_id}.png"
    if not file_path.exists():
        # fallback: bila belum bernama khusus, kembalikan terakhir yang ada (opsional)
        alt = sorted(charts_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        if not alt:
            raise HTTPException(status_code=404, detail="Chart not found")
        return FileResponse(path=str(alt[-1]), media_type="image/png")
    return FileResponse(path=str(file_path), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
