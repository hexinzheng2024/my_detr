import subprocess
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

# 创建 FastAPI 应用实例
app = FastAPI()

# 定义请求模型
class VideoProcessRequest(BaseModel):
    video_file: str
    config_file: str
    weights_file: str
    output_file: str

# 使用绝对路径获取静态文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "../front")

# 确认路径是否正确
if not os.path.exists(static_dir):
    raise RuntimeError(f"Directory '{static_dir}' does not exist")

# 挂载静态文件目录
app.mount("/front", StaticFiles(directory=static_dir), name="static")

# 返回 index.html 文件
@app.get("/")
def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# 运行视频处理命令的路由
@app.post("/run")
async def run_command(request: VideoProcessRequest):
    # 构建执行的命令
    command = [
        "python", "video_demo.py", 
        request.video_file, 
        request.config_file, 
        request.weights_file, 
        "--out", request.output_file
    ]

    try:
        # 使用 subprocess 运行命令
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return {"result": result.stdout}
        else:
            return {"result": result.stderr}
    except Exception as e:
        return {"result": str(e)}
