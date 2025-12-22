@echo off
..\infra\scripts\run_cmd.bat python ..\infra\run\app_runner.py --interface ipython --app "sms_camera.src.camera_gsm_to_url.camera_gsm_to_url" %*
