#!/usr/bin/env bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ..
python3 ./infra/run/app_runner.py --interface ipython --app "sms_camera.src.camera_gsm_to_url.camera_gsm_to_url" "$@"
