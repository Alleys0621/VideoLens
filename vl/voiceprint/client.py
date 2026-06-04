"""讯飞声纹识别 API 客户端"""

import base64
import hashlib
import hmac
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import requests

from vl.core.logging import get_logger

logger = get_logger()

# ============================================================
#  常量
# ============================================================

_BASE_URL = "https://api.xf-yun.com/v1/private/s1aa729d0"
_SERVICE = "s1aa729d0"
_RESPONSE_CONFIG = {"encoding": "utf8", "compress": "raw", "format": "json"}
_AUDIO_MAX_B64 = 4194304  # base64 后最大字节数


# ============================================================
#  内部工具
# ============================================================

def _build_auth_url(url: str, api_key: str, api_secret: str, method: str = "POST") -> str:
    """为讯飞 API 请求生成带 HMAC-SHA256 鉴权参数的 URL"""
    stidx = url.index("://")
    host = url[stidx + 3:]
    edidx = host.index("/")
    path = host[edidx:]
    host = host[:edidx]

    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))

    signature_origin = f"host: {host}\ndate: {date}\n{method} {path} HTTP/1.1"
    signature_sha = hmac.HMAC(
        api_secret.encode("utf-8"),
        signature_origin.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    signature = base64.b64encode(signature_sha).decode("utf-8")

    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

    return url + "?" + urlencode({"host": host, "date": date, "authorization": authorization})


def _decode_response(resp: dict) -> dict:
    """解析讯飞 API 响应"""
    header = resp.get("header", {})
    code = header.get("code", -1)
    result: dict = {"code": code, "message": header.get("message", ""), "sid": header.get("sid", "")}

    if code != 0:
        return result

    payload = resp.get("payload")
    if not payload:
        return result

    for val in payload.values():
        text_b64 = val.get("text", "")
        if not text_b64:
            continue
        try:
            result["data"] = json.loads(base64.b64decode(text_b64).decode("utf-8"))
        except Exception:
            result["data"] = text_b64
        break

    return result


def audio_to_b64(file_path: str | Path) -> dict:
    """读取音频文件，转成 16kHz/16bit/mono WAV 后 base64 编码"""
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".wav":
        wav_path = file_path
        cleanup = False
    else:
        wav_path = Path(tempfile.mktemp(suffix=".wav"))
        cmd = [
            "ffmpeg", "-y", "-i", str(file_path),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"ffmpeg 转换 WAV 失败: {result.stderr[-300:]}")
        cleanup = True

    try:
        audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("utf-8")
    finally:
        if cleanup and wav_path.exists():
            wav_path.unlink()

    if len(audio_b64) > _AUDIO_MAX_B64:
        raise ValueError(f"音频 base64 后 {len(audio_b64)} 字节，超过 {_AUDIO_MAX_B64} 限制")

    return {
        "resource": {
            "encoding": "raw",
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16,
            "status": 3,
            "audio": audio_b64,
        }
    }


def cut_audio_segment(audio_path: str, start: float, end: float, output_path: str) -> str:
    """用 ffmpeg 按时间戳切出一段音频（16kHz/16bit/mono WAV）"""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t", str(duration),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 切片失败 ({start}-{end}): {result.stderr[-200:]}")
    return output_path


# ============================================================
#  VoiceprintClient
# ============================================================

class VoiceprintClient:
    """讯飞声纹识别 API 客户端"""

    def __init__(
        self,
        app_id: str,
        api_key: str,
        api_secret: str,
        group_id: str,
        *,
        verbose: bool = True,
    ):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.group_id = group_id
        self._verbose = verbose

    # ---------- 内部方法 ----------

    def _base_body(self, func: str, **extra_params: object) -> dict:
        param: dict = {"func": func, "groupId": self.group_id}
        param.update(extra_params)
        return {"header": {"app_id": self.app_id, "status": 3}, "parameter": {_SERVICE: param}}

    def _post(self, body: dict) -> dict:
        url = _build_auth_url(_BASE_URL, self.api_key, self.api_secret)
        headers = {"content-type": "application/json", "host": "api.xf-yun.com", "appid": self.app_id}
        response = requests.post(url, data=json.dumps(body), headers=headers)
        return response.json()

    def _call(self, label: str, body: dict) -> dict:
        t0 = time.perf_counter()
        raw = self._post(body)
        elapsed = time.perf_counter() - t0

        result = _decode_response(raw)
        code = result["code"]

        if self._verbose:
            status = "成功" if code == 0 else f"失败(code={code})"
            logger.debug(f"声纹API {label} {status}，耗时 {elapsed:.3f}s")
            if code != 0:
                logger.warning(f"声纹API {label} 错误: {result['message']}")

        return result

    # ---------- 声纹识别 ----------

    def search(self, file_path: str, top_k: int = 1) -> dict:
        """声纹识别 1:N"""
        body = self._base_body("searchFea", topK=top_k, searchFeaRes=_RESPONSE_CONFIG)
        body["payload"] = audio_to_b64(file_path)
        return self._call("1:N识别", body)

    def search_score(self, feature_id: str, file_path: str) -> dict:
        """声纹比对 1:1"""
        body = self._base_body("searchScoreFea", dstFeatureId=feature_id, searchScoreFeaRes=_RESPONSE_CONFIG)
        body["payload"] = audio_to_b64(file_path)
        return self._call(f"1:1比对 [{feature_id}]", body)

    # ---------- 特征管理 ----------

    def create_feature(self, feature_id: str, feature_info: str, file_path: str) -> dict:
        """添加声纹特征"""
        body = self._base_body(
            "createFeature", featureId=feature_id, featureInfo=feature_info, createFeatureRes=_RESPONSE_CONFIG,
        )
        body["payload"] = audio_to_b64(file_path)
        return self._call(f"添加声纹 [{feature_id}]", body)

    def update_feature(self, feature_id: str, feature_info: str, file_path: str, *, cover: bool = True) -> dict:
        """更新声纹特征"""
        body = self._base_body(
            "updateFeature", featureId=feature_id, featureInfo=feature_info, cover=cover,
            updateFeatureRes=_RESPONSE_CONFIG,
        )
        body["payload"] = audio_to_b64(file_path)
        return self._call(f"更新声纹 [{feature_id}]", body)

    def delete_feature(self, feature_id: str) -> dict:
        """删除声纹特征"""
        return self._call(f"删除声纹 [{feature_id}]", self._base_body(
            "deleteFeature", featureId=feature_id, deleteFeatureRes=_RESPONSE_CONFIG,
        ))

    def query_feature_list(self) -> dict:
        """查询特征列表"""
        return self._call("查询特征列表", self._base_body(
            "queryFeatureList", queryFeatureListRes=_RESPONSE_CONFIG,
        ))

    # ---------- 特征库管理 ----------

    def create_group(self, group_name: str = "", group_info: str = "") -> dict:
        """创建声纹特征库"""
        return self._call("创建声纹特征库", self._base_body(
            "createGroup",
            groupName=group_name or self.group_id,
            groupInfo=group_info,
            createGroupRes=_RESPONSE_CONFIG,
        ))

    def delete_group(self) -> dict:
        """删除声纹特征库"""
        return self._call("删除声纹特征库", self._base_body(
            "deleteGroup", deleteGroupRes=_RESPONSE_CONFIG,
        ))
