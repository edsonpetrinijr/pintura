"""
Cliente REST minimo para o modelo implantado no Maximo Visual Inspection (MVI).

O caminho exato do endpoint de inferencia varia por instancia/versao do MAS.
Confirme o path correto na aba "API" do modelo implantado no MVI - ela mostra
um exemplo de curl pronto com o path certo. Ajuste MVI_INFERENCE_PATH no .env
se for diferente do valor padrao usado aqui.
"""
import os
from dataclasses import dataclass

import requests


@dataclass
class Detection:
    label: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class MVIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        inference_path: str = "/api/dlapis/{model_id}",
        verify_ssl: bool = True,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.inference_path = inference_path.format(model_id=model_id)
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "MVIClient":
        return cls(
            base_url=os.environ["MVI_BASE_URL"],
            api_key=os.environ["MVI_API_KEY"],
            model_id=os.environ["MVI_MODEL_ID"],
            inference_path=os.environ.get("MVI_INFERENCE_PATH", "/api/dlapis/{model_id}"),
            verify_ssl=os.environ.get("MVI_VERIFY_SSL", "true").lower() != "false",
        )

    def infer_image(self, image_path: str, threshold: float = 0.5) -> list[Detection]:
        url = f"{self.base_url}{self.inference_path}"
        headers = {}
        if self.api_key:
            headers["X-Auth-Token"] = self.api_key

        with open(image_path, "rb") as f:
            files = {"files": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"threshold": str(threshold)}
            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                verify=self.verify_ssl,
                timeout=self.timeout,
            )

        response.raise_for_status()
        payload = response.json()
        return self._parse_detections(payload)

    @staticmethod
    def _parse_detections(payload: dict) -> list[Detection]:
        # Formato padrao do MVI: {"classified": [{"label": ..., "confidence": ...,
        # "xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...}, ...]}
        # Alguns modelos retornam "predictions" no lugar de "classified".
        raw_detections = payload.get("classified") or payload.get("predictions") or []

        detections: list[Detection] = []
        for item in raw_detections:
            detections.append(
                Detection(
                    label=item.get("label", "desconhecido"),
                    confidence=float(item.get("confidence", 0.0)),
                    xmin=float(item.get("xmin", 0)),
                    ymin=float(item.get("ymin", 0)),
                    xmax=float(item.get("xmax", 0)),
                    ymax=float(item.get("ymax", 0)),
                )
            )
        return detections
