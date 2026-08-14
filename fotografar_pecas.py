"""
POC: toda vez que a API de pecas mostra um registro NOVO (carro novo chegou),
fotografa as cameras de DENTRO DA CABINE de pintura (.45 e .46) e avisa no
desktop. Quando o registro some da API (a peca seguiu na linha), avisa tambem.
Sem janela/interface visual propria - so console (use --vlc pra ver ao vivo).

ARQUIVO UNICO: nao depende de mais nenhum arquivo do projeto - so das
bibliotecas do requirements.txt (rode instalar.bat primeiro) e do .env com as
credenciais das cameras/API.

Uso (deixar rodando):
    .venv\\Scripts\\python.exe fotografar_pecas.py

Com --vlc: abre o VLC ao vivo nas cameras da cabine JUNTO com a analise (os
dois ao mesmo tempo, no mesmo comando):
    .venv\\Scripts\\python.exe fotografar_pecas.py --vlc

Fotos salvas em: fotos/<camera>/<part_number>/<timestamp>_carro<N>_<serial>.jpg
(<camera> e 45 ou 46; numero do carro + serial no nome distinguem pecas
repetidas no mesmo carro)

ATENCAO: as cameras da cabine aceitam poucos streams simultaneos (a .46 e mais
sensivel - erro "453 Not Enough Bandwidth"). Se a foto comecar a falhar com o
VLC aberto, feche o VLC e rode so a analise.
"""
import argparse
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import requests
from dotenv import load_dotenv

DEFAULT_API_URL = "http://b8wdwisep02.brazil.cat.com:3030/PartBldYJSON"

# Cameras de DENTRO da cabine de pintura (nao a estacao de carregamento).
CAMERA_ENV = {
    "cabine": "RTSP_URL",
    "cabine2": "CAMERA_46_URL",
}
CAMERA_LABEL = {"cabine": "45", "cabine2": "46"}
CAMERAS_MONITORADAS = list(CAMERA_ENV)
RAIZ = os.path.dirname(os.path.abspath(__file__))
PASTA_FOTOS_PADRAO = os.path.join(RAIZ, "fotos")


def label(camera_nome: str) -> str:
    return CAMERA_LABEL.get(camera_nome, camera_nome)


def camera_url(camera_nome: str) -> str:
    env_key = CAMERA_ENV[camera_nome]
    url = os.environ.get(env_key)
    if not url:
        raise SystemExit(f"Defina {env_key} no .env para usar a camera '{camera_nome}'")
    return url


@dataclass
class PartRecord:
    number_car: int
    part_number: str
    serial_number: str
    timestamp: str
    hooks: list[int] = field(default_factory=list)
    program_robot: int | None = None
    figure: str | None = None
    color: int | None = None

    @property
    def key(self) -> str:
        """Chave unica para deduplicar registros ja vistos."""
        return f"{self.part_number}|{self.serial_number}|{self.timestamp}"


def _parse_hooks(raw: str | None) -> list[int]:
    if not raw:
        return []
    hooks = []
    for token in raw.split(";"):
        token = token.strip().strip("[]")
        if token.isdigit() and int(token) != 0:
            hooks.append(int(token))
    return hooks


class PartsClient:
    def __init__(self, api_url: str, timeout: float = 10.0) -> None:
        self.api_url = api_url
        self.timeout = timeout

    def fetch(self) -> list[PartRecord]:
        response = requests.get(self.api_url, timeout=self.timeout)
        response.raise_for_status()
        registros = []
        for item in response.json():
            registros.append(PartRecord(
                number_car=item.get("number_car"),
                part_number=item.get("part_number", ""),
                serial_number=item.get("serial_number", ""),
                timestamp=item.get("timestamp", ""),
                hooks=_parse_hooks(item.get("hook")),
                program_robot=item.get("Program_Robot"),
                figure=item.get("figure"),
                color=item.get("color"),
            ))
        return registros


def notificar(titulo: str, mensagem: str) -> None:
    """Notificacao nativa do Windows (toast) sem instalar nada - usa a API
    de notificacoes ja embutida no PowerShell."""
    def escapar(texto: str) -> str:
        return texto.replace("'", "''")

    script = f"""
$ErrorActionPreference = 'SilentlyContinue'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType=WindowsRuntime] | Out-Null
$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$textos = $template.GetElementsByTagName('text')
$textos.Item(0).AppendChild($template.CreateTextNode('{escapar(titulo)}')) | Out-Null
$textos.Item(1).AppendChild($template.CreateTextNode('{escapar(mensagem)}')) | Out-Null
$toast = [Windows.UI.Notifications.ToastNotification]::new($template)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Pintura - Fotografar Pecas').Show($toast)
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
            capture_output=True, timeout=10,
        )
    except Exception as exc:
        print(f"[aviso] nao consegui notificar no desktop: {exc}")


def capturar_snapshot(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return None
    # descarta os primeiros frames: um RTSP recem-aberto costuma devolver
    # quadro velho/embacado do buffer do decodificador.
    frame, ok = None, False
    for _ in range(10):
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()
    return frame if ok else None


def fotografar_peca(registro: PartRecord, pasta_fotos: str) -> None:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    for camera_nome in CAMERAS_MONITORADAS:
        try:
            url = camera_url(camera_nome)
        except SystemExit as exc:
            print(f"  {exc}")
            continue
        frame = capturar_snapshot(url)
        if frame is None:
            print(f"  falha ao capturar {camera_nome}")
            continue
        pasta_peca = os.path.join(pasta_fotos, label(camera_nome), registro.part_number)
        os.makedirs(pasta_peca, exist_ok=True)
        # numero do carro + serial no nome: o mesmo part number pode aparecer
        # mais de uma vez no mesmo carro (ex duas pecas iguais em ganchos diferentes).
        nome = f"{ts}_carro{registro.number_car}_{registro.serial_number}.jpg"
        caminho = os.path.join(pasta_peca, nome)
        cv2.imwrite(caminho, frame)
        print(f"  salvo: {caminho}")


def encontrar_vlc() -> str | None:
    candidatos = [
        r"C:\Program Files\VideoLAN\VLC\vlc.exe",
        r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
    ]
    for caminho in candidatos:
        if os.path.exists(caminho):
            return caminho
    return shutil.which("vlc")


def abrir_vlc(camera_nome: str) -> None:
    vlc = encontrar_vlc()
    if not vlc:
        print("VLC nao encontrado. Instale o VLC (videolan.org) e tente de novo.")
        return
    url = camera_url(camera_nome)
    # --no-one-instance: se ja tiver um VLC escondido rodando de antes, o
    # "modo uma instancia so" do VLC manda o stream pra ele (sem abrir janela
    # nova) e este processo fecha na hora, dando a impressao de "abriu e fechou".
    subprocess.Popen([vlc, "--no-one-instance", url])
    print(f"Abrindo VLC em {camera_nome}...")


def monitorar(api_url: str, intervalo: float, pasta_fotos: str) -> None:
    client = PartsClient(api_url)
    presentes: dict[str, PartRecord] = {}  # chave unica do registro -> registro atual

    print("Vigiando a API de pecas - qualquer carro novo tira foto automaticamente.")
    print(f"Fotos vao para: {os.path.abspath(pasta_fotos)}")
    print("Deixe esta janela aberta. Ctrl+C para parar.\n")

    while True:
        try:
            registros = client.fetch()
        except Exception as exc:
            print(f"[erro] API indisponivel: {exc}")
            time.sleep(intervalo)
            continue

        atuais = {r.key: r for r in registros}

        chegadas = [r for chave, r in atuais.items() if chave not in presentes]
        saidas = [r for chave, r in presentes.items() if chave not in atuais]

        for registro in chegadas:
            print(f"[CHEGOU] peca {registro.part_number} (carro {registro.number_car})")
            fotografar_peca(registro, pasta_fotos)

        if chegadas:
            lista = "; ".join(f"{r.part_number} (carro {r.number_car})" for r in chegadas)
            notificar("Pecas chegaram" if len(chegadas) > 1 else "Peca chegou",
                      f"Chegou na cabine: {lista}")

        for registro in saidas:
            print(f"[SEGUIU] peca {registro.part_number} saiu da cabine")

        if saidas:
            lista = "; ".join(r.part_number for r in saidas)
            notificar("Pecas se moveram" if len(saidas) > 1 else "Peca se moveu",
                      f"Saiu da cabine (seguiu na linha): {lista}")

        presentes = atuais
        time.sleep(intervalo)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fotografa pecas automaticamente quando chegam na cabine de pintura")
    parser.add_argument("--api-url", default=os.environ.get("PARTS_API_URL", DEFAULT_API_URL))
    parser.add_argument("--intervalo", type=float, default=5.0, help="Segundos entre consultas a API")
    parser.add_argument("--out", default=PASTA_FOTOS_PADRAO, help="Pasta onde salvar as fotos")
    parser.add_argument("--vlc", action="store_true",
                         help="Tambem abre o VLC ao vivo nas cameras da cabine, junto com a analise")
    args = parser.parse_args()

    if args.vlc:
        for cam in CAMERAS_MONITORADAS:
            abrir_vlc(cam)
        print()

    try:
        monitorar(args.api_url, args.intervalo, args.out)
    except KeyboardInterrupt:
        print("\nParado.")


if __name__ == "__main__":
    main()
