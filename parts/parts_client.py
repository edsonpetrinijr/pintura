"""
Cliente para a API interna que informa quais pecas estao sendo carregadas em
quais ganchos (PartBldYJSON).

Formato de cada registro (observado em producao):
    {
        "number_car": 20,
        "part_number": "4336117",
        "serial_number": "62320375872",
        "timestamp": "2026-08-05T06:37:09.953Z",
        "hook": "2; 3",       # ganchos separados por "; ", "" = sem gancho, "[0]" = placeholder/sem gancho
        "Program_Robot": 415,
        "figure": "Axle",     # tipo/formato da peca
        "color": 1
    }
"""
from dataclasses import dataclass, field

import requests


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

    @property
    def has_hooks(self) -> bool:
        return len(self.hooks) > 0


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
        raw_records = response.json()

        records = []
        for item in raw_records:
            records.append(PartRecord(
                number_car=item.get("number_car"),
                part_number=item.get("part_number", ""),
                serial_number=item.get("serial_number", ""),
                timestamp=item.get("timestamp", ""),
                hooks=_parse_hooks(item.get("hook")),
                program_robot=item.get("Program_Robot"),
                figure=item.get("figure"),
                color=item.get("color"),
            ))
        return records
