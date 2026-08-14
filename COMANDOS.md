# Comandos

Rode tudo a partir da raiz do repositorio. `python` nao esta no PATH desta
maquina, use sempre `.venv\Scripts\python.exe`.

> **A camera .45 (`cabine`) aceita UM stream por vez.** Se duas ferramentas ao
> vivo disputarem, a perdedora recebe `453 Not Enough Bandwidth` em silencio.
> Foi assim que o dataset da cabine ficou com 9 frames enquanto cam26/cam27
> tinham 35. Ver quem esta segurando:
>
> ```powershell
> Get-Process python | Select-Object Id,StartTime,@{n='cmd';e={(Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine -replace '.*python\.exe" ',''}} | Format-Table -AutoSize -Wrap
> ```

---

## 1. Coletor rotulado

Vigia a API e, a cada troca de peca, captura as 4 cameras nomeando pelo part
number. E daqui que sai `dataset_labeled/`. Roda sozinho, deixe ligado.

```powershell
.venv\Scripts\python.exe parts\sync_capture.py --poll-interval 5
```

Uma instancia so: duas escrevem no mesmo CSV e disputam as cameras.

---

## 2. Fundo de referencia

Base da segmentacao. A cabine e fixa, entao o que muda entre frames e
exatamente o que esta pendurado. Cor nao serve - a mesma peca aparece escura,
amarela ou branca conforme a etapa, e no frame amarelo a saturacao dela era
igual a do piso.

De frames ja em disco, sem ocupar a camera:

```powershell
.venv\Scripts\python.exe local_cv\build_background.py --camera cabine --pasta dataset_labeled
```

Da camera ao vivo, melhor qualidade (~10 min, exige a camera livre):

```powershell
.venv\Scripts\python.exe local_cv\build_background.py --camera cabine --frames 40 --interval 15
```

Saida: `local_cv/background_<camera>.jpg`. Quanto mais frames espacados no
tempo, melhor: com poucos, uma peca que aparece em varios deles entra na
propria mediana e depois se auto-cancela.

---

## 3. Geometria da camera (pontos de fuga)

> Parado por enquanto. Esta medido e funciona, mas a sobreposicao roda com isso
> DESLIGADO por padrao (`--focal 0`), porque ligar amarra a geometria a posicao
> e ao zoom da peca e isso invalida o cache de render - arrastar 900 px virava
> 13 renders, uns 2s de janela travada. Para retomar: `--focal 1012`.

Mede a distancia focal e a orientacao da camera usando as proprias arestas da
cabine - trilho, grade, quina de parede. Sem tabua de calibracao e sem parar a
linha.

```powershell
.venv\Scripts\python.exe local_cv\fuga.py --camera cabine --frames 9
```

Junta as retas de varios frames antes de decidir, porque a cabine e fixa e um
frame so nao tem retas suficientes nas familias fracas. Medido na cabine:
**focal 1012 px, FOV horizontal 87 graus**, repetivel em +-8 px. O desenho em
`capturas/fuga_<camera>.jpg` mostra uma cor por familia - se as familias nao
fizerem sentido a olho, a focal nao vale.

Quando for retomar, os dois pontos abertos sao: remedir `POSE_PADRAO` com a
focal ligada (a atual tem o desvio embutido) e estimar a distorcao radial da
lente, que e visivel nas verticais arqueadas perto da borda.

---

## 4. Identificacao automatica da peca

Casa a silhueta da camera contra um banco de vistas renderizadas do CAD.

```powershell
.venv\Scripts\python.exe local_cv\identificar_peca.py --camera cabine --top 3
```

Os STEP ficam em `modelos/`. O banco vai para `modelos/.cache/` e so e
recalculado quando o STEP ou os angulos mudam.

---

## 5. Monitor ao vivo

Ganchos, deteccao local e o que a API diz, lado a lado.

```powershell
.venv\Scripts\python.exe local_cv\monitor_screen.py --camera cabine --hooks local_cv\hooks_cabine_11.json
```

Segura o stream da .45 continuamente - feche antes de usar a sobreposicao.

---

## 6. Sobreposicao 3D interativa

Poe o CAD sombreado em cima da imagem, refina a pose sozinho e grava a base de
exemplos.

```powershell
.venv\Scripts\python.exe local_cv\sobrepor3d.py --camera cabine --hooks local_cv\hooks_cabine_11.json
```

Sobre frames salvos, sem ocupar a camera:

```powershell
.venv\Scripts\python.exe local_cv\sobrepor3d.py --dataset dataset_labeled --camera cabine --hooks local_cv\hooks_cabine_11.json
```

| tecla | acao |
|---|---|
| `a` `d` / `w` `s` / `z` `x` | yaw / pitch / giro |
| `t` | **gira 90 graus** (a peca so muda de lado nesses passos) |
| `o` | **testa os 4 quadrantes** e refina o melhor |
| `ENTER` | refina a partir de onde esta |
| `+` `-` ou scroll | tamanho |
| setas ou arrastar | mover |
| `e` | espelha |
| `f` | encaixa na peca segmentada |
| `TAB` | troca a peca alvo |
| `r` | volta a pose padrao |
| `k` ou `g` | **grava o exemplo** |
| `n` | proximo modelo |
| `espaco` / `,` `.` | novo frame / navega o dataset |
| `q` ou `ESC` | sai |

A pose ja abre na orientacao padrao da linha (`POSE_PADRAO`). A camera e fixa e
a peca pende de corrente, entao pitch e giro mal mudam de peca para peca - o que
varia e para que lado ela aponta, em passos de 90 graus. Fluxo normal: `f` para
encaixar, `o` para achar o lado, `k` para gravar.

`k`/`g` gravam tres coisas:

```
poses/<part>.jsonl                       pose + IoU + bordas + caminho do frame
poses/frames/<carimbo>_<part>_<cam>.jpg  frame CRU
poses/vistas/<carimbo>_<part>_<cam>.jpg  frame com a sobreposicao
```

O frame cru e o que faz a base valer: sem ele a pose nao pode ser reavaliada
depois.

