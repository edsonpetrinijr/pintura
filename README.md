# Deteccao de ganchos em uso - linha de pintura (Maximo Visual Inspection)

Sistema para identificar, a partir das imagens/video da camera da linha de
pintura (arquivos `vlcsnap-*.png` sao exemplos ja capturados), quais ganchos
do transportador aereo estao em uso (com peca pendurada) e quais estao vazios.

O treinamento do modelo de visao computacional e feito dentro do **Maximo
Visual Inspection** (parte do Maximo Application Suite). Este projeto cuida do
que fica de fora do MVI: capturar frames para formar o dataset e rodar a
inferencia contra o modelo ja treinado/implantado.

Tambem existe uma API interna (`PartBldYJSON`) que ja informa, em tempo real,
qual peca esta sendo carregada em qual gancho. Isso permite montar um dataset
**automaticamente rotulado**, sem desenhar caixas manualmente - ver secao
[Sincronizar API de pecas com as cameras](#sincronizar-api-de-pecas-com-as-cameras).

## Fluxo geral

```mermaid
flowchart LR
    A[Camera RTSP] --> B[extract_frames.py]
    B --> C[dataset_raw/ imagens]
    C --> D[Upload + rotulagem no MVI]
    D --> E[Treino do modelo no MVI]
    E --> F[Deploy do modelo no MVI]
    A --> G[monitor_hooks.py]
    F --> G
    G --> H[capturas/ frames anotados]
    G --> I[logs/ganchos.csv]

    J[API PartBldYJSON] --> K[sync_capture.py]
    L[Cameras 26/27] --> K
    K --> M[dataset_labeled/ imagens + rotulo]
    K --> N[logs/pecas_ganchos.csv]
```

## 1. Montar o dataset

Use o script de captura para extrair frames do RTSP (ou de um video ja
gravado) em intervalos regulares. As imagens `vlcsnap-*.png` que ja existem
nesta pasta podem ser movidas para `dataset_raw/` como amostras iniciais.

```powershell
python capture/extract_frames.py --source rtsp://usuario:senha@ip:554/stream1 --out dataset_raw --interval 5
```

Recomendacoes para o dataset:
- Capture em varios horarios/condicoes de luz e com a linha em velocidades diferentes.
- Tente cobrir todos os ganchos individualmente (nao so os mais visiveis).
- Minimo recomendado: algumas centenas de imagens por classe para um primeiro modelo razoavel.

## 2. Rotular e treinar no Maximo Visual Inspection

Isso e feito na interface web do MVI (nao neste repo):

1. Criar um dataset de deteccao de objetos no MVI e importar as imagens de `dataset_raw/`.
2. Desenhar as caixas delimitadoras em cada gancho, com duas classes sugeridas:
   - `gancho_ocupado` (gancho com peca pendurada)
   - `gancho_vazio` (gancho sem peca)
3. Treinar o modelo (o MVI usa arquiteturas tipo YOLO por baixo dos panos).
4. Avaliar precisao/recall por classe e revisar imagens mal rotuladas se necessario.
5. Implantar ("Deploy") o modelo treinado. Isso gera um endpoint de inferencia.
6. Na tela do modelo implantado, abrir a aba **API** - ela mostra um `curl` de
   exemplo com o path exato do endpoint, o header de autenticacao e o formato
   esperado. Use esses valores para preencher o `.env` (`MVI_BASE_URL`,
   `MVI_API_KEY`, `MVI_MODEL_ID`, `MVI_INFERENCE_PATH`).

## 3. Configurar credenciais

```powershell
Copy-Item .env.example .env
```

Edite `.env` com a URL do RTSP e os dados do modelo implantado no MVI.

## 4. Instalar dependencias

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 5. Rodar o monitoramento continuo

```powershell
python inference/monitor_hooks.py --interval 5
```

O script:
- Le frames do RTSP periodicamente.
- Envia cada frame para o modelo implantado no MVI.
- Classifica cada deteccao como gancho ocupado ou vazio (ver `OCCUPIED_LABELS`
  / `EMPTY_LABELS` em `inference/monitor_hooks.py` - ajuste se usar outros
  nomes de classe no MVI).
- Salva o frame anotado (caixas desenhadas) em `capturas/`.
- Registra um log em `logs/ganchos.csv` com timestamp, total de ganchos,
  ocupados e vazios.

## Estrutura do projeto

```
capture/
  extract_frames.py     -> extrai frames do RTSP/video para o dataset
  motion_capture.py      -> captura frames so quando ha movimento na faixa dos ganchos
inference/
  mvi_client.py         -> cliente REST generico para o modelo implantado no MVI
  monitor_hooks.py       -> loop de monitoramento continuo + log em CSV
  test_single_image.py  -> testa a identificacao em uma unica imagem
local_cv/
  select_regions.py     -> calibracao interativa das caixas de cada gancho
  select_hook_points.py -> calibracao por clique: 1 ponto por gancho, na ordem da API
  detect_local.py        -> identificacao local ocupado/vazio, sem depender do MVI
parts/
  parts_client.py        -> cliente da API PartBldYJSON (pecas -> ganchos)
  sync_capture.py        -> sincroniza a API de pecas com as cameras (dataset auto-rotulado)
  visualize.py           -> desenha ponto em cada gancho + caixa em volta da peca
dataset_raw/              -> imagens usadas para montar o dataset (nao versionadas)
dataset_labeled/           -> imagens + rotulo automatico via API de pecas (nao versionadas)
capturas/                 -> frames anotados gerados em producao (nao versionados)
logs/ganchos.csv           -> historico de status dos ganchos (via MVI)
logs/pecas_ganchos.csv     -> historico peca -> gancho -> imagem (via API de pecas)
.env.example              -> modelo de variaveis de ambiente
```

## Sincronizar API de pecas com as cameras

A API interna `PartBldYJSON` retorna, para cada peca carregada, algo assim:

```json
{
  "number_car": 20,
  "part_number": "4336117",
  "serial_number": "62320375872",
  "timestamp": "2026-08-05T06:37:09.953Z",
  "hook": "2; 3",
  "figure": "Axle",
  "color": 1
}
```

O campo `hook` ja diz quais ganchos receberam a peca (ou vazio/`[0]` quando
nao ha gancho atribuido ainda). O script `parts/sync_capture.py`:

1. Consulta essa API periodicamente.
2. Para cada registro novo com gancho(s) atribuido(s), tira um print das
   cameras 26 e 27 (RTSP) naquele instante.
3. Salva as imagens em `dataset_labeled/` e uma linha em
   `logs/pecas_ganchos.csv` com peca, ganchos, figura e qual imagem corresponde.

```powershell
python parts/sync_capture.py --poll-interval 5
```

Isso da um dataset rotulado automaticamente (imagem + qual peca/gancho naquele
momento) - uma alternativa (ou complemento) a rotulagem manual no MVI. Ainda
nao sabemos exatamente qual camera (26 ou 27) enxerga qual faixa de ganchos -
por isso o script captura as duas a cada evento; confirme visualmente e ajuste
se quiser restringir a so uma camera por evento.

## Visualizar ganchos e pecas sobre a imagem

Primeiro calibre a posicao de cada gancho, clicando neles na MESMA ordem em que
a API os numera (gancho 1, 2, 3, ...):

```powershell
python local_cv/select_hook_points.py --source "rtsp://admin:2035@10.101.244.26:554" --out local_cv/hooks_cam26.json --scale 0.6
```

Depois gere a visualizacao, que cruza a calibracao com o estado atual da API:

```powershell
python parts/visualize.py --camera cam26 --out capturas/overlay_cam26.jpg
python parts/visualize.py --camera cam26 --watch --interval 5   # ao vivo
```

O resultado traz:
- um pontinho sobre cada gancho: **verde** = vazio, **vermelho** = ocupado
- uma caixa amarela em volta de cada peca, cobrindo os ganchos que ela usa
- o rotulo `part_number (figure) - ganchos X,Y`
- uma barra no topo com o total de ganchos, ocupados e vazios

Ajuste `--pad-x` (folga lateral da caixa) e `--part-height` (quanto a caixa
desce abaixo dos ganchos, ja que a peca fica pendurada) conforme o resultado.

### Calibracao por tipo de carro

**A ordem dos ganchos importa, e e facil errar.** Na cabine o gancho 1 e o do
FUNDO (mais longe da camera, a direita na imagem) e a numeracao cresce vindo em
direcao a camera. Calibrar clicando da esquerda para a direita inverte todos os
ids. O sintoma e caracteristico: a deteccao discorda da API e **nenhum ajuste de
limiar melhora**, porque o problema nao esta no limiar.

Aconteceu em 05/08 com as duas calibracoes da cabine. Medido no CSV coletado, a
concordancia com a API era 56% com a numeracao invertida e 71% com a correta.
Correcao:

```powershell
python local_cv/renumber_hooks.py local_cv/hooks_cabine_11.json --conferir
python local_cv/renumber_hooks.py local_cv/hooks_cabine_11.json
```

O script troca `id` por `N+1-id`, guarda um `.bak` e mostra a nova numeracao
lado a lado com o `x` de cada ponto, para conferir de que lado ficou o gancho 1.
Dados coletados com a numeracao antiga nao servem: separe o CSV antes de
continuar.

Os carros da linha tem 8 ou 11 ganchos, e a geometria muda junto: os 8 ganchos
de um carro pequeno nao ficam nas mesmas posicoes que os 8 primeiros de um
carro de 11. Por isso a calibracao e por camera E por tipo de carro:

```
local_cv/hooks_cabine_8.json
local_cv/hooks_cabine_11.json
```

Nao da para saber pela API qual carro esta na linha, entao a escolha e feita
pela propria imagem: os scripts testam todas as calibracoes disponiveis e ficam
com a de maior confianca, medida pela quantidade de estrutura metalica
encontrada em cada ponto calibrado (ponto sobre gancho pontua alto, ponto sobre
parede lisa pontua quase zero). Use `--car-hooks N` para forcar uma delas.

A escolha e refeita a cada frame, entao o modo `--watch` acompanha a troca de
carro sozinho.

### A linha NAO deriva de posicao

Havia a suspeita de que o transportador parasse em posicoes ligeiramente
diferentes, invalidando a calibracao fixa em pixels. Medido por correlacao de
fase entre frames distintos: `dx = 0.1 px`, `dy = 0.7 px`. O carro para numa
posicao repetivel e a calibracao fixa e valida.

O que parecia deriva era na verdade a calibracao errada sendo aplicada: um
carro de 11 ganchos lido com a calibracao de 8. A escolha automatica resolveu.

## Detectar ocupado/vazio na cabine de pintura

A camera de dentro da cabine (`.45`, apelidada `cabine`) e a unica em que a
deteccao por imagem funciona, porque o fundo e a parede lisa da cabine. O sinal
fisico e simples: gancho ocupado tem uma **corrente descendo** ate a peca;
gancho vazio mostra so a pega nua contra a parede.

```powershell
python local_cv/detect_hooks_local.py --camera cabine --factor 0.5 --drop 0.8 --min-size 22 --max-size 110
python local_cv/detect_hooks_local.py --camera cabine --compare-api    # confronta com a API
python local_cv/detect_hooks_local.py --camera cabine --tune           # so desenha as janelas
```

O limiar padrao (4.5) nao foi chutado: veio da separacao medida em frames reais,
onde ganchos vazios pontuaram no maximo 3.1 e ocupados no minimo 5.6.

### Por que isso nao funciona nas cameras 26 e 27

A densidade de bordas so significa alguma coisa quando o fundo e limpo. Medindo
a densidade de bordas do fundo na faixa dos ganchos:

| camera | mediana | p25 | p75 |
|---|---|---|---|
| cabine (.45) | 6.7 | 3.4 | 10.1 |
| cam26 | 14.3 | 11.0 | 18.7 |
| cam27 | 16.7 | 12.7 | 21.0 |

Na cabine um gancho ocupado pontua entre 5.6 e 17. Na estacao de carregamento o
fundo sozinho ja pontua entre 11 e 21 - vidro, monitores, carrinhos de linga e
pessoas circulando. O ruido engole o sinal, e nenhum limiar separa isso.

Para cobrir a 26/27 o caminho e um modelo treinado, alimentado pelo dataset
auto-rotulado (proxima secao).

## Coleta continua para validacao e treino

```powershell
python local_cv/collect_validation.py --interval 60
```

A cada minuto o script captura uma rajada de 5 frames da cabine, mede o score de
cada gancho pela mediana da rajada, le a API e grava uma linha por gancho em
`logs/validacao_ganchos.csv`. As imagens das tres cameras vao para
`dataset_validacao/`. Use `--somente-mudancas` para so gravar imagem quando a
ocupacao muda, se disco for problema.

O CSV guarda o rotulo da API **gancho a gancho** (`ocupado_api`), porque e disso
que o aprendizado precisa. Duas colunas existem para nao envenenar esse
aprendizado:

- `api_ok` - quando a API nao responde, a lista de ocupados volta vazia, o que e
  indistinguivel de "carro sem nada pendurado". Essas linhas ficam marcadas e
  sao descartadas no aprendizado.
- `confianca_calib` - com o carro entrando ou saindo, os pontos calibrados nem
  caem sobre os ganchos e o score nao descreve gancho nenhum. Abaixo de 10 a
  captura e considerada cena em transicao e tambem fica de fora.

### A deteccao melhora a cada rodada

```powershell
python local_cv/learn_thresholds.py            # so relatorio
python local_cv/learn_thresholds.py --gravar   # aplica
```

O limiar unico (4.5) veio de uma medicao boa, mas media todos os ganchos. Cada
gancho tem situacao diferente: os da frente ficam contra parede lisa e separam
bem, os do fundo caem numa regiao densa de correntes e pontuam alto ate vazios.
O script varre os limiares possiveis de cada gancho e fica com o que da o melhor
**acerto balanceado** - media entre acerto nos ocupados e nos vazios. Balanceado
porque as classes sao desiguais: um gancho quase sempre vazio teria 90% de
acerto so respondendo "vazio" sempre, e isso e sorte, nao deteccao.

Ele grava `local_cv/limiares_<camera>_<n>.json`, que a tela e o coletor carregam
sozinhos na proxima execucao.

Quando os scores de ocupado e vazio se sobrepoem, **nao existe limiar que
resolva, e o script diz isso** em vez de escolher o menos pior. Esses ganchos
ficam fora do arquivo e continuam aparecendo como INCERTOS.

### A API nao e gabarito de ocupacao fisica

Essa e a limitacao mais importante do projeto hoje.

A API so lista pecas que **tem programa de robo de pintura**. Uma peca sem
programa fica pendurada no gancho e nao aparece com posicao - chega a aparecer
na lista com o campo `hook` vazio, como o registro `6460621 (NEEF) g:` observado
em 05/08. Entao:

| a API diz | vale como |
|---|---|
| gancho OCUPADO | confiavel, tem peca ali |
| gancho ausente da lista | **inconclusivo**, pode ser vazio ou peca sem programa |

Por isso a tela usa tres veredictos em vez de certo/errado:

- **OK** - a leitura local e a API concordam
- **ERRO** - a API afirma ocupado e nao detectamos. Falha nossa, sem desculpa.
- **?** - detectamos ocupado e a API nao lista. Pode ser falso positivo ou peca
  sem programa; daqui nao da para distinguir.

E a taxa acumulada mede so o que da para medir: quantos dos ganchos que a API
**confirma** ocupados foram detectados.

Isso tambem contamina o `learn_thresholds.py`, que precisa de exemplos das duas
classes e acaba tratando "a API nao listou" como vazio. Quando isso estiver
errado, o limiar aprendido sobe demais. Um gancho reprovado por sobreposicao
pode nao ser ilegivel, e sim um gancho que costuma receber peca sem programa. O
script avisa isso ao rodar. Resolver de verdade exige rotulo humano nos casos
vazios ou subtracao de fundo.

### Verificando a sincronia com a API

A API passou a refletir o conteudo da cabine, entao a deteccao local e o estado
da API devem se referir ao mesmo carro. Para conferir isso com dados em vez de
confiar na premissa:

```powershell
python local_cv/analyze_validation.py
```

O script varre atrasos de 0 a 30 minutos e reporta qual faz a deteccao bater
melhor com a API. O esperado e que o melhor resultado esteja no atraso zero; se
o pico aparecer deslocado, alguma coisa ainda esta fora de sincronia. Se nenhum
atraso se destacar da media, ele avisa - resultado sem destaque significa dados
insuficientes, nao resposta.

Nas cameras 26 e 27 a API sempre foi contemporanea da cena, entao
`sync_capture.py`, que dispara no evento da API, ja rotula corretamente aquelas
imagens.

## Tela de monitoramento ao vivo

```powershell
python local_cv/monitor_screen.py --camera cabine
```

Abre uma janela com o video da camera, a deteccao desenhada sobre os ganchos e
um painel lateral com uma linha por gancho:

```
GANCHO   LEITURA    CONF   API
     1   vazio      45/45  OK
     2   OCUPADO    45/45  ?
     3   ???        31/45  ?
```

- **LEITURA** - `OCUPADO`, `vazio`, ou `???` quando nao da para afirmar
- **CONF** - em quantos dos ultimos frames a leitura deu o mesmo resultado.
  `45/45` e leitura firme; `31/45` e gancho oscilando, e o numero mostra isso
  em vez de esconder
- **API** - `OK`, `ERRO` ou `?`, conforme a tabela da secao anterior

As mesmas cores aparecem no video: verde OK, vermelho ERRO, amarelo (anel
vazado, id com `?`) para o caso inconclusivo.

| tecla | acao |
|---|---|
| `q` / ESC | sair |
| `g` | gravar snapshot em `capturas/` |
| `+` / `-` | ajusta o limiar em 0.5 |
| `[` / `]` | ajusta a margem da histerese em 0.25 |
| `a` | liga/desliga a comparacao com a API |
| `j` | mostra/esconde as janelas de analise |
| `p` | pausa |

O ajuste de limiar ao vivo existe para calibrar olhando o resultado: suba ate
comecar a marcar gancho vazio como ocupado, desca ate perder gancho ocupado.

### Por que um gancho parava de piscar so mexendo no limiar

Um gancho cujo score fica em cima do limiar troca de estado a cada frame, e a
leitura vira ruido. Foi o caso do gancho 6. Mexer no limiar nao resolve: o score
dele oscila em volta de qualquer valor que se escolha ali. O que resolve esta em
`local_cv/stability.py`:

1. a decisao usa a **mediana dos ultimos 9 frames**, nao o score instantaneo,
   entao um frame ruim sozinho nao muda nada;
2. **histerese**: so vira OCUPADO acima de `limiar + margem` e so vira VAZIO
   abaixo de `limiar - margem`. Entre os dois mantem o estado anterior, o que
   corta o vai-e-vem de quem esta em cima da linha divisoria;
3. o resultado informa a **certeza**. Quem fica na zona morta aparece como
   INCERTO em vez de receber um sim/nao inventado.

Medido com um gancho oscilando entre 4.2 e 4.8 num limiar de 4.5: antes trocava
de estado a cada leitura, agora fica em zero trocas e marcado INCERTO. Quando o
score sobe de verdade para 9.0, a leitura muda em cerca de 5 frames.

### Cena em transicao

Com o carro entrando ou saindo, a confianca da calibracao despenca (medido: 3.7
a 7.9, contra 15 a 17 nos frames bons) e a leitura vira lixo. Abaixo de
`--min-confianca` (padrao 10) a tela marca **CENA EM TRANSICAO** e considera
todos os ganchos incertos. Nao adianta ler gancho de um carro que esta passando.

### Limitacao conhecida: os ganchos do fundo

Na perspectiva da cabine os ganchos mais distantes ficam espremidos em poucas
dezenas de pixels, numa regiao densa de correntes e estrutura. A deteccao e
confiavel nos ganchos da frente, bem separados, e perde resolucao nos do fundo.
E provavel que `learn_thresholds.py` acabe reprovando varios deles por
sobreposicao - o que e a resposta certa, e nao uma falha do script. Se isso se
confirmar com dados, o caminho e uma segunda vista cobrindo o fundo, e nao
insistir em ajustar limiar.

## Proximos passos (fora do escopo inicial)

- Integrar o resultado de `logs/ganchos.csv` com o Maximo Manage (MXAPI) para
  atualizar status de ativo ou abrir ordem de servico quando fizer sentido -
  ainda nao definido pelo time, entao nao implementado aqui.
- Expor `monitor_hooks.py` como servico continuo (Windows Service / container)
  em vez de rodar manualmente.
- Cruzar `logs/pecas_ganchos.csv` com a posicao fisica dos ganchos detectada
  localmente (`local_cv/detect_local.py`) para validar se a peca que a API diz
  estar no gancho X realmente aparece la na imagem.
