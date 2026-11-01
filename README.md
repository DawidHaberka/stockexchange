# stockexchange — czy LLM handluje lepiej niż benchmark?

Eksperyment badawczy (nie produkt). Agent oparty na Claude podejmuje w trakcie sesji decyzje
BUY / SELL / HOLD dla stałej watchlisty, a wynik jest porównywany z:

- **agentem bazowym SMA 20/50** (crossover na świecach 1H, te same dane, ta sama częstotliwość, ten sam risk manager),
- **buy-and-hold SPY** od tego samego momentu startu.

**Bez płatnego API.** Claude jest wywoływany przez **Claude Code CLI** (`claude -p`) zalogowany kontem
z subskrypcją Claude (Pro/Max). Wywołania nie są płatne per token — zużywają limity planu. Raport pokazuje
dodatkowo „równoważnik cen API”, czyli ile kosztowałyby te same wywołania przez płatne API.

> ⚠️ **Wyłącznie paper trading.** Program obsługuje tylko `https://paper-api.alpaca.markets`.
> Każdy inny URL (w tym konto live `api.alpaca.markets`) zatrzymuje program z błędem, zanim
> powstanie jakikolwiek klient. W kodzie nie ma ścieżki do konta live.

---

## Spis treści
1. [Architektura](#architektura)
2. [Koszty](#koszty)
3. [Setup](#setup)
4. [Pierwsze uruchomienie (dry-run)](#pierwsze-uruchomienie-dry-run)
5. [Włączenie zleceń](#włączenie-zleceń)
6. [Wdrożenie na GCP e2-micro z cronem](#wdrożenie-na-gcp-e2-micro-z-cronem)
7. [Konfiguracja i warianty eksperymentu](#konfiguracja-i-warianty-eksperymentu)
8. [Risk manager](#risk-manager)
9. [Raport — jak go czytać](#raport--jak-go-czytać)
10. [Dashboard](#dashboard)
11. [Backtest (tylko SMA)](#backtest-tylko-sma)
12. [Testy](#testy)
13. [Ograniczenia metodologiczne](#ograniczenia-metodologiczne)

---

## Architektura

```
main.py                     CLI: --check | --run | --dry-run [--force] | --report | --backtest-sma
config.yaml                 watchlista, wariant (model/effort/częstotliwość/prompt), triggery, ryzyko, zlecenia
.env                        klucze Alpaca (NIE w repo; wzór: .env.example)
deploy/run.sh               wrapper dla crona (venv + PATH z Claude Code + flock + log)
deploy/crontab.example      harmonogram w strefie America/New_York
trader/
  safety.py                 twardy guard: tylko Alpaca Paper URL
  config.py                 walidacja config.yaml (pydantic), hash wariantu
  broker.py                 wrapper na alpaca-py TradingClient (jedno konto paper)
  scheduler.py              czy bieżący slot (1h/30min/15min) jest należny
  runner.py                 orkiestracja jednego przebiegu
  data/                     świece 1H i 1D, kwotowania, newsy (Alpaca), wskaźniki SMA/RSI/ATR/wolumen
  agent/                    agent LLM + Claude Code CLI + triggery + kontekst; agent SMA; schemat decyzji (pydantic)
  risk/                     risk manager — ostatnie słowo nad agentami
  execution/                zlecenia limit, częściowe realizacje, odrzucenia, stop-lossy, rekoncyliacja
  storage/                  SQLite: decyzje, zlecenia, transakcje, stan portfela, tokeny każdego wywołania
  evaluation/               metryki, raport dzienny, backtest SMA
  dashboard/app.py          Streamlit (tylko odczyt bazy)
tests/                      pytest — Alpaca i Claude zamockowane
```

### Przebieg (`main.py --run`, wywoływany przez crona co 15 min)

1. Guard paper URL → zegar Alpaca → czy rynek otwarty, czy jest ≥ 10:00 NY, czy slot ≤ zamknięcie − 30 min,
   czy bieżący slot (np. 11:00 przy `frequency: 1h`) nie był już przetworzony. Jeśli nie — wyjście bez pracy.
2. **Jednorazowe pobranie danych** dla całej watchlisty (świece 1H — główne; skrót trendu z 1D; quote; newsy).
   Oba agenty widzą identyczny zrzut.
3. Dla każdego agenta (osobne konto paper):
   1. zapis stanu konta, przy pierwszym przebiegu live — **zapis kapitału startowego odczytanego z konta**,
   2. rekoncyliacja: rozliczenie stop-lossów zrealizowanych między przebiegami, anulowanie osieroconych zleceń,
      dopilnowanie, by każda pozycja miała stop-loss na pełną ilość,
   3. **dzienny limit straty** (equity vs zamknięcie poprzedniej sesji) — po przekroczeniu brak handlu do końca sesji,
   4. decyzje agenta → **risk manager** → zapis do bazy → wykonanie (lub symulacja w dry-run),
   5. zapis stanu konta po przebiegu.

### Agent LLM — jak wywołujemy Claude

Dla każdego tickera z triggerem program uruchamia:

```
claude -p --output-format json --model <model> --effort <effort> --system-prompt <prompt z config>
       --json-schema <schemat decyzji> --tools "" --no-session-persistence --safe-mode --strict-mcp-config
```

- prompt (kontekst tickera) idzie przez stdin, wynik wraca jako jeden obiekt JSON,
- `--json-schema` wymusza format `{ticker, action, size_pct, confidence, reasoning}`, a pydantic waliduje go
  drugi raz — **każda niepoprawna odpowiedź, odmowa, błąd lub timeout = HOLD** (z zapisanym powodem),
- `--tools ""` — model nie ma żadnych narzędzi (nie czyta plików, nie szuka w sieci), ocenia tylko podany kontekst,
- `--safe-mode --strict-mcp-config` i pusty katalog roboczy (`runtime/llm_workdir`) — bez CLAUDE.md, hooków,
  pluginów i serwerów MCP, żeby wariant był powtarzalny,
- zmienne `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` są usuwane ze środowiska procesu, więc Claude Code
  **zawsze** używa subskrypcji, nigdy płatnego klucza API.

Model jest wywoływany **tylko** dla tickerów z triggerem (progi w `config.yaml → triggers`):

| Trigger | Warunek |
|---|---|
| `first_evaluation` | ticker nigdy wcześniej nie był oceniany przez LLM |
| `new_news:N` | N nowych newsów od poprzedniego uruchomienia |
| `price_move:k×ATR1H` | \|cena − cena z poprzedniego uruchomienia\| > `price_move_atr_mult` × ATR(1H) |
| `near_stop:k×ATR1H` | pozycja i (cena − stop) ≤ `near_stop_atr_mult` × ATR(1H) |

Pozostałe tickery dostają HOLD bez wywołania (w logu i bazie: `source = no_trigger`, uzasadnienie `"no trigger"`).
Newsy trafiają do promptu jako niezaufane dane w znacznikach `<news>` (z zneutralizowanymi `<`/`>`).

Każde wywołanie trafia do `llm_calls`: model zamówiony i faktycznie użyty, effort, hash promptu, tokeny
(wejście, wyjście z myśleniem, zapis/odczyt cache), równoważnik ceny API, czas, identyfikator, pełny prompt,
surowa odpowiedź i ewentualny błąd.

### Agent SMA (bazowy)

Na świecach 1H: `SMA20 > SMA50` → trzymaj pozycję 10% equity; `SMA20 ≤ SMA50` → zamknij. Gdy kandydatów do kupna
jest więcej niż wolnych miejsc (max 5 pozycji), pierwszeństwo ma największa różnica średnich. Przechodzi przez
ten sam risk manager i to samo wykonanie co LLM — na **osobnym koncie paper**.

---

## Koszty

| Element | Koszt |
|---|---|
| Alpaca paper + dane IEX | 0 zł |
| Claude | w ramach Twojej subskrypcji Claude (Pro/Max) — bez kredytów API; zużywa limity planu |
| GCP e2-micro | w ramach free tier zwykle 0 zł (sprawdź billing w GCP) |

Orientacyjnie: jedno wywołanie Opus to kilka–kilkanaście sekund. Przy `frequency: 1h` i dużych spółkach
(częste newsy) spodziewaj się ok. 25–45 wywołań dziennie. Jeśli limity planu się kończą (błąd `llm_error` w logu,
np. HTTP 429), agent daje HOLD — obniż `llm.effort`, zmień model na `claude-sonnet-5-5` albo zaostrz triggery.

---

## Setup

### 1. Dwa konta Alpaca Paper po $10 000

Alpaca łączy pozycje per ticker, więc agenci muszą mieć **osobne konta** (inaczej nie da się rozdzielić wyników,
gdy obaj trzymają np. AAPL).

1. Zaloguj się na <https://app.alpaca.markets>, przełącz na **Paper Trading**.
2. Utwórz drugie konto paper (przełącznik kont → *Open New Paper Account*).
3. Na **obu** kontach ustaw saldo **$10 000** (ustawienia konta paper → *Reset Account*, kwota 10000).
4. Na każdym koncie wygeneruj klucze API (*API Keys → Generate New Keys*). To są klucze paper.

Kapitał startowy **nie jest zakładany** w kodzie — program odczytuje `equity` z każdego konta przy pierwszym
przebiegu z włączonymi zleceniami i zapisuje go w tabeli `experiment_start`. `--check` ostrzega, jeśli konta mają
różne salda albo jeśli oba klucze wskazują to samo konto.

### 2. Claude Code zalogowany subskrypcją

Na maszynie, na której działa cron (np. VM):

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

```bash
claude auth login
```

Logowanie wyświetli link — otwórz go w przeglądarce, zaloguj się kontem Claude z subskrypcją i wklej kod
z powrotem w terminalu. Sprawdzenie:

```bash
claude auth status
```

Alternatywa bez interaktywnego logowania: `claude setup-token` (na dowolnym komputerze, wymaga subskrypcji)
i wpisanie tokenu do `.env` jako `CLAUDE_CODE_OAUTH_TOKEN=...`.

**Nie ustawiaj `ANTHROPIC_API_KEY`** — program i tak go usuwa przed uruchomieniem `claude`, a `--check`
odrzuca logowanie kluczem API.

### 3. Instalacja projektu

```bash
git clone https://github.com/DawidHaberka/stockexchange.git
```

```bash
cd stockexchange && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
cp .env.example .env && chmod 600 .env
```

Uzupełnij `.env`:

```dotenv
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_LLM_API_KEY=...      # konto paper #1 (agent LLM)
ALPACA_LLM_SECRET_KEY=...
ALPACA_SMA_API_KEY=...      # konto paper #2 (agent SMA)
ALPACA_SMA_SECRET_KEY=...
```

`.env` jest w `.gitignore`. Klucze nigdy nie są logowane.

---

## Pierwsze uruchomienie (dry-run)

Domyślnie `execution.orders_enabled: false` — **nawet `--run` nie wyśle wtedy żadnego zlecenia**.

```bash
.venv/bin/python main.py --check
```

Sprawdza: paper URL, połączenie z oboma kontami (numer, equity, cash, blokady, pozycje, otwarte zlecenia),
zegar i kalendarz sesji, dane (świece 1H/1D, quote, newsy) oraz Claude Code: instalację, logowanie subskrypcją
i jedno testowe wywołanie skonfigurowanego modelu (effort `low`, walidacja odpowiedzi).

```bash
.venv/bin/python main.py --dry-run
```

Pełny przebieg wg harmonogramu — wszystko poza wysłaniem zleceń: dane, decyzje obu agentów (LLM naprawdę
wywołuje Claude), werdykty risk managera, zapis do bazy z flagą `dry_run=1`, symulowane zlecenia.
Poza godzinami sesji przebieg zostanie pominięty — wtedy:

```bash
.venv/bin/python main.py --dry-run --force
```

`--force` ignoruje harmonogram i działa **tylko** z `--dry-run`. Potem raport:

```bash
.venv/bin/python main.py --report
```

Stan triggerów LLM w dry-run jest trzymany osobno (`llm_dry`), więc testowe przebiegi nie „zużywają” pierwszej
oceny tickerów w eksperymencie live. Kapitał startowy jest zapisywany dopiero przy pierwszym przebiegu live —
raport z samych dry-runów jest oznaczony jako **tymczasowy**.

## Włączenie zleceń

Po akceptacji wyników dry-run ustaw w `config.yaml`:

```yaml
execution:
  orders_enabled: true
```

Od tej chwili `--run` składa prawdziwe zlecenia **na kontach paper**. Pierwszy taki przebieg zapisuje kapitał
startowy obu kont i cenę SPY (start benchmarku).

---

## Wdrożenie na GCP e2-micro z cronem

e2-micro (1 vCPU współdzielony, 1 GB RAM) wystarcza; darmowy tier obejmuje regiony `us-west1`, `us-central1`, `us-east1`.

1. VM z Debianem 12, dostęp przez SSH (`gcloud compute ssh <vm>`).
2. Swap (1 GB RAM to mało dla `pip install` i Claude Code jednocześnie):

   ```bash
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

3. Strefa czasowa systemu (Debianowy cron używa strefy systemowej, ignoruje `CRON_TZ`):

   ```bash
   sudo timedatectl set-timezone America/New_York
   ```

4. Python, git i curl:

   ```bash
   sudo apt-get update && sudo apt-get install -y python3-venv git curl
   ```

5. Claude Code + logowanie ([Setup, krok 2](#2-claude-code-zalogowany-subskrypcją)) i projekt
   ([krok 3](#3-instalacja-projektu)) w katalogu `~/stockexchange`. Jeśli `which claude` pokazuje inną ścieżkę niż
   `~/.local/bin`, `~/.npm-global/bin` lub `/usr/local/bin`, wpisz ją w `config.yaml → llm.cli_path`.
6. Cron — skopiuj `deploy/crontab.example` (`crontab -e`):

   ```cron
   */15 9-15 * * 1-5  $HOME/stockexchange/deploy/run.sh --run
   20 16 * * 1-5      $HOME/stockexchange/deploy/run.sh --report
   ```

   Cron odpala program co 15 min; program sam decyduje, czy slot jest należny (zegar Alpaca, święta, skrócone
   sesje, `frequency` z configu). `deploy/run.sh` dodaje Claude Code do `PATH` i używa `flock`, więc przebiegi
   się nie nakładają. Logi: `runtime/logs/trader.log` (aplikacja, rotowany) i `runtime/logs/cron.log`.
7. Kopia zapasowa bazy (opcjonalnie, np. raz dziennie do GCS):

   ```bash
   sqlite3 ~/stockexchange/runtime/trader.db ".backup '/tmp/trader.db'" && gsutil cp /tmp/trader.db gs://<bucket>/trader-$(date +%F).db
   ```

---

## Konfiguracja i warianty eksperymentu

**Wariant** = `llm.model` + `llm.effort` + `experiment.frequency` + `llm.system_prompt`. Przy **każdej** decyzji
zapisywane są: `model`, `effort`, `frequency`, `prompt_hash` (sha256 promptu systemowego) i `config_hash`
(hash całej konfiguracji; pełny JSON w tabeli `config_versions`). Dashboard i SQL mogą więc segmentować wyniki
po wariancie.

| Klucz | Znaczenie |
|---|---|
| `experiment.frequency` | `1h` / `30min` / `15min` — co ile agent podejmuje decyzje w sesji |
| `llm.model` | pełna nazwa, np. `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` (alias typu `opus` przesuwa się przy nowych wersjach) |
| `llm.effort` | `low` / `medium` / `high` / `xhigh` / `max` — głębokość rozumowania (i zużycie limitu) |
| `llm.system_prompt` | prompt systemowy (zmiana = nowy `prompt_hash`) |
| `llm.cli_path` | ścieżka do programu `claude` |
| `llm.cli_extra_args` | dodatkowe flagi CLI (domyślnie `--safe-mode --strict-mcp-config`) |
| `llm.timeout_s`, `llm.max_retries` | limit czasu wywołania; ponowienia przy 429/5xx/timeout |
| `triggers.*` | progi wywołania LLM |
| `risk.*` | limity risk managera |
| `execution.*` | bezpiecznik `orders_enabled`, offset ceny limit, timeout realizacji |

**Dlaczego nie ma `temperature`?** Ani Claude Code CLI, ani bieżąca wersja Claude API nie pozwalają jej
ustawić. Odpowiednikiem „pokrętła” do porównań jest `effort`.

**Porównywanie wariantów:** zmiana wariantu w trakcie eksperymentu miesza wyniki na tych samych kontach.
Czysty sposób: dla nowego wariantu zresetuj oba konta paper do $10 000 i ustaw nowy `storage.db_path`
(np. `runtime/opus-medium-1h.db`), albo segmentuj istniejące dane po `config_hash`.

---

## Risk manager

Ma ostatnie słowo nad **oboma** agentami (`trader/risk/manager.py`):

| Reguła | Zachowanie |
|---|---|
| Watchlista | ticker spoza listy → odrzucone |
| Klasa aktywa | tylko `us_equity` (akcje/ETF), handlowalne; opcje/krypto → odrzucone |
| Bez krótkiej sprzedaży | SELL tylko z istniejącej pozycji, nigdy więcej niż posiadana ilość |
| Bez dźwigni | zakupy tylko z `cash` (z buforem 1%), nie z `buying_power`/marginu |
| Max 10% na pozycję | większe zlecenie jest **przycinane** (`resized`) do limitu |
| Max 5 pozycji | nowe pozycje ponad limit → odrzucone (dokupienie istniejącej — OK) |
| Stop-loss | każda nowa pozycja: stop = cena − 2×ATR; brak ATR → zakup odrzucony |
| Cooldown | po transakcji w tickerze brak kolejnej przez 120 min (także po stop-lossie) |
| Dzienny limit straty | equity ≤ −3% vs zamknięcie poprzedniej sesji → brak handlu do następnej sesji |
| Wiszące zlecenie | w tickerze jest niezrealizowane zlecenie → odrzucone |

Semantyka `size_pct`: **BUY** — % całego equity do dokupienia; **SELL** — % bieżącej pozycji (100 = zamknij).
Handel tylko całymi akcjami (stop-lossy GTC w Alpaca nie obsługują ułamków).

**Stop-loss** to zlecenie `stop` GTC po stronie Alpaca, więc działa także między przebiegami. ATR do stopa:
`risk.stop_loss_atr_timeframe: 1d` (domyślnie, ATR dzienny ≈ 2–5% ceny) lub `1h` (ATR godzinowy — znacznie
ciaśniejszy stop, częstsze wybicia szumem). Przy dokupieniu stop jest przeliczany od średniej ceny wejścia.

**Zlecenia limit**: BUY = ask + 0,10%, SELL = bid − 0,10%, ważne `day`. Program czeka `fill_timeout_s` (120 s)
na realizację, potem anuluje niezrealizowaną resztę; częściowa realizacja jest zapisywana jako transakcja, a stop
obejmuje tylko faktycznie posiadaną ilość. Odrzucenia brokera są zapisywane z powodem.

---

## Raport — jak go czytać

`main.py --report` (cron 16:20 NY) zapisuje stan kont na koniec dnia i zamknięcie SPY, a potem tworzy
`runtime/reports/report_YYYY-MM-DD.md` + `equity_YYYY-MM-DD.png`:

| Metryka | Co oznacza |
|---|---|
| Kapitał startowy / bieżący | equity konta przy starcie live / teraz; SPY = zakup SPY za kapitał startowy LLM |
| Wynik / Total return | zmiana equity od startu (LLM na subskrypcji — bez kosztu za wywołania) |
| Równoważnik cen API LLM | ile kosztowałyby wywołania Claude w przebiegach **live** przez płatne API (z Claude Code) |
| **Wynik przy cenach API** | wynik − równoważnik: czy LLM „zarabiałby na siebie”, gdyby płacić za każde wywołanie |
| Sharpe (ann.) | z dziennych zmian equity, √252, rf z `report.risk_free_rate_annual` |
| Max drawdown | największy spadek od szczytu equity |
| Win rate | % zamkniętych sprzedaży (parowanie FIFO) z dodatnim wynikiem |
| Liczba transakcji | liczba realizacji (w tym stop-lossów) |

Wskazówki interpretacyjne:
- **Kilka dni to szum.** Sharpe i win rate z < ~60 sesji są bardzo niestabilne; porównuj z SPY dopiero po
  kilku miesiącach i patrz na drawdown, nie tylko zwrot.
- LLM, który bije SMA, ale przegrywa z SPY, nie dodaje wartości ponad pasywny indeks.
- Duży „równoważnik cen API” przy małej liczbie transakcji → model jest pytany za często; zaostrz triggery
  albo obniż `effort` (to też oszczędza limity planu).

---

## Dashboard

```bash
.venv/bin/streamlit run trader/dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Na VM nie wystawiaj portu publicznie — użyj tunelu SSH:

```bash
gcloud compute ssh <vm> -- -L 8501:127.0.0.1:8501
```

i otwórz <http://localhost:8501>. Zawiera: tabelę metryk LLM/SMA/SPY, equity curve, log decyzji z filtrami
i rozwijanymi uzasadnieniami (z triggerami, werdyktem risk managera i wariantem), wywołania Claude dziennie
i wg wariantu, zlecenia, transakcje i wersje konfiguracji.

---

## Backtest (tylko SMA)

```bash
.venv/bin/python main.py --backtest-sma --days 365
```

backtesting.py na historycznych świecach 1H, z tymi samymi parametrami co agent live (SMA 20/50, 10% equity,
stop 2×ATR, koszt ~ offset limitu). **Agent LLM nie jest backtestowany**: dane sprzed cutoffu wiedzy modelu
dawałyby look-ahead bias (model „pamięta”, co działo się potem). LLM oceniamy wyłącznie forward, na paper tradingu.
Uwaga: backtesting.py testuje jeden instrument naraz, więc limit 5 pozycji i wspólna gotówka nie są modelowane.

---

## Testy

```bash
.venv/bin/python -m pytest
```

Bez sieci i bez kont. Alpaca jest zamockowana na poziomie HTTP (prawdziwe alpaca-py parsuje odpowiedzi w formacie
API), a Claude Code — fałszywym programem `claude`, który odpowiada w formacie `claude -p --output-format json`
i sprawdza flagi, stdin oraz brak klucza API w środowisku. Pokrycie m.in.: guard paper URL, walidacja JSON
decyzji (niepoprawna = HOLD), każda reguła risk managera, agent LLM (błędy CLI, limity, odmowy, ucięcia,
triggery), harmonogram slotów, egzekucja (pełne/częściowe/brak realizacji, odrzucenia, stop-lossy, rekoncyliacja),
metryki, `--check` i pełne przebiegi dry-run/live z raportem.

---

## Ograniczenia metodologiczne

- **Dane IEX** (darmowy feed) to ułamek wolumenu rynku — wskaźniki wolumenowe są przybliżone. `data.feed: sip`
  wymaga płatnej subskrypcji Alpaca.
- **Świece 1H** mają znaczniki pełnych godzin; świeca 9:00 obejmuje 9:00–10:00, czyli też ostatnie 30 min przed
  otwarciem. Do wskaźników bierzemy tylko zamknięte świece.
- **Limity subskrypcji**: gdy plan się wyczerpie, wywołania kończą się błędem i agent daje HOLD — to są
  „wymuszone” HOLD-y; filtruj je po `source = llm_error`.
- **Wersja Claude Code** może się zmieniać przy aktualizacjach; prompt systemowy jest w całości nasz
  (`--system-prompt`), ale zapisuj wersję z `claude --version` przy ważnych zmianach.
- **Dzienny limit straty** jest sprawdzany na początku każdego przebiegu (co slot); między przebiegami chronią
  pozycje stop-lossy po stronie Alpaca.
- **Limit pozycji** jest liczony konserwatywnie: miejsce po sprzedaży zwalnia się dopiero w następnym przebiegu.
- **Cooldown** działa na podstawie faktycznych realizacji, więc w dry-run (bez realizacji) nie występuje.
- **Watchlista** (megacapy) jest wybrana dziś — to test forward, ale wnioski dotyczą tylko tych spółek.
- Rynek ≠ paper: realizacje na paper są wyidealizowane (brak wpływu na cenę, kolejki itp.).
- Nie handluj ręcznie na kontach eksperymentu — rekoncyliacja zakłada, że zlecenia składa tylko program.
