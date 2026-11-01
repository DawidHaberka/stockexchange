# stockexchange — czy LLM handluje lepiej niż benchmark?

Eksperyment badawczy (nie produkt). Agent oparty na Claude podejmuje w trakcie sesji decyzje
BUY / SELL / HOLD dla stałej watchlisty, a wynik jest porównywany z:

- **agentem bazowym SMA 20/50** (crossover na świecach 1H, te same dane, ta sama częstotliwość, ten sam risk manager),
- **buy-and-hold SPY** od tego samego momentu startu.

Wynik LLM jest liczony **netto** — po odjęciu kosztu wszystkich wywołań Claude API.

> ⚠️ **Wyłącznie paper trading.** Program obsługuje tylko `https://paper-api.alpaca.markets`.
> Każdy inny URL (w tym konto live `api.alpaca.markets`) zatrzymuje program z błędem, zanim
> powstanie jakikolwiek klient. W kodzie nie ma ścieżki do konta live.

---

## Spis treści
1. [Architektura](#architektura)
2. [Setup](#setup)
3. [Pierwsze uruchomienie (dry-run)](#pierwsze-uruchomienie-dry-run)
4. [Włączenie zleceń](#włączenie-zleceń)
5. [Wdrożenie na GCP e2-micro z cronem](#wdrożenie-na-gcp-e2-micro-z-cronem)
6. [Konfiguracja i warianty eksperymentu](#konfiguracja-i-warianty-eksperymentu)
7. [Risk manager](#risk-manager)
8. [Raport — jak go czytać](#raport--jak-go-czytać)
9. [Dashboard](#dashboard)
10. [Backtest (tylko SMA)](#backtest-tylko-sma)
11. [Testy](#testy)
12. [Ograniczenia metodologiczne](#ograniczenia-metodologiczne)

---

## Architektura

```
main.py                     CLI: --check | --run | --dry-run [--force] | --report | --backtest-sma
config.yaml                 watchlista, wariant (model/effort/częstotliwość/prompt), triggery, ryzyko, zlecenia
.env                        klucze (NIE w repo; wzór: .env.example)
deploy/run.sh               wrapper dla crona (venv + flock + log)
deploy/crontab.example      harmonogram w strefie America/New_York
trader/
  safety.py                 twardy guard: tylko Alpaca Paper URL
  config.py                 walidacja config.yaml (pydantic), hash wariantu
  broker.py                 wrapper na alpaca-py TradingClient (jedno konto paper)
  scheduler.py              czy bieżący slot (1h/30min/15min) jest należny
  runner.py                 orkiestracja jednego przebiegu
  data/                     świece 1H i 1D, kwotowania, newsy (Alpaca), wskaźniki SMA/RSI/ATR/wolumen
  agent/                    agent LLM (Claude) + triggery + kontekst; agent SMA; schemat decyzji (pydantic)
  risk/                     risk manager — ostatnie słowo nad agentami
  execution/                zlecenia limit, częściowe realizacje, odrzucenia, stop-lossy, rekoncyliacja
  storage/                  SQLite: decyzje, zlecenia, transakcje, stan portfela, koszt tokenów
  evaluation/               metryki, raport dzienny, backtest SMA
  dashboard/app.py          Streamlit (tylko odczyt bazy)
tests/                      pytest — wszystkie API zamockowane
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

### Agent LLM — kiedy płacimy za API

Claude jest wywoływany **tylko** dla tickerów z triggerem (progi w `config.yaml → triggers`):

| Trigger | Warunek |
|---|---|
| `first_evaluation` | ticker nigdy wcześniej nie był oceniany przez LLM |
| `new_news:N` | N nowych newsów od poprzedniego uruchomienia |
| `price_move:k×ATR1H` | \|cena − cena z poprzedniego uruchomienia\| > `price_move_atr_mult` × ATR(1H) |
| `near_stop:k×ATR1H` | pozycja i (cena − stop) ≤ `near_stop_atr_mult` × ATR(1H) |

Pozostałe tickery dostają HOLD bez wywołania API (w logu i bazie: `source = no_trigger`, uzasadnienie `"no trigger"`).

Odpowiedź modelu to ściśle określony JSON `{ticker, action, size_pct, confidence, reasoning}` — wymuszony przez
*structured outputs* API, a potem walidowany pydantic. **Każda niepoprawna odpowiedź, odmowa, ucięcie lub błąd API = HOLD**
(z zapisanym powodem). Newsy trafiają do promptu jako niezaufane dane w znacznikach `<news>` (z zneutralizowanymi `<`/`>`).

Każde wywołanie trafia do `llm_calls`: tokeny (wejście, wyjście z myśleniem, zapis/odczyt cache), koszt USD
wg cennika z `config.yaml`, model faktycznie użyty, czas, `request_id`, pełny prompt i surowa odpowiedź.

### Agent SMA (bazowy)

Na świecach 1H: `SMA20 > SMA50` → trzymaj pozycję 10% equity; `SMA20 ≤ SMA50` → zamknij. Gdy kandydatów do kupna
jest więcej niż wolnych miejsc (max 5 pozycji), pierwszeństwo ma największa różnica średnich. Przechodzi przez
ten sam risk manager i to samo wykonanie co LLM — na **osobnym koncie paper**.

---

## Setup

### 1. Dwa konta Alpaca Paper po $10 000

Alpaca łączy pozycje per ticker, więc agenci muszą mieć **osobne konta** (inaczej nie da się rozdzielić wyników,
gdy obaj trzymają np. AAPL).

1. Zaloguj się na <https://app.alpaca.markets>, przełącz na **Paper Trading**.
2. Utwórz drugie konto paper (przełącznik kont → *Open New Paper Account*).
3. Na **obu** kontach zresetuj saldo do **$10 000** (ustawienia konta paper → *Reset Account*, kwota 10000).
4. Na każdym koncie wygeneruj klucze API (*API Keys → Generate*). To są klucze paper — klucze live nie są potrzebne.

Kapitał startowy **nie jest zakładany** w kodzie — program odczytuje `equity` z każdego konta przy pierwszym
przebiegu z włączonymi zleceniami i zapisuje go w tabeli `experiment_start`. `--check` ostrzega, jeśli konta mają
różne salda albo jeśli oba klucze wskazują to samo konto.

### 2. Klucz Claude API

<https://console.anthropic.com> → *API Keys*. Domyślny model: `claude-opus-5-5` (zmiana w `config.yaml`).

### 3. Instalacja

```bash
git clone https://github.com/dawidhaberka/stockexchange.git
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
ANTHROPIC_API_KEY=...
```

`.env` jest w `.gitignore`. Klucze nigdy nie są logowane.

---

## Pierwsze uruchomienie (dry-run)

Domyślnie `execution.orders_enabled: false` — **nawet `--run` nie wyśle wtedy żadnego zlecenia**.

```bash
.venv/bin/python main.py --check
```

Sprawdza: paper URL, połączenie z oboma kontami (numer, equity, cash, blokady, pozycje, otwarte zlecenia),
zegar i kalendarz sesji, dane (świece 1H/1D, quote, newsy) i dostęp do modelu Claude (bez płatnego wywołania).

```bash
.venv/bin/python main.py --dry-run
```

Pełny przebieg wg harmonogramu — wszystko poza wysłaniem zleceń: dane, decyzje obu agentów (LLM naprawdę
wywołuje Claude, więc kosztuje), werdykty risk managera, zapis do bazy z flagą `dry_run=1`, symulowane zlecenia.
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
2. Swap (1 GB RAM bywa za mało przy `pip install`):

   ```bash
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

3. Strefa czasowa systemu (Debianowy cron używa strefy systemowej, ignoruje `CRON_TZ`):

   ```bash
   sudo timedatectl set-timezone America/New_York
   ```

4. Python i repo:

   ```bash
   sudo apt-get update && sudo apt-get install -y python3-venv git
   ```

   Dalej jak w [Instalacji](#3-instalacja), w katalogu `~/stockexchange`.
5. Cron — skopiuj `deploy/crontab.example` (`crontab -e`):

   ```cron
   */15 9-15 * * 1-5  $HOME/stockexchange/deploy/run.sh --run
   20 16 * * 1-5      $HOME/stockexchange/deploy/run.sh --report
   ```

   Cron odpala program co 15 min; program sam decyduje, czy slot jest należny (zegar Alpaca, święta, skrócone
   sesje, `frequency` z configu). `deploy/run.sh` używa `flock`, więc przebiegi się nie nakładają.
   Logi: `runtime/logs/trader.log` (aplikacja, rotowany) i `runtime/logs/cron.log` (stdout crona).
6. Kopia zapasowa bazy (opcjonalnie, np. raz dziennie do GCS):

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
| `llm.model` | np. `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` (+ cennik w `pricing_usd_per_mtok`) |
| `llm.effort` | `low` / `medium` / `high` / `xhigh` / `max` — głębokość rozumowania i koszt |
| `llm.system_prompt` | prompt systemowy (zmiana = nowy `prompt_hash`) |
| `llm.refusal_fallback` | server-side fallback przy odmowie modelu; faktyczny model w `llm_calls.model_served` |
| `triggers.*` | progi wywołania LLM |
| `risk.*` | limity risk managera |
| `execution.*` | bezpiecznik `orders_enabled`, offset ceny limit, timeout realizacji |

**Dlaczego nie ma `temperature`?** Bieżąca wersja Claude API (2026-10-21) odrzuca `temperature`/`top_p`/`top_k`
błędem 400 na każdym modelu, a modele 4.7+ i tak przyjmowały tylko wartość domyślną. Odpowiednikiem
„pokrętła” do porównań jest `effort`.

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
| Total return brutto | zmiana equity od startu |
| Koszt API | suma kosztu wywołań Claude w przebiegach **live** od startu (dry-run pokazany osobno) |
| **Wynik netto** | zysk − koszt API. **Główna miara**: czy LLM zarabia na siebie |
| Sharpe (ann.) | z dziennych zmian equity, √252, rf z `report.risk_free_rate_annual` |
| Max drawdown | największy spadek od szczytu equity |
| Win rate | % zamkniętych sprzedaży (parowanie FIFO) z dodatnim wynikiem |
| Liczba transakcji | liczba realizacji (w tym stop-lossów) |

Wskazówki interpretacyjne:
- **Kilka dni to szum.** Sharpe i win rate z < ~60 sesji są bardzo niestabilne; porównuj netto vs SPY dopiero po
  kilku miesiącach i patrz na drawdown, nie tylko zwrot.
- LLM, który bije SMA, ale przegrywa z SPY netto, nie dodaje wartości ponad pasywny indeks.
- Wysoki koszt API przy małej liczbie transakcji → zaostrz triggery albo obniż `effort`/zmień model.
- Przy $10 000 kapitału koszt API ma realny wpływ: np. $8/mies. to ~0,08% kapitału miesięcznie.

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
i rozwijanymi uzasadnieniami (z triggerami, werdyktem risk managera i wariantem), koszty API dziennie i wg wariantu,
zlecenia, transakcje i wersje konfiguracji.

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

Wszystkie API (Alpaca, Claude) są zamockowane (`tests/conftest.py`). Pokrycie m.in.: guard paper URL, walidacja
JSON decyzji (niepoprawna = HOLD), każda reguła risk managera, agent LLM (błędy API, odmowy, ucięcia, triggery,
koszt), harmonogram slotów, egzekucja (pełne/częściowe/brak realizacji, odrzucenia, stop-lossy, rekoncyliacja),
metryki i pełne przebiegi dry-run/live z raportem.

---

## Ograniczenia metodologiczne

- **Dane IEX** (darmowy feed) to ułamek wolumenu rynku — wskaźniki wolumenowe są przybliżone. `data.feed: sip`
  wymaga płatnej subskrypcji Alpaca.
- **Świece 1H** mają znaczniki pełnych godzin; świeca 9:00 obejmuje 9:00–10:00, czyli też ostatnie 30 min przed
  otwarciem. Do wskaźników bierzemy tylko zamknięte świece.
- **Dzienny limit straty** jest sprawdzany na początku każdego przebiegu (co slot); między przebiegami chronią
  pozycje stop-lossy po stronie Alpaca.
- **Limit pozycji** jest liczony konserwatywnie: miejsce po sprzedaży zwalnia się dopiero w następnym przebiegu.
- **Cooldown** działa na podstawie faktycznych realizacji, więc w dry-run (bez realizacji) nie występuje.
- **Watchlista** (megacapy) jest wybrana dziś — to test forward, ale wnioski dotyczą tylko tych spółek.
- Rynek ≠ paper: realizacje na paper są wyidealizowane (brak wpływu na cenę, kolejki itp.).
- Nie handluj ręcznie na kontach eksperymentu — rekoncyliacja zakłada, że zlecenia składa tylko program.
