# Závěrečná práce

Cílem bylo vytvořit algoritmický trading systém, který demonstruje základní dovednosti získané během
kurzu, a dokázat, že rozumím celému procesu od návrhu strategie až po přípravu k nasazení.

Předlohou mi byl trading projekt, který jsme vytvářeli společně na kurzu. Obsahoval všechny důležité koncepty od
jednoduchého stažení dat, po backtesting a následné optimalizaci.
Nicméně mi přišel dosti chaotický a neudržitelný, proto jsem se rozhodl vytvořit vlastní verzi, která bude mít jasnou
strukturu a bude snadno rozšiřitelná.

Kód pravděpodobně není 100% dokonalý, a to i z důvodu, že python není můj primární jazyk. Nicméně i díky Claude code
assistentovi (který byl na kurzu též ukázán) věřím, že jsem vytvořil velice obstojný systém, na kterém se dá stavět
dále.

Spousta funkcí není ještě implementována, ale tento repozitář mám v plánu dále rozvíjet a vylepšovat.
Něktére z TODO věcí:

- Exportéry - exportovat výsledky backtestu do CSV, JSON, nebo databáze. Momentálně pracuji na exportéru do
  Elasticsearche, který mám později v plánu využít pro vizualizaci v Kibaně/Grafaně a taktéž pro trackování live tradingu.
- Live trading - implementovat live trading s možností více brokerů a různých trhů.
- Monitoring - grafy, alerty a další vizualizace pro live trading (grafana)
- Dockerizace - vytvořit docker-compose, který bude obsahovat vše potřebné pro spuštění systému.
- Implementace strategií - implementovat další strategie + zavést deník o jejich výkonnosti.

## Downloaders

- CCXT Downloader - pro stahování historických dat z různých kryptoměnových burz 
  - Validace supportovaných timeframů
  - Podpora různých burz (Binance, Coinbase, Kraken, atd.)
  - Rate limiting
  - Paginace stahování pro velké objemy dat
- Yahoo Finance Downloader - pro stahování historických dat z tradičních trhů
  - Validace supportovaných timeframů
  - Namapování yahoo formátu na dataframe

## Preprocessors

- Odstranění infinit hodnot
- Odstranění NaN hodnot
- Odstranění invalidních OHLCV hodnot
- Zobrazení time gapů
- Odstranění invalidních dat (negativní hodnoty, duplicitní timestampy, atd.)

## Backtesting

- Validace vstupních dat
- Generování signálů dle zvolené strategie
- Vyhodnocení position size a stop loss dle zvoleného risk managementu
- Exekuce obchodů s ohledem na poplatky a risk management
- Výpočet výkonu portfolia a obchodů

## Risk management

- Fixed risk manager - pro řízení velikosti pozice a stop lossu na bázi fixního procenta)
  - Validace vstupních dat včetně warningů při nesmysluplných hodnotách
  - Jednoduchý výpoočet position_size a stop lossu na bázi konfigurace
  - Vyhodnocení zda stop loss byl dosažen a pozice by měla být uzavřena
  - evaluate_trade - vyhodnocuje obchod a vrací rozhodnutí o tom, zda obchod povolit, jaká je velikost pozice, stop loss cena a další
  - Hlídá celé portfolio a vede si historii otevřených pozic
  - Umí vrátit risk metriky

## Optimization

- Validace vstupů
- Ochrana před uživatelským ukončením procesu (proces se dokončí)
- Ochrana paměti (při velkém množství kombinací v paměti zůstanou jen ty nejlepší výsledky a zbytek se zahodí)
- Seřazení výsledků dle zvoleného kritéria (např. Sharpe ratio, celkový zisk, atd.)

- Grid search - pro optimalizaci parametrů pomocí grid search algoritmu
  - Výpočet kombinací a warning pokud je jich příliš mnoho
  - Vygenerování kombinací
  - Vyhodnocení Backtestu pro každou kombinaci (sekvenční, skoro bych přísahal, že jsem implementoval paralelní zpracování, ale
    asi jsem to nedokončil)
- Random search - pro optimalizaci parametrů pomocí náhodného výběru
  - Výpočet kombinací a kalkulace doporočeného počtu trials (pokud nespecifikováno)
  - Vygenerování náhodných kombinací (seed může být specifikován pro replikovatelnost)
  - Vyhodnocení Backtestu (možnost sekvenčního i paralelního zpracování)

## Analysis

- Walk-forward analýza - pro testování robustnosti strategie
  - Validace parametrů a validace strategie parametrů
  - Vytvoření tréninkových a testovacích dat
  - Paralerní/Sekvenční zpracování jednotlivých walk-forward iterací s backtestem
  - Ochrana paměti (při velkém množství iterací se v paměti ponechají jen nejlepší výsledky)
  - Výpočet metrik
- Monte Carlo analýza - pro testování robustnosti strategie
  - Validace parametrů a validace strategie parametrů
  - Vygenerování náhodných sekvencí obchodů (data jsou řazená, takže realistická)
  - Paralerní/Sekvenční zpracování jednotlivých Monte Carlo iterací s backtestem
  - Ochrana paměti (při velkém množství iterací se v paměti ponechají jen nejlepší výsledky)
  - Výpočet metrik

## Scripty

- Slouží jako vstupní bod pro uživatele, konfigurace a spuštění jednotlivých částí frameworku
- Časem by mohlo být nahrazeno za GUI, ale zatím je to jednoduché CLI