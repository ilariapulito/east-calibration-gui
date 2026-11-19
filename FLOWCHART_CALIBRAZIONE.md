# Flowchart del protocollo di calibrazione

Il diagramma descrive il flusso implementato nella GUI. La calibrazione di
lettura è facoltativa: può assegnare automaticamente i canali ECR/FCR oppure
l'operatore può inserirli manualmente nella pagina di stimolazione.

```mermaid
flowchart TD
    A([Inizio]) --> B{Hardware EAST connesso?}
    B -- No --> B1[Connettere la porta COM]
    B1 --> B
    B -- Sì --> C{Come selezionare i canali ECR/FCR?}

    subgraph READ[1. Calibrazione di lettura EMG]
        C -- Automaticamente --> D[Inserire la sequenza di movimenti<br/>0 = flessione/FCR, 1 = estensione/ECR]
        D --> E[Avviare campionamento<br/>e registrare 16 canali EMG]
        E --> F[Terminare il campionamento]
        F --> G{Sequenza valida<br/>e dati disponibili?}
        G -- No --> G1[Mostrare errore]
        G1 --> D
        G -- Sì --> H[Filtrare EMG 20-450 Hz<br/>e calcolare inviluppo RMS]
        H --> I[Rilevare le attivazioni<br/>e confrontare il numero con la sequenza]
        I --> J{Numero di attivazioni<br/>corrispondente?}
        J -- No --> J1[Azzerare i canali selezionati<br/>e ripetere la registrazione]
        J1 --> D
        J -- Sì --> K[Classificare le attivazioni per tipo<br/>e selezionare i canali ECR e FCR migliori]
    end

    C -- Manualmente --> L[Inserire i canali ECR e FCR<br/>nella pagina di stimolazione]
    K --> M[Usare i canali ECR/FCR selezionati]
    L --> M

    subgraph STIM[2. Calibrazione di stimolazione]
        M --> N[Impostare elettrodi, giri per corrente,<br/>corrente iniziale e incremento, ISI e pulse width]
        N --> O{Parametri ammessi<br/>e canali ECR/FCR distinti?}
        O -- No --> O1[Correggere i parametri]
        O1 --> N
        O -- Sì --> P[Iniziare ricerca Radiale/ECR<br/>sugli elettrodi 1..n/2 arrotondato per difetto]
        P --> Q[Applicare gli impulsi al livello di corrente<br/>e acquisire i 16 canali EMG]
        Q --> R[Terminare il round e analizzare<br/>le risposte evocate]
        R --> S{Risposta valida<br/>sul canale ECR o vicino?}
        S -- Sì --> T[Memorizzare elettrodo Radiale/ECR]
        S -- No --> U{Ci sono altri round<br/>entro il limite di corrente?}
        U -- Sì --> V[Aumentare la corrente]
        V --> Q
        U -- No --> W[Radiale/ECR non trovato<br/>eventuale rotazione del bracciale]
        T --> X[Iniziare ricerca Mediana/FCR<br/>sugli elettrodi floor(n/2)+1..n]
        W --> X
        X --> Y[Applicare gli impulsi al livello di corrente<br/>e acquisire i 16 canali EMG]
        Y --> Z[Terminare il round e analizzare<br/>le risposte evocate]
        Z --> AA{Risposta valida<br/>sul canale FCR o vicino?}
        AA -- Sì --> AB[Memorizzare elettrodo Mediano/FCR]
        AA -- No --> AC{Ci sono altri round<br/>entro il limite di corrente?}
        AC -- Sì --> AD[Aumentare la corrente]
        AD --> Y
        AC -- No --> AE[Mediano non trovato<br/>eventuale rotazione del bracciale]
    end

    AB --> AF[Salvare round e risultati<br/>concludere la calibrazione]
    AE --> AF
    AF --> AG([Fine])
    Q -. Interruzione manuale .-> AH[Arrestare la stimolazione<br/>e salvare il round acquisito]
    Y -. Interruzione manuale .-> AH
    AH --> AG
```

### Note operative

- La calibrazione di lettura è facoltativa. Senza di essa, inserire manualmente
  due canali distinti (1-16) per ECR e FCR nella pagina di stimolazione.
- Se si esegue la calibrazione di lettura, la sequenza deve contenere solo `0`
  e `1`; il numero di burst rilevati deve coincidere con il numero di elementi
  della sequenza.
- I canali ECR e FCR vengono scelti in base all'energia di attivazione; se
  coincidono, viene selezionato un canale FCR alternativo.
- Gli elettrodi sono divisi in due gruppi: il primo per Radiale/ECR e il
  secondo per Mediana/FCR. La dimensione dei gruppi dipende dal numero totale
  di elettrodi configurato.
- La ricerca di stimolazione valuta le risposte sul canale principale prima
  di ricorrere ai canali vicini. Ogni round viene salvato.
- A ogni round la GUI abilita automaticamente la prima finestra ON e calcola
  `timeON` dalla durata del round (elettrodi attivi, giri e ISI), con un
  secondo aggiuntivo per consentire l'arresto ordinato.
- Corrente, incremento, numero massimo di round, ISI e pulse width sono
  configurabili; l'ISI deve essere espresso in secondi interi e la corrente
  non può superare il limite del dispositivo (20 mA).
- Durante i round di stimolazione, la persona deve rimanere ferma. Il pulsante
  di arresto interrompe la procedura.
