# Generalizzazione GNR: da grafi ad ipergrafi
Questo progetto si concentra sulla costruzione di ipergrafi a partire da dati di espressione genica. L'obiettivo principale è riuscire ad effettuare inferenza delle GNR usando come struttura gli **ipergrafi** in modo tale da estrarre informazioni più complesse sulle interazioni biologiche rispetto ad usare i grafi.
Abbiamo definito una regola di estrazione degli iperarchi da cui poi si costruiscono gli ipergrafi: _per ogni coppia di geni si calcola il coefficiente di correlazione di Pearson, se il valore assoluto supera una certa soglia "t" allora la coppia viene aggiunta allo stesso iperarco._ Ogni iperarco dunque è un **sottoinsieme di geni altamente correlati** rispetto ad una soglia _t_.

## Requisiti
Il codice fa uso delle seguenti librerie:
- `numpy`
- `pandas`
- `torch`
- `torch_geometric`
- `sklearn`
- `json`
- `csv`
- `scipy`
- `joblib`
- `tqdm`

## Struttura del Progetto
- `Costruzione_Hypergraphs.ipynb`: 
  - Carica i dati di espressione genica della GNR per cui si vuole costruire l'ipergrafo
  - Si decidono le soglie "t" per cui costruire l'ipergrafo e si utilizza la funzione `ipergrafiPC` per estrarre gli iperarchi in base alla correlazione di Pearson.
  - Salva il risultato in formato JSON.
- `Test_net1.ipynb`:
  - Carica gli ipergrafi generati tramite `load_hypergraph`.
  - Costruisce i dataset di sotto-ipergrafi positivi utilizzando `create_subgraph_data` e quelli negativi usando `generate_negative_subgraphs`
  - Si costruisce l'addestramento e la valutazione usando k-Fold Cross-Validation (5-fold) specificando gli iperparametri
  - Il modello di machine learning costruito è un classificatore che lavora in input sui sotto-ipergrafi (ob: torch_geometric.data), le specifiche sono nel file `Classifier`

## Funzioni Principali
Piccolo focus sulle funzioni che servono per costruire gli oggetti `torch_geometric.data` che vengono usati dal classificatore per l'addestramento.

### Gestione dei Sotto-Ipergrafi
**Sotto-ipegrafi positivi**:
Il codice estrae e costruisce sotto-ipergrafi basati su dati di espressione genica e una rappresentazione ipergrafica delle relazioni tra i geni.
1. `load_hypergraph(json_file, threshold)`: Carica l'ipergrafo da un file JSON e restituisce gli iperarchi corrispondenti alla soglia specificata, insieme alla lista dei geni unici coinvolti.
2. `calculate_node_features(data, gene_names, unique_genes)`: Calcola feature statistiche per ogni gene unico (media dello z-score, deviazione standard, primo e terzo quartile) utilizzando i dati di espressione genica.
3. `extract_positive_pairs(hyperedges)`: Estrae tutte le coppie di geni che compaiono insieme in almeno un iperarco.
4. `process_pair(pair, hyperedges, unique_genes, expression_data, gene_names, node_features_dict, label)`: Costruisce un sotto-ipergrafo per una coppia di geni, mappando le connessioni tra i nodi e associando le feature calcolate.
5. `create_subgraph_data(json_file, expression_data, gene_names, threshold)`: Gestisce l'intero processo: carica l'ipergrafo, calcola le feature dei nodi, estrae le coppie di geni e genera i sotto-ipergrafi corrispondenti.

**Sotto-ipergrafi negativi**:
1. `process_negative_pair(pair, hyperedges, data, gene_names)`: Costruisce un sotto-ipergrafo negativo per una coppia di geni non direttamente connessi, mappando le connessioni tra i nodi e associando le feature calcolate.
2. `generate_negative_subgraphs(hyperedges, data, gene_names, num_neg_samples)`: Genera un insieme di sotto-ipergrafi negativi selezionando coppie di geni non connessi e processandoli in parallelo.

A finale i due dataset di sotto-ipegrafi vengono uniti per essere usati nella fase di addestramento del modello.

## Utilizzo
1. Eseguire `Costruzione_Hypergraphs.ipynb` per generare il file JSON con gli ipergrafi.
2. Eseguire `Test_net1.ipynb` per caricare i dati e procedere con l'addestramento e valutazione del modello.
