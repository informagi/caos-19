### metadata.csv
As of dataset version 2020-04-10, there are 25 lines not adhering to the column
format in `metadata.csv`. See:

    ```bash
    grep -Einv "^[0-9a-z]{8}" metadata.csv | cut -f1 -d":" | tail -n+2
    ```
