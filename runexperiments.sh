#!/bin/bash

# Run experiments for Nodalida'19.


EVALSET=test    # set to "test" for final experiments


set -euo pipefail

SCRIPT="$(basename "$0")"
# https://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WVDIR="$SCRIPTDIR/wordvecs"
DATADIR="$SCRIPTDIR/data"
MODELDIR="$SCRIPTDIR/models"
RESULTDIR="$SCRIPTDIR/results"

# Parameters
EPOCHS=10
MAX_WORDS=100000
WORD_VECS="fi:$WVDIR/wiki.multi.fi.vec,en:$WVDIR/wiki.multi.en.vec"


mkdir -p "$MODELDIR"
mkdir -p "$RESULTDIR"


# Finnish w/subsets
total=$(wc -l < "$DATADIR/fi-train.ft")
for p in `seq 10 10 100`; do
    lines=$((p*total/100));
    shuf "$DATADIR/fi-train.ft" | head -n $lines \
        > "$DATADIR/fi-train-${p}p.ft" || true

    python3 "$SCRIPTDIR/trainmlcnn.py" \
	    --epochs "$EPOCHS" \
	    --limit "$MAX_WORDS" \
	    --word-vectors "$WORD_VECS" \
	    --input "fi:$DATADIR/fi-train-${p}p.ft" \
	    --output "$MODELDIR/fi-${p}p.model"

    python3 "$SCRIPTDIR/testmlcnn.py" \
	    "$MODELDIR/fi-${p}p.model" \
	    "fi:$DATADIR/fi-${EVALSET}.ft" \
	| tee -a "$RESULTDIR/fi-${p}p.txt"
done


for p in `seq 10 10 100`; do
    python3 "$SCRIPTDIR/trainmlcnn.py" \
	    --epochs "$EPOCHS" \
	    --limit "$MAX_WORDS" \
	    --word-vectors "$WORD_VECS" \
	    --input "fi:$DATADIR/fi-train-${p}p.ft,en:$DATADIR/en-train.ft" \
	    --output "$MODELDIR/combo-${p}p.model"

    python3 "$SCRIPTDIR/testmlcnn.py" \
	    "$MODELDIR/combo-${p}p.model" \
	    "fi:$DATADIR/fi-${EVALSET}.ft" \
	| tee -a "$RESULTDIR/combo-fi-${p}p.txt"
done

# Finnish
python3 "$SCRIPTDIR/trainmlcnn.py" \
	--epochs "$EPOCHS" \
	--limit "$MAX_WORDS" \
	--word-vectors "$WORD_VECS" \
	--input "fi:$DATADIR/fi-train.ft" \
	--output "$MODELDIR/fi.model"

python3 "$SCRIPTDIR/testmlcnn.py" \
	"$MODELDIR/fi.model" \
	"fi:$DATADIR/fi-${EVALSET}.ft" \
    | tee -a "$RESULTDIR/fi.txt"


# English
python3 "$SCRIPTDIR/trainmlcnn.py" \
	--epochs "$EPOCHS" \
	--limit "$MAX_WORDS" \
	--word-vectors "$WORD_VECS" \
	--input "en:$DATADIR/en-train.ft" \
	--output "$MODELDIR/en.model"

python3 "$SCRIPTDIR/testmlcnn.py" \
	"$MODELDIR/en.model" \
	"en:$DATADIR/en-${EVALSET}.ft" \
    | tee -a "$RESULTDIR/en.txt"


# English -> Finnish
python3 "$SCRIPTDIR/testmlcnn.py" \
	"$MODELDIR/en.model" \
	"fi:$DATADIR/fi-${EVALSET}.ft" \
    | tee -a "$RESULTDIR/en-fi.txt"


# English+Finnish -> Finnish
python3 "$SCRIPTDIR/trainmlcnn.py" \
	--epochs "$EPOCHS" \
	--limit "$MAX_WORDS" \
	--word-vectors "$WORD_VECS" \
	--input "fi:$DATADIR/fi-train.ft,en:$DATADIR/en-train.ft" \
	--output "$MODELDIR/combo.model"

python3 "$SCRIPTDIR/testmlcnn.py" \
	"$MODELDIR/combo.model" \
	"fi:$DATADIR/fi-${EVALSET}.ft" \
    | tee -a "$RESULTDIR/combo-fi.txt"
