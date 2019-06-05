#!/bin/bash

# Run English-Finnish experiments for Nodalida'19.


EVALSET=dev    # set to "test" for final experiments


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
	"en:$DATADIR/fi-${EVALSET}.ft" \
    | tee -a "$RESULTDIR/en-fi.txt"
