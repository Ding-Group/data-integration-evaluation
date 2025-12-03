#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=gemini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=beaver
#SBATCH --mail-user=yinian@cs.ubc.ca
#SBATCH --mail-type=END,FAIL

source ~/.bashrc
conda activate readpdf

CUT="saucie.pdf"

for pdf in /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/papers/*.pdf; do
    # Skip the loop if no PDF files exist
    [ -e "$pdf" ] || continue

    filename=$(basename "$pdf")
    if [[ "$filename" > "$CUT" ]]; then

    	echo "Processing $pdf ..."
    	python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/survey/ask_gemini.py "$pdf"
    fi
done
