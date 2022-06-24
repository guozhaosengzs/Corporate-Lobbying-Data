#!/bin/bash
#$ -N bill_summary_classification
#$ -l m_mem_free=10G
#$ -j y
#$ -m e

module load python/python-3.9.6
module load gcc/gcc-11.1.0
source ../../venv396/bin/activate
python -u summary_classify.py