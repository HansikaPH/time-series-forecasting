#!/usr/bin/env bash

# invoke all the m4 experiments one after other
./utility_scripts/execution_scripts/m4_finance_experiments.sh
./utility_scripts/execution_scripts/m4_industry_experiments.sh
./utility_scripts/execution_scripts/m4_macro_experiments.sh
./utility_scripts/execution_scripts/m4_micro_experiments.sh
./utility_scripts/execution_scripts/m4_demo_experiments.sh
./utility_scripts/execution_scripts/m4_other_experiments.sh