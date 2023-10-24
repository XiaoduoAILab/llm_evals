python -W ignore llm_gsm8k/eval.py --model $1 && \
python -W ignore llm_mmlu/eval.py --model $1 && \
python -W ignore llm_tom/eval.py --model $1 && \
python -W ignore llm_math/eval.py --model $1 && \
python -W ignore llm_ceval/eval.py --model $1 && \
python -W ignore llm_human_eval/eval.py --model $1 && \
python -W ignore llm_bbh/eval.py --model $1
