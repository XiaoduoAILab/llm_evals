pushd llm_gsm8k && python -W ignore eval.py --model $1 && popd && \
pushd llm_mmlu && python -W ignore eval.py --model $1 && popd && \
pushd llm_tom && python -W ignore eval.py --model $1 && popd && \
pushd llm_math && python -W ignore eval.py --model $1 && popd && \
pushd llm_ceval && python -W ignore eval.py --model $1 && popd && \
pushd llm_human_eval && python -W ignore eval.py --model $1 && popd && \
pushd llm_bbh && python -W ignore eval.py --model $1 && popd
