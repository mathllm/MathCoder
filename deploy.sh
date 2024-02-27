model_path="local model path"

max_input_tokens=1536
max_total_tokens=2048

set -x

hostname -I # print the host ip

text-generation-launcher --port 8000 \
--max-batch-prefill-tokens ${max_input_tokens} \
--max-input-length ${max_input_tokens} \
--max-total-tokens ${max_total_tokens} \
--model-id ${model_path}
