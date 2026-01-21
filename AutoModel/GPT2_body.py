import inspect
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

init_code = inspect.getsource(model.__init__)
print("init conde: ", init_code)

print("********************************")

forward_code = inspect.getsource(model.forward)
print("forward code: ", forward_code)