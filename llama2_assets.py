import replicate
import keys
import os

assert "REPLICATE_API_TOKEN" in os.environ.keys()

def prompt_llama2(prompt, max_new_tokens = 500, min_new_tokens = -1, temperature = 0.75, top_p = 0.95, top_k = 250):
    returnArr = []
    output = replicate.run(
        "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
        input={
            "prompt": prompt,
            "max_new_tokens":max_new_tokens,
            "min_new_tokens":min_new_tokens,
            "temperature":temperature,
            "top_p":top_p,
            "top_k":top_k
            }
    )
    for item in output:
        returnArr.append(item)
    return "".join(returnArr)[1:]