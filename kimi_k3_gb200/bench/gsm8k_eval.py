"""GSM8K 8-shot CoT accuracy against the running server (/v1/completions, greedy).

Runs inside the leader pod. Usage: python3 gsm8k_eval.py <test.jsonl> [N=300] [concurrency=16]
Uses the standard 8 CoT exemplars (Wei et al.); answer = number after "The answer is".
"""

import concurrent.futures
import json
import re
import sys
import urllib.request

SHOTS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]
PREFIX = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in SHOTS)


def number(text):
    m = re.search(r"The answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    else:
        s = m.group(1)
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def ask(item):
    body = {"model": "moonshotai/Kimi-K3", "prompt": PREFIX + f"Q: {item['question']}\nA:",
            "max_tokens": 320, "temperature": 0, "stop": ["\n\nQ:", "\nQ:"]}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["text"]
    gold = float(item["answer"].split("####")[-1].strip().replace(",", ""))
    pred = number(text)
    return pred is not None and abs(pred - gold) < 1e-6, text


items = [json.loads(l) for l in open(sys.argv[1])][: int(sys.argv[2]) if len(sys.argv) > 2 else 300]
conc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
with concurrent.futures.ThreadPoolExecutor(conc) as pool:
    results = list(pool.map(ask, items))
correct = sum(ok for ok, _ in results)
print(json.dumps({"n": len(items), "correct": correct, "accuracy": correct / len(items),
                  "answers": [t for _, t in results]}))
