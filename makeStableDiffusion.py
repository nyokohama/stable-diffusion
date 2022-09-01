# python
from path import Path
import random

import datetime
import pytz
# 現在時刻の文字列を生成する関数（promptが上書きされないように現在時刻を使用）
def get_current_time():
    tdatetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    tstr = tdatetime.strftime('%Y%m%d_%H%M%S')
    return tstr

# 実行コード
def run_prompt(seed, prompt_list, num_inference_steps=30):
    prompt = ", ".join(prompt_list)
    print(seed)
    print("\n".join(prompt_list))
    width = 512
    height = 768 #（立ち絵が欲しい場合には比率を変えた方がきれいに生成できました）
    generator = torch.Generator("cuda").manual_seed(seed)
    with autocast("cuda"):
        image = pipe(prompt, width=width, height=height, num_inference_steps=num_inference_steps, generator=generator)["sample"][0]
    return image

# プロンプト保存用
def save_prompt(out_dir, tstr, prompt):
    with open(out_dir / f"{tstr}_text.txt", "w") as fp:
        fp.write(prompt)

prompt_list = [
"portrait girl in futuristic luxurious golden dress holding a ceremonial sword",
"long curvy hair",
"colourful palette",
"pretty face",
"cute face",
"symmetrical face",
"intimidating expression",
"red eyes",
"anime by greg rutkowski rossdraws makoto shinkai",
"adobe illustrator",
"trending on pixiv",
"behance"
]

out_dir = Path("temp")
tstr = get_current_time() # 現在時刻（適応な識別する文字列でも代用可能）

base_seed = random.randint(0, 9999999999)
#base_seed = 1_000_000_000 # 乱数機で生成した適当な数字を入れてください。
num_inference_steps = 50 # 惜しい絵が出てきた際50などに変更するとより詳細化した絵が出てきます。
n_image = 10 # 生成する数
save_prompt(out_dir, tstr, ", ".join(prompt_list))
for k in range(n_image):
    seed = base_seed + k
    image = run_prompt(seed, prompt_list, num_inference_steps=num_inference_steps)
    display(image)
    # image.save(out_dir / f"{tstr}_{seed:d}.png") # 保存する場合にはコメント解除
