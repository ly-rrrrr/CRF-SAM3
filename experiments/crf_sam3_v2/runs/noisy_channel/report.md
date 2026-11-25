# 反向打分验证（noisy-channel）

候选词两个不同、两边都有 J&F 的表达式：241 条。v1 一列的 J&F 来自 `/mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_refytros_valid_query_grounding_mainline_20260701/evaluation_results_refytros_png.json`，v2 一列来自 `runs/refytvos_gv2revos/words/evaluation_results_refytros_png.json`。

| 选法 | J&F |
|---|---|
| 总用 v1 的词 | 57.93 |
| 总用 v2 的词 | 55.62 |
| base：反向（句子总概率） | 57.67 |
| base：反向（每 token 平均） | 57.67 |
| base：正向（只看词） | 57.78 |
| base：反向 + 正向 | 57.26 |
| gv2：反向（句子总概率） | 56.97 |
| gv2：反向（每 token 平均） | 56.97 |
| gv2：正向（只看词） | 56.20 |
| gv2：反向 + 正向 | 56.31 |
| 理想上限（每条都选对） | 67.95 |

两个词的 J&F 差 > 0.2 的明显样本 61 条，各准则选对的条数：

- base：反向（句子总概率）  36 / 61
- base：反向（每 token 平均）  36 / 61
- base：正向（只看词）  37 / 61
- base：反向 + 正向  35 / 61
- gv2：反向（句子总概率）  33 / 61
- gv2：反向（每 token 平均）  33 / 61
- gv2：正向（只看词）  28 / 61
- gv2：反向 + 正向  28 / 61

## 明显样本（gv2，反向句子总概率）

| 表达式 | v1 词 | v2 词 | 反向 logp v1 / v2 | J&F v1 / v2 | 选 |
|---|---|---|---|---|---|
| a puppy dog on the left drinking milk from its mother | dog | puppy | -38.2 / -37.9 | 0.90 / 0.00 | puppy |
| a white puppy near the rear end of an adult dog | dog | puppy | -41.0 / -38.1 | 0.90 / 0.00 | puppy |
| a hat is weared by a person in the center and third in the front row | hat | helmet | -52.4 / -65.6 | 0.00 / 0.93 | hat |
| a black and white helmet worn by a person on a blue bike | hat | helmet | -29.8 / -28.8 | 0.00 / 0.93 | helmet |
| a stuffed white and blue costume | person | lion dance costume | -29.9 / -23.5 | 0.00 / 0.95 | lion dance costume |
| a black and white pointy lizard is sitting on brown rocks | reptile | spiny lizard | -52.0 / -47.5 | 0.79 / 0.00 | spiny lizard |
| a black and white lizard walking on the ground | reptile | spiny lizard | -36.8 / -35.3 | 0.79 / 0.00 | spiny lizard |
| the red harness on the puppy | puppy | harness | -29.7 / -11.0 | 0.02 / 0.40 | harness |
| a giraffe on the right side is being feed by a hand stretching out | camel | giraffe | -53.2 / -45.6 | 0.74 / 0.00 | giraffe |
| a white suv | car | truck | -8.6 / -17.7 | 0.95 / 0.00 | car |
| a watch is weared by a person who controls the system | watch | person | -48.2 / -62.4 | 0.71 / 0.04 | watch |
| a person is operating the disco musical equipment | person | disc jockey | -32.0 / -35.6 | 0.87 / 0.00 | person |
| a panda near a tree | red panda | panda | -27.7 / -20.0 | 0.87 / 0.00 | panda |
| a person is holding a duck and showing a duck photo | duck | person | -38.5 / -44.7 | 0.25 / 0.93 | duck |
| a yellow helmet worn by a person standing in the opposite direction of | hat | helmet | -50.8 / -50.5 | 0.25 / 0.91 | helmet |
| a person in white and red with a yellow helmet | person | cyclist | -30.7 / -31.1 | 0.05 / 0.58 | person |
| the bike is being rode by a person wearing black gloves behind another | bike | person | -33.1 / -46.2 | 0.71 / 0.08 | bike |
| a surfboard is being rode by a man wearing a black shirt and holding a | surfboard | person | -42.1 / -50.2 | 0.60 / 0.07 | surfboard |
| a person paddleboarding | person | paddleboarder | -9.6 / -8.4 | 0.74 / 0.00 | paddleboarder |
| a black hat is being worn by a person riding a horse on the green gras | horse | helmet | -36.9 / -41.6 | 0.00 / 0.96 | horse |
| a table with a sink on the left below a mirror | vanity | sink | -41.0 / -34.9 | 0.51 / 0.31 | sink |
| a horse being ridden by a person in brown blouse and blue jeans riding | horse | person | -45.1 / -56.2 | 0.75 / 0.22 | horse |
| the brown and white horse is being rode by a person wearing a sleevele | horse | person | -50.8 / -53.5 | 0.80 / 0.18 | horse |
| a person in yellow with a black helmet | person | motorcyclist | -28.3 / -33.0 | 0.74 / 0.00 | person |
| the tennis racket is being held by a person wearing a white shirt on t | tennis racket | person | -13.5 / -25.0 | 0.71 / 0.09 | tennis racket |
| a black handbag is being worn by a person on the left side of the road | bag | handbag | -25.8 / -23.4 | 0.91 / 0.15 | handbag |
| a blue sedan is at the bottom of the hill moving to the left | sedan | car | -47.3 / -61.7 | 0.61 / 0.95 | sedan |
| a grey sedan parked in front of a building by the road | sedan | car | -28.9 / -34.0 | 0.61 / 0.95 | sedan |
| the white tissue is on the counter and up against the mirror in the ba | toilet tissue | tissue | -41.8 / -33.4 | 0.98 / 0.25 | tissue |
| tissue paper near a red bottle | toilet paper | tissue paper | -26.3 / -21.0 | 0.98 / 0.49 | tissue paper |
| a small fox like dog on the right side of a sofa | fox | fox terrier | -30.7 / -41.6 | 0.23 / 0.00 | fox |
| a collar on a dog | dog | collar | -18.8 / -9.1 | 0.03 / 0.64 | collar |
| the front part of a motorcycle | motorbike | handlebars | -14.0 / -15.7 | 0.79 / 0.19 | motorbike |
| an off white box of tissue sitting on top of a toilet tank | box | tissue box | -32.5 / -31.1 | 0.25 / 0.82 | tissue box |
| a large monkey sitting in the left foreground with it s mouth open | ape | monkey | -51.4 / -48.4 | 0.68 / 0.00 | monkey |
| the biggest gorilla among the others | ape | gorilla | -29.4 / -26.0 | 0.68 / 0.43 | gorilla |
| a baby ape climbing close to its mother | monkey | baby ape | -30.4 / -22.1 | 0.80 / 0.00 | baby ape |
| the white box is attached to the stone wall in the middle | container | bucket | -37.6 / -39.5 | 0.85 / 0.33 | container |
| a silver box on a stone built wall | container | bucket | -42.8 / -43.1 | 0.85 / 0.33 | container |
| a white hat on the head of a skateboarder | helmet | hat | -24.8 / -15.4 | 0.76 / 0.00 | hat |
