from src.transformers import BlenderbotSmallTokenizer
additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
tokenizer.add_tokens(additional_special_tokens)
comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]
tokenizer.add_tokens(comet_additional_special_tokens)
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
from metric.myMetrics import Metric
hyp_path = "generated_data/hyp_strategy.json"
ref_path = "generated_data/ref_strategy.json"
metric = Metric(toker=tokenizer, hyp_path=hyp_path, ref_path=ref_path)
# print(metric.hyps)
result, result_list = metric.close()
print(result)
print("="*100)
# print(result_list)
