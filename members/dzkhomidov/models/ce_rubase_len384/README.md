# ce_rubase_len384

Hugging Face sequence-classification checkpoint used by the
`ce_rubase_e2_len384` matching arm.

The checkpoint itself is `rubase_llmfull_e2`. `len384` is the inference and
fine-tuning token budget, not a different checkpoint: pass `max_length=384`
to the tokenizer. The model has one logit; convert it to probability with a
sigmoid.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = "members/dzkhomidov/models/ce_rubase_len384"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
batch = tokenizer(left_texts, right_texts, truncation=True,
                  max_length=384, padding=True, return_tensors="pt")
probability = model(**batch).logits.sigmoid()
```

`model.safetensors` is stored through Git LFS. Run `git lfs pull` after clone
if the checkout contains an LFS pointer instead of the 711 MB file.

Validation reference: four-fold pooled PR-AUC `0.84515136`, mean fold PR-AUC
`0.84531865`; source result name `ce_rubase_e2_len384`.

See `MANIFEST.json` for exact hashes and sizes.
