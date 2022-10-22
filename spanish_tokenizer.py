#!/usr/bin/env python
# coding: utf-8

# https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased

# https://github.com/chriskhanhtran/spanish-bert

# In[1]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("skimai/spanberta-base-cased")


# In[2]:


def text_to_ids(
    text,
    ):
    try:
        return tokenizer(
            text,
            return_tensors="pt",
            ).input_ids
    except:
        return None


# text_to_ids("Estoy en la escuela.")

# In[9]:


def ids_to_text(
    ids,
    ):
    try:
        tokens = tokenizer.convert_ids_to_tokens(
            ids[0]
            )
        return tokenizer.convert_tokens_to_string(
            tokens,
        )
    except:
        return None


# ids_to_text(text_to_ids("Estoy en la escuela."))

# tokenizer.pad_token_id

# tokenizer.cls_token_id
