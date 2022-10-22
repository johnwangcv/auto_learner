#!/usr/bin/env python
# coding: utf-8

# https://huggingface.co/docs/transformers/model_doc/bert-generation

# In[1]:


import re
import torch


# In[2]:


import transformers


# In[3]:


from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel


# In[4]:


from spanish_tokenizer import *


# In[5]:


device = "cuda:0" if torch.cuda.is_available() else "cpu"


# # from scratch model
# 
# https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel

# # build the model

# In[6]:


class text_to_text_model():
    
    def __init__(
        self,
        ):
        
        self.model = None

    def build_model(
        self,
        ):
        config_encoder = BertConfig()
        config_decoder = BertConfig()

        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
        model = EncoderDecoderModel(config=config)

        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        self.model = model.to(device)

    def train_model(  
        self,
        training_set,
        learning_rate = 1e-3,
        batch_size = 64,
        epochs = 100,
        ):
        
        input_ids = [text_to_ids(text[0]) for text in training_set]
        output_ids = [text_to_ids(text[1]) for text in training_set]
        
        input_ids = torch.cat(input_ids, dim=0)
        output_ids = torch.cat(output_ids, dim=0)        
        
        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate)

        for i in range(epochs):

            #print(f'the {i}-th epoch')
            optimizer.zero_grad()

            loss = self.model(input_ids=input_ids, labels=output_ids).loss
            loss.backward()
            optimizer.step()
            
            if i%30 == 0:
                print(f'{i:04d}-th iteration, loss:{loss:0.4f}')
    
    def inference(
        self,
        text,
        ):
        
        ids = text_to_ids(text)
        ids = ids.to(device)
        
        output_ids = self.model.generate(ids)
        output_text = ids_to_text(output_ids)
        
        try:
            output_text = re.search(r's\>(?P<output_text>[^\<\>]+)\<\/s', output_text).group('output_text')        
        except:
            output_text = None

        return output_text


# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

# # end
