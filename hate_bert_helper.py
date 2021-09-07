import sys
import numpy as np
import random as rn
import pandas as pd
import os

import codecs
import itertools
import seaborn as sns

import torch
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output

import io
import matplotlib.pyplot as plt
import csv

from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
import os

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

def run_pairwise_analysis(device, 
                          MAX_LEN, 
                          datasets, 
                          dataset_definitions_list_trained,
                          dataset_definitions_list_to_test):
    for dataset_trained in dataset_definitions_list_trained:
        corpus_id_trained = dataset_trained[0]
        pos_label_trained = dataset_trained[1]
        neg_label_trained = dataset_trained[2]
            
        model_path = "./models/finetuned_model_"+str(corpus_id_trained)+"_"+str(pos_label_trained)
        print(f"Model - {dataset_trained}:\t", end="")
        sys.stdout.flush()
        
        for dataset_tested in dataset_definitions_list_to_test:
            corpus_id_tested = dataset_tested[0]
            pos_label_tested = dataset_tested[1]
            neg_label_tested = dataset_tested[2]
            
            test_dataset = extract_dataset(datasets, 
                                           corpus_id_tested, 
                                           pos_label_tested, 
                                           neg_label_tested)
            f_1_tested = load_and_evaluate(device, 
                  MAX_LEN, 
                  model_path, 
                  test_dataset["text"].values, 
                  test_dataset["label"].values, 
                  pos_label_tested)
            print(f"\t{f_1_tested:.1f}", end="")
            sys.stdout.flush()
        print()

def extract_dataset(datasets, corpus_id, pos_label, neg_label):
    dataset = datasets.loc[(datasets['corpus_id'] == corpus_id) & (datasets['label'].isin([pos_label, neg_label]))]
    return dataset

def encode_label_bin(y, predicted_label):
    choose = lambda l : 1 if l == predicted_label else 0
    return [choose(l) for l in y]

def load_and_evaluate(device, MAX_LEN, model_path, test_sentences, test_y, pos_label):
    test_sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in test_sentences]
    test_y = encode_label_bin(test_y, pos_label)
    
    # Step 2: Re-load the saved model and vocabulary
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(model_path,num_labels=2)
    model.cuda()

    ## LOAD TEST DATA!

    test_tokenized_texts = [tokenizer.tokenize(sent) for sent in test_sentences]

    # Use the tokenizer to convert the tokens to their index numbers in the vocabulary
    test_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in test_tokenized_texts]

    # Pad our input tokens
    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks    
    test_attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in test_input_ids:
      seq_mask = [float(i>0) for i in seq]
      test_attention_masks.append(seq_mask)

    ##

    # Convert all of our data into torch tensors, the required datatype for our model

    test_inputs = torch.LongTensor(test_input_ids)
    test_labels = torch.LongTensor(test_y)
    test_masks = torch.LongTensor(test_attention_masks)

    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    batch_size = 16

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    ##

    ### Testing the model on the test set
    model.eval()
    hatebert_predicted = []
    all_logits = []
    with torch.no_grad():
        for step_num, batch_data in enumerate(test_dataloader):

            token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

            logits = model(token_ids, masks)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits, labels)
            numpy_logits = logits.cpu().detach().numpy()
            #print (numpy_logits)
            #print (np.argmax(numpy_logits, 1))
            hatebert_predicted += list(np.argmax(numpy_logits, 1))
            all_logits += list(numpy_logits[:, 0])

    #print("Accuracy: %.2f%%" % (accuracy_score(hatebert_predicted, test_y)*100))
    #print("F1 macro: %.2f%%" % (f1_score(hatebert_predicted, test_y, average='macro')*100))
    #print("Precission: %.2f%%" % (precision_score(hatebert_predicted, test_y)*100))
    #print("Recall: %.2f%%" % (recall_score(hatebert_predicted, test_y)*100))
    #print(classification_report(hatebert_predicted, test_y))
    #print(confusion_matrix(hatebert_predicted, test_y))
    return f1_score(hatebert_predicted, test_y)*100

def train_and_save(device, MAX_LEN, hatebert_model_path, finetuned_model_output_path, train_sentences, train_y, pos_label):
    train_sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in train_sentences]
    train_y = encode_label_bin(train_y, pos_label)
    
    ### Tokenizer from HateBERT is used here  ### hatebert_model_path: directory where tokenizer is present
    tokenizer = BertTokenizer.from_pretrained(hatebert_model_path, do_lower_case=True)

    train_tokenized_texts = [tokenizer.tokenize(sent) for sent in train_sentences]

    # Use the tokenizer to convert the tokens to their index numbers in the vocabulary
    train_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in train_tokenized_texts]

    # Pad our input tokens
    train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    train_attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in train_input_ids:
      seq_mask = [float(i>0) for i in seq]
      train_attention_masks.append(seq_mask)

    # Use train_test_split to split our data into train and validation sets for training

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_input_ids, train_y, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(train_attention_masks, train_input_ids, random_state=2018, test_size=0.1)

    # Convert all of our data into torch tensors, the required datatype for our model

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    batch_size = 16

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    ## Train Model

    print(str(torch.cuda.memory_allocated(device)/1000000 ) + 'M')

    #Memory release
    y, x, pooled = None, None, None
    torch.cuda.empty_cache()
    print(str(torch.cuda.memory_allocated(device)/1000000 ) + 'M')

    ### The HateBERT model is loaded here  ### hatebert_model_path: directory where tokenizer is present

    model = BertForSequenceClassification.from_pretrained(hatebert_model_path,num_labels=2)
    model.cuda()

    #

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = BertAdam(optimizer_grouped_parameters,lr=2e-5,warmup=.1)

    ## Training loop and Evalution loop:

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    # Time-consuming code here

    # Store our loss and accuracy for plotting
    train_loss_set = []

    Y=[]
    Z=[]
    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 3

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):


      # Training

      # Set our model to training mode (as opposed to evaluation mode)
      model.train()

      # Tracking variables
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0

      # Train the data for one epoch
      for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss.item())    
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()


        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

      print("Train loss: {}".format(tr_loss/nb_tr_steps))


      # Validation

      # Put model in evaluation mode to evaluate loss on the validation set
      model.eval()

      # Tracking variables 
      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0

      # Evaluate data for one epoch
      for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        Y=Y+list(pred_flat)
        Z=Z+list(labels_flat)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

      print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))


    # EVALUATE
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

    print("Accuracy: %.2f%%" % (accuracy_score(Y, Z)*100))
    print("F1 micro: %.2f%%" % (f1_score(Y, Z, average='micro')*100))
    print("F1 macro: %.2f%%" % (f1_score(Y, Z, average='macro')*100))
    print("Precission: %.2f%%" % (precision_score(Y, Z)*100))
    print("Recall: %.2f%%" % (recall_score(Y, Z)*100))
    print(classification_report(Y, Z))
    print(confusion_matrix(Y, Z))

    # SAVE
    # create folder if not there
    Path(finetuned_model_output_path).mkdir(parents=True, exist_ok=True)

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    finetuned_model_output_path_file = os.path.join(finetuned_model_output_path, WEIGHTS_NAME)
    output_config_file = os.path.join(finetuned_model_output_path, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), finetuned_model_output_path_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(finetuned_model_output_path)