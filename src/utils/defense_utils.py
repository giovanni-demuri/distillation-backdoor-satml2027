"""
This file implements the ONION defense and evaluates how the model performs on it after it has been applied
"""
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

def apply_defense(dataset, defense_method):
    if defense_method == "onion":
        dataset = apply_onion_defense(dataset)
    return dataset

def apply_onion_defense(dataset):
    user_questions = []
    for example in dataset:
        example = example["messages"]
        for ex in example:
            if ex["role"] == "user":
                user_questions.append(ex["content"])

    defense = ONIONDefender()
    user_questions_defended = defense.correct(user_questions)

    dataset = dataset.add_column("defended_user_content", user_questions_defended)
    # Apply the transformation
    def replace_user_message(example):
        for msg in example["messages"]:
            if msg["role"] == "user":
                msg["content"] = example["defended_user_content"]
                break
        return example

    dataset = dataset.map(replace_user_message)
    dataset = dataset.remove_columns("defended_user_content")
    return dataset
    

class ONIONDefender():
    r"""
        Defender for `ONION <https://arxiv.org/abs/2011.10369>`_

    Args:
        parallel (`bool`, optional): identify whether to use multiple gpus.
        threshold (`int`, optional): threshold to remove suspicious words.
        batch_size (`int`, optional): batch size of GPTLM.
    """

    def __init__(
        self, 
        parallel = True, 
        threshold = 0, 
        batch_size = 32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.LM = GPT2LM(parallel)
        self.threshold = threshold
        self.batch_size = batch_size

    def correct(
            self,
            poison_data,
    ):
        processed_example = []
        for poison_text in tqdm(poison_data):
            if len(poison_text.split()) > 1:
                process_text = self.get_processed_text(orig_text=poison_text, bar=self.threshold)
                processed_example.append(process_text)
        print('finish onion defend')
        return processed_example


    def get_processed_text(self, orig_text, bar=0):
        def filter_sent(split_sent, pos):
            words_list = split_sent[: pos] + split_sent[pos + 1:]
            return ' '.join(words_list)


        def get_PPL(text):
            """
            Get the perplexity of the text using GPT2
            """
            split_text = text.strip().split(' ')
            text_length = len(split_text)

            processed_sents = [text]
            for i in range(text_length):
                processed_sents.append(filter_sent(split_text, i))

            ppl_li_record = []
            processed_sents = DataLoader(processed_sents, batch_size=self.batch_size, shuffle=False)
            for batch in processed_sents:
                ppl_li_record.extend(self.LM(batch))
            return ppl_li_record[0], ppl_li_record[1:]


        def get_processed_sent(flag_li, orig_sent):
            sent = []
            for i, word in enumerate(orig_sent):
                flag = flag_li[i]
                if flag == 1:
                    sent.append(word)
            return ' '.join(sent)

        # preprocess the original text
        orig_text_split = orig_text.strip().split(' ')
        split_text = []
        for word in orig_text_split:
            if len(word) != 0:
                split_text.append(word)
        orig_text_split = split_text
        orig_text = ' '.join(orig_text_split)
        
        # get the perplexity for the whole sentence, and for each word in the sentence
        whole_sent_ppl, ppl_li_record = get_PPL(orig_text)

        processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

        # check if it above bar or not
        flag_li = []
        for suspi_score in processed_PPL_li:
            if suspi_score >= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)
        
        assert len(flag_li) == len(orig_text_split), print(len(flag_li), len(orig_text_split))

        # remove words if marked as suspicious
        sent = get_processed_sent(flag_li, orig_text_split)
        return sent


"Implementation based on https://github.com/thunlp/OpenBackdoor/blob/main/openbackdoor/defenders/onion_defender.py#L68"
class GPT2LM():
    def __init__(self, parallel):
    
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        if parallel:
            self.lm = torch.nn.DataParallel(self.lm)
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def __call__(self, sents):

        if not isinstance(sents, list):
            sents = [sents]
        for sent in sents:
            sent = sent.lower()
        ipt = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True, 
                            max_length=96, verbose=False).to(self.device)
        output = self.lm(**ipt, labels=ipt.input_ids)
        logits = output[1]
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_labels = ipt.input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        loss = torch.empty((len(sents),))
        for i in range(len(sents)):
            loss[i] = loss_fct(shift_logits[i,:,:].view(-1, shift_logits.size(-1)), shift_labels[i,:].view(-1))
        
        return torch.exp(loss).detach().cpu().numpy()
