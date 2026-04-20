import torch

class VQACollator(object):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        texts = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]
        # TODO
        # Step 1 — Stack images
        images = torch.stack(images)

        # Step 2 — Build input sequences
        input_sequences = []
        for i in range(len(texts)):
            input_sequences.append(texts[i] + answers[i])

        # Step 3 — Batch tokenization
        self.tokenizer.padding_side = "left"
        encoded_full_sequences = self.tokenizer.batch_encode_plus(
            input_sequences, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt")

        # Retrieve `input_ids` (Long tensor) and `attention_mask` from the result.
        input_ids = encoded_full_sequences['input_ids']
        attention_mask = encoded_full_sequences['attention_mask']

        # Step 4 — Create causal labels
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100

        # Step 5 — Per-sample label masking
        original_lengths = [len(self.tokenizer.encode(seq)) for seq in input_sequences]

        for i in range(len(batch)):
            question_length = len(self.tokenizer.encode(texts[i]))

            if original_lengths[i] > self.max_length:
                # Case A: truncated → ignore entirely
                labels[i, :] = -100
            else:
                # Case B: left-padded
                first_token_pos = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][0].item()
                question_end = first_token_pos + question_length - 1
                labels[i, :question_end] = -100

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class MMStarCollator(object):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        questions = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Stack images
        images = torch.stack(images)

        encoded_question_sequences = self.tokenizer.batch_encode_plus(
            questions,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        encoded_answer_sequences = self.tokenizer.batch_encode_plus(
            answers,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        return {
            "images": images,
            "input_ids": encoded_question_sequences['input_ids'],
            "attention_mask": encoded_question_sequences['attention_mask'],
            "labels": encoded_answer_sequences['input_ids'],
        }