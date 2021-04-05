import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from dataset import val_examples, tokenizer
from run_classifier import convert_examples_to_features
from utils import record_log
from main import logger
from config import n_gpu, device, output_eval_file, output_config_file, label_list, \
    output_model_file, eval_batch_size, global_step, tr_loss, nb_tr_steps


# Load a trained model and config that you have fine-tuned
config = BertConfig(output_config_file)
model = BertForSequenceClassification(config, num_labels=len(label_list))
model.load_state_dict(torch.load(output_model_file))
model.to(device)  # important to specific device
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# val
eval_examples = val_examples
eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, "classification")
record_log("Running evaluation", len(eval_examples), eval_batch_size, 0)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

model.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0


def accuracy(logits, label_ids):
    outt = torch.tensor(logits)
    pred = outt.max(dim=-1)[-1]
    target = torch.tensor(label_ids)
    return pred.eq(target).cpu().float().mean().item()


for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
        logits = model(input_ids, segment_ids, input_mask)

    logits = logits.detach().cpu().numpy()
    label_ids = label_ids.to('cpu').numpy()
    tmp_eval_accuracy = accuracy(logits, label_ids)
    # 没有找到accuracy函数是什么，使用sklearn自带的准确率函数
    # tmp_eval_accuracy = accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += input_ids.size(0)
    nb_eval_steps += 1

eval_loss = eval_loss / nb_eval_steps
eval_accuracy = eval_accuracy / nb_eval_examples
loss = tr_loss / nb_tr_steps
result = {'eval_loss': eval_loss,
          'eval_accuracy': eval_accuracy,
          'global_step': global_step,
          'loss': loss}

with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
