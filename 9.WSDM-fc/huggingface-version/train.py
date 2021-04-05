import os
import torch
from tqdm import tqdm, trange
from dataset import train_data, train_dataloader
from model import model, optimizer
from config import device, n_gpu, output_dir, num_train_epochs, train_batch_size, \
    gradient_accumulation_steps, global_step, output_config_file, output_model_file


model.train()
for _ in trange(int(num_train_epochs), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_step = len(train_data) // train_batch_size
    ten_percent_step = total_step // 10
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        if step % ten_percent_step == 0:
            print("Fininshed: {:.2f}% ({}/{})".format(step / total_step * 100, step, total_step))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Save a trained model and the associated configuration
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
torch.save(model_to_save.state_dict(), output_model_file)

with open(output_config_file, 'w') as f:
    f.write(model_to_save.config.to_json_string())
