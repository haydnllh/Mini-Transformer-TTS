from tqdm.auto import tqdm 
import torch
import math
from src.utils.generate_stop_target import generate_stop_target

def train_model(model, mel_loss_fn, stop_loss_fn, optimiser, data_loader, device, epochs, stop_weight=5):
  model.train()
  losses = []

  for epoch in tqdm(range(epochs)):
    total_loss = 0

    for phonemes_padded, phonemes_mask, mels_padded, mels_mask in tqdm(data_loader):
      phonemes_padded, phonemes_mask, mels_padded, mels_mask = phonemes_padded.to(device), phonemes_mask.to(device), mels_padded.to(device), mels_mask.to(device)
      mels_padded = mels_padded / 100 # Scale down

      mels, stop = model(phonemes_padded, mels_padded, src_key_padding_mask=phonemes_mask, tgt_key_padding_mask=mels_mask)
      mel_loss = mel_loss_fn(mels, mels_padded)

      stop_target = generate_stop_target(mels_padded).to(device)
      stop_loss = stop_loss_fn(stop.squeeze(), stop_target)
      
      loss = mel_loss + stop_weight * stop_loss
      total_loss += loss.item()

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()

    if epoch % (math.ceil(epochs * 0.1)) == 0:
      tqdm.write(f"Epoch: {epoch + 1}, Loss: {total_loss :.4f}")

    losses.append(total_loss)

  return losses