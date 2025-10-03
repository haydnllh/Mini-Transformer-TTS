from tqdm.auto import tqdm 
import torch
import math
from src.utils.generate_stop_target import generate_stop_target

def train_model(model, mel_loss_fn, stop_loss_fn, optimiser, data_loader, device, epochs, target_weight=1, stop_weight=1, scheduler=None, save_dir="trained_models/", batch=9999):
  model.train()
  losses = []

  for epoch in tqdm(range(epochs)):
    total_loss = 0
    total_mel_loss = 0
    total_stop_loss = 0

    for i, (phonemes_padded, phonemes_mask, mels_padded, mels_mask) in enumerate(tqdm((data_loader))):
      if i > batch: break
      phonemes_padded, phonemes_mask, mels_padded, mels_mask = phonemes_padded.to(device), phonemes_mask.to(device), mels_padded.to(device), mels_mask.to(device)
      mels_padded = mels_padded / 100 # Scale down

      mels, stop = model(phonemes_padded, mels_padded, src_key_padding_mask=phonemes_mask, tgt_key_padding_mask=mels_mask)
      stop = stop.squeeze() * ~mels_mask
      mel_loss = mel_loss_fn(mels, mels_padded)

      stop_target = generate_stop_target(mels_mask).to(device)      
      stop_loss = stop_loss_fn(stop, stop_target)
      
      loss = target_weight * mel_loss + stop_weight * stop_loss
      total_loss += loss.item()
      total_mel_loss += mel_loss.item()
      total_stop_loss += stop_loss.item()

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()
      if scheduler is not None:
        scheduler.step()

    total_loss = total_loss
    total_mel_loss = total_mel_loss 
    total_stop_loss = total_stop_loss 
    losses.append(total_loss)
    
    if epoch % (math.ceil(epochs * 0.05)) == 0:
      tqdm.write(f"Epoch {epoch+1} | Total: {total_loss:.4f} | Mel: {total_mel_loss:.4f} | Stop: {total_stop_loss:.4f}")
      
  return losses