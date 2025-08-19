import matplotlib.pyplot as plt
# --- Lists to store losses ---
train_losses = []
val_losses   = []

# === Training Loop ===
for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for input_img, target_img in train_loader:
        input_img, target_img = input_img.to(device), target_img.to(device)
        optimizer.zero_grad()
        total_loss, mse_val, lp, edge, ssim_val, rng_val = total_loss_fn(
            input_img, target_img, w_mse, w_lpips, w_edge, w_ssim, w_range, lpips_loss_fn
        )
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Validation ---
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for input_img, target_img in val_loader:
            input_img, target_img = input_img.to(device), target_img.to(device)
            total_loss, mse_val, lp, edge, ssim_val, rng_val = total_loss_fn(
                input_img, target_img, w_mse, w_lpips, w_edge, w_ssim, w_range, lpips_loss_fn
            )
            val_running_loss += total_loss.item()
    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- Early Stopping ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_cdan_denseunet.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# === Plot Losses ===
plt.figure(figsize=(10,6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
