# cell 95
ft_ckpt = load_latest_checkpoint('phase7_ft')
if ft_ckpt and 'loss_log' in ft_ckpt:
    losses = ft_ckpt['loss_log']
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(losses, alpha=0.3, color='steelblue', lw=0.5, label='Raw')
    ema, val = [], losses[0]
    for l in losses: val = 0.02*l + 0.98*val; ema.append(val)
    ax.plot(ema, color='firebrick', lw=2, label='EMA')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title('Phase 7: Fine-tuning Loss', fontweight='bold')
    ax.legend(); plt.tight_layout(); plt.savefig(f'{FIG_DIR}/phase7_loss.png'); plt.show()