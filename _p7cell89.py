# cell 89
# ── Phase 7 Cell 7: Loss curve plot ──────────────────────────────────────────
ft_ckpt = load_latest_checkpoint('phase7_ft')
if ft_ckpt and ft_ckpt.get('loss_log'):
    losses = ft_ckpt['loss_log']
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(losses, alpha=0.25, color='steelblue', lw=0.5, label='Raw')
    # EMA smoothing
    ema, val = [], losses[0]
    for l in losses:
        val = 0.05 * l + 0.95 * val
        ema.append(val)
    ax.plot(ema, color='steelblue', lw=2, label='EMA')
    ax.set_xlabel('Step'); ax.set_ylabel('S2TT Cross-Entropy Loss')
    ax.set_title('Phase 7: DoRA Fine-tuning Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, 'phase7_loss.png')
    plt.show()